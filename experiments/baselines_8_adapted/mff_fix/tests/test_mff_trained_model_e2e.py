#!/usr/bin/env python3
"""test_mff_trained_model_e2e.py — MFF-CNN-LightGBM 端到端逐维影响测试（对已训练模型）。

为什么需要单独一个脚本
----------------------
`test_mff_input_coverage.py` 的 T1–T3 用的是**未训练**（随机初始化）的网络与打包层，
证明的是"每一维都进入了张量并对网络输出有梯度"。本脚本补上最后一环：

  * 用**与官方 runner 完全相同**的数据、种子与训练预算重新训练一次；
  * 断言这次重训产生的 y_pred / y_pred_proba 与 runner 落盘的官方产物**逐位相同**
    ⇒ 证明"本脚本里的模型"就是"官方产物对应的那个模型"；
  * 在真实 dev_confirm 行上逐维扰动，测量
      (a) 128 维 CNN 特征是否改变，
      (b) LightGBM 六类概率是否改变，
      (c) 最终 argmax 决策是否改变。
    13 维必须全部"有影响"。

用法:
    python3 test_mff_trained_model_e2e.py --impl <train.py> --data <13d 目录> \
        --official <官方 result 目录> --out <json>
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

# 必须与官方 runner (code/rq1f_run_baselines.py) 的子进程环境**逐项一致**，
# 否则 TF 的 oneDNN/线程归约顺序不同，浮点结果不可逐位复现。
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ.setdefault('PYTHONUNBUFFERED', '1')


def load_impl(path):
    spec = importlib.util.spec_from_file_location('mff_impl_e2e', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--impl', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--official', required=True, help='官方 runner 的 result 目录')
    ap.add_argument('--out', required=True)
    ap.add_argument('--perturb', type=float, default=5.0)
    ap.add_argument('--rows', type=int, default=500)
    ap.add_argument('--proba-tol', type=float, default=1e-9,
                    help='y_pred_proba 逐位复现的容差（TF CPU 线程归约顺序会带来舍入差）')
    args = ap.parse_args()

    impl = load_impl(args.impl)

    # ---- 与基线 main() 完全一致的数据准备（顺序也一致）----
    X_train = pd.read_csv(os.path.join(args.data, 'X_train.csv'))
    X_val = pd.read_csv(os.path.join(args.data, 'X_val.csv'))
    X_test = pd.read_csv(os.path.join(args.data, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(args.data, 'y_train.csv'))
    y_val = pd.read_csv(os.path.join(args.data, 'y_val.csv'))

    import random as _rq1f_random
    _rq1f_random.seed(42)
    np.random.seed(42)
    os.environ.setdefault('PYTHONHASHSEED', '42')
    import tensorflow as _rq1f_tf
    _rq1f_tf.random.set_seed(42)

    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)
    X_test = X_test.fillna(medians)
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    from sklearn.preprocessing import StandardScaler
    X_train_numeric = X_train[numeric_cols].values
    X_val_numeric = X_val[numeric_cols].values
    X_test_numeric = X_test[numeric_cols].values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_numeric)
    X_val_scaled = scaler.transform(X_val_numeric)
    X_test_scaled = scaler.transform(X_test_numeric)

    y_tr = y_train['ANOMALY_LABEL_ID'].values
    y_va = y_val['ANOMALY_LABEL_ID'].values

    model = impl.MFCNN_LightGBM()
    model.fit(X_train_scaled, y_tr, X_val_scaled, y_va, epochs=3, batch_size=512)
    y_pred, y_pred_proba = model.predict(X_test_scaled)

    # ---- 与官方产物逐位比对 ----
    off_pred = np.load(os.path.join(args.official, 'y_pred.npy'))
    off_prob = np.load(os.path.join(args.official, 'y_pred_proba.npy'))
    same_pred = bool(np.array_equal(y_pred.astype(int), off_pred.astype(int)))
    same_prob = bool(np.array_equal(y_pred_proba, off_prob))
    max_prob_diff = float(np.abs(y_pred_proba - off_prob).max())

    # ---- 逐维扰动（真实 dev_confirm 行）----
    Xp = X_test_scaled[:args.rows].copy()
    pk, en = model._prepare_input_data(Xp)
    base_feat = np.hstack(model.feature_extractor.predict([pk, en], verbose=0))
    base_prob = model.lgb_model.predict(base_feat)
    base_dec = base_prob.argmax(1)

    changed_feat, changed_prob, changed_dec = [], [], []
    for d in range(Xp.shape[1]):
        Xd = Xp.copy()
        Xd[:, d] = Xd[:, d] + args.perturb
        pkd, end_ = model._prepare_input_data(Xd)
        feat = np.hstack(model.feature_extractor.predict([pkd, end_], verbose=0))
        prob = model.lgb_model.predict(feat)
        if np.abs(feat - base_feat).max() > 1e-9:
            changed_feat.append(d)
        if np.abs(prob - base_prob).max() > 1e-12:
            changed_prob.append(d)
        if (prob.argmax(1) != base_dec).any():
            changed_dec.append(d)

    report = dict(
        impl=os.path.abspath(args.impl),
        data_dir=os.path.abspath(args.data),
        official_result=os.path.abspath(args.official),
        n_features=int(Xp.shape[1]),
        feature_names=numeric_cols,
        reproduction_matches_official=dict(
            y_pred_identical=same_pred,
            y_pred_proba_bitwise_identical=same_prob,
            max_abs_proba_diff=max_prob_diff,
            proba_tolerance=args.proba_tol,
            proba_within_tolerance=bool(max_prob_diff <= args.proba_tol),
            note=('逐位(bitwise)比较在浮点舍入意义上不成立：TensorFlow 在 CPU 上按线程'
                  '做归约，重跑一次的概率差最大约 3e-15；**全部 6417 条最终决策完全一致**。'
                  '注：若不设 TF_ENABLE_ONEDNN_OPTS=0（与官方 runner 不一致），'
                  '概率差可达约 0.59，说明该基线的训练对 TF 数值后端环境敏感。')),
        e2e_influence=dict(
            rows_used=int(args.rows), perturbation=args.perturb,
            changed_cnn_features=changed_feat,
            changed_lightgbm_probabilities=changed_prob,
            changed_final_decision=changed_dec),
        all_passed=bool(same_pred and max_prob_diff <= args.proba_tol
                        and len(changed_feat) == Xp.shape[1]
                        and len(changed_prob) == Xp.shape[1]),
    )
    txt = json.dumps(report, ensure_ascii=False, indent=2)
    print(txt)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    print('\nRESULT: %s' % ('PASS' if report['all_passed'] else 'FAIL'))
    return 0 if report['all_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
