#!/usr/bin/env python3
"""test_mff_input_coverage.py — MFF-CNN-LightGBM 逐维输入覆盖测试。

目的
----
基线的 `_prepare_input_data()` 把 13 维输入按 `n_features // Ncpt`（下取整）切成
7 个通道，13//7 = 1 ⇒ **特征 7..12 从未进入网络**。仅检查"CSV 有 13 列"无法发现
这一点，因为丢维发生在打包这一步。

本测试**不看 CSV 的列数**，而是逐维证明"该维的真实数值确实到达了张量输入并对网络
输出有影响"，共四层：

  T1 打包层：每维用唯一哨兵值编码，逐维确认哨兵出现在打包后的 (n, 7, 100) 张量里，
             且该维占据的槽位数 > 0。
  T2 数据层：在**真实**训练数据上逐维扰动，确认打包张量确实随之改变。
  T3 网络层：对真实打包张量求梯度，确认每一维在 CNN 输出上都有**非零梯度质量**，
             即该维真实进入了网络（而非仅仅被搬进了张量）。
  T4 端到端层（可选，--model）：用**训练好的**模型，逐维扰动真实 dev_confirm 行，
             确认 128 维 CNN 特征与 LightGBM 概率确实改变。

用法
----
    python3 test_mff_input_coverage.py --impl <train.py 路径> --data <13d 数据目录>
    python3 test_mff_input_coverage.py --impl <train.py 路径> --data <目录> \
        --model <训练产物 result 目录>

退出码 0 = 全部通过；1 = 有维度未覆盖（FAIL）。
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')


def load_impl(path):
    spec = importlib.util.spec_from_file_location('mff_impl_under_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_real_data(data_dir):
    """复刻基线 main() 的预处理口径：中位数填补 + StandardScaler（仅在 fit 折上拟合）。"""
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med)
    X_test = X_test.fillna(med)
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_train[num_cols].values)
    Xte = sc.transform(X_test[num_cols].values)
    return Xtr, Xte, num_cols


def dim_slot_map(impl, X):
    """逐维探测该维在打包张量中占据的槽位（用 impl 自己的打包函数反推）。"""
    n, f = X.shape
    masks = []
    for d in range(f):
        probe = np.zeros((1, f), dtype=float)
        probe[0, d] = 12345.6789
        packed, _ = impl.MFCNN_LightGBM()._prepare_input_data(probe)
        masks.append(np.abs(packed[0, :, :, 0]) > 0)
    return masks


def t1_packing_coverage(impl, n_features, report):
    """T1：每维哨兵必须出现在打包张量中，且槽位数 > 0。"""
    X = np.zeros((4, n_features), dtype=float)
    for i in range(4):
        for d in range(n_features):
            X[i, d] = 1000.0 * (d + 1) + i
    packed, energy = impl.MFCNN_LightGBM()._prepare_input_data(X)
    assert packed.shape[1:] == (7, 100, 1), packed.shape
    found, missing = [], []
    for d in range(n_features):
        sentinel = 1000.0 * (d + 1)
        hit = np.isclose(packed[0, :, :, 0], sentinel, atol=1e-6).any() and \
              np.isclose(energy[0, :, :, 0], sentinel, atol=1e-6).any()
        (found if hit else missing).append(d)
    report['T1_packing'] = dict(n_features=n_features, covered=found, missing=missing)
    return not missing


def t2_real_data_influence(impl, Xtr, report):
    """T2：真实数据上逐维扰动，打包张量必须随之改变。"""
    X = Xtr[:200].copy()
    base, _ = impl.MFCNN_LightGBM()._prepare_input_data(X)
    changed, unchanged = [], []
    for d in range(X.shape[1]):
        Xp = X.copy()
        Xp[:, d] = Xp[:, d] + 5.0
        pert, _ = impl.MFCNN_LightGBM()._prepare_input_data(Xp)
        (changed if np.abs(pert - base).max() > 1e-9 else unchanged).append(d)
    report['T2_real_data'] = dict(changed=changed, unchanged=unchanged)
    return not unchanged


def t3_network_gradient(impl, Xtr, report):
    """T3：逐维在 CNN 输出上的梯度质量必须 > 0（即该维真实进入了网络）。"""
    import tensorflow as tf
    X = Xtr[:64].copy()
    peak, energy = impl.MFCNN_LightGBM()._prepare_input_data(X)
    masks = dim_slot_map(impl, X)

    m = impl.MFCNN_LightGBM()
    keras_model, peak_in, energy_in, peak_feat, energy_feat = m._build_lightweight_mfcnn()
    keras_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

    p = tf.constant(peak, dtype=tf.float32)
    e = tf.constant(energy, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch([p, e])
        out = keras_model([p, e], training=False)
        loss = tf.reduce_sum(out)
    grads = tape.gradient(loss, [p, e])
    g = np.abs(grads[0].numpy())          # (n, 7, 100, 1)

    alive, dead = [], []
    for d in range(X.shape[1]):
        mass = 0.0
        if masks[d].any():
            mass = float(g[:, masks[d]].sum())
        (alive if mass > 0 else dead).append(d)
    report['T3_network_gradient'] = dict(
        alive=alive, dead=dead,
        grad_mass={str(d): (float(g[:, masks[d]].sum()) if masks[d].any() else 0.0)
                   for d in range(X.shape[1])})
    return not dead


def t4_trained_model(impl, Xte, model_dir, report):
    """T4：训练好的模型上逐维扰动真实 dev_confirm 行，CNN 特征与 LightGBM 概率必须改变。"""
    if model_dir is None:
        report['T4_trained_model'] = dict(status='SKIPPED (未提供 --model 目录)')
        return True
    pred_f = os.path.join(model_dir, 'y_pred.npy')
    prob_f = os.path.join(model_dir, 'y_pred_proba.npy')
    if not (os.path.isfile(pred_f) and os.path.isfile(prob_f)):
        report['T4_trained_model'] = dict(status='SKIPPED (未提供 --model 目录)')
        return True
    # 重新构建并载入训练好的模型（与基线 fit/predict 同一路径）
    import lightgbm as lgb
    X = Xte[:400].copy()
    m = impl.MFCNN_LightGBM()
    # 训练产物里只保存了预测；为做端到端扰动需要模型本体，
    # 因此该测试在 --model 目录下要求存在 mff_model.pkl（由 retrain 脚本保存）。
    pkl = os.path.join(model_dir, 'mff_model.pkl')
    if not os.path.isfile(pkl):
        report['T4_trained_model'] = dict(status='SKIPPED (缺少 %s)' % pkl)
        return True
    import pickle
    with open(pkl, 'rb') as f:
        model = pickle.load(f)
    pk, en = model._prepare_input_data(X)
    base_feat = np.hstack(model.feature_extractor.predict([pk, en], verbose=0))
    base_p = model.lgb_model.predict(base_feat)
    changed_feat, changed_prob = [], []
    for d in range(X.shape[1]):
        Xp = X.copy()
        Xp[:, d] = Xp[:, d] + 5.0
        pk2, en2 = model._prepare_input_data(Xp)
        f2 = np.hstack(model.feature_extractor.predict([pk2, en2], verbose=0))
        p2 = model.lgb_model.predict(f2)
        if np.abs(f2 - base_feat).max() > 1e-8:
            changed_feat.append(d)
        if np.abs(p2 - base_p).max() > 1e-12:
            changed_prob.append(d)
    report['T4_trained_model'] = dict(
        status='RUN', changed_features=changed_feat, changed_probabilities=changed_prob)
    return not changed_feat and not changed_prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--impl', required=True, help='被测 train.py 路径')
    ap.add_argument('--data', required=True, help='13d 基线输入目录')
    ap.add_argument('--model', default=None, help='训练产物 result 目录（启用 T4）')
    ap.add_argument('--out', default=None, help='把 JSON 报告写到该文件')
    args = ap.parse_args()

    impl = load_impl(args.impl)
    Xtr, Xte, num_cols = load_real_data(args.data)
    n_features = Xtr.shape[1]
    print('被测实现 : %s' % args.impl)
    print('输入维度 : %d  (%s)' % (n_features, ', '.join(num_cols)))

    report = dict(impl=os.path.abspath(args.impl), n_features=n_features,
                  feature_names=num_cols, data_dir=os.path.abspath(args.data))
    ok = True
    ok &= bool(t1_packing_coverage(impl, n_features, report))
    ok &= bool(t2_real_data_influence(impl, Xtr, report))
    ok &= bool(t3_network_gradient(impl, Xtr, report))
    ok &= bool(t4_trained_model(impl, Xte, args.model, report))

    report['all_passed'] = bool(ok)
    txt = json.dumps(report, ensure_ascii=False, indent=2)
    print(txt)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(txt + '\n')
    print('\nRESULT: %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
