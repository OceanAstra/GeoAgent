"""rq1f_run_baselines.py — RQ1 正式验证：在**全量**统一数据上运行八个外部基线（公平性修复版）

与试点轮 (run_fixed_baselines.py) 的差别：
  * 输入来自 RQ2-S2-LARGE-DEV-CONFIRM-v1 的 manifest（train=fit 29,941 / val=calib 6,411 /
    test=dev_confirm 6,417），**sealed 1,065 行完全不参与**；
  * 记录每个基线的**训练时间**与**预测时间**（通过环境变量 RQ1_TIME_FILE 让基线自身写时序；
    未写时序的基线用外层包裹计时，见 collect()）；
  * 保留完整逐样本预测、分数矩阵、运行日志与状态 JSON。

用法：python3 code/rq1f_run_baselines.py --variant 13d --all
      python3 code/rq1f_run_baselines.py --variant 13d SMDPSO-SVM --smdpso-profile full
"""
import os
import sys
import json
import time
import shutil
import argparse
import subprocess

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BL = os.path.join(HERE, 'code', 'baselines_fixed')
LOGS = os.path.join(HERE, 'logs', 'baselines')
RUNS = os.path.join(HERE, 'baseline_out')
PRED = os.path.join(HERE, 'predictions')
for _d in (LOGS, RUNS, PRED):
    os.makedirs(_d, exist_ok=True)

SIX = ['CORROSION', 'EQUIPMENT_FAILURE', 'EXTERNAL_DAMAGE',
       'MATERIAL_DEFECT', 'OPERATIONAL', 'OTHER']

VARIANTS = {
    '13d': os.path.join(HERE, 'data', 'baseline_data_13d'),
    '9d': os.path.join(HERE, 'data', 'baseline_data_9d'),
}

DEFAULT_TIMEOUT = {
    'AUTDL': 3600, 'CIC': 3600, 'DeepPipe': 5400, 'DeepPipeNet': 5400,
    'SART-1DCNN': 5400, 'MFF-CNN-LightGBM': 5400,
    '1DCNN-DBO-LSTM': 10800, 'SMDPSO-SVM': 10800,
}
ALL = list(DEFAULT_TIMEOUT)
CPU_ONLY = {'DeepPipe', 'MFF-CNN-LightGBM'}


def load_rowmap(variant):
    """row_id / y_true / y_true_label / accident_group（事故级统计分组）"""
    rm = pd.read_csv(os.path.join(VARIANTS[variant], 'rowmap_test.csv'))
    # [Tier3 dev] 事故级分组改用**权威新主表** master_v2_consistent.csv
    # (sha256 d9ef45f2...)，不再依赖 RQ2 旧清单 (c6bcde43...)。
    # accident_group 在两版之间逐行未变（见 GeOAD_ReEval_Planning_20261028/results/d8_run_manifest.json
    # 断言 "列 accident_group 与基础清单逐行一致"），故分组统计口径不变。
    man = pd.read_csv(ARCH_ROOT + '/GeOAD_ReEval_Planning_20261028/'
                      'data/master_v2_consistent.csv').set_index('row_id')
    rm = rm.copy()
    rm['accident_group'] = man.loc[rm.row_id, 'accident_group'].to_numpy()
    return rm


def collect(slug, variant, rm, elapsed, detail=None):
    """收集基线的逐样本预测 -> 统一格式 CSV"""
    base = os.path.join(RUNS, variant, slug)
    cands = [os.path.join(base, 'result'), os.path.join(base, 'result', 'baseline')]
    predf = None
    for d in cands:
        if os.path.isfile(os.path.join(d, 'y_pred.npy')):
            predf = os.path.join(d, 'y_pred.npy')
            break
    if predf is None:
        return 'NOT_RUN', '未产出 y_pred.npy'
    pred = np.load(predf).astype(int)
    probaf, proba_kind = None, None
    for d in cands:
        for nm in ['y_pred_proba.npy', 'y_pred_scores.npy']:
            if os.path.isfile(os.path.join(d, nm)):
                probaf, proba_kind = os.path.join(d, nm), nm
                break
        if probaf:
            break
    if probaf is None:
        return 'PARTIAL', '有预测但无概率/决策分数文件'
    P = np.load(probaf)
    score_type = 'probability' if proba_kind == 'y_pred_proba.npy' else 'decision_score'
    if P.ndim == 1:
        P = np.column_stack([-P, P])
    if P.shape[1] != 6:
        return 'PARTIAL', f'{proba_kind} 维度 {P.shape[1]} != 6'
    if len(pred) != len(rm):
        return 'PARTIAL', f'预测行数 {len(pred)} != {len(rm)}'
    agree = float((P.argmax(1) == pred).mean())

    df = pd.DataFrame({
        'row_id': rm.row_id.to_numpy(),
        'accident_group': rm.accident_group.to_numpy(),
        'y_true': rm.y_true.to_numpy(),
        'y_true_label': rm.y_true_label.to_numpy(),
        'pred': pred,
        'pred_label': [SIX[i] for i in pred],
        'correct': (pred == rm.y_true.to_numpy()).astype(int),
    })
    for i, c in enumerate(SIX):
        df[f'score_{c}'] = P[:, i].astype(float)
    df['score_type'] = score_type
    df['argmax_agrees_with_pred'] = agree
    out = os.path.join(PRED, variant, f'{slug}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    return 'RUN', dict(csv=os.path.relpath(out, HERE), score_file=proba_kind,
                       score_type=score_type, argmax_agreement=round(agree, 6),
                       n=len(df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('slugs', nargs='*')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--variant', default='13d', choices=list(VARIANTS))
    ap.add_argument('--timeouts', nargs='*', default=[])
    ap.add_argument('--smdpso-profile', default='full',
                    choices=['full', 'reduced', 'smoke'])
    args = ap.parse_args()

    to = dict(DEFAULT_TIMEOUT)
    for t in args.timeouts:
        k, v = t.split('=')
        to[k] = int(v)

    variant = args.variant
    data_dir = VARIANTS[variant]
    slugs = ALL if args.all else (args.slugs or ALL)
    rm = load_rowmap(variant)

    status_path = os.path.join(HERE, 'data', f'rq1f_run_status_{variant}.json')
    status = json.load(open(status_path, encoding='utf-8')) if os.path.isfile(status_path) else {}
    status.setdefault(variant, {})

    for slug in slugs:
        if slug not in to:
            print(f'!! 未知基线 {slug}', flush=True)
            continue
        d = os.path.join(BL, slug)
        for sub in ('result',):
            p = os.path.join(d, sub)
            if os.path.isdir(p):
                shutil.rmtree(p)
        os.makedirs(os.path.join(d, 'result'), exist_ok=True)
        os.makedirs(os.path.join(d, 'log'), exist_ok=True)

        logp = os.path.join(LOGS, f'{variant}_{slug}.log')
        print(f'=== [{variant}] 运行 {slug} (timeout={to[slug]}s) ===', flush=True)
        t0 = time.time()
        env = dict(os.environ)
        env['TF_CPP_MIN_LOG_LEVEL'] = '3'
        env['PYTHONUNBUFFERED'] = '1'
        env['CUDA_VISIBLE_DEVICES'] = '' if slug in CPU_ONLY else '0'
        env['TF_ENABLE_ONEDNN_OPTS'] = '0'
        env['RQ1_DATA_DIR'] = data_dir
        env['RQ1_SMDPSO_PROFILE'] = args.smdpso_profile
        try:
            p = subprocess.run([sys.executable, 'train.py'], cwd=d, env=env,
                               stdout=open(logp, 'w', encoding='utf-8'),
                               stderr=subprocess.STDOUT, timeout=to[slug])
            rc, st, detail = p.returncode, None, f'exit={p.returncode}'
        except subprocess.TimeoutExpired:
            rc, st, detail = None, 'NOT_RUN', f'超时（>{to[slug]}s）——计算预算阻塞，未用其它算法替代'
        sec = time.time() - t0

        dest = os.path.join(RUNS, variant, slug)
        os.makedirs(dest, exist_ok=True)
        if os.path.isdir(os.path.join(d, 'result')):
            shutil.rmtree(os.path.join(dest, 'result'), ignore_errors=True)
            shutil.move(os.path.join(d, 'result'), os.path.join(dest, 'result'))
        for f in (os.listdir(os.path.join(d, 'log')) if os.path.isdir(os.path.join(d, 'log')) else []):
            shutil.copy2(os.path.join(d, 'log', f), os.path.join(dest, f))

        if st is None:
            if rc == 0:
                st, sdet = collect(slug, variant, rm, sec)
            else:
                st, sdet = 'NOT_RUN', f'进程退出码 {rc}（见 {os.path.relpath(logp, HERE)} 末段）'
        else:
            sdet = detail
        rec = dict(slug=slug, variant=variant, data_dir=os.path.relpath(data_dir, HERE),
                   status=st, seconds=round(sec, 1), timeout_sec=to[slug], detail=sdet,
                   smdpso_profile=(args.smdpso_profile if slug == 'SMDPSO-SVM' else None),
                   log=os.path.relpath(logp, HERE))
        status[variant][slug] = rec
        print(f'  -> {st}  {sdet}  ({sec:.0f}s)', flush=True)
        with open(status_path, 'w', encoding='utf-8') as f:
            json.dump(status, f, ensure_ascii=False, indent=2)

    print('\n汇总：')
    for slug, v in status.get(variant, {}).items():
        print(f'  {slug:20s} {v["status"]:8s} {v.get("seconds")}s  {v.get("detail")}')


if __name__ == '__main__':
    main()
