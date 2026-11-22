"""rq1f_smdpso_9d.py — SMDPSO-SVM 的 9 维补充口径（独立工作副本，不与 13d 主口径抢目录）

与 `rq1f_run_baselines.py --variant 9d` 的唯一差别：**工作目录是独立副本**
（`baseline_out/_9d_smdpso/`），因此可与 13d 主口径的 SMDPSO **并行**运行，
不需要等主口径释放 `code/baselines_fixed/SMDPSO-SVM/`。

训练规则与主口径完全一致：原实现自带的 5,000 行训练上限 + full 搜索档位。
输出：predictions/9d/SMDPSO-SVM.csv、results/rq1f_smdpso_9d.json
"""
import os
import json
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BL = os.path.join(HERE, 'code', 'baselines_fixed')
DATA = os.path.join(HERE, 'data', 'baseline_data_9d')
WORK = os.path.join(HERE, 'baseline_out', '_9d_smdpso')
SLUG = 'SMDPSO-SVM'
PROFILE = os.environ.get('RQ1F_PROFILE', 'full')
SIX = ['CORROSION', 'EQUIPMENT_FAILURE', 'EXTERNAL_DAMAGE',
       'MATERIAL_DEFECT', 'OPERATIONAL', 'OTHER']


def main():
    d = os.path.join(WORK, f'{SLUG}_{PROFILE}')
    if os.path.isdir(d):
        shutil.rmtree(d)
    shutil.copytree(os.path.join(BL, SLUG), d)
    shutil.rmtree(os.path.join(d, 'result'), ignore_errors=True)
    os.makedirs(os.path.join(d, 'result'), exist_ok=True)

    env = dict(os.environ)
    env['TF_CPP_MIN_LOG_LEVEL'] = '3'
    env['PYTHONUNBUFFERED'] = '1'
    env['RQ1_DATA_DIR'] = DATA
    env['RQ1_SMDPSO_PROFILE'] = PROFILE
    env.pop('RQ1_SMDPSO_TRAIN_CAP', None)   # 主口径：原实现自带 5,000 行上限
    logp = os.path.join(WORK, f'{SLUG}_9d_{PROFILE}.log')
    t0 = time.time()
    p = subprocess.run([sys.executable, 'train.py'], cwd=d, env=env,
                       stdout=open(logp, 'w', encoding='utf-8'),
                       stderr=subprocess.STDOUT, timeout=10800)
    el = time.time() - t0
    if p.returncode != 0:
        print(f'!! 运行失败 rc={p.returncode}，见 {logp}', flush=True)
        return

    yp = np.load(os.path.join(d, 'result', 'y_pred.npy')).astype(int)
    sc = np.load(os.path.join(d, 'result', 'y_pred_scores.npy')).astype(float)
    timing = None
    tp = os.path.join(d, 'result', 'timing.json')
    if os.path.isfile(tp):
        timing = json.load(open(tp))

    rm = pd.read_csv(os.path.join(DATA, 'rowmap_test.csv'))
    man = pd.read_csv(ARCH_ROOT + '/GeOAD_RQ2_SecondStage_LargeDevConfirm_20261025/'
                      'protocol/data_row_manifest.csv').set_index('row_id')
    g = man.loc[rm.row_id, 'accident_group'].to_numpy()
    df = pd.DataFrame({'row_id': rm.row_id.to_numpy(), 'accident_group': g,
                       'y_true': rm.y_true.to_numpy(), 'y_true_label': rm.y_true_label.to_numpy(),
                       'pred': yp, 'pred_label': [SIX[i] for i in yp],
                       'correct': (yp == rm.y_true.to_numpy()).astype(int)})
    for i, c in enumerate(SIX):
        df[f'score_{c}'] = sc[:, i]
    df['score_type'] = 'decision_score'
    df['argmax_agrees_with_pred'] = float((sc.argmax(1) == yp).mean())
    out_csv = os.path.join(HERE, 'predictions', '9d', f'{SLUG}.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    from sklearn.metrics import f1_score
    rec = {'slug': SLUG, 'variant': '9d', 'profile': PROFILE,
           'train_cap': 5000, 'seconds': round(el, 1),
           'accuracy': float((yp == rm.y_true.to_numpy()).mean()),
           'macro_f1': float(f1_score(rm.y_true.to_numpy(), yp, average='macro', zero_division=0)),
           'timing': timing, 'csv': os.path.relpath(out_csv, HERE)}
    with open(os.path.join(HERE, 'results', 'rq1f_smdpso_9d.json'), 'w',
              encoding='utf-8') as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print(json.dumps(rec, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
