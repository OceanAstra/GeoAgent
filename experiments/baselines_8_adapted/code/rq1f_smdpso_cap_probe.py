"""rq1f_smdpso_cap_probe.py — SMDPSO-SVM「训练样本上限敏感性」补充探针

主口径用原实现自带的上限 5,000 行（全量 29,941 行 ⇒ 只覆盖 17%）。
本脚本用**同一套代码、同一搜索档位**，把上限放宽到 10,000 行再跑一遍，
用于回答"该基线的成绩是否由 5,000 行上限造成"。

**这是补充探针，不进入主结果表**，单独输出到 predictions/supplementary/。

输出：predictions/supplementary/SMDPSO-SVM-cap10000.csv
      results/rq1f_smdpso_cap_probe.json
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
DATA = os.path.join(HERE, 'data', 'baseline_data_13d')
OUT = os.path.join(HERE, 'predictions', 'supplementary')
WORK = os.path.join(HERE, 'baseline_out', '_cap_probe')
SLUG = 'SMDPSO-SVM'
CAP = os.environ.get('RQ1F_CAP', '10000')
PROFILE = os.environ.get('RQ1F_PROFILE', 'full')
SIX = ['CORROSION', 'EQUIPMENT_FAILURE', 'EXTERNAL_DAMAGE',
       'MATERIAL_DEFECT', 'OPERATIONAL', 'OTHER']


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    d = os.path.join(WORK, f'{SLUG}_{PROFILE}')
    if os.path.isdir(d):
        shutil.rmtree(d)
    shutil.copytree(os.path.join(BL, SLUG), d)
    for sub in ('result',):
        p = os.path.join(d, sub)
        if os.path.isdir(p):
            shutil.rmtree(p)
    os.makedirs(os.path.join(d, 'result'), exist_ok=True)

    env = dict(os.environ)
    env['TF_CPP_MIN_LOG_LEVEL'] = '3'
    env['PYTHONUNBUFFERED'] = '1'
    env['RQ1_DATA_DIR'] = DATA
    env['RQ1_SMDPSO_PROFILE'] = PROFILE
    env['RQ1_SMDPSO_TRAIN_CAP'] = CAP
    logp = os.path.join(WORK, f'{SLUG}_cap{CAP}_{PROFILE}.log')
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
    out_csv = os.path.join(OUT, f'{SLUG}-cap{CAP}-{PROFILE}.csv')
    df.to_csv(out_csv, index=False)

    from sklearn.metrics import f1_score
    rec = {'slug': f'{SLUG}-cap{CAP}', 'train_cap': int(CAP), 'profile': PROFILE,
           'n_train_actual': int(CAP), 'seconds': round(el, 1),
           'accuracy': float((yp == rm.y_true.to_numpy()).mean()),
           'macro_f1': float(f1_score(rm.y_true.to_numpy(), yp, average='macro', zero_division=0)),
           'timing': timing, 'csv': os.path.relpath(out_csv, HERE),
           'note': '补充探针，不进入主结果表'}
    with open(os.path.join(HERE, 'results', 'rq1f_smdpso_cap_probe.json'), 'w',
              encoding='utf-8') as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print(json.dumps(rec, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
