"""rq1f_smdpso_reduced_fallback.py — SMDPSO-SVM 的 reduced 档备用运行（计算限制预案）

用途：主口径的 **full 档** PSO 搜索若在 3 小时预算内未能完成（超时），
按指令 §4「允许沿用已有 reduced 档，但必须明确标注计算限制」，
用 **reduced 档**（适应度子采样 400 / 验证子采样 150，粒子 20 × 迭代 30 不变）
在同一份全量统一输入上重跑 SMDPSO-SVM，得到可用的主表成绩。

与主口径的唯一差别：**搜索档位**（计算预算），不是训练数据、不是特征、不是评价口径。

* 若 full 档已成功 ⇒ 本脚本的产物**只作为补充**，不进入主表。
* 若 full 档超时 ⇒ 本脚本的产物进入主表，并在文档中明确标注
  「reduced 档 / 计算限制」。

输出：predictions/supplementary/SMDPSO-SVM-reduced13d.csv
      results/rq1f_smdpso_reduced_fallback.json
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
WORK = os.path.join(HERE, 'baseline_out', '_reduced_fallback')
SLUG = 'SMDPSO-SVM'
SIX = ['CORROSION', 'EQUIPMENT_FAILURE', 'EXTERNAL_DAMAGE',
       'MATERIAL_DEFECT', 'OPERATIONAL', 'OTHER']


def main():
    d = os.path.join(WORK, SLUG)
    if os.path.isdir(d):
        shutil.rmtree(d)
    shutil.copytree(os.path.join(BL, SLUG), d)
    shutil.rmtree(os.path.join(d, 'result'), ignore_errors=True)
    os.makedirs(os.path.join(d, 'result'), exist_ok=True)

    env = dict(os.environ)
    env['TF_CPP_MIN_LOG_LEVEL'] = '3'
    env['PYTHONUNBUFFERED'] = '1'
    env['RQ1_DATA_DIR'] = DATA
    env['RQ1_SMDPSO_PROFILE'] = 'reduced'
    env.pop('RQ1_SMDPSO_TRAIN_CAP', None)
    logp = os.path.join(WORK, f'{SLUG}_reduced13d.log')
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
    out_csv = os.path.join(HERE, 'predictions', 'supplementary', f'{SLUG}-reduced13d.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    from sklearn.metrics import f1_score
    rec = {'slug': f'{SLUG}-reduced', 'variant': '13d', 'profile': 'reduced',
           'train_cap': 5000, 'seconds': round(el, 1),
           'accuracy': float((yp == rm.y_true.to_numpy()).mean()),
           'macro_f1': float(f1_score(rm.y_true.to_numpy(), yp, average='macro', zero_division=0)),
           'timing': timing, 'csv': os.path.relpath(out_csv, HERE),
           'note': 'reduced 档（计算限制预案）；full 档若成功则本行仅作补充，不进入主表'}
    with open(os.path.join(HERE, 'results', 'rq1f_smdpso_reduced_fallback.json'), 'w',
              encoding='utf-8') as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print(json.dumps(rec, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
