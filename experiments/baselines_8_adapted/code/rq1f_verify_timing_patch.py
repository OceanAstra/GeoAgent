"""rq1f_verify_timing_patch.py — 证明"加计时插桩"**不改变任何基线的输出**

方法：把试点数据（3,000/500/1,000）喂给**加过计时插桩**的 train.py，
      逐个基线与试点轮已归档的逐样本预测做**逐行逐类**比对：
        - 完全一致（预测标签与分数） => 插桩行为中性；
        - 有差异 => 插桩影响了行为，必须排查。

注意：随机性来源（torch/np 种子、DBO/PSO 采样、DataLoader shuffle）会让"逐值完全相同"
      不总是成立，因此分两级判定：
        L1 硬判定：y_pred 完全一致（准确率相同且预测标签逐行相同）；
        L2 参考判定：准确率差 <= 0.002（属随机性抖动范围）。
      L1 不通过但 L2 通过的，标为 RANDOM_TOLERABLE 并在报告中说明。

输出：results/rq1f_timing_patch_verification.json
"""
import os
import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOT = ARCH_ROOT + '/GeOAD_RQ1_EightBaselines_Pilot_20261025'
BL = os.path.join(HERE, 'code', 'baselines_fixed')
PILOT_DATA = os.path.join(PILOT, 'data', 'baseline_data_13d')
PILOT_PRED = os.path.join(PILOT, 'predictions_fair', '13d')
WORK = os.path.join(HERE, 'baseline_out', '_timing_verify')

ALL = ['AUTDL', 'CIC', 'DeepPipe', 'DeepPipeNet', 'SART-1DCNN',
       'MFF-CNN-LightGBM', '1DCNN-DBO-LSTM', 'SMDPSO-SVM']
CPU_ONLY = {'DeepPipe', 'MFF-CNN-LightGBM'}
TIMEOUT = 5400


def main():
    os.makedirs(WORK, exist_ok=True)
    out = {}
    for slug in ALL:
        ref_path = os.path.join(PILOT_PRED, f'{slug}.csv')
        if not os.path.isfile(ref_path):
            out[slug] = {'status': 'NO_REFERENCE'}
            print(f'[{slug}] 无试点参照，跳过', flush=True)
            continue
        ref = pd.read_csv(ref_path).reset_index(drop=True)
        rm_pilot = pd.read_csv(os.path.join(PILOT_DATA, 'rowmap_test.csv'))
        order_ok = bool((ref.row_id.values == rm_pilot.row_id.values).all())

        d = os.path.join(BL, slug)
        for sub in ('result',):
            p = os.path.join(d, sub)
            if os.path.isdir(p):
                shutil.rmtree(p)
        os.makedirs(os.path.join(d, 'result'), exist_ok=True)

        env = dict(os.environ)
        env['TF_CPP_MIN_LOG_LEVEL'] = '3'
        env['PYTHONUNBUFFERED'] = '1'
        env['CUDA_VISIBLE_DEVICES'] = '' if slug in CPU_ONLY else '0'
        env['RQ1_DATA_DIR'] = PILOT_DATA
        env['RQ1_SMDPSO_PROFILE'] = 'reduced'  # 与试点轮实际档一致
        logp = os.path.join(WORK, f'{slug}.log')
        t0 = __import__('time').time()
        try:
            p = subprocess.run([sys.executable, 'train.py'], cwd=d, env=env,
                               stdout=open(logp, 'w', encoding='utf-8'),
                               stderr=subprocess.STDOUT, timeout=TIMEOUT)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = 'TIMEOUT'
        el = __import__('time').time() - t0

        # 找到产出
        new = None
        for cand in [os.path.join(d, 'result'), os.path.join(d, 'result', 'baseline')]:
            if os.path.isfile(os.path.join(cand, 'y_pred.npy')):
                new = cand
                break
        if new is None or rc != 0:
            out[slug] = {'status': 'RUN_FAILED', 'rc': str(rc), 'seconds': round(el, 1),
                         'log': os.path.relpath(logp, HERE)}
            print(f'[{slug}] 运行失败 rc={rc}（{el:.0f}s）', flush=True)
            continue

        yp = np.load(os.path.join(new, 'y_pred.npy')).astype(int)
        timing = None
        for cand in [new, os.path.join(d, 'result')]:
            tp = os.path.join(cand, 'timing.json')
            if os.path.isfile(tp):
                timing = json.load(open(tp))
                break

        rp = ref.pred.to_numpy().astype(int)
        same = (len(yp) == len(rp)) and bool((yp == rp).all())
        acc_new = float((yp == ref.y_true.to_numpy()).mean())
        acc_ref = float(ref.correct.mean())
        verdict = 'L1_EXACT' if same else ('L2_RANDOM_TOLERABLE' if abs(acc_new - acc_ref) <= 0.002
                                           else 'DIFFERS')
        if not order_ok:
            verdict = 'ROW_ORDER_MISMATCH'
        out[slug] = {'status': 'OK', 'verdict': verdict, 'seconds': round(el, 1),
                     'acc_reference': acc_ref, 'acc_new': acc_new,
                     'row_order_matches_reference': order_ok,
                     'n_pred_changed': int((yp != rp).sum()) if len(yp) == len(rp) else None,
                     'timing': timing}
        print(f'[{slug}] {verdict}  ref_acc={acc_ref:.4f} new_acc={acc_new:.4f} '
              f'changed={out[slug]["n_pred_changed"]}  timing={timing}', flush=True)

    with open(os.path.join(HERE, 'results', 'rq1f_timing_patch_verification.json'), 'w',
              encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('\n' + json.dumps({k: v.get('verdict', v.get('status')) for k, v in out.items()},
                             ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
