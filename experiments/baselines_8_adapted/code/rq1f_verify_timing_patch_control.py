"""rq1f_verify_timing_patch_control.py — 计时插桩"行为中性"的对照实验（对照组）

问题：rq1f_verify_timing_patch.py 中 4 个 torch 基线（DeepPipe / DeepPipeNet /
      MFF-CNN-LightGBM / 1DCNN-DBO-LSTM）复跑结果与试点归档不完全一致。
      需排除"插桩改变了基线行为"这一可能。

对照设计（不改动任何基线逻辑）：
  1. 从试点目录复制**未插桩的原始 train.py**（pilot 的 code/baselines_fixed/<slug>）到临时目录；
  2. 在**同一份试点数据**上跑两遍（run_A、run_B）；
  3. 与试点归档预测三方比对：
       - pristine_run_A vs pristine_run_B   → 纯复跑抖动（未插桩 vs 未插桩）
       - pristine_run_A vs 试点归档         → 未插桩的复跑漂移
       - patched_run  vs 试点归档           → 插桩后的复跑漂移（rq1f_verify 的结果）
  若 (1) 的抖动幅度与 (3) 同量级 ⇒ 差异来自 **torch/numpy 随机性**，
  而非计时插桩；插桩本身在 4 个 L1_EXACT 基线上已被证明逐行零差异。

输出：results/rq1f_timing_patch_control.json
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
PILOT_BL = os.path.join(PILOT, 'code', 'baselines_fixed')
PILOT_DATA = os.path.join(PILOT, 'data', 'baseline_data_13d')
PILOT_PRED = os.path.join(PILOT, 'predictions_fair', '13d')
WORK = os.path.join(HERE, 'baseline_out', '_timing_control')
PRISTINE = os.path.join(WORK, 'pristine')

TARGETS = ['DeepPipe', 'DeepPipeNet', 'MFF-CNN-LightGBM', '1DCNN-DBO-LSTM']
CPU_ONLY = {'DeepPipe', 'MFF-CNN-LightGBM'}
TIMEOUT = 5400


def run_pristine(slug, tag):
    """把未插桩的原始 train.py 复制到独立目录跑一遍，返回 (pred, acc, seconds)"""
    d = os.path.join(PRISTINE, slug)
    if os.path.isdir(d):
        shutil.rmtree(d)
    shutil.copytree(os.path.join(PILOT_BL, slug), d)
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
    env['RQ1_SMDPSO_PROFILE'] = 'reduced'
    logp = os.path.join(WORK, f'{slug}_{tag}.log')
    import time as _t
    t0 = _t.time()
    p = subprocess.run([sys.executable, 'train.py'], cwd=d, env=env,
                       stdout=open(logp, 'w', encoding='utf-8'),
                       stderr=subprocess.STDOUT, timeout=TIMEOUT)
    el = _t.time() - t0
    if p.returncode != 0:
        return None, el, logp, f'returncode={p.returncode}'
    yp = None
    for cand in [os.path.join(d, 'result'), os.path.join(d, 'result', 'baseline')]:
        if os.path.isfile(os.path.join(cand, 'y_pred.npy')):
            yp = np.load(os.path.join(cand, 'y_pred.npy')).astype(int)
            break
    return yp, el, logp, None


def main():
    os.makedirs(WORK, exist_ok=True)
    out = {}
    for slug in TARGETS:
        ref = pd.read_csv(os.path.join(PILOT_PRED, f'{slug}.csv'))
        rp = ref.pred.to_numpy().astype(int)
        acc_ref = float(ref.correct.mean())

        pa, el_a, log_a, err_a = run_pristine(slug, 'runA')
        pb, el_b, log_b, err_b = run_pristine(slug, 'runB')
        if pa is None or pb is None:
            out[slug] = {'status': 'RUN_FAILED', 'err': err_a or err_b}
            print(f'[{slug}] 对照运行失败', flush=True)
            continue

        def cmp(p, q):
            changed = int((p != q).sum()) if len(p) == len(q) else None
            acc_p = float((p == ref.y_true.to_numpy()).mean())
            acc_q = float((q == ref.y_true.to_numpy()).mean())
            return {'n_changed': changed, 'acc_p': acc_p, 'acc_q': acc_q,
                    'abs_acc_diff': abs(acc_p - acc_q)}

        rec = {
            'status': 'OK',
            'acc_archived': acc_ref,
            'pristine_runA_seconds': round(el_a, 1),
            'pristine_runB_seconds': round(el_b, 1),
            'pristineA_vs_pristineB': cmp(pa, pb),
            'pristineA_vs_archived': cmp(pa, rp),
            'pristineB_vs_archived': cmp(pb, rp),
            'pristineB_vs_pristineA_n_changed': int((pa != pb).sum()),
        }
        out[slug] = rec
        print(f'[{slug}] archived={acc_ref:.4f} | pristineA={rec["pristineA_vs_archived"]["acc_p"]:.4f} '
              f'pristineB={rec["pristineB_vs_archived"]["acc_p"]:.4f} | '
              f'A-vs-B changed={rec["pristineB_vs_pristineA_n_changed"]}', flush=True)

    with open(os.path.join(HERE, 'results', 'rq1f_timing_patch_control.json'), 'w',
              encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('\n' + json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
