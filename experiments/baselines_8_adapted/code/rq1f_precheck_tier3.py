"""rq1f_precheck_tier3.py — Tier 3 (阶段 D rev2 / G4) 正式重评的**运行前合法性检查**

与 RQ1 的 rq1f_precheck.py 的差别（唯一且必要）：
  * 参考主表由 **RQ2 旧清单 (c6bcde43...)** 改为 **权威主表 master_v2_consistent.csv
    (sha256 d9ef45f2...)**。旧清单是 INTERNAL_INSPECTION_IND 修复**之前**的版本，
    用它比较会必然报 274 行差异（那正是本次修复本身），不构成缺陷。
  * 只检查 13d 正式对照变体（9d 历史变体本轮授权范围外，且**不运行**）。
  * 其余检查项与 RQ1 precheck 同义（行数 / row_id 逐行顺序 / 标签 / 9 数值列逐值 /
    4 类别列词表编码未知→-1 / 各文件 sha256 与 fair_manifest 一致 / 封存池排除）。

用法: python3 code/rq1f_precheck_tier3.py
"""
import os
import sys
import json
import hashlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, 'data', 'baseline_data_13d')
MASTER = (ARCH_ROOT + '/GeOAD_ReEval_Planning_20261028/'
          'data/master_v2_consistent.csv')
MASTER_SHA = 'd9ef45f215b429300628dd01d96f84b1e7f703b69ab35bccf523fa5f01c4046a'

SIX = ['CORROSION', 'EQUIPMENT_FAILURE', 'EXTERNAL_DAMAGE',
       'MATERIAL_DEFECT', 'OPERATIONAL', 'OTHER']
MAIN_13 = ['PIPE_DIAMETER', 'PIPE_WALL_THICKNESS', 'PIPE_SMYS', 'PIPE_SPECIFICATION',
           'PIPE_SEAM_TYPE', 'PIPE_COATING_TYPE', 'PIPE_TYPE', 'ACCIDENT_PSIG', 'MOP_PSIG',
           'INTERNAL_INSPECTION_IND', 'PIPE_AGE_AT_ACCIDENT',
           'MANUFACTURE_AGE_AT_ACCIDENT', 'PRESSURE_RATIO']
NUMERIC = ['PIPE_DIAMETER', 'PIPE_WALL_THICKNESS', 'PIPE_SMYS', 'ACCIDENT_PSIG',
           'MOP_PSIG', 'INTERNAL_INSPECTION_IND', 'PIPE_AGE_AT_ACCIDENT',
           'MANUFACTURE_AGE_AT_ACCIDENT', 'PRESSURE_RATIO']
CATEGORICAL = ['PIPE_SPECIFICATION', 'PIPE_SEAM_TYPE', 'PIPE_COATING_TYPE', 'PIPE_TYPE']
SPLITS = [('fit', 'train', 29941), ('calib', 'val', 6411), ('dev_confirm', 'test', 6417)]

FAILS = []


def ck(name, cond, detail=''):
    ok = bool(cond)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILS.append(name)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    print('== Tier3 precheck (参考主表 = master_v2_consistent.csv) ==')
    ck('权威主表 sha256 == d9ef45f2...', sha256_file(MASTER) == MASTER_SHA,
       sha256_file(MASTER)[:12])
    man = pd.read_csv(MASTER)
    ck('主表行数 == 43,834', len(man) == 43834, len(man))
    ck('主表 row_id 唯一', man.row_id.is_unique, int(man.row_id.nunique()))
    ck('主表划分计数 fit/calib/dev_confirm/sealed',
       man.split.value_counts().to_dict() == {'fit': 29941, 'dev_confirm': 6417,
                                              'calib': 6411, 'sealed': 1065},
       str(man.split.value_counts().to_dict()))
    ck('封存池 1,065 行标签从未读取 (y=-1 / label=SEALED)',
       (man.loc[man.split == 'sealed', 'y'] == -1).all()
       and (man.loc[man.split == 'sealed', 'label'] == 'SEALED').all(), '')

    fm = json.load(open(os.path.join(DATA, 'fair_manifest.json'), encoding='utf-8'))
    vocab = fm['vocab']

    for split, tag, n in SPLITS:
        sub = man[man.split == split].reset_index(drop=True)
        X = pd.read_csv(os.path.join(DATA, f'X_{tag}.csv'))
        rm = pd.read_csv(os.path.join(DATA, f'rowmap_{tag}.csv'))
        yv = pd.read_csv(os.path.join(DATA, f'y_{tag}.csv'))
        ck(f'{split}: 行数 == {n}', len(X) == n == len(sub) == len(rm) == len(yv),
           f'{len(X)}/{len(sub)}/{len(rm)}/{len(yv)}')
        ck(f'{split}: row_id 逐行一致', rm.row_id.tolist() == sub.row_id.tolist(), '')
        ck(f'{split}: X 列顺序 == MAIN_13（协议锁顺序）', list(X.columns) == MAIN_13,
           f'前3列 {list(X.columns)[:3]}')
        lab = {l: i for i, l in enumerate(SIX)}
        yt = sub['label'].map(lab).to_numpy()
        ck(f'{split}: 标签与主表逐行一致', (yt == rm.y_true.to_numpy()).all(), '')
        ck(f'{split}: y_*.csv(ANOMALY_LABEL_ID) 与 rowmap 一致',
           (yv['ANOMALY_LABEL_ID'].to_numpy() == rm.y_true.to_numpy()).all(), '')
        bad = []
        for c in NUMERIC:
            a = pd.to_numeric(X[c], errors='coerce').to_numpy(dtype=float)
            b = pd.to_numeric(sub[c], errors='coerce').to_numpy(dtype=float)
            if not (np.array_equal(np.isnan(a), np.isnan(b))
                    and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], equal_nan=True)):
                bad.append(c)
        ck(f'{split}: 9 个数值列与主表逐值一致（含 NaN 位置）', not bad, str(bad))
        badc = []
        for c in CATEGORICAL:
            m = vocab[c]
            enc = X[c].to_numpy()
            ref = sub[c].map(lambda v: m.get(v, -1)).to_numpy()
            if not np.array_equal(enc, ref):
                badc.append(c)
        ck(f'{split}: 4 个类别列词表编码正确（未知→-1）', not badc, str(badc))
        for f in (f'X_{tag}.csv', f'y_{tag}.csv', f'rowmap_{tag}.csv'):
            h = sha256_file(os.path.join(DATA, f))
            ck(f'{split}/{f} sha256 与 fair_manifest 一致',
               fm['files'][tag]['sha256'].get(f) == h, h[:12])

    print('\n== 结论 ==')
    if FAILS:
        print(f'[FAIL] {len(FAILS)} 项未通过: {FAILS}')
        return 1
    print('[PASS] 全部通过 —— Tier 3 (G4) 具备运行条件')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
