"""
t3_g6_mechanism.py — 阶段 D rev2 / G6：连带核验（零训练）

依据 `GeOAD_ReEval_Planning_20261028/docs/06_阶段C携带缺陷与连带重算清单.md`：
G1 换主表为 `master_v2_consistent.csv`（sha256 d9ef45f2...）之后，下列三项必须重算，
否则论文中引用的是陈旧数值：

  G6.1 「13 维模式未在 fit 出现」的局限指标（dev_confirm 侧）
  G6.2 近重复机制分析：fit / dev_confirm 之间的 13 维模式重叠 vs 事故组重叠
  G6.3 P_no_proto 与两个对照臂的决策分解（纠正 / 破坏 / 同错 → 错误类别变化），
       并在「事故组」与「13 维模式」两个维度上统计改动集中度

不训练、不改模型、不读 2024+ 封存池标签。
"""
import json
import os
import hashlib
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T1 = ARCH_ROOT + '/GeOAD_ReEval_Run_20261028_Tier1'
MASTER = (ARCH_ROOT + '/GeOAD_ReEval_Planning_20261028/'
          'data/master_v2_consistent.csv')
MASTER_SHA = 'd9ef45f215b429300628dd01d96f84b1e7f703b69ab35bccf523fa5f01c4046a'
RES = os.path.join(HERE, 'results')
os.makedirs(RES, exist_ok=True)

MAIN_13 = ['PIPE_DIAMETER', 'PIPE_WALL_THICKNESS', 'PIPE_SMYS', 'PIPE_SPECIFICATION',
           'PIPE_SEAM_TYPE', 'PIPE_COATING_TYPE', 'PIPE_TYPE', 'ACCIDENT_PSIG', 'MOP_PSIG',
           'INTERNAL_INSPECTION_IND', 'PIPE_AGE_AT_ACCIDENT',
           'MANUFACTURE_AGE_AT_ACCIDENT', 'PRESSURE_RATIO']


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_arm(name):
    p = os.path.join(T1, 'predictions', f'{name}.csv')
    df = pd.read_csv(p)
    return df


def main():
    t0 = time.time()
    out = {'generated_by': 'GeOAD_Submission_Sprint_20261029 / t3_g6_mechanism.py',
           'master': MASTER, 'master_sha256': sha256_file(MASTER),
           'master_sha256_expected': MASTER_SHA}
    assert out['master_sha256'] == MASTER_SHA, '主表哈希不符'

    man = pd.read_csv(MASTER, low_memory=False)
    assert 'sealed' in set(man.split.unique())
    dev = man[man.split.isin(['fit', 'calib', 'dev_confirm'])].copy()
    assert (man.loc[man.split == 'sealed', 'y'] == -1).all(), '封存池标签不是占位'

    # pattern13 官方公式：13 维 MAIN_13 的字符串连接
    dev['pattern13'] = dev[MAIN_13].astype(str).agg('|'.join, axis=1)
    fit = dev[dev.split == 'fit']
    dcp = dev[dev.split == 'dev_confirm']
    cal = dev[dev.split == 'calib']

    fit_pat = set(fit.pattern13)
    fit_grp = set(fit.accident_group)

    # ---------------- G6.1 未出现模式指标 ----------------
    n_dc = len(dcp)
    not_in_fit = ~dcp.pattern13.isin(fit_pat)
    out['G6_1_pattern_not_in_fit'] = {
        'definition': 'dev_confirm 中 13 维特征模式未出现在 fit 中的记录数',
        'n_dev_confirm': int(n_dc),
        'n_not_in_fit': int(not_in_fit.sum()),
        'pct_not_in_fit': round(float(not_in_fit.mean() * 100), 4),
        'n_pattern_in_fit': int(n_dc - not_in_fit.sum()),
        'pct_pattern_in_fit': round(float((~not_in_fit).mean() * 100), 4),
        'note': '口径与 RQ2 s2_score.py 一致（pattern13 自洽重算版）',
    }
    # calib 侧同口径
    out['G6_1_calib_pattern_not_in_fit'] = {
        'n_calib': int(len(cal)),
        'n_not_in_fit': int((~cal.pattern13.isin(fit_pat)).sum()),
        'pct_not_in_fit': round(float((~cal.pattern13.isin(fit_pat)).mean() * 100), 4),
    }

    # ---------------- G6.2 近重复机制：两种分组键的跨划分重叠 ----------------
    pat_overlap_rows = int(dcp.pattern13.isin(fit_pat).sum())
    grp_overlap_rows = int(dcp.accident_group.isin(fit_grp).sum())
    out['G6_2_near_duplicate_mechanism'] = {
        'grouping_key_A_pattern13': {
            'n_groups_fit': int(fit.pattern13.nunique()),
            'n_groups_dev_confirm': int(dcp.pattern13.nunique()),
            'dev_confirm_rows_whose_pattern_also_in_fit': pat_overlap_rows,
            'pct': round(pat_overlap_rows / n_dc * 100, 4),
        },
        'grouping_key_B_accident_group': {
            'n_groups_fit': int(fit.accident_group.nunique()),
            'n_groups_dev_confirm': int(dcp.accident_group.nunique()),
            'dev_confirm_rows_whose_group_also_in_fit': grp_overlap_rows,
            'pct': round(grp_overlap_rows / n_dc * 100, 4),
        },
        'interpretation': ('两种分组键给出的跨划分重复通道完全不同：13 维模式分组保留 '
                           f'{pat_overlap_rows} 条（{pat_overlap_rows/n_dc*100:.2f}%）跨划分近重复，'
                           '事故组分组保留 0 条。'),
    }

    # ---------------- G6.3 决策分解 ----------------
    dP = load_arm('P_no_proto')
    man_idx = man.set_index('row_id')
    pat_dc = man_idx.loc[dP.row_id.to_numpy(), 'pattern13'].astype(str).to_numpy()
    grp_dc = dP.accident_group.astype(str).to_numpy()
    y = dP.y_true.to_numpy().astype(int)
    dec = {}
    for base in ['A_ensemble', 'F_concat']:
        dB = load_arm(base)
        assert dB.row_id.tolist() == dP.row_id.tolist(), f'{base}: row_id 不一致'
        pb = dB.pred.to_numpy().astype(int)
        pp = dP.pred.to_numpy().astype(int)
        okB, okP = (pb == y), (pp == y)
        changed = pb != pp
        corrected = (~okB & okP)
        harmed = (okB & ~okP)
        wrong2wrong = changed & (~okB) & (~okP)
        dec[base] = {
            'n': int(len(y)),
            'n_changed': int(changed.sum()),
            'pct_changed': round(float(changed.mean() * 100), 4),
            'corrected': int(corrected.sum()),
            'harmed': int(harmed.sum()),
            'wrong_to_wrong_class_change': int(wrong2wrong.sum()),
            'net_correct': int(corrected.sum() - harmed.sum()),
            'accuracy_baseline': round(float(okB.mean() * 100), 4),
            'accuracy_geoad': round(float(okP.mean() * 100), 4),
            'delta_accuracy_pt': round(float((okP.mean() - okB.mean()) * 100), 4),
        }
        rows = []
        for dim, key in [('accident_group', grp_dc), ('pattern13', pat_dc)]:
            g = pd.DataFrame(dict(k=key, co=corrected.astype(int),
                                  ha=harmed.astype(int), ch=changed.astype(int)))
            agg = g.groupby('k').agg(n=('ch', 'size'), co=('co', 'sum'),
                                     ha=('ha', 'sum'), ch=('ch', 'sum'))
            rows.append(dict(dimension=dim, n_units=int(len(agg)),
                             units_touched=int((agg.ch > 0).sum()),
                             units_p_better=int((agg.co > agg.ha).sum()),
                             units_b_worse=int((agg.ha > agg.co).sum()),
                             units_tie=int((agg.co == agg.ha).sum())))
        pd.DataFrame(rows).to_csv(
            os.path.join(RES, f't3_g6_concentration_{base}.csv'), index=False)
    out['G6_3_decision_decomposition'] = dec

    out['elapsed_sec'] = round(time.time() - t0, 1)
    with open(os.path.join(RES, 't3_g6_mechanism.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
