"""
t2_macro_perm.py — Tier 2 · 针对 **Macro-F1** 的事故组配对置换检验（零训练）

============================================================================
为什么需要这份脚本
------------------------------------------------------------------------------
`t2_score.py` 中的 McNemar 精确二项检验与事故组配对符号翻转置换检验检验的都是
**逐行正确性（净正确数 / Accuracy）**。它们的 p 值**不能**被用来宣称
"Macro-F1 的差异显著 / 不显著"（Tier 1 rev1 已确认的口径纪律）。

本脚本把 Tier 1 rev1 的**同一构造**（事故组内配对交换，within-group swap
randomization）推广到 Tier 2 的预注册主检验族：

  * 统计量 = Macro-F1 差值（b − a），六类 Macro-F1 的等权平均；
  * 零分布 = 对每个事故组独立地以 1/2 概率把模型 a 与模型 b 在该组内的**逐行预测互换**，
    再重算 Macro-F1 差值；零假设 = 两模型在事故组内不可区分；
  * 依赖单位 = accident_group；
  * 仅使用**已有**的 dev_confirm 6,417 条逐样本预测，**不训练、不改模型、不读封存数据**；
  * Holm 族 = **运行前写定**的 {E1, E2}（见 docs/00 预注册）；
  * 内置**一致性校验**：同一置换机制下的 Accuracy 置换 p 须与组符号翻转 p 一致。

**性质声明（必须随结果一并报告）**：开发集（dev_confirm）上的检验，
dev_confirm 不是盲测、不是独立验证集；2024+ 封存池未读标签、未评分。

输出
  results/t2_macro_permutation.csv
  results/t2_macro_permutation.json
  logs/t2_macro_perm_log.txt
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'code'))
import s2_common as S2  # noqa: E402

PRED = os.path.join(HERE, 'predictions')
RES = os.path.join(HERE, 'results')
LOGP = os.path.join(HERE, 'logs')
os.makedirs(RES, exist_ok=True)
os.makedirs(LOGP, exist_ok=True)
LOGF = open(os.path.join(LOGP, 't2_macro_perm_log.txt'), 'w', encoding='utf-8')

N_PERM = 20000
SEED = 42
BLOCK = 2000
# 预注册主检验族（运行前写定）：(id, a=基线, b=被比较方)
PRIMARY = [('E1', 'ABL_B', 'P_no_proto'), ('E2', 'A_single', 'ABL_B')]
# 参照比较（不入 Holm 族）
REFERENCE = [('R1', 'A_ensemble', 'P_no_proto'), ('R2', 'A_single', 'A_ensemble')]
ARMS = ['P_no_proto', 'A_ensemble', 'ABL_B', 'A_single', 'F_concat']


def log(msg):
    print(msg, flush=True)
    LOGF.write(str(msg) + '\n')
    LOGF.flush()


def load_arm(name):
    return pd.read_csv(os.path.join(PRED, f'{name}.csv'))


def group_matrices(y, pred, groups, n_classes=6):
    """每个事故组 × 每个类别的计数矩阵。

      TP[g, c] : 组 g 内"预测为 c 且真值为 c"的行数
      PC[g, c] : 组 g 内"预测为 c"的行数（预测正例数）
    """
    G = len(np.unique(groups))
    inv = np.unique(groups, return_inverse=True)[1]
    TP = np.zeros((G, n_classes), dtype=np.float64)
    PC = np.zeros((G, n_classes), dtype=np.float64)
    for c in range(n_classes):
        pc = (pred == c)
        TP[:, c] = np.bincount(inv[pc & (y == c)], minlength=G)
        PC[:, c] = np.bincount(inv[pc], minlength=G)
    return TP, PC


def macro_f1_from_counts(TP, PC, support):
    den = PC + support[None, :]
    f1 = np.where(den > 0, 2.0 * TP / np.maximum(den, 1e-12), 0.0)
    return f1.mean(axis=1)


def perm_macro_f1(TP_a, PC_a, TP_b, PC_b, support, n_perm, seed):
    """事故组内配对交换置换：返回 (obs, nulls)。"""
    G = TP_a.shape[0]
    rng = np.random.default_rng(seed)
    obs = float(macro_f1_from_counts(TP_b.sum(axis=0)[None, :], PC_b.sum(axis=0)[None, :],
                                     support)[0]
                - macro_f1_from_counts(TP_a.sum(axis=0)[None, :], PC_a.sum(axis=0)[None, :],
                                       support)[0])
    nulls = np.empty(n_perm, dtype=np.float64)
    for s in range(0, n_perm, BLOCK):
        k = min(BLOCK, n_perm - s)
        flip = (rng.random((k, G)) < 0.5)
        f = flip.astype(np.float64)
        one_minus = 1.0 - f
        TP_bp = one_minus @ TP_b + f @ TP_a
        PC_bp = one_minus @ PC_b + f @ PC_a
        TP_ap = one_minus @ TP_a + f @ TP_b
        PC_ap = one_minus @ PC_a + f @ PC_b
        fb = macro_f1_from_counts(TP_bp, PC_bp, support)
        fa = macro_f1_from_counts(TP_ap, PC_ap, support)
        nulls[s:s + k] = fb - fa
    return obs, nulls


def acc_stat_null(y, pa, pb, groups, n_perm, seed):
    """同一套置换机制下的 Accuracy 差值零分布（与既有组符号翻转检验对照）。"""
    G = len(np.unique(groups))
    inv = np.unique(groups, return_inverse=True)[1]
    ok_a = (pa == y).astype(np.float64)
    ok_b = (pb == y).astype(np.float64)
    d = ok_b - ok_a
    per_g = np.bincount(inv, weights=d, minlength=G)
    rng = np.random.default_rng(seed)
    nulls = np.empty(n_perm, dtype=np.float64)
    for s in range(0, n_perm, BLOCK):
        k = min(BLOCK, n_perm - s)
        flip = (rng.random((k, G)) < 0.5)
        sgn = 1.0 - 2.0 * flip.astype(np.float64)
        nulls[s:s + k] = sgn @ per_g / len(y)
    return float(per_g.sum() / len(y)), nulls


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        v = (m - rank) * pvals[i]
        run = max(run, v)
        adj[i] = min(1.0, run)
    return adj, adj <= 0.05


def main():
    t0 = time.time()
    log('=== Tier 2 · Macro-F1 事故组配对置换检验（零训练，仅用已有预测）===')
    log(f'置换次数 = {N_PERM}，随机种子 = {SEED}，机制 = 事故组内配对交换')
    log('检验对象 = Macro-F1 差值（b − a）；零假设 = 两模型在事故组内不可区分')
    log('Holm 族 = 预注册的 {E1, E2}；R1/R2 为参照，不入族')
    log('性质：开发集（dev_confirm）上的检验 —— 非盲测、非独立验证；封存集未读')

    D = {a: load_arm(a) for a in ARMS}
    ref = D['P_no_proto']
    y = ref['y_true'].to_numpy().astype(int)
    groups = ref['accident_group'].astype(str).to_numpy()
    for a in ARMS:
        assert (D[a].row_id.to_numpy() == ref.row_id.to_numpy()).all(), f'{a} 行序不一致'
        assert (D[a].accident_group.to_numpy() == groups).all(), f'{a} 事故组不一致'
        assert (D[a].y_true.to_numpy() == y).all(), f'{a} 真值不一致'
    support = np.bincount(y, minlength=6).astype(np.float64)
    log(f'dev_confirm: {len(y)} 行，事故组 {len(np.unique(groups))} 个，'
        f'各类 support = {support.astype(int).tolist()}')

    mats = {a: group_matrices(y, D[a]['pred'].to_numpy().astype(int), groups) for a in ARMS}

    rows = []
    for cid, a, b in PRIMARY + REFERENCE:
        TP_a, PC_a = mats[a]
        TP_b, PC_b = mats[b]
        obs, nulls = perm_macro_f1(TP_a, PC_a, TP_b, PC_b, support, N_PERM, SEED)
        p_two = float((np.abs(nulls) >= abs(obs) - 1e-15).mean())
        p_two_plus1 = float((1 + int((np.abs(nulls) >= abs(obs) - 1e-15).sum())) / (1 + N_PERM))
        p_one = float((nulls >= obs - 1e-15).mean())
        obs_direct = float(f1_score(y, D[b]['pred'], average='macro', zero_division=0)
                           - f1_score(y, D[a]['pred'], average='macro', zero_division=0))
        assert abs(obs - obs_direct) < 1e-12, (obs, obs_direct)
        pa = D[a]['pred'].to_numpy().astype(int)
        pb = D[b]['pred'].to_numpy().astype(int)
        acc_obs, acc_nulls = acc_stat_null(y, pa, pb, groups, N_PERM, SEED)
        acc_p = float((np.abs(acc_nulls) >= abs(acc_obs) - 1e-12).mean())
        rows.append(dict(
            comparison_id=cid, comparison=f'{b} vs {a}',
            family='primary_Holm' if cid in ('E1', 'E2') else 'reference',
            n_rows=int(len(y)), n_groups=int(len(np.unique(groups))),
            macro_f1_a=float(f1_score(y, pa, average='macro', zero_division=0)),
            macro_f1_b=float(f1_score(y, pb, average='macro', zero_division=0)),
            macro_f1_diff_b_minus_a=obs,
            macro_f1_diff_pt=round(obs * 100, 4),
            macro_f1_perm_p_two_sided=p_two,
            macro_f1_perm_p_two_sided_plus1=p_two_plus1,
            macro_f1_perm_p_one_sided_b_better=p_one,
            macro_f1_null_mean=float(nulls.mean()),
            macro_f1_null_sd=float(nulls.std(ddof=1)),
            macro_f1_null_q025=float(np.percentile(nulls, 2.5)),
            macro_f1_null_q975=float(np.percentile(nulls, 97.5)),
            n_perm=N_PERM, seed=SEED,
            acc_obs_diff=acc_obs, acc_perm_p_two_sided=acc_p,
            acc_perm_p_consistency_note='与 t2_score.py 的组符号翻转 p 应一致（构造等价）'))
        log(f'\n[{cid}] {b} vs {a}')
        log(f'  Macro-F1: a={rows[-1]["macro_f1_a"]:.6f}  b={rows[-1]["macro_f1_b"]:.6f}'
            f'  Δ={obs:+.6f}（{obs*100:+.4f} pt）')
        log(f'  零分布: mean={nulls.mean():+.3e}  sd={nulls.std(ddof=1):.6f}'
            f'  2.5%/97.5% = [{np.percentile(nulls,2.5):.6f}, {np.percentile(nulls,97.5):.6f}]')
        log(f'  置换 p（双侧）= {p_two:.6f}  （双侧 +1 修正 = {p_two_plus1:.6f}）'
            f'  单侧(b 更优) = {p_one:.6f}')
        log(f'  [一致性校验] 同机制下 Accuracy 置换 p = {acc_p:.6f}'
            f'（应与 t2_score.py 的组符号翻转 p 一致）')

    df = pd.DataFrame(rows)
    sub = df[df.family == 'primary_Holm'].copy()
    adj, rej = holm(sub.macro_f1_perm_p_two_sided.to_numpy())
    df['macro_f1_holm_p'] = np.nan
    df['macro_f1_holm_reject_0p05'] = False
    df.loc[sub.index, 'macro_f1_holm_p'] = adj
    df.loc[sub.index, 'macro_f1_holm_reject_0p05'] = rej
    df.to_csv(os.path.join(RES, 't2_macro_permutation.csv'), index=False)

    with open(os.path.join(RES, 't2_macro_permutation.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(
            generated_by='t2_macro_perm.py',
            statistic='Macro-F1 差值（b − a）',
            null_mechanism='事故组内配对交换（within-group swap randomization）',
            n_perm=N_PERM, seed=SEED,
            holm_family=['E1', 'E2'],
            reference_not_in_family=['R1', 'R2'],
            n_groups=int(len(np.unique(groups))),
            scope='开发集 dev_confirm（6,417 行）上的检验 —— **非盲测、非独立验证**',
            caveats=[
                'McNemar / 组符号翻转 p 检验的是逐行正确性（Accuracy 口径），'
                '不得用于宣称 Macro-F1 差异显著/不显著；本脚本不替换那两个 p 值',
                '组级 Bootstrap 区间沿用 t2_score.py 的既有结果，未重算、未替换',
                'E1/E2 的检验族在训练前写定（docs/00 预注册）',
                'dev_confirm 已多轮参与模型开发，本轮属开发集机制分析，'
                '**不是**新独立验证；p 值不得表述为独立验证的显著性',
                '未训练、未改模型、未使用封存集（2024+，1,065 行）'],
            seconds=round(time.time() - t0, 2)), f, ensure_ascii=False, indent=2)

    log('\n=== 汇总（Holm 族 = {E1, E2}）===')
    log(df[['comparison_id', 'comparison', 'family', 'macro_f1_diff_b_minus_a',
            'macro_f1_diff_pt', 'macro_f1_perm_p_two_sided', 'macro_f1_holm_p',
            'macro_f1_holm_reject_0p05', 'acc_perm_p_two_sided']].to_string(index=False))
    log(f'\n总耗时 {time.time() - t0:.1f}s')
    LOGF.close()


if __name__ == '__main__':
    main()
