"""
t1_macro_perm.py — 统计报告补正：针对 **Macro-F1** 的事故组配对置换检验

============================================================================
为什么需要这份脚本
------------------------------------------------------------------------------
现有 `t1_score.py` 的两项配对检验（McNemar 精确二项、事故组配对符号翻转置换）
检验的都是**逐行正确性**（净正确数 / Accuracy）。因此它们的 p 值**不能**被用来
宣称"Macro-F1 的差异不显著"。本次补正：

  * 统计量改为 **Macro-F1 差值**（b − a）；
  * 零分布由**事故组内配对交换**（within-group swap randomization）生成：
    对每个事故组，以 1/2 概率把模型 a 与模型 b 在该组内的逐行预测**互换**，
    重算 Macro-F1 差值。零假设 = 两模型在组内不可区分。
    这与既有符号翻转检验是**同一个构造**（对 Accuracy 而言二者等价），
    但可以作用于任何"逐行可加"型统计量（Macro-F1 即属此类）。
  * 仅使用**已有**的 dev_confirm 6,417 条逐样本预测，**不训练、不改模型、
    不读封存数据**；
  * 保留现有组级 Bootstrap 区间（不重算、不替换）；
  * 对运行前写定的两项主比较 {C1, C2} 分别做 **Holm 校正**。

**性质声明（必须随结果一并报告）**
  这是**开发集（dev_confirm）上的补充事后检验**：检验对象（Macro-F1）在看到
  结果之后才被指定，**不是事前预注册**；dev_confirm 是开发性验证集，
  **不是独立验证集**；因此 p 值不能被表述为"独立验证的统计显著性"。

输出
  results/t1_macro_permutation.csv     两项主比较 + 一项辅助比较的逐项结果
  results/t1_macro_permutation.json    口径、零分布诊断、一致性校验
  logs/t1_macro_perm_log.txt            真实运行日志
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
LOGF = open(os.path.join(LOGP, 't1_macro_perm_log.txt'), 'w', encoding='utf-8')

N_PERM = 20000
SEED = 42
BLOCK = 2000
# 预定主比较族（运行前写定）：(id, a=基线, b=被比较方)
PRIMARY = [('C1', 'A_ensemble', 'P_no_proto'), ('C2', 'F_concat', 'P_no_proto')]
AUX = [('A', 'A_single', 'P_no_proto')]


def log(msg):
    print(msg, flush=True)
    LOGF.write(str(msg) + '\n')
    LOGF.flush()


def load_arm(name):
    df = pd.read_csv(os.path.join(PRED, f'{name}.csv'))
    return df


def group_matrices(y, pred, groups, n_classes=6):
    """每个事故组 × 每个类别的计数矩阵。

    返回
      TP[g, c] : 组 g 内"预测为 c 且真值为 c"的行数
      PC[g, c] : 组 g 内"预测为 c"的行数（= 预测正例数，按组分摊）
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
    """Macro-F1（按组聚合后计算；support 为各类真值数，不随置换改变）。"""
    den = PC + support[None, :]
    f1 = np.where(den > 0, 2.0 * TP / np.maximum(den, 1e-12), 0.0)
    return f1.mean(axis=1)


def perm_macro_f1(TP_a, PC_a, TP_b, PC_b, support, n_perm, seed):
    """事故组内配对交换置换：返回 (obs, null, flips_used)。

    每次置换对每个事故组独立地以 1/2 概率交换两模型在该组内的预测
    （等价于对该组的"逐行差值"翻转符号）。
    """
    G = TP_a.shape[0]
    rng = np.random.default_rng(seed)
    # 观测值（不交换）
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
        # 交换后：b 的 TP/PC 取自 a 的对应计数（逐组择一）
        TP_bp = one_minus @ TP_b + f @ TP_a
        PC_bp = one_minus @ PC_b + f @ PC_a
        TP_ap = one_minus @ TP_a + f @ TP_b
        PC_ap = one_minus @ PC_a + f @ PC_b
        fb = macro_f1_from_counts(TP_bp, PC_bp, support)
        fa = macro_f1_from_counts(TP_ap, PC_ap, support)
        nulls[s:s + k] = fb - fa
    return obs, nulls, G


def acc_stat_null(y, pa, pb, groups, n_perm, seed):
    """同一套置换机制下的 Accuracy 差值零分布（用于与既有符号翻转检验对照）。"""
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
    log('=== 统计报告补正：Macro-F1 的事故组配对置换检验（零训练，仅用已有预测）===')
    log(f'置换次数 = {N_PERM}，随机种子 = {SEED}，机制 = 事故组内配对交换')
    log('检验对象 = Macro-F1 差值（b − a）；零假设 = 两模型在事故组内不可区分')
    log('性质：开发集（dev_confirm）上的**补充事后检验**，非事前预注册、非独立验证')

    D = {a: load_arm(a) for a in ['A_single', 'A_ensemble', 'F_concat', 'P_no_proto']}
    ref = D['P_no_proto']
    y = ref['y_true'].to_numpy().astype(int)
    groups = ref['accident_group'].astype(str).to_numpy()
    for a, df in D.items():
        assert (df.row_id.to_numpy() == ref.row_id.to_numpy()).all(), f'{a} 行序不一致'
        assert (df.accident_group.to_numpy() == groups).all(), f'{a} 事故组不一致'
        assert (df.y_true.to_numpy() == y).all(), f'{a} 真值不一致'
    support = np.bincount(y, minlength=6).astype(np.float64)
    log(f'dev_confirm: {len(y)} 行，事故组 {len(np.unique(groups))} 个，'
        f'各类 support = {support.astype(int).tolist()}')

    mats = {}
    for a, df in D.items():
        mats[a] = group_matrices(y, df['pred'].to_numpy().astype(int), groups)

    rows, diag = [], {}
    for cid, a, b in PRIMARY + AUX:
        TP_a, PC_a = mats[a]
        TP_b, PC_b = mats[b]
        obs, nulls, G = perm_macro_f1(TP_a, PC_a, TP_b, PC_b, support, N_PERM, SEED)
        # 双侧 p（零分布对称于 0）
        p_two = float((np.abs(nulls) >= abs(obs) - 1e-15).mean())
        p_two_plus1 = float((1 + int((np.abs(nulls) >= abs(obs) - 1e-15).sum())) / (1 + N_PERM))
        # 单侧（b 优于 a 方向）
        p_one = float((nulls >= obs - 1e-15).mean())
        obs_direct = float(f1_score(y, D[b]['pred'], average='macro', zero_division=0)
                           - f1_score(y, D[a]['pred'], average='macro', zero_division=0))
        assert abs(obs - obs_direct) < 1e-12, (obs, obs_direct)
        # 一致性校验：同一置换机制下的 Accuracy 零分布应复现既有符号翻转检验
        pa = D[a]['pred'].to_numpy().astype(int)
        pb = D[b]['pred'].to_numpy().astype(int)
        acc_obs, acc_nulls = acc_stat_null(y, pa, pb, groups, N_PERM, SEED)
        acc_p = float((np.abs(acc_nulls) >= abs(acc_obs) - 1e-12).mean())
        rows.append(dict(
            comparison_id=cid, comparison=f'{b} vs {a}',
            family='primary_Holm' if cid in ('C1', 'C2') else 'auxiliary',
            n_rows=int(len(y)), n_groups=int(G),
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
            acc_perm_p_consistency_note='与 t1_score.py 的组符号翻转 p 应一致（构造等价）'))
        diag[cid] = dict(macro_f1_null_sd=float(nulls.std(ddof=1)),
                         acc_perm_p=acc_p)
        log(f'\n[{cid}] {b} vs {a}')
        log(f'  Macro-F1: a={rows[-1]["macro_f1_a"]:.6f}  b={rows[-1]["macro_f1_b"]:.6f}'
            f'  Δ={obs:+.6f}（{obs*100:+.4f} pt）')
        log(f'  零分布: mean={nulls.mean():+.3e}  sd={nulls.std(ddof=1):.6f}'
            f'  2.5%/97.5% = [{np.percentile(nulls,2.5):.6f}, {np.percentile(nulls,97.5):.6f}]')
        log(f'  置换 p（双侧）= {p_two:.6f}   （双侧 +1 修正 = {p_two_plus1:.6f}）'
            f'   单侧(b 更优) = {p_one:.6f}')
        log(f'  [一致性校验] 同机制下 Accuracy 置换 p = {acc_p:.6f}'
            f'（应与既有组符号翻转 p 一致）')

    df = pd.DataFrame(rows)
    # Holm：族 = 运行前写定的两项主比较 {C1, C2}
    sub = df[df.family == 'primary_Holm'].copy()
    adj, rej = holm(sub.macro_f1_perm_p_two_sided.to_numpy())
    df['macro_f1_holm_p'] = np.nan
    df['macro_f1_holm_reject_0p05'] = False
    df.loc[sub.index, 'macro_f1_holm_p'] = adj
    df.loc[sub.index, 'macro_f1_holm_reject_0p05'] = rej
    df.to_csv(os.path.join(RES, 't1_macro_permutation.csv'), index=False)

    with open(os.path.join(RES, 't1_macro_permutation.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(
            generated_by='t1_macro_perm.py',
            statistic='Macro-F1 差值（b − a）',
            null_mechanism='事故组内配对交换（within-group swap randomization）',
            n_perm=N_PERM, seed=SEED,
            holm_family=['C1', 'C2'],
            n_groups=int(len(np.unique(groups))),
            scope='开发集 dev_confirm（6,417 行）上的**补充事后检验**',
            caveats=[
                '检验对象（Macro-F1）在看到结果之后才被指定 —— **不是事前预注册**',
                'dev_confirm 是开发性验证集，**不是独立验证集**；p 值不得表述为独立验证的显著性',
                '既有 McNemar / 组符号翻转 p 检验的是逐行正确性（Accuracy 口径），'
                '不得用于宣称 Macro-F1 差异不显著；本脚本不替换那两个 p 值，只补充 Macro-F1 口径',
                '组级 Bootstrap 区间沿用 t1_score.py 的既有结果，未重算、未替换',
                '未训练、未改模型、未使用封存集（2024+，1,065 行）'],
            diagnostics=diag,
            seconds=round(time.time() - t0, 2)), f, ensure_ascii=False, indent=2)

    log('\n=== 汇总（Holm 族 = {C1, C2}）===')
    log(df[['comparison_id', 'comparison', 'family', 'macro_f1_diff_b_minus_a',
            'macro_f1_diff_pt', 'macro_f1_perm_p_two_sided', 'macro_f1_holm_p',
            'macro_f1_holm_reject_0p05', 'acc_perm_p_two_sided']].to_string(index=False))
    log(f'\n总耗时 {time.time() - t0:.1f}s')
    LOGF.close()


if __name__ == '__main__':
    main()
