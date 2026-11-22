"""
t1_score.py — GeOAD 统一重评 Tier 1 · 独立统计核算（零训练）

============================================================================
**唯一评价入口**：所有指标都从 `predictions/<arm>.csv`（逐样本真值 + 6 维概率）
重新计算，**不从训练日志取任何 best 值**。

主比较族（运行前写定，Holm 在这 2 项上校正）
    C1: P_no_proto vs A_ensemble
    C2: P_no_proto vs F_concat
辅助比较（报告，不进入 Holm 族）
    A : P_no_proto vs A_single

统计方法（逐字沿用 RQ2 第二小阶段 s2_score.py）
    * 事故组整群 bootstrap 95% CI（依赖单位 = accident_group，2,000 次，seed 42）——主口径
    * 行级 McNemar 精确二项检验（仅作行级补充）
    * 事故组配对符号翻转置换检验（20,000 次，seed 42）——净正确的 p 值
    * Holm 步降法（族 = {C1, C2}）

修复前后并列
    修复前 = RQ2 第二小阶段归档 predictions_s2/<arm>.csv 与 results/s2_metrics.csv
    修复后 = 本次运行 predictions/<arm>.csv

输出
    results/t1_metrics.csv            四臂主表
    results/t1_per_class.csv          六类 P/R/F1/support
    results/t1_confusion.csv          混淆矩阵
    results/t1_paired_diffs.csv       配对差异（纠错/破坏/错误转错误 + CI + 检验）
    results/t1_holm.csv               Holm 校正
    results/t1_pre_post.csv           修复前后并列（指标 + 逐臂变化）
    results/t1_pre_post_paired.csv    修复前后逐样本配对（旧预测 vs 新预测）
    results/t1_score_meta.json        口径与计时
    logs/t1_score_log.txt             真实运行日志
"""
import json
import os
import sys
import time
from math import comb

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix, matthews_corrcoef)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'code'))
import s2_common as S2  # noqa: E402

PRED = os.path.join(HERE, 'predictions')
RES = os.path.join(HERE, 'results')
LOGP = os.path.join(HERE, 'logs')
os.makedirs(RES, exist_ok=True)
os.makedirs(LOGP, exist_ok=True)
LOGF = open(os.path.join(LOGP, 't1_score_log.txt'), 'w', encoding='utf-8')

# 归档源目录（可选）。打包件不附带该目录；缺省时相关对照段落跳过。
RQ2 = os.environ.get('ZHL_GEOAD_RQ2_DIR', '')
HAS_RQ2_ARCHIVE = bool(RQ2) and os.path.isdir(RQ2)

ARMS = ['A_single', 'A_ensemble', 'F_concat', 'P_no_proto']
HARD = ['MATERIAL_DEFECT', 'OPERATIONAL']
# (id, a=基线, b=被比较方)；主比较族 = {C1, C2}，运行前写定
PRIMARY = [('C1', 'A_ensemble', 'P_no_proto'), ('C2', 'F_concat', 'P_no_proto')]
AUX = [('A', 'A_single', 'P_no_proto')]


def log(msg):
    print(msg, flush=True)
    LOGF.write(str(msg) + '\n')
    LOGF.flush()


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return float(min(1.0, 2 * tail))


def group_confusion(y, pred, groups):
    uniq, inv = np.unique(groups, return_inverse=True)
    conf = np.zeros((len(uniq), 6, 6), dtype=np.int64)
    np.add.at(conf, (inv, y, pred), 1)
    return uniq, conf


def f1_acc_from_conf(conf):
    tp = np.diag(conf).astype(float)
    support = conf.sum(axis=1).astype(float)
    pred_pos = conf.sum(axis=0).astype(float)
    prec = np.where(pred_pos > 0, tp / np.maximum(pred_pos, 1), 0.0)
    rec = np.where(support > 0, tp / np.maximum(support, 1), 0.0)
    den = prec + rec
    f1 = np.where(den > 0, 2 * prec * rec / np.maximum(den, 1e-12), 0.0)
    n = conf.sum()
    acc = tp.sum() / n if n else 0.0
    return float(acc), float(f1.mean())


def paired_group_bootstrap(y, pred_a, pred_b, groups, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    _, ca = group_confusion(y, pred_a, groups)
    _, cb_ = group_confusion(y, pred_b, groups)
    n_g = ca.shape[0]
    accs, f1s, nets, hards = [], [], [], []
    ok_a = (pred_a == y)
    ok_b = (pred_b == y)
    uniq, inv = np.unique(groups, return_inverse=True)
    net_by_g = np.zeros(n_g)
    np.add.at(net_by_g, inv, (ok_b.astype(int) - ok_a.astype(int)))
    hard_idx = [S2.CLS2IDX[c] for c in HARD]

    def hard_f1(conf):
        fs = []
        for ci in hard_idx:
            tp = conf[ci, ci]
            sup = conf[ci].sum()
            pp = conf[:, ci].sum()
            pr = tp / pp if pp else 0.0
            rc = tp / sup if sup else 0.0
            fs.append(2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0)
        return float(np.mean(fs))

    for _ in range(n_boot):
        pick = rng.integers(0, n_g, n_g)
        sa = ca[pick].sum(axis=0)
        sb = cb_[pick].sum(axis=0)
        aa, fa = f1_acc_from_conf(sa)
        ab, fb = f1_acc_from_conf(sb)
        accs.append(ab - aa)
        f1s.append(fb - fa)
        hards.append(hard_f1(sb) - hard_f1(sa))
        nets.append(float(net_by_g[pick].sum()))
    accs, f1s, nets, hards = np.array(accs), np.array(f1s), np.array(nets), np.array(hards)
    return dict(
        acc_diff=float(ok_b.sum() - ok_a.sum()) / len(y),
        acc_lo=float(np.percentile(accs, 2.5)), acc_hi=float(np.percentile(accs, 97.5)),
        macro_f1_diff=float(f1_score(y, pred_b, average='macro', zero_division=0)
                            - f1_score(y, pred_a, average='macro', zero_division=0)),
        f1_lo=float(np.percentile(f1s, 2.5)), f1_hi=float(np.percentile(f1s, 97.5)),
        net_lo=float(np.percentile(nets, 2.5)), net_hi=float(np.percentile(nets, 97.5)),
        hard_f1_diff=float(np.mean([
            f1_score(y, pred_b, labels=[S2.CLS2IDX[c]], average='macro', zero_division=0)
            - f1_score(y, pred_a, labels=[S2.CLS2IDX[c]], average='macro', zero_division=0)
            for c in HARD])),
        hard_lo=float(np.percentile(hards, 2.5)), hard_hi=float(np.percentile(hards, 97.5)),
        n_groups=int(n_g), n_boot=n_boot)


def group_permutation_p(y, pred_a, pred_b, groups, n_perm=20000, seed=42):
    rng = np.random.default_rng(seed)
    ok_a, ok_b = (pred_a == y), (pred_b == y)
    uniq, inv = np.unique(groups, return_inverse=True)
    d = (ok_b.astype(int) - ok_a.astype(int)).astype(float)
    per_g = np.zeros(len(uniq))
    np.add.at(per_g, inv, d)
    obs = per_g.sum()
    n_g = len(uniq)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n_g))
    null = signs @ per_g
    p = float((np.abs(null) >= abs(obs) - 1e-12).mean())
    return p, float(obs)


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


def load_arm(name, root=PRED, subdir=''):
    df = pd.read_csv(os.path.join(root, subdir, f'{name}.csv'))
    probs = df[[f'prob_{c}' for c in S2.SIX]].to_numpy(dtype=float)
    return df, probs


def main():
    t0 = time.time()
    log('=== GeOAD 统一重评 Tier 1 · 独立统计核算（零训练）===')
    with open(os.path.join(HERE, 'protocol', 'protocol_lock.json'), encoding='utf-8') as f:
        lock = json.load(f)
    log(f'协议锁: {lock["protocol_id"]}')

    D = {}
    for a in ARMS:
        df, probs = load_arm(a)
        D[a] = dict(df=df, probs=probs)
    ref = D['P_no_proto']['df']
    y = ref['y_true'].to_numpy().astype(int)
    groups = ref['accident_group'].astype(str).to_numpy()

    for a in ARMS:
        assert (D[a]['df'].row_id.to_numpy() == ref.row_id.to_numpy()).all(), f'{a} 行序不一致'
        assert (D[a]['df'].accident_group.to_numpy() == groups).all(), f'{a} 事故组不一致'
        assert (D[a]['df'].y_true.to_numpy() == y).all(), f'{a} 真值不一致'
    log('[核查] 四臂逐样本文件行序 / 事故组 / 真值完全对齐')

    # --- 概率合法性 ---
    leg = []
    for a in ARMS:
        p = D[a]['probs']
        leg.append(dict(arm=a, all_finite=bool(np.isfinite(p).all()),
                        all_nonneg=bool((p >= 0).all()),
                        max_rowsum_dev=float(np.abs(p.sum(axis=1) - 1).max()),
                        min_prob=float(p.min()), max_prob=float(p.max())))
    legdf = pd.DataFrame(leg)
    legdf.to_csv(os.path.join(RES, 't1_prob_legality.csv'), index=False)
    assert legdf.all_finite.all() and legdf.all_nonneg.all()
    assert (legdf.max_rowsum_dev < 1e-6).all(), '概率行和不等于 1'
    log(f'[核查] 概率合法性全部通过（最大行和偏差 {legdf.max_rowsum_dev.max():.2e}）')

    # --- 主表 ---
    rows, perclass, confs = [], [], []
    for a in ARMS:
        df = D[a]['df']
        pred = df['pred'].to_numpy().astype(int)
        rows.append(dict(arm=a, n=len(y),
                         accuracy=float(accuracy_score(y, pred)),
                         macro_f1=float(f1_score(y, pred, average='macro', zero_division=0)),
                         weighted_f1=float(f1_score(y, pred, average='weighted', zero_division=0)),
                         mcc=float(matthews_corrcoef(y, pred)),
                         hard_class_f1=float(np.mean([
                             f1_score(y, pred, labels=[S2.CLS2IDX[c]], average='macro',
                                      zero_division=0) for c in HARD])),
                         n_correct=int((pred == y).sum()),
                         n_sites=int(len(set(groups)))))
        p_, r_, f_, s_ = precision_recall_fscore_support(y, pred, labels=range(6),
                                                         zero_division=0)
        for i, c in enumerate(S2.SIX):
            perclass.append(dict(arm=a, cls=c, precision=float(p_[i]), recall=float(r_[i]),
                                 f1=float(f_[i]), support=int(s_[i])))
        cm = confusion_matrix(y, pred, labels=range(6))
        confs.append(dict(arm=a, matrix=json.dumps(cm.tolist())))
    mdf = pd.DataFrame(rows)
    mdf.to_csv(os.path.join(RES, 't1_metrics.csv'), index=False)
    pd.DataFrame(perclass).to_csv(os.path.join(RES, 't1_per_class.csv'), index=False)
    pd.DataFrame(confs).to_csv(os.path.join(RES, 't1_confusion.csv'), index=False)
    log('\n主表（dev_confirm %d 行，修复后；开发性验证集，非盲测）:\n%s'
        % (len(y), mdf.to_string(index=False)))

    # --- 配对比较 ---
    def paired_stats(cid, a, b, kind, pa, pb, tag='post'):
        ok_a, ok_b = (pa == y), (pb == y)
        fixed = int((~ok_a & ok_b).sum())       # 有益纠错
        broken = int((ok_a & ~ok_b).sum())      # 有害翻转
        both_wrong = (~ok_a & ~ok_b)
        e2e_same = int((both_wrong & (pa == pb)).sum())   # 错误转错误（同类）
        e2e_diff = int((both_wrong & (pa != pb)).sum())   # 错误转错误（异类）
        changed = int((pa != pb).sum())
        bs = paired_group_bootstrap(y, pa, pb, groups)
        pm = mcnemar_exact(fixed, broken)
        pg, obs = group_permutation_p(y, pa, pb, groups)
        return dict(comparison_id=cid, comparison=f'{b} vs {a}', kind=kind,
                    data_version=tag,
                    a=a, b=b, a_correct=int(ok_a.sum()), b_correct=int(ok_b.sum()),
                    b_net_vs_a=fixed - broken,
                    direction=f'{b} 相对 {a} 净多 {fixed - broken} 条'
                              f'（有益纠错 {fixed} / 有害翻转 {broken}）',
                    beneficial_corrections=fixed, harmful_flips=broken,
                    error_to_error_same_class=e2e_same,
                    error_to_error_diff_class=e2e_diff,
                    error_to_error_total=e2e_same + e2e_diff,
                    changed_decisions=changed,
                    mcnemar_p=pm, group_permutation_p=pg, group_net_observed=obs,
                    acc_diff_a_to_b=bs['acc_diff'],
                    acc_ci95=f"[{bs['acc_lo']:.5f}, {bs['acc_hi']:.5f}]",
                    macro_f1_diff_a_to_b=bs['macro_f1_diff'],
                    macro_f1_ci95=f"[{bs['f1_lo']:.5f}, {bs['f1_hi']:.5f}]",
                    hard_class_f1_diff_a_to_b=bs['hard_f1_diff'],
                    hard_class_f1_ci95=f"[{bs['hard_lo']:.5f}, {bs['hard_hi']:.5f}]",
                    net_ci95=f"[{bs['net_lo']:.0f}, {bs['net_hi']:.0f}]",
                    n_groups=bs['n_groups'], n_boot=bs['n_boot'])

    def paired(cid, a, b, kind):
        return paired_stats(cid, a, b, kind,
                            D[a]['df']['pred'].to_numpy().astype(int),
                            D[b]['df']['pred'].to_numpy().astype(int), 'post_fix')

    diffs = [paired(cid, a, b, 'primary') for cid, a, b in PRIMARY] + \
            [paired(cid, a, b, 'auxiliary') for cid, a, b in AUX]
    ddf = pd.DataFrame(diffs)
    ddf.to_csv(os.path.join(RES, 't1_paired_diffs.csv'), index=False)
    log('\n配对差异（净正确方向明确标注；CI = 事故组整群 bootstrap 2000 次）:\n'
        + ddf[['comparison_id', 'comparison', 'kind', 'a_correct', 'b_correct',
               'b_net_vs_a', 'beneficial_corrections', 'harmful_flips',
               'error_to_error_same_class', 'error_to_error_diff_class',
               'mcnemar_p', 'group_permutation_p', 'acc_diff_a_to_b', 'acc_ci95',
               'macro_f1_diff_a_to_b', 'macro_f1_ci95',
               'hard_class_f1_diff_a_to_b']].to_string(index=False))

    # --- Holm（族 = {C1, C2}，运行前写定）---
    sub = ddf[ddf.kind == 'primary'].copy()
    adj_m, rej_m = holm(sub.mcnemar_p.to_numpy())
    adj_g, rej_g = holm(sub.group_permutation_p.to_numpy())
    hdf = pd.DataFrame(dict(comparison_id=sub.comparison_id.values,
                            comparison=sub.comparison.values,
                            mcnemar_p=sub.mcnemar_p.values,
                            holm_p_mcnemar=adj_m, reject_mcnemar_0p05=rej_m,
                            group_permutation_p=sub.group_permutation_p.values,
                            holm_p_group_perm=adj_g, reject_groupperm_0p05=rej_g,
                            net=sub.b_net_vs_a.values, direction=sub.direction.values,
                            acc_ci95=sub.acc_ci95.values,
                            macro_f1_ci95=sub.macro_f1_ci95.values))
    hdf.to_csv(os.path.join(RES, 't1_holm.csv'), index=False)
    log('\nHolm 校正（族 = 运行前写定的 {C1, C2}）:\n' + hdf.to_string(index=False))

    # ---------------- 修复前（RQ2 归档）同族比较（并列口径）----------------
    pre_diffs = []
    for cid, a, b in PRIMARY + AUX:
        od_a = pd.read_csv(os.path.join(RQ2, 'predictions_s2', f'{a}.csv'))
        od_b = pd.read_csv(os.path.join(RQ2, 'predictions_s2', f'{b}.csv'))
        pre_diffs.append(paired_stats(
            cid, a, b, 'primary' if cid in ('C1', 'C2') else 'auxiliary',
            od_a['pred'].to_numpy().astype(int),
            od_b['pred'].to_numpy().astype(int), 'pre_fix'))
    predf = pd.DataFrame(pre_diffs)
    predf.to_csv(os.path.join(RES, 't1_pre_post_primary.csv'), index=False)
    log('\n修复前(RQ2 归档) 同族比较:\n'
        + predf[['comparison_id', 'comparison', 'data_version', 'b_correct',
                 'b_net_vs_a', 'mcnemar_p', 'group_permutation_p',
                 'acc_diff_a_to_b', 'acc_ci95', 'macro_f1_diff_a_to_b',
                 'macro_f1_ci95']].to_string(index=False))

    # ---------------- 修复前后并列 ----------------
    old_dir = os.path.join(RQ2, 'predictions_s2')
    old_metrics = pd.read_csv(os.path.join(RQ2, 'results', 's2_metrics.csv'))
    old_metrics = old_metrics[old_metrics.arm.isin(ARMS)].set_index('arm')
    pre = []
    for a in ARMS:
        o = old_metrics.loc[a]
        n = mdf[mdf.arm == a].iloc[0]
        pre.append(dict(arm=a,
                        acc_pre=float(o.accuracy), acc_post=float(n.accuracy),
                        acc_delta_pt=round((float(n.accuracy) - float(o.accuracy)) * 100, 4),
                        macro_f1_pre=float(o.macro_f1), macro_f1_post=float(n.macro_f1),
                        macro_f1_delta_pt=round((float(n.macro_f1) - float(o.macro_f1)) * 100, 4),
                        w_f1_pre=float(o.weighted_f1), w_f1_post=float(n.weighted_f1),
                        w_f1_delta_pt=round((float(n.weighted_f1) - float(o.weighted_f1)) * 100, 4),
                        mcc_pre=float(o.mcc), mcc_post=float(n.mcc),
                        correct_pre=int(o.n_correct), correct_post=int(n.n_correct),
                        hard_f1_pre=float(o.hard_class_f1),
                        hard_f1_post=float(n.hard_class_f1),
                        hard_f1_delta_pt=round((float(n.hard_class_f1)
                                                - float(o.hard_class_f1)) * 100, 4)))
    pdf = pd.DataFrame(pre)
    pdf.to_csv(os.path.join(RES, 't1_pre_post.csv'), index=False)
    log('\n修复前(RQ2 归档) vs 修复后(本次) 并列:\n' + pdf.to_string(index=False))

    # 逐样本配对：旧预测 vs 新预测（同一 row_id 顺序）
    pp = []
    for a in ARMS:
        old_df, _ = load_arm(a, root=old_dir)
        assert (old_df.row_id.to_numpy() == ref.row_id.to_numpy()).all(), f'{a} 修复前后行序不一致'
        po = old_df['pred'].to_numpy().astype(int)
        pn = D[a]['df']['pred'].to_numpy().astype(int)
        ok_o, ok_n = (po == y), (pn == y)
        pp.append(dict(arm=a,
                       n_correct_pre=int(ok_o.sum()), n_correct_post=int(ok_n.sum()),
                       net_vs_pre=int(ok_n.sum() - ok_o.sum()),
                       pre_broken_by_fix=int((~ok_o & ok_n).sum()),
                       pre_broken_by_regression=int((ok_o & ~ok_n).sum()),
                       changed_decisions=int((po != pn).sum()),
                       changed_and_still_wrong=int(((po != pn) & (~ok_o) & (~ok_n)).sum()),
                       acc_pre=float(ok_o.mean()), acc_post=float(ok_n.mean())))
    ppdf = pd.DataFrame(pp)
    ppdf.to_csv(os.path.join(RES, 't1_pre_post_paired.csv'), index=False)
    log('\n修复前后逐样本配对:\n' + ppdf.to_string(index=False))

    # 六类 support（自然分布，未做任何平衡）
    dist = pd.DataFrame({'cls': S2.SIX,
                          'n_dev_confirm': [int((y == i).sum()) for i in range(6)],
                          'pct': [round(float((y == i).mean()) * 100, 2) for i in range(6)]})
    dist.to_csv(os.path.join(RES, 't1_class_distribution.csv'), index=False)

    with open(os.path.join(RES, 't1_score_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(protocol=lock['protocol_id'], n_dev_confirm=int(len(y)),
                       n_groups=int(len(set(groups))), arms=ARMS,
                       holm_family=['C1', 'C2'],
                       primary_pairs=[f'{b} vs {a}' for _, a, b in PRIMARY],
                       auxiliary_pairs=[f'{b} vs {a}' for _, a, b in AUX],
                       ci_method='事故组整群 bootstrap 2000 次（seed 42）；'
                                 'McNemar 精确二项仅作行级补充；'
                                 '组级配对符号翻转置换 20000 次（seed 42）',
                       pre_fix_source=RQ2, seconds=round(time.time() - t0, 2)),
                  f, ensure_ascii=False, indent=2)
    log(f'\n总耗时 {time.time() - t0:.1f}s')
    LOGF.close()


if __name__ == '__main__':
    main()
