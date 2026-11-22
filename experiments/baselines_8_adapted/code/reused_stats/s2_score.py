"""
s2_score.py — RQ2 第二小阶段 · 门槛 C：独立统计与机制核算（零训练）

============================================================================
**唯一评价入口**：所有指标都从 `predictions_s2/<arm>.csv`（逐样本真值 + 概率）
重新计算，**不从训练日志取任何 best 值**。

主比较（协议锁 analysis_plan.primary_pairs）
    P_full vs A_ensemble
    P_full vs F_concat
辅比较（auxiliary_pairs）
    P_full vs A_single、P_full vs 三个重训消融

统计方法
    * 行级 McNemar 精确二项检验（**仅作补充**，需声明适用前提）
    * **事故组整群 bootstrap 95% CI**（依赖单位 = accident_group，主口径）
    * 三个重训消融做 **Holm 多重比较校正**
    * 方向标签一律写成"X 相对 Y 净多/净少 N 条"，避免第一小阶段的符号歧义

输出
    results/s2_metrics.csv                七臂主表
    results/s2_per_class.csv              六类 P/R/F1/support
    results/s2_confusion.csv              全部混淆矩阵
    results/s2_paired_diffs.csv            配对差异（纠错/破坏/净正确 + CI + 检验）
    results/s2_holm.csv                   三消融 Holm 校正
    results/s2_pattern_unseen.csv          13 维模式未在 fit 出现的 dev_confirm 子集
    results/s2_prob_legality.csv          概率合法性核查
    results/s2_score_meta.json            核查与计时
    logs/s2_score_run_log.txt             真实运行日志
"""
import os
import sys
import json
import time
from math import comb

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix, matthews_corrcoef)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'code'))
import s2_common as S2  # noqa: E402

PRED = os.path.join(HERE, 'predictions_s2')
RES = os.path.join(HERE, 'results')
LOGP = os.path.join(HERE, 'logs')
os.makedirs(RES, exist_ok=True)
os.makedirs(LOGP, exist_ok=True)

_LOGF = open(os.path.join(LOGP, 's2_score_run_log.txt'), 'w', encoding='utf-8')


def log(msg):
    print(msg, flush=True)
    _LOGF.write(str(msg) + '\n')
    _LOGF.flush()


ARMS = ['A_single', 'A_ensemble', 'F_concat', 'P_full',
        'P_no_proto', 'P_no_rule', 'P_no_neural']
HARD = ['MATERIAL_DEFECT', 'OPERATIONAL']
PRIMARY = [('P_full', 'A_ensemble'), ('P_full', 'F_concat')]
AUXILIARY = [('P_full', 'A_single')]
ABLATION = [('P_full', 'P_no_proto'), ('P_full', 'P_no_rule'), ('P_full', 'P_no_neural')]


# ------------------------------------------------------------------ 读取
def load_arm(name, subdir=''):
    d = os.path.join(PRED, subdir, f'{name}.csv')
    df = pd.read_csv(d)
    probs = df[[f'prob_{c}' for c in S2.SIX]].to_numpy(dtype=float)
    return df, probs


def mcnemar_exact(b, c):
    """McNemar 精确二项检验（b/c 为不一致计数）。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return float(min(1.0, 2 * tail))


# ------------------------------------------------------------------ 组级 bootstrap
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
    """事故组整群 bootstrap：返回 acc / macro-F1 差值的 95% CI（b − a）。"""
    rng = np.random.default_rng(seed)
    _, ca = group_confusion(y, pred_a, groups)
    _, cb = group_confusion(y, pred_b, groups)
    n_g = ca.shape[0]
    accs, f1s, nets, hards = [], [], [], []
    ok_a = (pred_a == y)
    ok_b = (pred_b == y)
    uniq, inv = np.unique(groups, return_inverse=True)
    net_by_g = np.zeros(n_g)
    np.add.at(net_by_g, inv, (ok_b.astype(int) - ok_a.astype(int)))
    hard_idx = [S2.CLS2IDX[c] for c in HARD]

    def hard_f1(conf):
        f1s_ = []
        for ci in hard_idx:
            tp = conf[ci, ci]
            sup = conf[ci].sum()
            pp = conf[:, ci].sum()
            pr = tp / pp if pp else 0.0
            rc = tp / sup if sup else 0.0
            f1s_.append(2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0)
        return float(np.mean(f1s_))

    for _ in range(n_boot):
        pick = rng.integers(0, n_g, n_g)
        sa = ca[pick].sum(axis=0)
        sb = cb[pick].sum(axis=0)
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
    """事故组层面的**配对符号翻转置换检验**（主统计量的 p 值）。

    依赖单位 = accident_group：在每个事故组内随机翻转 (b−a) 的差值符号，
    重采样组总和，得到"零假设下净正确差"的零分布。
    """
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
    """Holm 步降法，返回 (校正后 p, 是否拒绝 0.05)。"""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        v = (m - rank) * pvals[i]
        run = max(run, v)
        adj[i] = min(1.0, run)
    return adj, adj <= 0.05


# ------------------------------------------------------------------ 主流程
def main():
    t0 = time.time()
    log('=== RQ2 第二小阶段 · 门槛 C：独立统计与机制核算（零训练）===')
    lock = S2.load_lock()
    log(f'协议锁: {lock["protocol_id"]}')

    D = {}
    for a in ARMS:
        df, probs = load_arm(a)
        D[a] = dict(df=df, probs=probs)
    ref = D['P_full']['df']
    y = ref['y_true'].to_numpy().astype(int)
    groups = ref['accident_group'].astype(str).to_numpy()

    # --- 对齐核查 ---
    for a in ARMS:
        assert (D[a]['df'].row_id.to_numpy() == ref.row_id.to_numpy()).all(), f'{a} 行序不一致'
        assert (D[a]['df'].accident_group.to_numpy() == groups).all(), f'{a} 事故组不一致'
        assert (D[a]['df'].y_true.to_numpy() == y).all(), f'{a} 真值不一致'
    log('[核查] 七臂逐样本文件行序 / 事故组 / 真值完全对齐')

    # --- 概率合法性 ---
    leg = []
    for a in ARMS:
        p = D[a]['probs']
        leg.append(dict(arm=a,
                        all_finite=bool(np.isfinite(p).all()),
                        all_nonneg=bool((p >= 0).all()),
                        max_rowsum_dev=float(np.abs(p.sum(axis=1) - 1).max()),
                        min_prob=float(p.min()), max_prob=float(p.max())))
    legdf = pd.DataFrame(leg)
    legdf.to_csv(os.path.join(RES, 's2_prob_legality.csv'), index=False)
    assert legdf.all_finite.all() and legdf.all_nonneg.all()
    assert (legdf.max_rowsum_dev < 1e-6).all(), '概率行和不等于 1'
    log(f'[核查] 概率合法性全部通过（最大行和偏差 {legdf.max_rowsum_dev.max():.2e}）')

    # --- 主表 ---
    rows, perclass, confs, conf_long = [], [], [], []
    for a in ARMS:
        df = D[a]['df']
        pred = df['pred'].to_numpy().astype(int)
        m = dict(arm=a,
                 n=len(y),
                 accuracy=float(accuracy_score(y, pred)),
                 macro_f1=float(f1_score(y, pred, average='macro', zero_division=0)),
                 weighted_f1=float(f1_score(y, pred, average='weighted', zero_division=0)),
                 mcc=float(matthews_corrcoef(y, pred)),
                 hard_class_f1=float(np.mean([
                     f1_score(y, pred, labels=[S2.CLS2IDX[c]], average='macro',
                              zero_division=0) for c in HARD])),
                 n_correct=int((pred == y).sum()),
                 n_sites=int(len(set(groups))))
        rows.append(m)
        p_, r_, f_, s_ = precision_recall_fscore_support(y, pred, labels=range(6),
                                                         zero_division=0)
        for i, c in enumerate(S2.SIX):
            perclass.append(dict(arm=a, cls=c, precision=float(p_[i]), recall=float(r_[i]),
                                 f1=float(f_[i]), support=int(s_[i])))
        cm = confusion_matrix(y, pred, labels=range(6))
        confs.append(dict(arm=a, matrix=json.dumps(cm.tolist())))
        for i, c in enumerate(S2.SIX):
            for j, cp in enumerate(S2.SIX):
                conf_long.append(dict(arm=a, true_class=c, pred_class=cp, n=int(cm[i, j])))
    mdf = pd.DataFrame(rows)
    mdf.to_csv(os.path.join(RES, 's2_metrics.csv'), index=False)
    pd.DataFrame(perclass).to_csv(os.path.join(RES, 's2_per_class.csv'), index=False)
    pd.DataFrame(confs).to_csv(os.path.join(RES, 's2_confusion.csv'), index=False)
    pd.DataFrame(conf_long).to_csv(os.path.join(RES, 's2_confusion_long.csv'), index=False)
    log('\n主表（dev_confirm %d 行，开发性验证集，非盲测）:\n%s'
        % (len(y), mdf.to_string(index=False)))

    # --- 自然类分布 ---
    dist = pd.DataFrame({'cls': S2.SIX,
                          'n_dev_confirm': [int((y == i).sum()) for i in range(6)],
                          'pct': [round(float((y == i).mean()) * 100, 2) for i in range(6)]})
    log('\n自然类分布（保持原分布，未做任何平衡）:\n' + dist.to_string(index=False))
    dist.to_csv(os.path.join(RES, 's2_class_distribution.csv'), index=False)

    # --- 配对比较 ---
    def paired(a, b, kind):
        """b 相对 a：纠错（b 修正 a 的错）、破坏（b 破坏 a 的对）、净正确。"""
        pa = D[a]['df']['pred'].to_numpy().astype(int)
        pb = D[b]['df']['pred'].to_numpy().astype(int)
        ok_a, ok_b = (pa == y), (pb == y)
        fixed = int((~ok_a & ok_b).sum())
        broken = int((ok_a & ~ok_b).sum())
        changed = int((pa != pb).sum())
        bs = paired_group_bootstrap(y, pa, pb, groups)
        pm = mcnemar_exact(fixed, broken)
        pg, obs = group_permutation_p(y, pa, pb, groups)
        return dict(comparison=f'{b} vs {a}', kind=kind,
                    a=a, b=b,
                    a_correct=int(ok_a.sum()), b_correct=int(ok_b.sum()),
                    b_net_vs_a=fixed - broken,
                    direction=f'{b} 相对 {a} 净多 {fixed - broken} 条'
                              f'（纠错 {fixed} / 破坏 {broken}）',
                    changed_decisions=changed,
                    mcnemar_p=pm,
                    group_permutation_p=pg,
                    group_net_observed=obs,
                    acc_diff_a_to_b=bs['acc_diff'],
                    acc_ci95=f"[{bs['acc_lo']:.5f}, {bs['acc_hi']:.5f}]",
                    macro_f1_diff_a_to_b=bs['macro_f1_diff'],
                    macro_f1_ci95=f"[{bs['f1_lo']:.5f}, {bs['f1_hi']:.5f}]",
                    hard_class_f1_diff_a_to_b=bs['hard_f1_diff'],
                    hard_class_f1_ci95=f"[{bs['hard_lo']:.5f}, {bs['hard_hi']:.5f}]",
                    net_ci95=f"[{bs['net_lo']:.0f}, {bs['net_hi']:.0f}]",
                    n_groups=bs['n_groups'], n_boot=bs['n_boot'])

    diffs = [paired(a, b, 'primary') for a, b in PRIMARY] + \
            [paired(a, b, 'auxiliary') for a, b in AUXILIARY] + \
            [paired(a, b, 'ablation') for a, b in ABLATION]
    ddf = pd.DataFrame(diffs)
    ddf.to_csv(os.path.join(RES, 's2_paired_diffs.csv'), index=False)
    log('\n配对差异（净正确方向明确标注；CI = 事故组整群 bootstrap 2000 次）:\n'
        + ddf[['comparison', 'kind', 'b_correct', 'b_net_vs_a', 'changed_decisions',
               'mcnemar_p', 'group_permutation_p', 'acc_diff_a_to_b', 'acc_ci95',
               'macro_f1_diff_a_to_b',
               'macro_f1_ci95', 'hard_class_f1_diff_a_to_b',
               'hard_class_f1_ci95']].to_string(index=False))

    # --- Holm（三个重训消融） ---
    sub = ddf[ddf.kind == 'ablation'].copy()
    adj_m, rej_m = holm(sub.mcnemar_p.to_numpy())
    adj_g, rej_g = holm(sub.group_permutation_p.to_numpy())
    hdf = pd.DataFrame(dict(comparison=sub.comparison.values,
                            mcnemar_p=sub.mcnemar_p.values,
                            holm_p_mcnemar=adj_m, reject_mcnemar_0p05=rej_m,
                            group_permutation_p=sub.group_permutation_p.values,
                            holm_p_group_perm=adj_g, reject_groupperm_0p05=rej_g,
                            net=sub.b_net_vs_a.values,
                            direction=sub.direction.values,
                            acc_ci95=sub.acc_ci95.values,
                            macro_f1_ci95=sub.macro_f1_ci95.values))
    hdf.to_csv(os.path.join(RES, 's2_holm.csv'), index=False)
    log('\nHolm 校正（三个重训消融）:\n' + hdf.to_string(index=False))

    # --- 13 维模式未在 fit 出现的 dev_confirm 子集（敏感性，非主分数）---
    man = S2.load_manifest()
    fit_pat = set(man[man.split == 'fit'].pattern13.astype(str))
    pat = ref['row_id'].map(man.set_index('row_id').pattern13.astype(str))
    unseen = ~pat.isin(fit_pat).to_numpy()
    log(f'\n[敏感性] dev_confirm 中 13 维模式未在 fit 出现的行: {unseen.sum()}/{len(y)} '
        f'({unseen.mean()*100:.1f}%)')
    sub_rows = []
    for a in ARMS:
        pred = D[a]['df']['pred'].to_numpy().astype(int)
        yu = y[unseen]
        pu = pred[unseen]
        sub_rows.append(dict(arm=a, n=int(unseen.sum()),
                             accuracy_unseen=float(accuracy_score(yu, pu)),
                             macro_f1_unseen=float(f1_score(yu, pu, average='macro',
                                                           zero_division=0))))
    udf = pd.DataFrame(sub_rows)
    udf.to_csv(os.path.join(RES, 's2_pattern_unseen.csv'), index=False)
    log(udf.to_string(index=False))

    with open(os.path.join(RES, 's2_score_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(protocol=lock['protocol_id'],
                       n_dev_confirm=int(len(y)),
                       n_groups=int(len(set(groups))),
                       arms=ARMS,
                       primary_pairs=[f'{b} vs {a}' for a, b in PRIMARY],
                       auxiliary_pairs=[f'{b} vs {a}' for a, b in AUXILIARY],
                       ablation_pairs=[f'{b} vs {a}' for a, b in ABLATION],
                       ci_method='事故组整群 bootstrap 2000 次（seed 42）；'
                                 'McNemar 精确二项仅作行级补充',
                       holm_applied_to='三个重训消融（mcnemar_p）',
                       seconds=round(time.time() - t0, 2)), f, ensure_ascii=False, indent=2)
    log(f'\n总耗时 {time.time() - t0:.1f}s')
    _LOGF.close()


if __name__ == '__main__':
    main()
