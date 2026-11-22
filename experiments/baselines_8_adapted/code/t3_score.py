"""
t3_score.py — 阶段 D rev2 / G5：Tier 3 八个外部基线的同口径统计（零训练）

唯一评价入口：所有指标都从**逐样本预测 CSV**（真值 + 6 维分数）重新计算，
不从训练日志取任何 best 值。

口径与 Tier 1 完全一致（沿用同一套统计构件）：
  * 事故组整群 bootstrap 95% CI（依赖单位 = accident_group，2,000 次，seed 42）
  * 行级 McNemar 精确二项检验（Accuracy 族）
  * 事故组配对符号翻转置换检验（Macro-F1 族；统计量 = Macro-F1 差值 b − a，
    20,000 次，seed 42，加一修正）
  * Holm 步降法，**按指标分族**：Accuracy 族与 Macro-F1 族各自校正。
    比较族在运行前写定：P_no_proto vs 八个外部基线（8 项），两族各 8 项。

不训练、不改模型、不读 2024+ 封存池标签。

输出（results/ 下）：
  t3_baseline_metrics.csv        八基线主表
  t3_baseline_per_class.csv      八基线逐类 P/R/F1/support
  t3_baseline_confusion.csv      八基线混淆矩阵
  t3_baseline_vs_geoad.csv       逐基线 vs P_no_proto 的配对比较（两族）
  t3_holm.csv                    Holm 校正结果
  t3_macro_permutation.csv        Macro-F1 事故组置换检验明细
  t3_new_vs_old.csv              新旧基线结果并列（10-25 旧归档 vs 本轮）
  t3_score_meta.json             口径、输入哈希、计时
"""
import json
import os
import sys
import time
import hashlib
from math import comb
from fractions import Fraction

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix, matthews_corrcoef)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T1 = ARCH_ROOT + '/GeOAD_ReEval_Run_20261028_Tier1'
OLD = ARCH_ROOT + '/GeOAD_RQ1_FormalValidation_20261025'
sys.path.insert(0, os.path.join(T1, 'code'))

PRED = os.path.join(HERE, 'predictions', '13d')
RES = os.path.join(HERE, 'results')
LOGP = os.path.join(HERE, 'logs')
os.makedirs(RES, exist_ok=True)
os.makedirs(LOGP, exist_ok=True)
LOGF = open(os.path.join(LOGP, 't3_score_log.txt'), 'w', encoding='utf-8')

SIX = ['CORROSION', 'EQUIPMENT_FAILURE', 'EXTERNAL_DAMAGE',
       'MATERIAL_DEFECT', 'OPERATIONAL', 'OTHER']
HARD = ['MATERIAL_DEFECT', 'OPERATIONAL']
BASELINES = ['AUTDL', 'CIC', 'DeepPipe', 'DeepPipeNet', 'SART-1DCNN',
             'MFF-CNN-LightGBM', '1DCNN-DBO-LSTM', 'SMDPSO-SVM']
OFFICIAL = 'P_no_proto'          # 官方模型（Tier 1，修复后数据）
N_PERM = 20000
SEED = 42
BLOCK = 2000
N_BOOT = 2000


def log(msg):
    print(msg, flush=True)
    LOGF.write(str(msg) + '\n')
    LOGF.flush()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ 统计构件
def mcnemar_exact(b, c):
    """Exact (binomial) McNemar two-sided p-value.

    Uses exact integer arithmetic (math.comb / Fraction): the naive
    2.0 ** n form overflows for the discordant-pair counts seen here
    (n of several hundred)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = Fraction(sum(comb(n, i) for i in range(0, k + 1)), 1 << n)
    return float(min(1.0, 2 * tail))


def group_confusion(y, pred, groups):
    uniq, inv = np.unique(groups, return_inverse=True)
    conf = np.zeros((len(uniq), 6, 6), dtype=np.int64)
    np.add.at(conf, (inv, y, pred), y * 0 + 1)
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
    return float(tp.sum() / n if n else 0.0), float(f1.mean())


def paired_group_bootstrap(y, pred_a, pred_b, groups, n_boot=N_BOOT, seed=SEED):
    """事故组整群 bootstrap：同时给出 Accuracy 差与 Macro-F1 差 的 95% CI。"""
    rng = np.random.default_rng(seed)
    _, ca = group_confusion(y, pred_a, groups)
    _, cb = group_confusion(y, pred_b, groups)
    n_g = ca.shape[0]
    da, db = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n_g, n_g)
        da.append(f1_acc_from_conf(ca[idx].sum(axis=0)))
        db.append(f1_acc_from_conf(cb[idx].sum(axis=0)))
    da = np.array(da)
    db = np.array(db)
    d_acc = db[:, 0] - da[:, 0]
    d_f1 = db[:, 1] - da[:, 1]
    obs_a = f1_acc_from_conf(ca.sum(axis=0))
    obs_b = f1_acc_from_conf(cb.sum(axis=0))
    return dict(
        acc_diff=obs_b[0] - obs_a[0],
        acc_ci=[float(np.quantile(d_acc, 0.025)), float(np.quantile(d_acc, 0.975))],
        f1_diff=obs_b[1] - obs_a[1],
        f1_ci=[float(np.quantile(d_f1, 0.025)), float(np.quantile(d_f1, 0.975))],
        n_groups=int(n_g))


def group_matrices(y, pred, groups, n_classes=6):
    G = len(np.unique(groups))
    inv = np.unique(groups, return_inverse=True)[1]
    TP = np.zeros((G, n_classes))
    PC = np.zeros((G, n_classes))
    for c in range(n_classes):
        pc = (pred == c)
        TP[:, c] = np.bincount(inv[pc & (y == c)], minlength=G)
        PC[:, c] = np.bincount(inv[pc], minlength=G)
    return TP, PC


def macro_f1_from_counts(TP, PC, support):
    den = PC + support[None, :]
    f1 = np.where(den > 0, 2.0 * TP / np.maximum(den, 1e-12), 0.0)
    return f1.mean(axis=1)


def perm_macro_f1(TP_a, PC_a, TP_b, PC_b, support, n_perm=N_PERM, seed=SEED):
    G = TP_a.shape[0]
    rng = np.random.default_rng(seed)
    obs = float(macro_f1_from_counts(TP_b.sum(axis=0)[None, :], PC_b.sum(axis=0)[None, :],
                                     support)[0]
                - macro_f1_from_counts(TP_a.sum(axis=0)[None, :], PC_a.sum(axis=0)[None, :],
                                       support)[0])
    nulls = np.empty(n_perm)
    for s in range(0, n_perm, BLOCK):
        k = min(BLOCK, n_perm - s)
        f = (rng.random((k, G)) < 0.5).astype(np.float64)
        om = 1.0 - f
        nulls[s:s + k] = (macro_f1_from_counts(om @ TP_b + f @ TP_a,
                                               om @ PC_b + f @ PC_a, support)
                          - macro_f1_from_counts(om @ TP_a + f @ TP_b,
                                                 om @ PC_a + f @ PC_b, support))
    p_two = float((np.sum(np.abs(nulls) >= abs(obs)) + 1) / (n_perm + 1))
    p_one = float((np.sum(nulls >= obs) + 1) / (n_perm + 1))
    return obs, nulls, p_two, p_one


def holm(pvals, names):
    """Holm 步降法；返回 {name: (holm_p, reject)}，保持字典序输入顺序。"""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    out = {}
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[i])
        running = max(running, adj)
        out[names[i]] = (float(running), bool(running <= 0.05))
    return out


# ------------------------------------------------------------------ 载入
def load_pred(path):
    df = pd.read_csv(path)
    cols = list(df.columns)
    if 'y_true' not in cols:
        raise RuntimeError(f'{path}: 缺 y_true')
    sc = [c for c in cols if c.startswith('prob_')] or \
         [c for c in cols if c.startswith('score_') and c != 'score_type']
    if len(sc) != 6:
        raise RuntimeError(f'{path}: 分数列 {sc} 不是 6 列')
    S = df[sc].to_numpy(float)
    # 基线自身的决策列 pred 是权威的（runner 已写出）；仅当缺失时才由分数取 argmax。
    if 'pred' in cols:
        pred = df['pred'].to_numpy(int)
    else:
        pred = S.argmax(1)
    df = df.copy()
    df['pred'] = pred
    return df, df['y_true'].to_numpy(int), pred, groups_of(df), sc


def groups_of(df):
    if 'accident_group' in df.columns:
        return df['accident_group'].astype(str).to_numpy()
    raise RuntimeError('缺 accident_group')


def main():
    t0 = time.time()
    log('=== Tier 3 / G5 同口径统计开始 ===')
    meta = {'n_perm': N_PERM, 'seed': SEED, 'n_boot': N_BOOT,
            'official_model': OFFICIAL,
            'official_source': T1,
            'baseline_dir': PRED,
            'inputs': {}}

    # 官方模型
    op = os.path.join(T1, 'predictions', f'{OFFICIAL}.csv')
    dfg, yg, pg, gg, scg = load_pred(op)
    meta['inputs'][OFFICIAL] = {'path': op, 'sha256': sha256_file(op)}

    rows, perclass, confs = [], [], []
    for slug in BASELINES:
        p = os.path.join(PRED, f'{slug}.csv')
        if not os.path.isfile(p):
            log(f'!! 缺预测文件 {p} —— 该基线未纳入本次统计')
            continue
        df, y, pred, groups, sc = load_pred(p)
        meta['inputs'][slug] = {'path': p, 'sha256': sha256_file(p)}
        assert len(df) == len(dfg), (slug, len(df), len(dfg))
        assert df.row_id.tolist() == dfg.row_id.tolist(), f'{slug}: row_id 顺序不一致'
        acc = accuracy_score(y, pred)
        macro = f1_score(y, pred, average='macro', zero_division=0)
        w = f1_score(y, pred, average='weighted', zero_division=0)
        hard = f1_score(y, pred, average='macro', labels=[SIX.index(c) for c in HARD],
                        zero_division=0)
        rows.append(dict(arm=slug, n=len(y), accuracy=acc, macro_f1=macro,
                         weighted_f1=w, hard_class_f1=hard,
                         mcc=matthews_corrcoef(y, pred), n_correct=int((y == pred).sum()),
                         n_groups=int(len(np.unique(groups))),
                         score_columns='|'.join(sc)))
        pr, rc, f1, sup = precision_recall_fscore_support(
            y, pred, labels=list(range(6)), zero_division=0)
        for i, c in enumerate(SIX):
            perclass.append(dict(arm=slug, cls=c, precision=pr[i], recall=rc[i],
                                 f1=f1[i], support=int(sup[i])))
        cm = confusion_matrix(y, pred, labels=list(range(6)))
        for i, c in enumerate(SIX):
            confs.append(dict(arm=slug, true_cls=c,
                               **{f'pred_{SIX[j]}': int(cm[i, j]) for j in range(6)}))
        log(f'  {slug:20s} Acc {acc*100:.4f}%  MacroF1 {macro*100:.4f}%  '
            f'W-F1 {w*100:.4f}%  hard-F1 {hard*100:.4f}%')

    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(RES, 't3_baseline_metrics.csv'), index=False)
    pd.DataFrame(perclass).to_csv(os.path.join(RES, 't3_baseline_per_class.csv'), index=False)
    pd.DataFrame(confs).to_csv(os.path.join(RES, 't3_baseline_confusion.csv'), index=False)

    # 官方模型一行并入主表
    accg = accuracy_score(yg, pg)
    rows_g = [dict(arm=OFFICIAL + ' (official GeOAD)', n=len(yg), accuracy=accg,
                   macro_f1=f1_score(yg, pg, average='macro', zero_division=0),
                   weighted_f1=f1_score(yg, pg, average='weighted', zero_division=0),
                   hard_class_f1=f1_score(yg, pg, average='macro',
                                          labels=[SIX.index(c) for c in HARD], zero_division=0),
                   mcc=matthews_corrcoef(yg, pg), n_correct=int((yg == pg).sum()),
                   n_groups=int(len(np.unique(gg))),
                   score_columns='|'.join(scg))]
    pd.DataFrame(rows_g + rows).to_csv(os.path.join(RES, 't3_all_arms_metrics.csv'), index=False)

    # ---------------- 配对比较：P_no_proto (b) vs 各基线 (a) ----------------
    support = np.bincount(yg, minlength=6).astype(float)
    comp_rows, perm_rows = [], []
    acc_p, f1_p, names = [], [], []
    for slug in BASELINES:
        p = os.path.join(PRED, f'{slug}.csv')
        if not os.path.isfile(p):
            continue
        df, y, pred, groups, sc = load_pred(p)
        ok_a = (pred == y)
        ok_b = (pg == y)
        b_only = int((~ok_a & ok_b).sum())      # 基线错、GeOAD 对 -> 纠正
        c_only = int((ok_a & ~ok_b).sum())      # 基线对、GeOAD 错 -> 破坏
        net = b_only - c_only
        pmc = mcnemar_exact(c_only, b_only)
        bs = paired_group_bootstrap(y, pred, pg, groups)
        TP_a, PC_a = group_matrices(y, pred, groups)
        TP_b, PC_b = group_matrices(y, pg, groups)
        obs, nulls, p_two, p_one = perm_macro_f1(TP_a, PC_a, TP_b, PC_b, support)
        acc_p.append(pmc)
        f1_p.append(p_two)
        names.append(slug)
        comp_rows.append(dict(
            baseline=slug, comparison=f'P_no_proto vs {slug}',
            acc_diff_pt=bs['acc_diff'] * 100, acc_ci95_pt=[round(v * 100, 4) for v in bs['acc_ci']],
            mcnemar_p=pmc, corrected=b_only, harmed=c_only, net=net,
            macro_f1_diff_pt=bs['f1_diff'] * 100,
            macro_f1_ci95_pt=[round(v * 100, 4) for v in bs['f1_ci']],
            macro_f1_perm_p=p_two, macro_f1_perm_p_plus1=float((np.sum(np.abs(nulls) >= abs(obs)) + 1) / (N_PERM + 1)),
            macro_f1_perm_p_one_sided=p_one,
            n_groups=int(len(np.unique(groups)))))
        perm_rows.append(dict(
            comparison=f'P_no_proto vs {slug}', n_rows=len(y),
            n_groups=int(len(np.unique(groups))),
            macro_f1_a=float(f1_score(y, pred, average='macro', zero_division=0)),
            macro_f1_b=float(f1_score(yg, pg, average='macro', zero_division=0)),
            macro_f1_diff_pt=obs * 100, macro_f1_perm_p_two_sided=p_two,
            macro_f1_null_mean=float(nulls.mean()), macro_f1_null_sd=float(nulls.std()),
            macro_f1_null_q025=float(np.quantile(nulls, 0.025)),
            macro_f1_null_q975=float(np.quantile(nulls, 0.975)),
            n_perm=N_PERM, seed=SEED,
            acc_obs_diff_pt=bs['acc_diff'] * 100))
        log(f'  cmp {slug:20s} dAcc {bs["acc_diff"]*100:+.4f}pt (McNemar p={pmc:.4f})  '
            f'dMacroF1 {bs["f1_diff"]*100:+.4f}pt (perm p={p_two:.5f})  '
            f'corr {b_only} / harm {c_only}')

    holm_acc = holm(acc_p, names)
    holm_f1 = holm(f1_p, names)
    for r in comp_rows:
        s = r['baseline']
        r['holm_p_mcnemar_acc'] = holm_acc[s][0]
        r['reject_acc_0p05'] = holm_acc[s][1]
        r['holm_p_macro_perm'] = holm_f1[s][0]
        r['reject_macro_perm_0p05'] = holm_f1[s][1]
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(os.path.join(RES, 't3_baseline_vs_geoad.csv'), index=False)
    pd.DataFrame(perm_rows).to_csv(os.path.join(RES, 't3_macro_permutation.csv'), index=False)
    comp[['comparison', 'mcnemar_p', 'holm_p_mcnemar_acc', 'reject_acc_0p05',
          'macro_f1_perm_p', 'holm_p_macro_perm', 'reject_macro_perm_0p05',
          'corrected', 'harmed', 'net']].to_csv(os.path.join(RES, 't3_holm.csv'), index=False)

    # ---------------- 新旧并列（10-25 旧归档 vs 本轮修复后重训） ----------------
    old_rows = []
    for slug in BASELINES:
        op_old = os.path.join(OLD, 'predictions', '13d', f'{slug}.csv')
        if not os.path.isfile(op_old):
            continue
        do, yo, po, go, _ = load_pred(op_old)
        old_rows.append(dict(arm=slug,
                             old_accuracy=accuracy_score(yo, po),
                             old_macro_f1=f1_score(yo, po, average='macro', zero_division=0),
                             old_n=len(yo)))
        pn = os.path.join(PRED, f'{slug}.csv')
        dn, yn, pn_, gn, _ = load_pred(pn)
        old_rows[-1]['new_accuracy'] = accuracy_score(yn, pn_)
        old_rows[-1]['new_macro_f1'] = f1_score(yn, pn_, average='macro', zero_division=0)
        old_rows[-1]['new_n'] = len(yn)
        old_rows[-1]['delta_acc_pt'] = (old_rows[-1]['new_accuracy']
                                        - old_rows[-1]['old_accuracy']) * 100
        old_rows[-1]['delta_macro_f1_pt'] = (old_rows[-1]['new_macro_f1']
                                             - old_rows[-1]['old_macro_f1']) * 100
        old_rows[-1]['same_row_ids'] = bool(do.row_id.tolist() == dn.row_id.tolist())
    if old_rows:
        pd.DataFrame(old_rows).to_csv(os.path.join(RES, 't3_new_vs_old.csv'), index=False)
        log('  新旧并列已写出：results/t3_new_vs_old.csv')

    meta['elapsed_sec'] = round(time.time() - t0, 1)
    meta['note'] = ('Holm 按指标分族：Accuracy 族 = 8 项（McNemar），'
                     'Macro-F1 族 = 8 项（事故组配对置换，20,000 次，seed 42）。'
                     '比较族在运行前写定，不因结果调整。'
                     'dev_confirm 为开发性验证集，非独立盲测。')
    with open(os.path.join(RES, 't3_score_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f'=== G5 统计完成，用时 {meta["elapsed_sec"]}s ===')


if __name__ == '__main__':
    main()
