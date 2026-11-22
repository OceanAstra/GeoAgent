"""rq1f_analyze.py — RQ1 正式验证：统一指标 + 配对统计 + 六问结论

数据来源（全部只读）：
  * 内部对照 A_single / A_ensemble / F_concat / P_full
        → GeOAD_RQ2_SecondStage_LargeDevConfirm_20261025/predictions_s2/*.csv
        （协议、特征、样本 ID、评价定义与本轮完全一致 ⇒ 允许直接复用）
  * 八个外部基线（13 维为主口径，9 维为补充对照）
        → 本包 predictions/13d/*.csv、predictions/9d/*.csv

统计口径与 RQ2 第二小阶段完全一致（直接 import s2_score 的实现，避免口径漂移）：
  * 事故组整群 bootstrap 2000 次（seed 42）→ acc / macro-F1 差值 95% CI
  * 事故组层面配对符号翻转置换 20,000 次（seed 42）→ p 值
  * McNemar 精确二项检验 → p 值
  * Holm 步降法对 10 个主比较做多重比较校正

输出（results/）：
  rq1f_main_table.csv      统一主结果表（所有模型 × 全部指标 + 训练/预测耗时）
  rq1f_per_class.csv       逐类 Precision/Recall/F1/support（所有模型）
  rq1f_confusion.csv       混淆矩阵长表（所有模型）
  rq1f_paired_stats.csv    P_full 对每个对照的配对差异 + CI + p 值
  rq1f_holm.csv            Holm 校正结果
  rq1f_timing.csv          训练/预测耗时表
  rq1f_era_secondary.csv   事前定义的时代分层（次要结果，附总体成绩）
  rq1f_summary.json        机器可读汇总
"""
import os
import sys
import json
import importlib.util

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, classification_report)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RQ2 = ARCH_ROOT + '/GeOAD_RQ2_SecondStage_LargeDevConfirm_20261025'
RES = os.path.join(HERE, 'results')
os.makedirs(RES, exist_ok=True)

# ---- 复用 RQ2 的统计实现（口径不漂移）----------------------------------------
sys.path.insert(0, os.path.join(RQ2, 'code'))
import s2_common as S2  # noqa: E402
import s2_score as SC  # noqa: E402

SIX = S2.SIX
HARD = ['MATERIAL_DEFECT', 'OPERATIONAL']  # 用户要求必须完整保留的两个类别

INTERNAL = {
    'GeOAD_P (P_full)': 'P_full',
    'A_ensemble (CatBoost集成)': 'A_ensemble',
    'A_single (CatBoost单种子)': 'A_single',
    'F_concat (非几何F)': 'F_concat',
}
EXTERNAL = ['AUTDL', 'CIC', 'DeepPipe', 'DeepPipeNet', 'SART-1DCNN',
            'MFF-CNN-LightGBM', '1DCNN-DBO-LSTM', 'SMDPSO-SVM']
# 主比较集合：P 对八个外部基线 + A_ensemble + F_concat = 10 个
MAIN_COMPARATORS = EXTERNAL + ['A_ensemble (CatBoost集成)', 'F_concat (非几何F)']


def load_internal(name):
    df = pd.read_csv(os.path.join(RQ2, 'predictions_s2', f'{name}.csv'))
    return df


def load_external(variant, slug):
    p = os.path.join(HERE, 'predictions', variant, f'{slug}.csv')
    if not os.path.isfile(p):
        return None
    return pd.read_csv(p)


def mcnemar_exact_stable(b, c):
    """McNemar 精确二项检验（与 s2_score.mcnemar_exact **同一个统计量**）。

    s2_score 的实现用 `2.0 ** n` 构造二项分母，在不一致对子数 n > 1023 时会 OverflowError；
    本函数用 scipy 的等价二项检验替代（p = 2·P(X ≤ min(b,c)) 上限截断到 1），
    在小 n 上已与原实现逐位核对一致（见 results/rq1f_mcnemar_equivalence.json）。
    """
    from scipy.stats import binomtest
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, binomtest(k, n, 0.5).pvalue))


def metrics_block(y, pred, prefix):
    cm = confusion_matrix(y, pred, labels=list(range(6)))
    row = {
        f'{prefix}accuracy': accuracy_score(y, pred),
        f'{prefix}macro_f1': f1_score(y, pred, average='macro', zero_division=0),
        f'{prefix}weighted_f1': f1_score(y, pred, average='weighted', zero_division=0),
        f'{prefix}macro_precision': precision_score(y, pred, average='macro', zero_division=0),
        f'{prefix}macro_recall': recall_score(y, pred, average='macro', zero_division=0),
    }
    for c in HARD:
        i = SIX.index(c)
        row[f'{prefix}{c}_f1'] = f1_score(y, pred, labels=[i], average='macro', zero_division=0)
    return row, cm


def main():
    print('=== RQ1 正式验证：统一指标与配对统计 ===', flush=True)
    man = pd.read_csv(os.path.join(RQ2, 'protocol', 'data_row_manifest.csv'))
    man = man.set_index('row_id')

    models = {}          # display name -> dataframe
    kind = {}
    for disp, arm in INTERNAL.items():
        models[disp] = load_internal(arm)
        kind[disp] = 'internal'
    for slug in EXTERNAL:
        df = load_external('13d', slug)
        if df is not None:
            models[slug] = df
            kind[slug] = 'external_13d'
    models9 = {}
    for slug in EXTERNAL:
        df = load_external('9d', slug)
        if df is not None:
            models9[slug] = df

    # ---- 对齐检查：所有模型必须在同一批 6,417 行、同一标签、同一事故组上 --------
    base = models['GeOAD_P (P_full)']
    rid0 = base.row_id.to_numpy()
    y0 = base.y_true.to_numpy()
    g0 = base.accident_group.to_numpy()
    align = []
    for disp, df in list(models.items()) + [(f'9d::{k}', v) for k, v in models9.items()]:
        same = (len(df) == len(base) and np.array_equal(df.row_id.to_numpy(), rid0)
                and np.array_equal(df.y_true.to_numpy(), y0)
                and np.array_equal(df.accident_group.to_numpy(), g0))
        align.append({'model': disp, 'aligned': bool(same), 'n': int(len(df))})
    align_df = pd.DataFrame(align)
    align_df.to_csv(os.path.join(RES, 'rq1f_alignment_check.csv'), index=False)
    bad = align_df[~align_df.aligned]
    if len(bad):
        print('!! 未对齐模型：', bad.model.tolist())
        raise SystemExit(1)
    print(f'[OK] {len(align)} 个模型全部对齐到同一 {len(base)} 行 dev_confirm', flush=True)

    # ---- 主表 -----------------------------------------------------------------
    rows, cms, percls = [], {}, []
    for disp, df in models.items():
        y, pred = df.y_true.to_numpy(), df.pred.to_numpy()
        m, cm = metrics_block(y, pred, '')
        m.update({'model': disp, 'kind': kind.get(disp, 'external_13d'),
                  'n': len(df), 'score_type': df.get('score_type', pd.Series(['n/a'] * len(df))).iloc[0]})
        rows.append(m)
        cms[disp] = cm
        rep = classification_report(y, pred, target_names=SIX, labels=list(range(6)),
                                    output_dict=True, zero_division=0)
        for c in SIX:
            percls.append({'model': disp, 'class': c,
                           'precision': rep[c]['precision'], 'recall': rep[c]['recall'],
                           'f1': rep[c]['f1-score'], 'support': int(rep[c]['support'])})
    main_tab = pd.DataFrame(rows)
    main_tab.to_csv(os.path.join(RES, 'rq1f_main_table.csv'), index=False)
    pd.DataFrame(percls).to_csv(os.path.join(RES, 'rq1f_per_class.csv'), index=False)

    conf_rows = []
    for disp, cm in cms.items():
        for i in range(6):
            for j in range(6):
                conf_rows.append({'model': disp, 'true_class': SIX[i], 'pred_class': SIX[j],
                                  'n': int(cm[i, j])})
    pd.DataFrame(conf_rows).to_csv(os.path.join(RES, 'rq1f_confusion.csv'), index=False)

    # ---- 9d 补充 --------------------------------------------------------------
    rows9 = []
    for slug, df in models9.items():
        y, pred = df.y_true.to_numpy(), df.pred.to_numpy()
        m, _ = metrics_block(y, pred, '')
        m.update({'model': slug, 'kind': 'external_9d', 'n': len(df)})
        rows9.append(m)
    if rows9:
        pd.DataFrame(rows9).to_csv(os.path.join(RES, 'rq1f_main_table_9d.csv'), index=False)

    # ---- 配对统计：P 对每个对照 -------------------------------------------------
    p_y, p_pred, p_g = y0, base.pred.to_numpy(), g0
    prows = []
    for disp in MAIN_COMPARATORS:
        if disp not in models:
            print(f'!! 缺少对照 {disp}，跳过', flush=True)
            continue
        df = models[disp]
        pred = df.pred.to_numpy()
        fixed = (p_pred == p_y) & (pred != p_y)
        broken = (p_pred != p_y) & (pred == p_y)
        b, c = int(fixed.sum()), int(broken.sum())
        pm = mcnemar_exact_stable(b, c)
        # s2_score.paired_group_bootstrap 的差值方向是 (b − a)：
        #   传入 (a=对照, b=P) ⇒ 返回的 diff 与 CI 都是 P − 对照，与本表列名一致。
        bs = SC.paired_group_bootstrap(p_y, pred, p_pred, p_g)
        pg, obs = SC.group_permutation_p(p_y, p_pred, pred, p_g)
        prows.append({
            'comparator': disp, 'kind': kind.get(disp, 'external_13d'),
            'acc_A': float((p_pred == p_y).mean()),
            'acc_B': float((pred == p_y).mean()),
            'acc_diff_P_minus_B': float((p_pred == p_y).mean() - (pred == p_y).mean()),
            'acc_ci95_lo': bs['acc_lo'], 'acc_ci95_hi': bs['acc_hi'],
            'macro_f1_A': float(f1_score(p_y, p_pred, average='macro', zero_division=0)),
            'macro_f1_B': float(f1_score(p_y, pred, average='macro', zero_division=0)),
            'macro_f1_diff_P_minus_B': bs['macro_f1_diff'],
            'macro_f1_ci95_lo': bs['f1_lo'], 'macro_f1_ci95_hi': bs['f1_hi'],
            'hard_f1_diff_P_minus_B': bs['hard_f1_diff'],
            'hard_f1_ci95_lo': bs['hard_lo'], 'hard_f1_ci95_hi': bs['hard_hi'],
            'mcnemar_b_P_only_correct': b, 'mcnemar_c_B_only_correct': c,
            'mcnemar_p': pm, 'group_permutation_p': pg,
            'n_groups': bs['n_groups'], 'n_boot': bs['n_boot'],
        })
    paired = pd.DataFrame(prows)
    adj_m, rej_m = SC.holm(paired.mcnemar_p.to_numpy())
    adj_g, rej_g = SC.holm(paired.group_permutation_p.to_numpy())
    paired['holm_p_mcnemar'] = adj_m
    paired['reject_mcnemar_0p05'] = rej_m
    paired['holm_p_group_perm'] = adj_g
    paired['reject_groupperm_0p05'] = rej_g
    paired.to_csv(os.path.join(RES, 'rq1f_paired_stats.csv'), index=False)
    paired[['comparator', 'mcnemar_p', 'holm_p_mcnemar', 'reject_mcnemar_0p05',
            'group_permutation_p', 'holm_p_group_perm', 'reject_groupperm_0p05',
            'acc_diff_P_minus_B', 'acc_ci95_lo', 'acc_ci95_hi',
            'macro_f1_diff_P_minus_B', 'macro_f1_ci95_lo', 'macro_f1_ci95_hi'
            ]].to_csv(os.path.join(RES, 'rq1f_holm.csv'), index=False)

    # ---- 耗时 -----------------------------------------------------------------
    status = json.load(open(os.path.join(HERE, 'data', 'rq1f_run_status_13d.json'),
                            encoding='utf-8')) if os.path.isfile(
        os.path.join(HERE, 'data', 'rq1f_run_status_13d.json')) else {}
    trows = []
    for slug in EXTERNAL:
        rec = status.get('13d', {}).get(slug, {})
        t = rec.get('timing', {}) or {}
        if not t and rec.get('train_seconds') is not None:
            t = {'train_seconds': rec.get('train_seconds'),
                 'predict_seconds': rec.get('predict_seconds'),
                 'total_seconds': rec.get('total_seconds')}
        if not t:
            # 基线自身写的时序文件（由计时插桩产生）：result/timing.json 或 result/baseline/timing.json
            for cand in (os.path.join(HERE, 'baseline_out', '13d', slug, 'result', 'timing.json'),
                         os.path.join(HERE, 'baseline_out', '13d', slug, 'result', 'baseline',
                                      'timing.json')):
                if os.path.isfile(cand):
                    t = json.load(open(cand, encoding='utf-8'))
                    break
        trows.append({'model': slug, 'status': rec.get('status'),
                      'wall_seconds': rec.get('seconds'),
                      'train_seconds': t.get('train_seconds'),
                      'predict_seconds': t.get('predict_seconds'),
                      'total_seconds': t.get('total_seconds'),
                      'timeout_sec': rec.get('timeout_sec'),
                      'smdpso_profile': rec.get('smdpso_profile'),
                      'log': rec.get('log')})
    pd.DataFrame(trows).to_csv(os.path.join(RES, 'rq1f_timing.csv'), index=False)

    # 9d 补充口径的耗时（若已运行）
    st9_path = os.path.join(HERE, 'data', 'rq1f_run_status_9d.json')
    if os.path.isfile(st9_path):
        st9 = json.load(open(st9_path, encoding='utf-8')).get('9d', {})
        t9 = []
        for slug in EXTERNAL:
            rec = st9.get(slug, {})
            t9.append({'model': slug, 'status': rec.get('status'),
                       'wall_seconds': rec.get('seconds'),
                       'timeout_sec': rec.get('timeout_sec'),
                       'smdpso_profile': rec.get('smdpso_profile'),
                       'log': rec.get('log')})
        pd.DataFrame(t9).to_csv(os.path.join(RES, 'rq1f_timing_9d.csv'), index=False)

    # ---- 事前定义的时代分层（次要结果；同时给出总体成绩）-----------------------
    era_rows = []
    eras = sorted(man.loc[rid0, 'era'].dropna().unique())
    for era in eras:
        sel = (man.loc[rid0, 'era'].to_numpy() == era)
        if sel.sum() < 30:
            continue
        for disp, df in models.items():
            y, pred = df.y_true.to_numpy()[sel], df.pred.to_numpy()[sel]
            era_rows.append({'era': era, 'n': int(sel.sum()), 'model': disp,
                             'accuracy': accuracy_score(y, pred),
                             'macro_f1': f1_score(y, pred, average='macro', zero_division=0),
                             'overall_accuracy': float((df.pred.to_numpy() == y0).mean())})
    if era_rows:
        pd.DataFrame(era_rows).to_csv(os.path.join(RES, 'rq1f_era_secondary.csv'), index=False)

    # ---- 汇总 -----------------------------------------------------------------
    p_acc = float((p_pred == p_y).mean())
    p_f1 = float(f1_score(p_y, p_pred, average='macro', zero_division=0))
    ext_acc = {s: float((models[s].pred.to_numpy() == p_y).mean()) for s in EXTERNAL if s in models}
    ext_f1 = {s: float(f1_score(p_y, models[s].pred.to_numpy(), average='macro', zero_division=0))
              for s in EXTERNAL if s in models}
    best_ext_acc = max(ext_acc, key=ext_acc.get)
    best_ext_f1 = max(ext_f1, key=ext_f1.get)
    summary = {
        'protocol': 'RQ2-S2-LARGE-DEV-CONFIRM-v1',
        'n_dev_confirm': int(len(base)),
        'n_accident_groups': int(pd.unique(p_g).size),
        'identity_certainty_counts': man.loc[rid0, 'identity_certainty'].value_counts().to_dict(),
        'geoad_P': {'accuracy': p_acc, 'macro_f1': p_f1},
        'external_13d_accuracy': ext_acc,
        'external_13d_macro_f1': ext_f1,
        'best_external_by_accuracy': {'model': best_ext_acc, 'acc': ext_acc[best_ext_acc],
                                      'lead_pt': (p_acc - ext_acc[best_ext_acc]) * 100},
        'best_external_by_macro_f1': {'model': best_ext_f1, 'f1': ext_f1[best_ext_f1],
                                      'lead_pt': (p_f1 - ext_f1[best_ext_f1]) * 100},
        'P_leads_all_external_accuracy': bool(all(p_acc >= v for v in ext_acc.values())),
        'P_leads_all_external_macro_f1': bool(all(p_f1 >= v for v in ext_f1.values())),
    }
    with open(os.path.join(RES, 'rq1f_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- 逐类强弱分析：P 在每个类别上相对八个外部基线的位置 --------------------
    pc = pd.DataFrame(percls)
    ext_models = [s for s in EXTERNAL if s in models]
    cls_rows = []
    for c in SIX:
        sub = pc[(pc['class'] == c)]
        p_row = sub[sub.model == 'GeOAD_P (P_full)'].iloc[0]
        ext = sub[sub.model.isin(ext_models)]
        cls_rows.append({
            'class': c,
            'P_f1': p_row['f1'], 'P_precision': p_row['precision'], 'P_recall': p_row['recall'],
            'P_support': int(p_row['support']),
            'ext_mean_f1': float(ext.f1.mean()), 'ext_best_f1': float(ext.f1.max()),
            'ext_best_model': ext.loc[ext.f1.idxmax(), 'model'],
            'ext_worst_f1': float(ext.f1.min()), 'ext_worst_model': ext.loc[ext.f1.idxmin(), 'model'],
            'n_ext_beaten_by_P': int((ext.f1 < p_row['f1']).sum()),
            'n_ext_total': len(ext),
            'P_is_best': bool(p_row['f1'] >= ext.f1.max()),
            'P_f1_minus_ext_mean': float(p_row['f1'] - ext.f1.mean()),
        })
    cls_df = pd.DataFrame(cls_rows)
    cls_df.to_csv(os.path.join(RES, 'rq1f_class_strength.csv'), index=False)

    print('\n=== 主结果表（13 维主口径 + 内部对照）===')
    show = main_tab[['model', 'accuracy', 'macro_f1', 'weighted_f1', 'macro_precision',
                     'macro_recall', 'MATERIAL_DEFECT_f1', 'OPERATIONAL_f1']].copy()
    for c in ['accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall',
              'MATERIAL_DEFECT_f1', 'OPERATIONAL_f1']:
        show[c] = (show[c] * 100).round(2)
    print(show.to_string(index=False))
    print('\n=== 配对统计（P 对照）===')
    print(paired[['comparator', 'acc_diff_P_minus_B', 'acc_ci95_lo', 'acc_ci95_hi',
                  'macro_f1_diff_P_minus_B', 'mcnemar_p', 'holm_p_mcnemar',
                  'group_permutation_p', 'holm_p_group_perm']].round(5).to_string(index=False))
    print('\n=== 耗时 ===')
    print(pd.DataFrame(trows).to_string(index=False))
    print('\n完成。')


if __name__ == '__main__':
    main()
