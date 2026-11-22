"""rq1f_make_report.py — 由 results/ 自动生成 docs/04 与 docs/05（避免手抄数字出错）

输出：docs/04_主结果与配对统计.md、docs/05_六问结论.md
"""
import os
import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, 'results')
DOCS = os.path.join(HERE, 'docs')
SIX = ['CORROSION', 'EQUIPMENT_FAILURE', 'EXTERNAL_DAMAGE',
       'MATERIAL_DEFECT', 'OPERATIONAL', 'OTHER']
DISPLAY = {
    'GeOAD_P (P_full)': 'GeOAD P',
    'A_ensemble (CatBoost集成)': 'A_ensemble (CatBoost 集成)',
    'A_single (CatBoost单种子)': 'A_single (CatBoost 单种子)',
    'F_concat (非几何F)': 'F_concat (非几何 F)',
    'AUTDL': 'AUTDL', 'CIC': 'CIC', 'DeepPipe': 'DeepPipe', 'DeepPipeNet': 'DeepPipeNet',
    'SART-1DCNN': 'SART-1DCNN', 'MFF-CNN-LightGBM': 'MFF-CNN-LightGBM',
    '1DCNN-DBO-LSTM': '1DCNN-DBO-LSTM', 'SMDPSO-SVM': 'SMDPSO-SVM',
}
ORDER = ['GeOAD_P (P_full)', 'A_ensemble (CatBoost集成)', 'A_single (CatBoost单种子)',
         'F_concat (非几何F)', 'AUTDL', 'CIC', 'DeepPipe', 'DeepPipeNet', 'SART-1DCNN',
         'MFF-CNN-LightGBM', '1DCNN-DBO-LSTM', 'SMDPSO-SVM']


def pct(x):
    return f'{x*100:.2f}'


def md_table(df, cols=None, headers=None, fmts=None):
    if cols:
        df = df[cols]
    headers = headers or list(df.columns)
    fmts = fmts or {}
    out = ['| ' + ' | '.join(headers) + ' |',
           '|' + '|'.join(['---'] * len(headers)) + '|']
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if c in fmts:
                cells.append(fmts[c](v))
            elif isinstance(v, float):
                cells.append(f'{v:.4f}')
            else:
                cells.append(str(v))
        out.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


def confusion_md(conf, model):
    sub = conf[conf.model == model]
    m = np.zeros((6, 6), dtype=int)
    for _, r in sub.iterrows():
        m[SIX.index(r.true_class), SIX.index(r.pred_class)] = int(r.n)
    lines = [f'**{DISPLAY.get(model, model)}** — 行=真值，列=预测', '',
             '| 真值 \\ 预测 | ' + ' | '.join(SIX) + ' | 合计 |',
             '|' + '|'.join(['---'] * 8) + '|']
    for i, c in enumerate(SIX):
        lines.append('| ' + ' | '.join([c] + [str(x) for x in m[i]] + [str(m[i].sum())]) + ' |')
    lines.append('| **合计** | ' + ' | '.join([str(m[:, j].sum()) for j in range(6)]) +
                 f' | {m.sum()} |')
    return '\n'.join(lines)


def main():
    os.makedirs(DOCS, exist_ok=True)
    main_tab = pd.read_csv(os.path.join(RES, 'rq1f_main_table.csv'))
    percls = pd.read_csv(os.path.join(RES, 'rq1f_per_class.csv'))
    conf = pd.read_csv(os.path.join(RES, 'rq1f_confusion.csv'))
    paired = pd.read_csv(os.path.join(RES, 'rq1f_paired_stats.csv'))
    timing = pd.read_csv(os.path.join(RES, 'rq1f_timing.csv'))
    cls_str = pd.read_csv(os.path.join(RES, 'rq1f_class_strength.csv'))
    summary = json.load(open(os.path.join(RES, 'rq1f_summary.json'), encoding='utf-8'))
    era = pd.read_csv(os.path.join(RES, 'rq1f_era_secondary.csv')) \
        if os.path.isfile(os.path.join(RES, 'rq1f_era_secondary.csv')) else None

    main_tab['disp'] = main_tab.model.map(lambda m: DISPLAY.get(m, m))
    main_tab = main_tab.set_index('model').reindex([m for m in ORDER if m in set(main_tab.model)]).reset_index()

    # ---------------- 04 ----------------
    L = []
    L.append('# 04 · 统一主结果、逐类指标、混淆矩阵与配对统计\n')
    L.append('> 所有数字由 `code/rq1f_analyze.py` 计算、由 `code/rq1f_make_report.py` 自动生成本文件。\n')
    L.append(f"评价集合：**dev_confirm {summary['n_dev_confirm']} 行**"
             f"（{summary['n_accident_groups']} 个事故组），六类自然分布，未做任何子集筛选。\n")

    L.append('\n## 1. 统一主结果表（13 维主口径 + 内部强对照）\n')
    t = main_tab.copy()
    for c in ['accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall',
              'MATERIAL_DEFECT_f1', 'OPERATIONAL_f1']:
        t[c] = t[c].map(lambda v: f'{v*100:.2f}')
    L.append(md_table(t, ['disp', 'accuracy', 'macro_f1', 'weighted_f1', 'macro_precision',
                           'macro_recall', 'MATERIAL_DEFECT_f1', 'OPERATIONAL_f1'],
                       ['模型', 'Acc(%)', 'Macro-F1(%)', 'Weighted-F1(%)', 'Macro-P(%)',
                        'Macro-R(%)', 'MATERIAL_DEFECT F1(%)', 'OPERATIONAL F1(%)']))
    L.append('\n\n*内部对照（A_* / F_concat / P_full）直接复用 RQ2 第二小阶段的预测：'
             '协议、特征、样本 ID、标签与事故组逐行一致（已由 100 项预检与 '
             '`rq1f_alignment_check.csv` 双重核对）。*\n')

    L.append('\n### 1a. 输出语义（分数类型与 argmax 一致性）\n')
    L.append('> 本轮所有指标**只由 `pred`（模型自身的 `predict()` 输出）计算**，'
             '从不使用分数列反推标签。分数列仅用于口径登记与后续分析。\n')
    sem = main_tab[['model', 'score_type']].copy()
    ARM = {'GeOAD_P (P_full)': 'P_full', 'A_ensemble (CatBoost集成)': 'A_ensemble',
           'A_single (CatBoost单种子)': 'A_single', 'F_concat (非几何F)': 'F_concat'}
    agree = []
    for m in main_tab.model:
        p = os.path.join(HERE, 'predictions', '13d', f'{m}.csv')
        if not os.path.isfile(p):
            p = os.path.join(HERE, 'predictions_s2_reused', f'{ARM.get(m, m)}.csv')
        if os.path.isfile(p):
            d = pd.read_csv(p)
            agree.append(float(d.argmax_agrees_with_pred.iloc[0]) if 'argmax_agrees_with_pred' in d else None)
        else:
            agree.append(None)
    sem['argmax_agreement'] = agree
    sem['argmax_agreement'] = sem['argmax_agreement'].map(
        lambda v: '—' if v is None else f'{v:.6f}')
    sem['score_type'] = sem['score_type'].map(
        lambda v: 'n/a（内部对照：复用 RQ2 的 prob_* 概率列）'
        if (v is None or (isinstance(v, float) and np.isnan(v))) else v)
    sem['model'] = sem.model.map(lambda m: DISPLAY.get(m, m))
    L.append(md_table(sem, ['model', 'score_type', 'argmax_agreement'],
                       ['模型', 'score_type', 'argmax 与 predict 一致率']))
    L.append('\n\n*SMDPSO-SVM 的 `score_type = decision_score`：'
             '公平性修复 F5 把 `decision_function_shape` 设为 `\'ovr\'` 后导出 6 列 OVR 决策分数，'
             '**这些分数不是概率，且跨类不可比**（一致率 0.9857 而非 1.0 即由此而来）。'
             '该模型的标签一律取自 `y_pred.npy`（`predict()` 输出），故不影响任何指标。*\n')

    if os.path.isfile(os.path.join(RES, 'rq1f_main_table_9d.csv')):
        t9 = pd.read_csv(os.path.join(RES, 'rq1f_main_table_9d.csv'))
        t9['disp'] = t9.model.map(lambda m: DISPLAY.get(m, m))
        for c in ['accuracy', 'macro_f1']:
            t9[c] = t9[c].map(lambda v: f'{v*100:.2f}')
        L.append('\n## 1b. 9 维补充对照（仅数值列，去掉 4 个类别列）\n')
        L.append(md_table(t9, ['disp', 'accuracy', 'macro_f1'],
                           ['模型', 'Acc(%)', 'Macro-F1(%)']))
        L.append('\n')

    L.append('\n## 2. 逐类 Precision / Recall / F1 / support\n')
    for model in ORDER:
        if model not in set(percls.model):
            continue
        sub = percls[percls.model == model].copy()
        sub = sub.set_index('class').reindex(SIX).reset_index()
        L.append(f'\n**{DISPLAY.get(model, model)}**\n')
        L.append(md_table(sub, ['class', 'precision', 'recall', 'f1', 'support'],
                           ['类别', 'Precision', 'Recall', 'F1', 'support'],
                           {'precision': lambda v: f'{v:.4f}', 'recall': lambda v: f'{v:.4f}',
                            'f1': lambda v: f'{v:.4f}', 'support': lambda v: f'{int(v)}'}))

    L.append('\n\n## 3. 混淆矩阵\n')
    for model in ORDER:
        if model in set(conf.model):
            L.append('\n' + confusion_md(conf, model) + '\n')

    L.append('\n## 4. P 对各对照的配对差异（行级 McNemar + 事故组整群 bootstrap + 组级置换 + Holm）\n')
    pt = paired.copy()
    for c in ['acc_A', 'acc_B', 'acc_diff_P_minus_B', 'macro_f1_A', 'macro_f1_B',
              'macro_f1_diff_P_minus_B', 'hard_f1_diff_P_minus_B']:
        pt[c] = pt[c].map(lambda v: f'{v*100:+.2f}')
    for c in ['acc_ci95_lo', 'acc_ci95_hi', 'macro_f1_ci95_lo', 'macro_f1_ci95_hi',
              'hard_f1_ci95_lo', 'hard_f1_ci95_hi']:
        pt[c] = pt[c].map(lambda v: f'{v*100:+.2f}')
    for c in ['mcnemar_p', 'holm_p_mcnemar', 'group_permutation_p', 'holm_p_group_perm']:
        pt[c] = pt[c].map(lambda v: f'{v:.4g}')
    pt['comparator'] = pt.comparator.map(lambda m: DISPLAY.get(m, m))
    L.append(md_table(pt, ['comparator', 'acc_A', 'acc_B', 'acc_diff_P_minus_B',
                            'acc_ci95_lo', 'acc_ci95_hi', 'macro_f1_diff_P_minus_B',
                            'macro_f1_ci95_lo', 'macro_f1_ci95_hi', 'mcnemar_p',
                            'holm_p_mcnemar', 'group_permutation_p', 'holm_p_group_perm'],
                       ['对照', 'P Acc(%)', '对照 Acc(%)', 'ΔAcc(pt)', 'Acc CI95 lo', 'Acc CI95 hi',
                        'ΔMacroF1(pt)', 'F1 CI lo', 'F1 CI hi', 'McNemar p', 'Holm p',
                        '置换 p', 'Holm p']))
    L.append('\n\n*Δ = P − 对照（正= P 更好）；CI 为事故组整群 bootstrap 2,000 次（seed 42）；'
             'Holm 对 10 个主比较做步降校正；「CI 跨 0 ≠ 等价」。*\n')

    L.append('\n## 5. 逐类强弱（相对八个外部基线）\n')
    cs = cls_str.copy()
    for c in ['P_f1', 'P_precision', 'P_recall', 'ext_mean_f1', 'ext_best_f1',
              'ext_worst_f1', 'P_f1_minus_ext_mean']:
        cs[c] = cs[c].map(lambda v: f'{v:.4f}')
    cs['P_is_best'] = cs.P_is_best.map(lambda v: '✅' if v else '❌')
    cs['ext_best_model'] = cs.ext_best_model.map(lambda m: DISPLAY.get(m, m))
    cs['ext_worst_model'] = cs.ext_worst_model.map(lambda m: DISPLAY.get(m, m))
    L.append(md_table(cs, ['class', 'P_f1', 'P_precision', 'P_recall', 'P_support',
                            'ext_mean_f1', 'ext_best_f1', 'ext_best_model', 'ext_worst_f1',
                            'ext_worst_model', 'n_ext_beaten_by_P', 'n_ext_total', 'P_is_best'],
                       ['类别', 'P F1', 'P P', 'P R', 'support', '外部均值F1', '外部最好F1',
                        '最好者', '外部最差F1', '最差者', 'P 击败的外部基线数', '外部基线数',
                        'P 是否最优']))

    L.append('\n\n## 6. 训练与预测耗时\n')
    tt = timing.copy()
    for c in ['train_seconds', 'predict_seconds']:
        tt[c] = tt[c].map(lambda v: '—' if pd.isna(v) else f'{v:.1f}')
    tt['wall_seconds'] = tt.wall_seconds.map(lambda v: '—' if pd.isna(v) else f'{v:.1f}')
    L.append(md_table(tt, ['model', 'status', 'train_seconds', 'predict_seconds',
                            'wall_seconds', 'smdpso_profile'],
                       ['模型', '状态', '训练(s)', '预测(s)', '墙钟(s)', 'SMDPSO 档位']))

    if era is not None and len(era):
        L.append('\n\n## 7. 事前定义的时代分层（**次要**结果，同时给出总体成绩）\n')
        L.append('> 时代分层为**事前定义**（manifest 的 `era` 字段），'
                 '每一层都附总体成绩；**不作为唯一主结果**。\n')
        piv = era.pivot_table(index=['era', 'n'], columns='model', values='accuracy')
        piv = piv.reset_index()
        piv.columns = ['era', 'n'] + [DISPLAY.get(c, c) for c in piv.columns[2:]]
        for c in piv.columns[2:]:
            piv[c] = piv[c].map(lambda v: f'{v*100:.2f}')
        piv = piv.sort_values('era')
        L.append(md_table(piv, list(piv.columns)))

    with open(os.path.join(DOCS, '04_主结果与配对统计.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print('[OK] docs/04_主结果与配对统计.md')


if __name__ == '__main__':
    main()
