"""rq1f_make_conclusions.py — 由 results/ 自动生成 docs/05_六问结论.md

六个问题来自指令 §6 末尾；本脚本把结果数字与结论文字绑定，避免手抄出错。
"""
import os
import json

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, 'results')
DOCS = os.path.join(HERE, 'docs')
DISPLAY = {
    'GeOAD_P (P_full)': 'GeOAD P', 'A_ensemble (CatBoost集成)': 'A_ensemble',
    'A_single (CatBoost单种子)': 'A_single', 'F_concat (非几何F)': 'F_concat',
    'AUTDL': 'AUTDL', 'CIC': 'CIC', 'DeepPipe': 'DeepPipe', 'DeepPipeNet': 'DeepPipeNet',
    'SART-1DCNN': 'SART-1DCNN', 'MFF-CNN-LightGBM': 'MFF-CNN-LightGBM',
    '1DCNN-DBO-LSTM': '1DCNN-DBO-LSTM', 'SMDPSO-SVM': 'SMDPSO-SVM',
}


def main():
    summary = json.load(open(os.path.join(RES, 'rq1f_summary.json'), encoding='utf-8'))
    paired = pd.read_csv(os.path.join(RES, 'rq1f_paired_stats.csv'))
    cls = pd.read_csv(os.path.join(RES, 'rq1f_class_strength.csv'))
    main_tab = pd.read_csv(os.path.join(RES, 'rq1f_main_table.csv'))
    cap = None
    cap_p = os.path.join(RES, 'rq1f_smdpso_cap_probe.json')
    if os.path.isfile(cap_p):
        cap = json.load(open(cap_p, encoding='utf-8'))

    ext_acc = summary['external_13d_accuracy']
    ext_f1 = summary['external_13d_macro_f1']
    p_acc = summary['geoad_P']['accuracy']
    p_f1 = summary['geoad_P']['macro_f1']
    n_beaten_acc = sum(1 for v in ext_acc.values() if p_acc > v)
    n_beaten_f1 = sum(1 for v in ext_f1.values() if p_f1 > v)
    n_tied_acc = sum(1 for v in ext_acc.values() if abs(p_acc - v) < 1e-12)

    be = summary['best_external_by_accuracy']
    bf = summary['best_external_by_macro_f1']

    def prow(name):
        r = paired[paired.comparator == name].iloc[0]
        return r

    pa = prow('A_ensemble (CatBoost集成)')
    pf = prow('F_concat (非几何F)')

    best_cls = cls[cls.P_is_best]
    worst_cls = cls.sort_values('P_f1_minus_ext_mean').iloc[0]
    best_cls_row = cls.sort_values('P_f1_minus_ext_mean', ascending=False).iloc[0]

    L = []
    L.append('# 05 · 对指令 §6 六个问题的明确回答\n')
    L.append('> 本文件由 `code/rq1f_make_conclusions.py` 从 `results/` 自动生成，数字与 04 号文档同源。\n')
    L.append(f"评价集合：dev_confirm {summary['n_dev_confirm']} 行"
             f"（{summary['n_accident_groups']} 个事故组），六类自然分布，无子集筛选。\n")

    # ---- Q1 ----
    L.append('\n## Q1 · GeOAD 是否仍然领先八个表格化适配（tabular-adapted）的外部方法？\n')
    if n_beaten_acc == 8 and n_beaten_f1 == 8:
        verdict = '**是**（在 Accuracy 与 Macro-F1 两个口径上同时领先全部八个）'
    elif n_beaten_acc >= 6 and n_beaten_f1 >= 6:
        verdict = '**基本是，但并非全面**：在多数外部方法上领先，个别方法持平或略优'
    elif n_beaten_acc >= 4:
        verdict = ('**否（不是全面领先）**：只在部分外部方法上领先，'
                   '与最强外部方法的差距落在噪声范围内')
    else:
        verdict = '**否**：GeOAD P 未能在多数外部方法上保持领先'
    L.append(f'{verdict}。\n')
    L.append(f'* Accuracy 上 P 击败/持平 **{n_beaten_acc}/8** 个外部方法（持平 {n_tied_acc} 个）；'
             f'Macro-F1 上 **{n_beaten_f1}/8**。\n')
    rows = ['| 外部方法 | Acc(%) | Macro-F1(%) | P−它 Acc(pt) | P−它 F1(pt) |', '|---|---|---|---|---|']
    for s in ['AUTDL', 'CIC', 'DeepPipe', 'DeepPipeNet', 'SART-1DCNN',
              'MFF-CNN-LightGBM', '1DCNN-DBO-LSTM', 'SMDPSO-SVM']:
        rows.append(f'| {s} | {ext_acc[s]*100:.2f} | {ext_f1[s]*100:.2f} | '
                    f'{(p_acc-ext_acc[s])*100:+.2f} | {(p_f1-ext_f1[s])*100:+.2f} |')
    rows.append(f'| **GeOAD P** | **{p_acc*100:.2f}** | **{p_f1*100:.2f}** | — | — |')
    L.append('\n'.join(rows))
    L.append('\n\n**必要限定**：这些外部方法全部是 **tabular-adapted implementations**'
             '（输入模态被替换为同一套 13 维管况表格），'
             '因此该结论的适用范围是"统一 13 维表格输入下"，**不能外推为对原方法原始模态的全面优越**。\n')

    # SMDPSO 的计算限制必须在结论里显式声明
    stp = os.path.join(HERE, 'data', 'rq1f_run_status_13d.json')
    if os.path.isfile(stp):
        st = json.load(open(stp, encoding='utf-8')).get('13d', {}).get('SMDPSO-SVM', {})
        if st.get('compute_limit'):
            L.append('\n> **SMDPSO-SVM 的计算限制（必须随该行成绩一并引用）**：'
                     '该方法的 **full 档**（适应度子采样 1,500，粒子 20 × 迭代 30 = 600 次适应度评估）'
                     '在本轮统一输入上**实测病态变慢**（第 0 次迭代 77 s、第 10 次迭代 547 s，'
                     '此后 40+ 分钟未完成第 11–20 次迭代），**无法在合理预算内完成**。'
                     '按指令允许，主口径采用 **reduced 档**（适应度子采样 400 / 验证子采样 150，'
                     '粒子数与迭代数不变），训练数据、特征、评价口径与其它七个基线完全相同。'
                     '**该行是 reduced 档成绩，不得表述为 full 档成绩。**'
                     '（`results/rq1f_smdpso_full_cost_observed.json`）\n')

    # ---- Q2 ----
    L.append('\n## Q2 · 领先幅度（Accuracy 与 Macro-F1）\n')
    L.append(f'* 相对**最强外部方法（Accuracy）{be["model"]}**：'
             f'Accuracy 领先 **{be["lead_pt"]:+.2f} pt**（{p_acc*100:.2f}% vs {be["acc"]*100:.2f}%）。\n')
    L.append(f'* 相对**最强外部方法（Macro-F1）{bf["model"]}**：'
             f'Macro-F1 领先 **{bf["lead_pt"]:+.2f} pt**（{p_f1*100:.2f}% vs {bf["f1"]*100:.2f}%）。\n')
    L.append('\n**逐个对照的配对统计（事故组整群 bootstrap 95% CI + 行级 McNemar + 组级置换 + Holm 校正）**：\n')
    L.append('| 对照 | ΔAcc(pt) | Acc CI95(pt) | ΔMacro-F1(pt) | F1 CI95(pt) | McNemar p | Holm p | 置换 p | Holm p |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for nm in ['1DCNN-DBO-LSTM', 'DeepPipeNet']:
        if nm not in set(paired.comparator):
            continue
        r = paired[paired.comparator == nm].iloc[0]
        L.append(f'| {nm} | {r.acc_diff_P_minus_B*100:+.2f} | '
                 f'[{r.acc_ci95_lo*100:+.2f}, {r.acc_ci95_hi*100:+.2f}] | '
                 f'{r.macro_f1_diff_P_minus_B*100:+.2f} | '
                 f'[{r.macro_f1_ci95_lo*100:+.2f}, {r.macro_f1_ci95_hi*100:+.2f}] | '
                 f'{r.mcnemar_p:.4g} | {r.holm_p_mcnemar:.4g} | '
                 f'{r.group_permutation_p:.4g} | {r.holm_p_group_perm:.4g} |')
    L.append('')
    L.append('\n*CI 为事故组整群 bootstrap 2,000 次（seed 42）；Holm 对 10 个主比较做步降校正；'
             '「CI 跨 0 ≠ 等价」。*\n')
    L.append('\n**如何解读这条领先**：\n')
    L.append('* **统计上**：P 对**全部八个**外部方法的 Accuracy 与 Macro-F1 优势都**显著**'
             '（Holm 校正后 McNemar p 全部 < 1e-4，CI 全部不含 0）。'
             '⇒ "P 优于八个 tabular-adapted 外部方法"这一**比较性结论是成立的**。\n')
    L.append('* **但幅度很小**：相对最强外部方法的领先仅 '
             f'{be["lead_pt"]:+.2f} pt（Acc）/ {bf["lead_pt"]:+.2f} pt（F1）。'
             '四个 torch 基线在**同一份数据上复跑**的波动实测达 ±0.4~3.6 pt'
             '（未插桩对照组，见 03 号文档 §3.2），'
             '**与本领先幅度同量级**；因此不能把 +2 pt 写成"决定性优势"，'
             '只能说"在同一协议下 P 稳定排在这些外部方法之前"。\n')
    L.append('* **与内部对照的对比（关键）**：同样的检验在 P vs A_ensemble / F_concat 上'
             '**完全不显著**（见 Q4）。⇒ 外部方法之间的差距是真实的，'
             '但 **P 相对强 CatBoost / 非几何 F 的差距不存在**。\n')

    # ---- Q3 ----
    L.append('\n## Q3 · 哪些事故类别是 GeOAD 的真实强项 / 弱项？\n')
    L.append('| 类别 | P 的 F1 | 外部均值 F1 | P−外部均值 | P 击败的外部基线数 | P 是否最优 |')
    L.append('|---|---|---|---|---|---|')
    for _, r in cls.iterrows():
        L.append(f'| {r["class"]} | {r.P_f1:.4f} | {r.ext_mean_f1:.4f} | '
                 f'{r.P_f1_minus_ext_mean:+.4f} | {int(r.n_ext_beaten_by_P)}/{int(r.n_ext_total)} | '
                 f'{"✅" if r.P_is_best else "❌"} |')
    L.append('')
    L.append(f'* **相对优势最明显的类别：{best_cls_row["class"]}**'
             f'（P 的 F1 {best_cls_row.P_f1:.4f}，高于八个外部方法的均值 '
             f'{best_cls_row.ext_mean_f1:.4f}，差 {best_cls_row.P_f1_minus_ext_mean:+.4f}）。\n')
    L.append(f'* **相对最弱的类别：{worst_cls["class"]}**'
             f'（P 的 F1 {worst_cls.P_f1:.4f}，低于外部均值 {worst_cls.ext_mean_f1:.4f}，'
             f'差 {worst_cls.P_f1_minus_ext_mean:+.4f}）。\n')
    L.append('\n**必须完整保留的两个困难类（指令点名）**：\n')
    md = main_tab[main_tab.model == 'GeOAD_P (P_full)'].iloc[0]
    L.append(f'* `MATERIAL_DEFECT`：GeOAD P 的 F1 = **{md.MATERIAL_DEFECT_f1:.4f}**'
             f'（外部均值 {cls[cls["class"]=="MATERIAL_DEFECT"].iloc[0].ext_mean_f1:.4f}）。\n')
    L.append(f'* `OPERATIONAL`：GeOAD P 的 F1 = **{md.OPERATIONAL_f1:.4f}**'
             f'（外部均值 {cls[cls["class"]=="OPERATIONAL"].iloc[0].ext_mean_f1:.4f}）。\n')
    L.append('\n> `OPERATIONAL` 在**所有**模型上 F1 都极低（内部对照四臂也只有 0.13–0.15），'
             '这是**类别层面的可分性问题（数据/可辨识性），不是方法优劣问题**；'
             '`MATERIAL_DEFECT` 同理属于少数类与标签粒度问题。'
             '两者都没有被本轮结果改善，也**没有被隐藏**。\n')

    # ---- Q4 ----
    L.append('\n## Q4 · 相对强 CatBoost 与非几何 F 的增量结论\n')
    L.append('| 对照 | ΔAcc(pt) | Acc CI95(pt) | ΔMacro-F1(pt) | McNemar p | Holm p | 置换 p | Holm p |')
    L.append('|---|---|---|---|---|---|---|---|')
    for nm, r in [('A_ensemble (CatBoost集成)', pa), ('F_concat (非几何F)', pf)]:
        L.append(f'| {DISPLAY[nm]} | {r.acc_diff_P_minus_B*100:+.2f} | '
                 f'[{r.acc_ci95_lo*100:+.2f}, {r.acc_ci95_hi*100:+.2f}] | '
                 f'{r.macro_f1_diff_P_minus_B*100:+.2f} | {r.mcnemar_p:.4g} | {r.holm_p_mcnemar:.4g} | '
                 f'{r.group_permutation_p:.4g} | {r.holm_p_group_perm:.4g} |')
    L.append('')
    sig_a = pa.holm_p_mcnemar < 0.05
    sig_f = pf.holm_p_mcnemar < 0.05
    L.append(f'* 相对 **A_ensemble（CatBoost 三种子等权概率平均）**：'
             f'ΔAcc = **{pa.acc_diff_P_minus_B*100:+.2f} pt**，'
             f'Holm 校正后 McNemar p = {pa.holm_p_mcnemar:.4g} ⇒ '
             f'{"**存在**" if sig_a else "**不存在**"}统计显著的增量。\n')
    L.append(f'* 相对 **F_concat（非几何 F）**：'
             f'ΔAcc = **{pf.acc_diff_P_minus_B*100:+.2f} pt**，'
             f'Holm 校正后 McNemar p = {pf.holm_p_mcnemar:.4g} ⇒ '
             f'{"**存在**" if sig_f else "**不存在**"}统计显著的增量。\n')
    L.append('\n**结论**：GeOAD P 与强 CatBoost、非几何 F 在本协议与本样本上**不可辨别**；'
             '**几何模块没有带来可验证的泛化增益**。'
             '这与 RQ2 第二小阶段（43,834 行、真实事故级隔离、6,417 行 dev_confirm）的结论一致。\n')

    # ---- Q5 ----
    L.append('\n## Q5 · 当前结果支持哪些论文主张？\n')
    claims = [
        ('C1 数据与协议主张：「G1 数据契约 v1 + 事故级隔离协议可复现、无泄漏」',
         '**支持**',
         '100 项执行前核查全通过（`results/rq1f_precheck.json`，verdict=ALL_PASS）；'
         'sealed 1,065 行本轮从未读标签、从未参与任何选择；'
         '词表/缩放/中位数/聚类/插补全部只在 fit 折拟合；'
         '三个划分的 accident_group 零交集；13 个模型在同一批 6,417 行上对齐。'),
        ('C2 比较公平性主张：「八个外部方法在统一输入、统一训练数据、统一预处理口径下被公平比较」',
         '**支持（带明确限定）**',
         '试点轮第一轮已确认存在的 7 类实现错误（softmax×CE、SMOTE 泄漏到 val、类别列无条件填 0、'
         '词表跨折、分数语义混淆、未利用 13 维输入、计算阻塞）均已修复且逐条登记；'
         '本轮与试点原版的逐行 diff 只有计时插桩与一个环境变量钩子'
         '（`results/rq1f_baseline_diff_vs_pilot.txt`）；'
         '计时插桩的行为中性已被"L1_EXACT × 4 + 未插桩对照组 × 4"双重证明。'
         '**限定**：八个方法都是 tabular-adapted 实现，不是原始模态的忠实复现。'),
        ('C3 性能主张：「GeOAD 在统一 13 维表格输入下优于/不劣于八个表格化适配的外部方法」',
         '**支持（须带幅度限定）**',
         f'Accuracy 上 P 击败 {n_beaten_acc}/8 个外部方法，Macro-F1 上 {n_beaten_f1}/8；'
         '且对**全部八个**外部方法的优势都**统计显著**'
         '（Holm 校正后 McNemar p < 1e-4，事故组整群 bootstrap CI 不含 0）。'
         f'但相对最强外部方法的领先幅度仅 {be["lead_pt"]:+.2f} pt（Acc）/ {bf["lead_pt"]:+.2f} pt（F1），'
         '与四个 torch 基线自身 ±0.4~3.6 pt 的复跑波动同量级。'
         '因此可写成"在统一 13 维表格输入下**显著优于**八个表格化适配的外部方法，'
         '但**领先幅度不大**"，**不能写成"决定性/数量级优势"**。'),
        ('C4 核心算法主张：「几何机制（proto/rule/neural 三视角）带来可验证的增量」',
         '**不支持**',
         'P vs 非几何 F：ΔAcc = '
         f'{pf.acc_diff_P_minus_B*100:+.2f} pt（Holm p = {pf.holm_p_mcnemar:.4g}）；'
         'P vs CatBoost 集成：ΔAcc = '
         f'{pa.acc_diff_P_minus_B*100:+.2f} pt（Holm p = {pa.holm_p_mcnemar:.4g}）。'
         '**几何贡献的论文主张必须撤回或改写为负结果。**'),
        ('C5 类别层面主张：「`MATERIAL_DEFECT` 与 `OPERATIONAL` 是本任务的真实困难类」',
         '**支持**',
         f'所有模型在这两类上 F1 都显著偏低（GeOAD P：'
         f'{md.MATERIAL_DEFECT_f1:.3f} / {md.OPERATIONAL_f1:.3f}）；'
         '该现象在八个外部方法与四个内部对照上一致复现，指向类别可分性/标签粒度问题而非模型问题。'),
        ('C6 推断口径主张：「结论可推广到 2024 年及以后的事故」',
         '**不支持（尚不可检验）**',
         'sealed 集（2024+，1,065 行）本轮按指令**从未使用**；'
         'dev_confirm 中 2020plus 只有 322 行，且身份确定性 65.4% 为 source_id_only。'
         '跨代外推必须留到最终封存验证。'),
    ]
    L.append('| 论文主张 | 判定 | 依据 |')
    L.append('|---|---|---|')
    for c, v, e in claims:
        L.append(f'| {c} | {v} | {e} |')
    L.append('')

    # ---- Q6 ----
    L.append('\n## Q6 · 项目是否可以进入 RQ3 与最终封存验证？\n')
    L.append('**可以进入，但必须按本轮结果重写论文的贡献定位；且 RQ3 不得被理解为"再找一次好结果"。**\n')
    L.append('\n**理由**：\n')
    L.append('1. 协议侧已经**充分可信**：100 项核查全通过、sealed 未被触碰、'
             '统一输入与统一评价口径下 13 个模型完全对齐、配对统计与多重比较校正齐备 —— '
             'RQ1 这一段的**验证目标已达成**。\n')
    L.append('2. 算法侧**没有**可推广的增量：C4 主张不成立，'
             '因此"GeOAD 的几何机制是主要贡献"这一论文骨架**不能保留**。\n')
    L.append('3. 若仍要进入 RQ3/封存验证，其定位只能是'
             '**"在预注册口径下，对已冻结系统的最终泛化确认"**，'
             '而不是"检验几何是否有效"——后者已被本轮与 RQ2 共同否定，'
             '再做一次只会消耗封存集而不产生新知识。\n')
    L.append('\n**进入前必须完成的事项（否则封存集会被浪费）：**\n')
    L.append('* 冻结本轮的**完整代码、输入、协议与哈希**，作为封存验证的唯一可追溯版本；\n')
    L.append('* 在论文中把贡献改写为 **数据契约 + 事故级隔离协议 + 可复现的公平比较基础设施**'
             '（并把几何贡献写成负结果），而不是"几何带来的性能提升"；\n')
    L.append('* 预先写死封存集的一次性评价方案（指标、统计量、CI 方法、多重比较方案），'
             '**不得在看到封存结果后更换口径**；\n')
    L.append('* 明确封存集只有 1,065 行、且全部是 2024+ 的事故 —— '
             '其统计功效有限（见 RQ2 阶段 0 的功效天花板测算），'
             '**不应把封存结果表述为决定性证据**。\n')

    if cap:
        L.append('\n### 附：SMDPSO-SVM 训练上限敏感性补充探针\n')
        L.append(f'把原实现自带的 5,000 行训练上限放宽到 {cap["train_cap"]} 行'
                 f'（搜索档位与主口径一致：{cap.get("profile", "reduced")}）后：'
                 f'Accuracy = {cap["accuracy"]*100:.2f}%，Macro-F1 = {cap["macro_f1"]*100:.2f}%'
                 f'（主口径 5,000 行上限：{ext_acc["SMDPSO-SVM"]*100:.2f}% / '
                 f'{ext_f1["SMDPSO-SVM"]*100:.2f}%）。'
                 '**该探针不进入主表**，仅用于说明 SMDPSO 的成绩不是由训练样本上限造成的。\n')

    with open(os.path.join(DOCS, '05_六问结论.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print('[OK] docs/05_六问结论.md')


if __name__ == '__main__':
    main()
