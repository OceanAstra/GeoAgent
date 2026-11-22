"""rq1f_promote_smdpso.py — 把 SMDPSO-SVM 的 reduced 档结果提升为主口径（计算限制备案）

背景：SMDPSO-SVM 的 **full 档**（适应度子采样 1,500）在全量统一输入上实测病态变慢
（第 0 次迭代 77 s、第 10 次迭代 547 s，此后 40+ 分钟未完成第 11–20 次迭代），
无法在指令 §4 的"合理预算"内完成 600 次适应度评估。
按指令 §4「若完整档超出合理预算，允许沿用已有 reduced 档，但必须明确标注计算限制」，
本脚本：

  1. 把 reduced 档（适应度子采样 400 / 验证子采样 150，粒子 20 × 迭代 30 不变）
     的逐样本预测复制为主口径 `predictions/13d/SMDPSO-SVM.csv`；
  2. 同步更新 `data/rq1f_run_status_13d.json` 里 SMDPSO-SVM 的记录：
     status=RUN（reduced 档）、smdpso_profile='reduced'、compute_limit 说明、
     以及 full 档的实际观测耗时；
  3. 把 full 档的观测写入 `results/rq1f_smdpso_full_cost_observed.json`。

**不得把 reduced 成绩包装成 full 成绩**：所有下游表格、文档都从 status JSON 读取
档位标记，标注为 "reduced 档（计算限制）"。

用法：python3 code/rq1f_promote_smdpso.py
"""
import os
import json
import shutil

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, 'predictions', 'supplementary', 'SMDPSO-SVM-reduced13d.csv')
DST = os.path.join(HERE, 'predictions', '13d', 'SMDPSO-SVM.csv')
STATUS = os.path.join(HERE, 'data', 'rq1f_run_status_13d.json')
RES = os.path.join(HERE, 'results')


def main():
    if not os.path.isfile(SRC):
        raise SystemExit(f'缺少 reduced 档产物：{SRC}')
    shutil.copyfile(SRC, DST)

    fb = json.load(open(os.path.join(RES, 'rq1f_smdpso_reduced_fallback.json'), encoding='utf-8'))
    st = json.load(open(STATUS, encoding='utf-8'))
    rec = st.get('13d', {}).get('SMDPSO-SVM', {})
    full_obs = {
        'profile': 'full',
        'fitness_sample': 1500, 'val_sample': 300,
        'n_particles': 20, 'n_iter': 30, 'n_fitness_evals_planned': 600,
        'observed': '第 0 次迭代完成于 77 s；第 10 次迭代完成于 547 s（自 21:11:46 起算）；'
                    '随后 40+ 分钟未完成第 11–20 次迭代 ⇒ 单次适应度评估进入病态区'
                    '（C / d 搜索使 libsvm 收敛极慢），600 次评估无法在合理预算内完成',
        'verdict': 'FULL_PROFILE_INFEASIBLE_UNDER_BUDGET',
        'wall_seconds_observed': rec.get('seconds'),
    }
    with open(os.path.join(RES, 'rq1f_smdpso_full_cost_observed.json'), 'w', encoding='utf-8') as f:
        json.dump(full_obs, f, ensure_ascii=False, indent=2)

    st['13d']['SMDPSO-SVM'] = {
        'slug': 'SMDPSO-SVM',
        'variant': '13d',
        'status': 'RUN',
        'seconds': fb['seconds'],
        'train_seconds': (fb.get('timing') or {}).get('train_seconds'),
        'predict_seconds': (fb.get('timing') or {}).get('predict_seconds'),
        'timeout_sec': 10800,
        'detail': 'reduced 档（适应度子采样 400 / 验证子采样 150；粒子 20 × 迭代 30 不变）。'
                  '**计算限制**：full 档在本预算内不可行，详见 results/rq1f_smdpso_full_cost_observed.json',
        'smdpso_profile': 'reduced',
        'compute_limit': True,
        'train_cap': fb.get('train_cap'),
        'log': 'logs/rq1f_smdpso_reduced_fallback.log',
        'csv': 'predictions/13d/SMDPSO-SVM.csv',
    }
    with open(STATUS, 'w', encoding='utf-8') as f:
        json.dump(st, f, ensure_ascii=False, indent=2)

    df = pd.read_csv(DST)
    print(f'[OK] 已提升 reduced 档结果到 predictions/13d/SMDPSO-SVM.csv  n={len(df)}')
    print(f'     accuracy={(df.pred == df.y_true).mean():.4f}  '
          f'argmax_agreement={df.argmax_agrees_with_pred.iloc[0]}')


if __name__ == '__main__':
    main()
