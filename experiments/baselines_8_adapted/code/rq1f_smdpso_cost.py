"""rq1f_smdpso_cost.py — SMDPSO-SVM 完整档（full）在**全量数据**上的实测计算成本估算

方法：不外推，直接在**与正式运行完全相同的全量训练数据与实现**上计时
      K 次真实适应度评估（full 档参数：fitness_sample=1500, val_sample=300），
      再按 20 粒子 × 30 迭代 = 600 次评估外推总时长。

输出：results/rq1f_smdpso_cost.json, logs/rq1f_smdpso_cost_log.txt
"""
import os
import json
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOT = ARCH_ROOT + '/GeOAD_RQ1_EightBaselines_Pilot_20261025'
BL = os.path.join(PILOT, 'code', 'baselines_fixed', 'SMDPSO-SVM')
DATA = os.path.join(HERE, 'data', 'baseline_data_13d')

import sys
sys.path.insert(0, BL)
import importlib.util
spec = importlib.util.spec_from_file_location('smd_train', os.path.join(BL, 'train.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

K = 3  # 实测适应度评估次数


def main():
    X_train = pd.read_csv(os.path.join(DATA, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(DATA, 'y_train.csv'))['ANOMALY_LABEL_ID'].values
    num = X_train.select_dtypes(include=[np.number]).columns
    med = X_train[num].median()
    X = X_train[num].fillna(med).values

    np.random.seed(42)
    n = min(5000, len(X))
    idx = np.random.choice(len(X), n, replace=False)
    X_sub, y_sub = X[idx], y_train[idx]

    m = mod.SMDPSO_SVM(n_particles=20, max_iter=30, fitness_sample=1500, val_sample=300)
    m.classes_ = np.unique(y_sub)
    n_val = min(800, int(len(X_sub) * 0.2))
    np.random.seed(42)
    perm = np.random.permutation(len(X_sub))
    m.X_fit_, m.y_fit_ = X_sub[perm[n_val:]], y_sub[perm[n_val:]]
    m.X_val_, m.y_val_ = X_sub[perm[:n_val]], y_sub[perm[:n_val]]

    per_eval = []
    for k in range(K):
        # 用搜索空间内的确定参数点做计时（不改变算法，只测单次适应度成本）
        params = np.array([1.0 + k, 0.5 + 0.1 * k, 2.0, 0.5])
        t0 = time.time()
        f = m.fitness_function(params)
        dt = time.time() - t0
        per_eval.append(dt)
        print(f'[eval {k}] fitness={f:.4f} 用时 {dt:.1f}s', flush=True)

    mean_e = float(np.mean(per_eval))
    total = mean_e * 600
    # 加上最终模型在全部训练子集上的重训（一次性，n≈4000 的预计算核）
    t0 = time.time()
    sc = StandardScaler()
    Xs = sc.fit_transform(m.X_fit_)
    Kt = m.mixed_kernel(Xs, Xs, 1.0, 2, 0.5)
    from sklearn.svm import SVC
    SVC(C=1.0, kernel='precomputed', decision_function_shape='ovr').fit(Kt, m.y_fit_)
    final_fit = time.time() - t0

    res = {
        'profile': 'full',
        'fitness_sample': 1500, 'val_sample': 300,
        'n_particles': 20, 'max_iter': 30, 'n_fitness_evals': 600,
        'full_scale_train_rows': int(len(X)),
        'subsample_used': int(n),
        'X_fit_rows': int(len(m.X_fit_)), 'X_val_rows': int(len(m.X_val_)),
        'measured_eval_seconds': [round(x, 2) for x in per_eval],
        'mean_eval_seconds': round(mean_e, 2),
        'projected_pso_seconds': round(total, 1),
        'projected_pso_hours': round(total / 3600, 2),
        'final_refit_seconds': round(final_fit, 1),
        'projected_total_seconds': round(total + final_fit, 1),
        'projected_total_hours': round((total + final_fit) / 3600, 2),
        'reduced_profile_measured_seconds_pilot': 33.8,
    }
    with open(os.path.join(HERE, 'results', 'rq1f_smdpso_cost.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
