"""rq1f_smdpso_cost2.py — SMDPSO-SVM 完整档（full）在全量数据上的**稳健**成本估算

背景：rq1f_smdpso_cost.py 用 3 个固定参数点测得 ~1.0s/次，但 SMDPSO 的搜索范围是
      (0.01, 10)；若 PSO 选到 C 大、sigma 小、alpha 大的难分参数，SVC 收敛会显著变慢。
      本脚本在**搜索空间内均匀随机抽 K 个参数点**（含极端点）实测单次适应度成本，
      给出 p50 / p90 / max，并据此外推 600 次评估的总时长。

输出：results/rq1f_smdpso_cost_robust.json
"""
import os
import json
import time
import importlib.util

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOT = ARCH_ROOT + '/GeOAD_RQ1_EightBaselines_Pilot_20261025'
BL = os.path.join(PILOT, 'code', 'baselines_fixed', 'SMDPSO-SVM')
DATA = os.path.join(HERE, 'data', 'baseline_data_13d')

spec = importlib.util.spec_from_file_location('smd_train2', os.path.join(BL, 'train.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

K = 12
SEED = 42


def main():
    X_train = pd.read_csv(os.path.join(DATA, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(DATA, 'y_train.csv'))['ANOMALY_LABEL_ID'].values
    num = X_train.select_dtypes(include=[np.number]).columns
    med = X_train[num].median()
    X = X_train[num].fillna(med).values

    np.random.seed(SEED)
    n = min(5000, len(X))
    idx = np.random.choice(len(X), n, replace=False)
    X_sub, y_sub = X[idx], y_train[idx]

    m = mod.SMDPSO_SVM(n_particles=20, max_iter=30, fitness_sample=1500, val_sample=300)
    n_val = min(800, int(len(X_sub) * 0.2))
    np.random.seed(SEED)
    perm = np.random.permutation(len(X_sub))
    m.X_fit_, m.y_fit_ = X_sub[perm[n_val:]], y_sub[perm[n_val:]]
    m.X_val_, m.y_val_ = X_sub[perm[:n_val]], y_sub[perm[:n_val]]

    rng = np.random.RandomState(SEED)
    # 搜索空间 (0.01, 10)；含 4 个显式极端点（最难的情形）
    pts = [list(rng.uniform(0.01, 10, 4)) for _ in range(K - 4)]
    pts += [[10.0, 0.01, 5, 0.9], [10.0, 0.05, 5, 0.9],
            [0.01, 10.0, 1, 0.1], [10.0, 10.0, 5, 0.5]]

    times = []
    for p in pts:
        C = max(0.01, p[0]); sigma = max(0.01, p[1])
        d = max(1, min(5, int(np.round(p[2])))); alpha = max(0.1, min(0.9, p[3]))
        t0 = time.time()
        idxs = np.random.choice(len(m.X_fit_), min(1500, len(m.X_fit_)), replace=False)
        Xs = StandardScaler().fit_transform(m.X_fit_[idxs])
        Kt = m.mixed_kernel(Xs, Xs, sigma, d, alpha)
        SVC(C=C, kernel='precomputed', gamma='scale').fit(Kt, m.y_fit_[idxs])
        dt = time.time() - t0
        times.append(dt)
        print(f'  C={C:.3f} sigma={sigma:.3f} d={d} alpha={alpha:.2f} -> {dt:.2f}s', flush=True)

    a = np.array(times)
    p50, p90, mx = float(np.percentile(a, 50)), float(np.percentile(a, 90)), float(a.max())
    # 总时长估计：600 次评估，保守取 p90 与 max 两个口径
    est_p50 = p50 * 600
    est_p90 = p90 * 600
    est_max = mx * 600

    # 最终模型重训（一次）：n=4200 的预计算核
    t0 = time.time()
    sc = StandardScaler()
    Xs = sc.fit_transform(m.X_fit_)
    Kt = m.mixed_kernel(Xs, Xs, 1.0, 2, 0.5)
    SVC(C=1.0, kernel='precomputed', decision_function_shape='ovr').fit(Kt, m.y_fit_)
    refit = time.time() - t0

    res = {
        'profile': 'full', 'fitness_sample': 1500, 'val_sample': 300,
        'n_particles': 20, 'max_iter': 30, 'n_fitness_evals': 600,
        'n_measured_points': len(times),
        'measured_seconds': [round(x, 2) for x in times],
        'p50_seconds': round(p50, 2), 'p90_seconds': round(p90, 2), 'max_seconds': round(mx, 2),
        'projected_total_hours_p50': round((est_p50 + refit) / 3600, 2),
        'projected_total_hours_p90': round((est_p90 + refit) / 3600, 2),
        'projected_total_hours_max': round((est_max + refit) / 3600, 2),
        'final_refit_seconds': round(refit, 1),
    }
    with open(os.path.join(HERE, 'results', 'rq1f_smdpso_cost_robust.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
