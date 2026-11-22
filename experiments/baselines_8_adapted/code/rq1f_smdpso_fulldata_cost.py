"""rq1f_smdpso_fulldata_cost.py — SMDPSO-SVM「用全部 29,941 行训练」的预计计算成本

背景：SMDPSO-SVM 的 run_experiment 有 `train_sample_size = min(5000, len(X_train))`
      这一**原实现自带**的上限（train_pre_fix.py 里就有，不是试点轮修出来的）。
      试点轮训练集只有 3,000 行 ⇒ 上限未生效；全量 29,941 行 ⇒ 只用 17%。
      按指令 §4「不得故意减少某个基线的输入信息/训练质量/正常能力」，
      必须**先报告完整配置的预计计算成本**。

本脚本按**真实代码路径**计时（mixed_kernel + SVC(kernel='precomputed').fit），
在若干 n 上测量，再外推到 n = 29,941（其中 80% 进 fit ⇒ 23,952 行）。

输出：results/rq1f_smdpso_fulldata_cost.json
"""
import os
import json
import time

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, 'results')
DATA = os.path.join(HERE, 'data', 'baseline_data_13d')


def mixed_kernel(X1, X2, sigma, d, alpha):
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    k_gaussian = np.exp(-dist_sq / (2 * sigma ** 2))
    k_poly = np.dot(X1, X2.T) ** d
    return alpha * k_gaussian + (1 - alpha) * k_poly


def main():
    import pandas as pd
    Xdf = pd.read_csv(os.path.join(DATA, 'X_train.csv'))
    y = pd.read_csv(os.path.join(DATA, 'y_train.csv'))['ANOMALY_LABEL_ID'].to_numpy()
    # 与 SMDPSO-SVM/train.py:320-324 完全一致的中位数插补（只用训练折中位数）
    numeric_cols = Xdf.select_dtypes(include=[np.number]).columns
    med = Xdf[numeric_cols].median()
    X = Xdf[numeric_cols].fillna(med).to_numpy(dtype=np.float64)
    n_total = len(X)
    print(f'训练集 {n_total} 行, {X.shape[1]} 维')

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    meas = []
    for n in [2000, 4000, 8000]:
        Xs_sub = Xs[:n]
        ys = y[:n]
        t0 = time.time()
        K = mixed_kernel(Xs_sub, Xs_sub, 0.5, 3, 0.5)
        t1 = time.time()
        m = SVC(C=1.0, kernel='precomputed', gamma='scale', decision_function_shape='ovr')
        m.fit(K, ys)
        t2 = time.time()
        rec = {'n': n, 'kernel_seconds': round(t1 - t0, 2), 'svm_fit_seconds': round(t2 - t1, 2),
               'total_seconds': round(t2 - t0, 2),
               'kernel_GB': round(n * n * 13 * 8 / 1e9, 2)}
        meas.append(rec)
        print(rec, flush=True)
        del K

    # 外推到全量（80% 进 fit）
    n_full_fit = int(n_total * 0.8)
    base = meas[-1]
    ratio = (n_full_fit / base['n']) ** 2          # 核构造 O(n^2)
    # libsvm 求解成本按 O(n^2)–O(n^3) 之间，取平方律保守下界与立方律上界
    extrap = {
        'n_fit_at_full_train': n_full_fit,
        'kernel_GB_at_full_train': round(n_full_fit ** 2 * 13 * 8 / 1e9, 1),
        'kernel_matrix_GB': round(n_full_fit ** 2 * 8 / 1e9, 1),
        'kernel_seconds_extrap_o2': round(base['kernel_seconds'] * ratio, 1),
        'svm_fit_seconds_extrap_o2': round(base['svm_fit_seconds'] * ratio, 1),
        'svm_fit_seconds_extrap_o3': round(base['svm_fit_seconds'] * ratio ** 1.5, 1),
    }
    out = {'measured': meas, 'extrapolation': extrap,
           'note': '全量训练变体需一次性构造 24k x 24k 混合核（float64 混合核矩阵本身约 %.1f GB，'
                   '差分张量约 %.1f GB），并被 test 侧复用 3 次；'
                   '内存可行（机器 RAM 755 GB），但 libsvm 求解成本随 n 超线性上升。'
                   % (extrap['kernel_matrix_GB'], extrap['kernel_GB_at_full_train'])}
    with open(os.path.join(RES, 'rq1f_smdpso_fulldata_cost.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
