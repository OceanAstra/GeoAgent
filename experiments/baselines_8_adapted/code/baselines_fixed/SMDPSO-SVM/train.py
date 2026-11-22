"""
SMDPSO-SVM: Saturated and Mixed-Delay Particle Swarm Optimization based SVM
For oil pipeline leak detection (binary classification: normal vs anomaly)
"""

import numpy as np
import pandas as pd
import json
import time as _rq1f_time
def _rq1f_mark():
    return _rq1f_time.time()
import logging
import os
from datetime import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings

# Release fix (2026-11-21): the historical default pointed at `<...>/code/data/baseline_data`,
# which is neither where `rq1f_build_baseline_data.py` writes (it writes
# `<repo>/experiments/baselines_8_adapted/data/baseline_data_13d`) nor a directory that ships
# with the repository, so a fresh clone could not start any baseline without manually setting
# RQ1_DATA_DIR. This resolves the shipped layout, keeps RQ1_DATA_DIR as the override, and
# never changes data content, splits, features or hyper-parameters.
def _resolve_data_dir():
    override = os.environ.get("RQ1_DATA_DIR")
    if override:
        return override
    here = os.path.dirname(os.path.abspath(__file__))
    # .../GeoAgent/experiments/baselines_8_adapted/code/baselines_fixed/<name>/train.py
    # baselines_fixed/<name> -> baselines_fixed -> code -> baselines_8_adapted
    root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    for leaf in ("baseline_data_13d", "baseline_data_9d", "baseline_data"):
        cand = os.path.join(root, "data", leaf)
        if os.path.isdir(cand):
            return cand
    return os.path.join(root, "data", "baseline_data_13d")

warnings.filterwarnings('ignore')

LOG_DIR = os.path.join(os.path.dirname(__file__), 'log')
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f'smdpso_svm_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.4f - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SMDPSO:
    """
    Saturated and Mixed-Delay Particle Swarm Optimization (SMDPSO)
    """

    def __init__(self, n_particles=20, dim=4, max_iter=100, search_range=(-10, 10)):
        self.n_particles = n_particles
        self.dim = dim
        self.max_iter = max_iter
        self.search_range = search_range

        self.vmax = 20.0
        self.zeta = 1e-4

        self.w_max = 0.731
        self.w_min = 0.5

        self.c1i = 2.5
        self.c1f = 0.5
        self.c2i = 0.5
        self.c2f = 2.5

        self.dm = 1
        self.dM = 5
        self.N = 10

        self.gbest_x = None
        self.gbest_f = float('inf')

    def saturation_func(self, v):
        if np.abs(v) < 1e-10:
            return self.zeta
        sign_v = np.sign(v)
        abs_v = np.abs(v)
        if abs_v > self.vmax:
            return sign_v * self.vmax
        return v

    def calc_evolutionary_factor(self, positions):
        n = self.n_particles
        di_list = []

        for i in range(n):
            dist_sum = 0.0
            for j in range(n):
                if i != j:
                    dist_sum += np.sqrt(np.sum((positions[i] - positions[j]) ** 2))
            di_list.append(dist_sum / (n - 1) if n > 1 else 0)

        di_array = np.array(di_list)
        dg = di_array.min()
        dmax = di_array.max()
        dmin = di_array.min()

        if dmax - dmin < 1e-10:
            return 0.5

        delta = (dg - dmin) / (dmax - dmin)
        return delta

    def get_evolutionary_state(self, delta):
        if delta < 0.25:
            return 1
        elif delta < 0.5:
            return 2
        elif delta < 0.75:
            return 3
        else:
            return 4

    def optimize(self, fitness_func):
        logger.info(f"SMDPSO starting: n_particles={self.n_particles}, dim={self.dim}, max_iter={self.max_iter}")

        positions = np.random.uniform(
            self.search_range[0], self.search_range[1],
            (self.n_particles, self.dim)
        )
        velocities = np.random.uniform(-2, 2, (self.n_particles, self.dim))

        pbest_x = positions.copy()
        pbest_f = np.array([float('inf')] * self.n_particles)

        self.gbest_x = positions[0].copy()
        self.gbest_f = float('inf')

        for iteration in range(self.max_iter):
            delta = self.calc_evolutionary_factor(positions)
            state = self.get_evolutionary_state(delta)

            c1 = (self.c1i - self.c1f) * (self.max_iter - iteration) / self.max_iter + self.c1f
            c2 = (self.c2i - self.c2f) * (self.max_iter - iteration) / self.max_iter + self.c2f

            w = 1.0 / (1.0 + np.exp(-delta))
            w = np.clip(w, self.w_min, self.w_max)

            for i in range(self.n_particles):
                f = fitness_func(positions[i])
                if f < pbest_f[i]:
                    pbest_f[i] = f
                    pbest_x[i] = positions[i].copy()
                if f < self.gbest_f:
                    self.gbest_f = f
                    self.gbest_x = positions[i].copy()

            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)

            for i in range(self.n_particles):
                vi = velocities[i].copy()
                xi = positions[i].copy()
                pi = pbest_x[i].copy()
                pg = self.gbest_x.copy()

                vi_saturated = np.array([self.saturation_func(v) for v in vi])

                if state == 1:
                    vi_new = w * vi_saturated + c1 * r1 * (pi - xi) + c2 * r2 * (pg - xi)
                elif state == 2:
                    vi_new = w * vi_saturated + c1 * r1 * (pi - xi) + c2 * r2 * (pg - xi)
                elif state == 3:
                    delay_sum = np.zeros(self.dim)
                    for _ in range(self.N):
                        delay_sum += pg
                    vi_new = w * vi_saturated + c1 * r1 * (pi - xi) + c2 * r2 * (delay_sum / self.N - xi)
                else:
                    delay_sum = np.zeros(self.dim)
                    for _ in range(self.N):
                        delay_sum += pg
                    vi_new = w * vi_saturated + c1 * r1 * (pi - xi) + c2 * r2 * (delay_sum / self.N - xi)

                velocities[i] = vi_new
                positions[i] = xi + vi_new
                positions[i] = np.clip(positions[i], self.search_range[0], self.search_range[1])

            if iteration % 10 == 0 or iteration == self.max_iter - 1:
                logger.info(f"Iter {iteration:4d}: best_f={self.gbest_f:.4f}, state={state}, w={w:.4f}")

        logger.info(f"SMDPSO finished: best_f={self.gbest_f:.4f}")
        return self.gbest_x, self.gbest_f


class SMDPSO_SVM(BaseEstimator, ClassifierMixin):
    """
    SMDPSO optimized SVM with Mixed Kernel (Gaussian + Polynomial)
    """

    def __init__(self, n_particles=20, max_iter=30, C=1.0, sigma=1.0, d=2, alpha=0.5,
                 fitness_sample=400, val_sample=150):
        self.n_particles = n_particles
        self.max_iter = max_iter
        # [RQ1-FAIR F7] 成本参数：适应度评估的子采样规模（默认 reduced 档）
        self.fitness_sample = fitness_sample
        self.val_sample = val_sample
        self.C = C
        self.sigma = sigma
        self.d = d
        self.alpha = alpha

        self.scaler = StandardScaler()
        self.svm_model = None
        self.classes_ = None
        self.X_train_scaled_ = None

    def set_params(self, params):
        self.C = max(0.01, params[0])
        self.sigma = max(0.01, params[1])
        self.d = max(1, min(5, int(np.round(params[2]))))
        self.alpha = max(0.1, min(0.9, params[3]))

    def get_params(self):
        return {'C': self.C, 'sigma': self.sigma, 'd': self.d, 'alpha': self.alpha}

    def mixed_kernel(self, X1, X2, sigma, d, alpha):
        """Mixed kernel: alpha * Gaussian + (1-alpha) * Polynomial"""
        diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        k_gaussian = np.exp(-dist_sq / (2 * sigma ** 2))
        k_poly = np.dot(X1, X2.T) ** d
        return alpha * k_gaussian + (1 - alpha) * k_poly

    def fitness_function(self, params):
        """Fitness function for SMDPSO: classification accuracy"""
        try:
            C = max(0.01, params[0])
            sigma = max(0.01, params[1])
            d = max(1, min(5, int(np.round(params[2]))))
            alpha = max(0.1, min(0.9, params[3]))

            # [RQ1-FAIR F7] 适应度评估的子采样规模可配置。
            # 算法定义（SMDPSO 搜索 + 混合核 SVM + 验证准确率适应度）完全不变，
            # 只降低"每次适应度评估"所拟合的 SVC 规模，从而把成本从 O(n^2)+O(n^3) 降下来。
            n_sample = min(self.fitness_sample, len(self.X_fit_))
            indices = np.random.choice(len(self.X_fit_), n_sample, replace=False)
            X_sub = self.X_fit_[indices]
            y_sub = self.y_fit_[indices]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_sub)

            K = self.mixed_kernel(X_scaled, X_scaled, sigma, d, alpha)
            model = SVC(C=C, kernel='precomputed', gamma='scale')
            model.fit(K, y_sub)

            n_val = min(self.val_sample, len(self.X_val_))
            val_indices = np.random.choice(len(self.X_val_), n_val, replace=False)
            X_val_sub = scaler.transform(self.X_val_[val_indices])
            K_val = self.mixed_kernel(X_val_sub, X_scaled, sigma, d, alpha)
            y_pred = model.predict(K_val)
            acc = accuracy_score(self.y_val_[val_indices], y_pred)

            return -acc

        except Exception as e:
            logger.warning(f"Fitness evaluation failed: {e}")
            return 1.0

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        n_val = min(800, int(len(X) * 0.2))
        np.random.seed(42)
        indices = np.random.permutation(len(X))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        self.X_fit_ = X[train_idx]
        self.y_fit_ = y[train_idx]
        self.X_val_ = X[val_idx]
        self.y_val_ = y[val_idx]

        logger.info(f"SMDPSO-SVM training: X_train={len(self.X_fit_)}, X_val={len(self.X_val_)}")

        # [RQ1-FAIR F7] 搜索结构与预算（粒子数 20 / 迭代 30）与原实现一致，
        # 成本削减只作用于每次适应度评估的子采样规模。
        optimizer = SMDPSO(n_particles=self.n_particles, dim=4, max_iter=self.max_iter,
                          search_range=(0.01, 10))
        best_params, best_f = optimizer.optimize(self.fitness_function)

        self.set_params(best_params)
        logger.info(f"Best parameters: C={self.C:.4f}, sigma={self.sigma:.4f}, d={self.d}, alpha={self.alpha:.4f}")

        # Retrain on training subset for final model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_fit_)
        self.scaler = scaler
        self.X_train_scaled_ = X_scaled

        K_train = self.mixed_kernel(X_scaled, X_scaled, self.sigma, self.d, self.alpha)
        # [RQ1-FAIR F5] decision_function_shape='ovr'：六列决策分数与类别顺序对齐
        self.svm_model = SVC(C=self.C, kernel='precomputed', gamma='scale',
                             decision_function_shape='ovr')
        self.svm_model.fit(K_train, self.y_fit_)

        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        K_test = self.mixed_kernel(X_scaled, self.X_train_scaled_, self.sigma, self.d, self.alpha)
        return self.svm_model.predict(K_test)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


def run_experiment():
    """Run SMDPSO-SVM experiment on oil pipeline leak detection data"""
    logger.info("=" * 60)
    logger.info("SMDPSO-SVM Oil Pipeline Leak Detection Experiment")
    logger.info("=" * 60)

    data_dir = _resolve_data_dir()
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    X_val = pd.read_csv(os.path.join(data_dir, 'X_val.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    y_val = pd.read_csv(os.path.join(data_dir, 'y_val.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))

    # [RQ1-6CLASS] 二分类标签 -> 六类标签（0..5）
    y_train = y_train['ANOMALY_LABEL_ID'].values
    y_val = y_val['ANOMALY_LABEL_ID'].values
    y_test = y_test['ANOMALY_LABEL_ID'].values

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    train_medians = X_train[numeric_cols].median()
    X_train = X_train[numeric_cols].fillna(train_medians).values
    X_val = X_val[numeric_cols].fillna(train_medians).values
    X_test = X_test[numeric_cols].fillna(train_medians).values

    logger.info(f"Data loaded: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    logger.info(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

    np.random.seed(42)
    # [RQ1-FORMAL] 训练子采样上限：默认 5000 = **原实现自带**（train_pre_fix.py 同值）；
    #   仅供『上限敏感性』补充探针通过环境变量覆盖，主口径不使用覆盖值。
    _cap = int(os.environ.get('RQ1_SMDPSO_TRAIN_CAP', '5000'))
    train_sample_size = min(_cap, len(X_train))
    train_indices = np.random.choice(len(X_train), train_sample_size, replace=False)
    X_train_sub = X_train[train_indices]
    y_train_sub = y_train[train_indices]

    logger.info(f"Training on {train_sample_size} samples")

    # [RQ1-FAIR F7] 成本档位：
    #   full    : 原规模（粒子 20 × 迭代 30，适应度子采样 1500）—— 与原稿一致，代价极高
    #   reduced : 粒子/迭代不变，仅把适应度子采样降到 400（本次实际运行档）
    #   smoke   : 2 粒子 × 2 迭代，仅用于验证代码通路
    profile = os.environ.get('RQ1_SMDPSO_PROFILE', 'reduced')
    if profile == 'full':
        n_particles, max_iter, fit_sample, val_sample = 20, 30, 1500, 300
    elif profile == 'smoke':
        n_particles, max_iter, fit_sample, val_sample = 2, 2, 150, 100
    else:
        n_particles, max_iter, fit_sample, val_sample = 20, 30, 400, 150
    logger.info(f"[RQ1-FAIR F7] SMDPSO profile={profile}: particles={n_particles}, "
                f"iters={max_iter}, fitness_sample={fit_sample}, val_sample={val_sample}")

    logger.info("Training SMDPSO-SVM model...")
    model = SMDPSO_SVM(n_particles=n_particles, max_iter=max_iter,
                       fitness_sample=fit_sample, val_sample=val_sample)
    _rq1f_t0 = _rq1f_mark()
    model.fit(X_train_sub, y_train_sub)
    _rq1f_train_s = _rq1f_mark() - _rq1f_t0

    _rq1f_t1 = _rq1f_mark()
    y_pred = model.predict(X_test)
    _rq1f_pred_s = _rq1f_mark() - _rq1f_t1
    # Result directory is not shipped with the repository; create it before the
    # first write. Previously os.makedirs() came only later in this function,
    # so a fresh clone crashed here after the full PSO search had completed.
    os.makedirs(os.path.join(os.path.dirname(__file__), "result"), exist_ok=True)
    json.dump({"train_seconds": round(_rq1f_train_s, 3),
               "predict_seconds": round(_rq1f_pred_s, 3),
               "total_seconds": round(_rq1f_pred_s + _rq1f_train_s, 3)},
              open(os.path.join(os.path.dirname(__file__), "result", "timing.json"), "w"))
    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Get decision function scores for AUC
    X_test_scaled = model.scaler.transform(X_test)
    K_test = model.mixed_kernel(X_test_scaled, model.X_train_scaled_,
                               model.sigma, model.d, model.alpha)
    y_scores = model.svm_model.decision_function(K_test)
    try:
        auc = roc_auc_score(y_test, y_scores, multi_class='ovr', average='macro')
    except Exception:
        auc = float('nan')  # [RQ1-6CLASS] 多分类 OVR AUC

    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(f"MCC: {mcc:.4f}")
    logger.info(f"AUC: {auc:.4f}")

    report = classification_report(y_test, y_pred, target_names=['CORROSION', 'EQUIPMENT_FAILURE', 'EXTERNAL_DAMAGE', 'MATERIAL_DEFECT', 'OPERATIONAL', 'OTHER'],
                                   digits=4, output_dict=False)
    report_dict = classification_report(y_test, y_pred, target_names=['CORROSION', 'EQUIPMENT_FAILURE', 'EXTERNAL_DAMAGE', 'MATERIAL_DEFECT', 'OPERATIONAL', 'OTHER'],
                                        digits=4, output_dict=True)

    logger.info("\nClassification Report:")
    logger.info(report)

    result_dir = os.path.join(os.path.dirname(__file__), 'result')
    os.makedirs(result_dir, exist_ok=True)

    report_file = os.path.join(result_dir, 'classification_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("SMDPSO-SVM Oil Pipeline Leak Detection Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: SMDPSO-SVM (Saturated Mixed-Delay PSO optimized SVM)\n")
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"MCC: {mcc:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        f.write("Detailed Metrics:\n")
        for key, value in report_dict.items():
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for k, v in value.items():
                    f.write(f"  {k}: {v:.4f}\n")
            else:
                f.write(f"{key}: {value:.4f}\n")
        f.write(f"\nMCC: {mcc:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")

    # [RQ1-6CLASS] 导出统一格式的逐样本预测（test 行序）
    result_dir = os.path.join(os.path.dirname(__file__), 'result')
    os.makedirs(result_dir, exist_ok=True)
    # [RQ1-FAIR F5] 决策分数：**不是概率**。原实现用默认 OvO 的 decision_function，
    # 六分类下返回的是 5 列（两两投票差），却被注释成"列序 = 0..5"，语义错误。
    # 改为 decision_function_shape='ovr'，得到 6 列、与 model.classes_ 对齐的 OVR 决策分数。
    _scores = model.svm_model.decision_function(
        model.mixed_kernel(model.scaler.transform(X_test), model.X_train_scaled_,
                           model.sigma, model.d, model.alpha))
    _scores = np.asarray(_scores)
    if _scores.ndim == 1:
        _scores = np.column_stack([-_scores, _scores])
    np.save(os.path.join(result_dir, 'y_pred.npy'), np.asarray(y_pred))
    np.save(os.path.join(result_dir, 'y_pred_scores.npy'), _scores)
    # [RQ1-FAIR F5] 语义说明（放在 np.save 之后：元数据失败不得影响预测落盘）
    with open(os.path.join(result_dir, 'score_semantics.json'), 'w', encoding='utf-8') as _f:
        json.dump({'y_pred.npy': 'hard labels (argmax of decision_function)',
                   'y_pred_scores.npy': 'OVR decision scores, 6 columns in class order; NOT probabilities',
                   'score_type': 'decision_score'}, _f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to: {report_file}")

    return acc, report_dict


if __name__ == "__main__":
    run_experiment()