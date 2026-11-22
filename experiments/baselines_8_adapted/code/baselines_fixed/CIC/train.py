"""
CIC-based Binary Classification for Pipeline Failure Detection (RQ1)
Based on: "Semi-supervised learning framework for oil and gas pipeline failure detection"
Alobaidi, Meguid & Zayed (2022)
"""

import os
import sys
import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import json as _rq1f_json
import time as _rq1f_time
def _rq1f_mark():
    return _rq1f_time.time()
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, matthews_corrcoef, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import MiniBatchKMeans

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


# Setup logging
log_dir = os.path.join(os.path.dirname(__file__), 'log')
result_dir = os.path.join(os.path.dirname(__file__), 'result')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'cic_train.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MatrixImputation:
    """Column mean imputation for missing values"""

    def __init__(self):
        self.col_means_ = None

    def fit(self, X, missing_mask):
        n_features = X.shape[1]
        self.col_means_ = np.zeros(n_features)
        for j in range(n_features):
            valid_mask = ~missing_mask[:, j]
            if valid_mask.any():
                self.col_means_[j] = np.mean(X[valid_mask, j])

    def transform(self, X, missing_mask):
        X_filled = X.copy()
        for j in range(X.shape[1]):
            X_filled[missing_mask[:, j], j] = self.col_means_[j]
        return X_filled


class CICClassifier:
    """Cluster-Impute-Classify pipeline"""

    def __init__(self):
        self.clusterer = None
        self.imputers = {}
        self.label_encoders = {}
        self.scaler = None
        self.classifier = None
        self.threshold = 0.5

    def fit(self, X_train, y_train, X_val, y_val):
        logger.info("Starting CIC training")

        # Encode categorical features
        cat_features = X_train.select_dtypes(include='object').columns.tolist()
        for col in cat_features:
            le = LabelEncoder()
            # [RQ1-FAIR F4] 词表只允许从训练折拟合（原实现 concat 了 val，属验证集泄漏）
            le.fit(X_train[col].astype(str).unique())
            X_train[col] = le.transform(X_train[col].astype(str))
            X_val[col] = le.transform(X_val[col].astype(str))
            self.label_encoders[col] = le

        # Fill missing values
        X_train_filled = X_train.fillna(-999).values.astype(np.float64)
        X_val_filled = X_val.fillna(-999).values.astype(np.float64)
        train_missing_mask = X_train.isna().values
        val_missing_mask = X_val.isna().values

        # Stage 1: Clustering
        logger.info("Stage 1: K-Means Clustering")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_filled)
        X_val_scaled = self.scaler.transform(X_val_filled)

        self.clusterer = MiniBatchKMeans(n_clusters=3, random_state=42, batch_size=1024)
        train_clusters = self.clusterer.fit_predict(X_train_scaled)
        val_clusters = self.clusterer.predict(X_val_scaled)
        logger.info(f"Cluster distribution: {np.bincount(train_clusters)}")

        # Stage 2: Imputation per cluster
        logger.info("Stage 2: Matrix Imputation")
        X_train_imputed = np.zeros_like(X_train_filled)
        for cluster_id in range(3):
            cluster_mask = train_clusters == cluster_id
            if cluster_mask.sum() > 0:
                cluster_data = X_train_filled[cluster_mask]
                cluster_missing = train_missing_mask[cluster_mask]
                imputer = MatrixImputation()
                imputer.fit(cluster_data, cluster_missing)
                X_train_imputed[np.where(cluster_mask)[0]] = imputer.transform(cluster_data, cluster_missing)
                self.imputers[cluster_id] = imputer
                logger.info(f"  Cluster {cluster_id}: {cluster_mask.sum()} samples")

        X_val_imputed = np.zeros_like(X_val_filled)
        for cluster_id in range(3):
            cluster_mask = val_clusters == cluster_id
            if cluster_mask.sum() > 0:
                cluster_data = X_val_filled[cluster_mask]
                cluster_missing = val_missing_mask[cluster_mask]
                X_val_imputed[np.where(cluster_mask)[0]] = self.imputers[cluster_id].transform(cluster_data, cluster_missing)

        # Stage 3: Classification with RUSBoost
        logger.info("Stage 3: RUSBoost Classification")
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        from imblearn.under_sampling import RandomUnderSampler

        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X_train_imputed, y_train)

        base_estimator = DecisionTreeClassifier(max_depth=4, random_state=42)
        self.classifier = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=30,
            random_state=42,
            algorithm='SAMME'
        )
        self.classifier.fit(X_resampled, y_resampled)

        # [RQ1-6CLASS] 二分类阈值搜索在六分类下无定义；改为在 val 上按 argmax 决策
        self.threshold = None

        logger.info(f"Optimal threshold: {self.threshold}")
        logger.info("CIC training completed")

    def predict(self, X_test):
        cat_features = X_test.select_dtypes(include='object').columns.tolist()
        for col in cat_features:
            le = self.label_encoders[col]
            # [RQ1-FAIR F4] 训练折未见的取值 -> -1（不再用 -999 哨兵，与共享编码方案一致）
            X_test[col] = X_test[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        X_test_filled = X_test.fillna(-999).values.astype(np.float64)
        X_test_scaled = self.scaler.transform(X_test_filled)
        test_clusters = self.clusterer.predict(X_test_scaled)
        test_missing_mask = X_test.isna().values

        X_test_imputed = np.zeros_like(X_test_filled)
        for cluster_id in range(3):
            cluster_mask = test_clusters == cluster_id
            if cluster_mask.sum() > 0:
                cluster_data = X_test_filled[cluster_mask]
                cluster_missing = test_missing_mask[cluster_mask]
                X_test_imputed[np.where(cluster_mask)[0]] = self.imputers[cluster_id].transform(cluster_data, cluster_missing)

        probs = self.classifier.predict_proba(X_test_imputed)  # [RQ1-6CLASS] 6 维概率
        return probs.argmax(1)

    def predict_proba(self, X_test):
        cat_features = X_test.select_dtypes(include='object').columns.tolist()
        for col in cat_features:
            le = self.label_encoders[col]
            # [RQ1-FAIR F4] 训练折未见的取值 -> -1（不再用 -999 哨兵，与共享编码方案一致）
            X_test[col] = X_test[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        X_test_filled = X_test.fillna(-999).values.astype(np.float64)
        X_test_scaled = self.scaler.transform(X_test_filled)
        test_clusters = self.clusterer.predict(X_test_scaled)
        test_missing_mask = X_test.isna().values

        X_test_imputed = np.zeros_like(X_test_filled)
        for cluster_id in range(3):
            cluster_mask = test_clusters == cluster_id
            if cluster_mask.sum() > 0:
                cluster_data = X_test_filled[cluster_mask]
                cluster_missing = test_missing_mask[cluster_mask]
                X_test_imputed[np.where(cluster_mask)[0]] = self.imputers[cluster_id].transform(cluster_data, cluster_missing)

        return self.classifier.predict_proba(X_test_imputed)  # [RQ1-6CLASS] 6 维概率


def main():
    data_dir = _resolve_data_dir()
    result_dir = os.path.join(os.path.dirname(__file__), 'result')

    logger.info("Loading data...")
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    X_val = pd.read_csv(os.path.join(data_dir, 'X_val.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))['ANOMALY_LABEL_ID'].values
    y_val = pd.read_csv(os.path.join(data_dir, 'y_val.csv'))['ANOMALY_LABEL_ID'].values
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))['ANOMALY_LABEL_ID'].values

    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {len(y_train)}")
    logger.info(f"X_val shape: {X_val.shape}, y_val shape: {len(y_val)}")
    logger.info(f"X_test shape: {X_test.shape}, y_test shape: {len(y_test)}")

    logger.info(f"Train class distribution: {np.bincount(y_train, minlength=6).tolist()}")
    logger.info(f"Val class distribution: {np.bincount(y_val, minlength=6).tolist()}")
    logger.info(f"Test class distribution: {np.bincount(y_test, minlength=6).tolist()}")

    model = CICClassifier()
    _rq1f_t0 = _rq1f_mark()
    model.fit(X_train, y_train, X_val, y_val)
    _rq1f_train_s = _rq1f_mark() - _rq1f_t0

    _rq1f_t1 = _rq1f_mark()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    _rq1f_pred_s = _rq1f_mark() - _rq1f_t1
    _rq1f_json.dump({"train_seconds": round(_rq1f_train_s, 3),
                     "predict_seconds": round(_rq1f_pred_s, 3),
                     "total_seconds": round(_rq1f_pred_s + _rq1f_train_s, 3)},
                    open(os.path.join(result_dir, "timing.json"), "w"))
    logger.info("Predictions generated")

    # [RQ1-6CLASS] 导出统一格式的逐样本预测（test 行序）
    np.save(os.path.join(result_dir, 'y_pred.npy'), np.asarray(y_pred))
    np.save(os.path.join(result_dir, 'y_pred_proba.npy'), np.asarray(y_proba))

    # Compute MCC and AUC for binary classification
    mcc = matthews_corrcoef(y_test, y_pred)
    try:
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    except:
        roc_auc = None
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    logger.info(f"MCC: {mcc:.4f}")
    logger.info(f"ROC-AUC: {roc_auc:.4f}" if roc_auc else "ROC-AUC: N/A")
    logger.info(f"Balanced Accuracy: {bal_acc:.4f}")

    report = classification_report(y_test, y_pred, target_names=['CORROSION', 'EQUIPMENT_FAILURE', 'EXTERNAL_DAMAGE', 'MATERIAL_DEFECT', 'OPERATIONAL', 'OTHER'], digits=4)
    logger.info("\n" + report)

    report_path = os.path.join(result_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("CIC Binary Classification Report (RQ1 - Anomaly Detection)\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        f.write("\nAdditional Metrics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"MCC (Matthews Correlation Coefficient): {mcc:.4f}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n" if roc_auc else "ROC-AUC: N/A\n")
        f.write(f"Balanced Accuracy: {bal_acc:.4f}\n")
    logger.info(f"Classification report saved to {report_path}")


if __name__ == '__main__':
    main()