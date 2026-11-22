"""
MFF-CNN-LightGBM for pipeline safety early warning
RQ1: Binary classification - detect if anomaly exists (IS_ANOMALY)

Based on paper:
"Pipeline Safety Early Warning by Multifeature-Fusion CNN and LightGBM
Analysis of Signals From Distributed Optical Fiber Sensors"
"""

import os
import sys
import logging
import numpy as np
import json as _rq1f_json
import time as _rq1f_time
def _rq1f_mark():
    return _rq1f_time.time()
import pandas as pd
from sklearn.metrics import classification_report, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import StandardScaler
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Setup logging
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, 'log')
RESULT_DIR = os.path.join(SCRIPT_DIR, 'result')

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'mfcnn_lightgbm_train.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MFCNN_LightGBM:
    """
    Multi-Feature-Fusion CNN and LightGBM fusion model
    Based on paper's architecture with lightweight CNN
    """

    def __init__(self, input_shape=(7, 100, 1)):
        self.input_shape = input_shape
        self.keras_model = None
        self.feature_extractor = None
        self.lgb_model = None

    def _build_lightweight_mfcnn(self):
        """Build lightweight MFCNN to reduce training time"""
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input, Conv2D, BatchNormalization, MaxPooling2D,
            Flatten, Dense, Concatenate, GlobalAveragePooling2D
        )

        peak_input = Input(shape=self.input_shape, name='peak_input')
        energy_input = Input(shape=self.input_shape, name='energy_input')

        # Peak branch - simplified
        x1 = Conv2D(16, (3, 3), activation='relu', padding='same')(peak_input)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D((2, 2))(x1)
        x1 = GlobalAveragePooling2D()(x1)
        x1 = Dense(64, activation='relu', name='peak_feat')(x1)

        # Energy branch - simplified
        x2 = Conv2D(16, (3, 3), activation='relu', padding='same')(energy_input)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D((2, 2))(x2)
        x2 = GlobalAveragePooling2D()(x2)
        x2 = Dense(64, activation='relu', name='energy_feat')(x2)

        concatenated = Concatenate()([x1, x2])
        # [MFF-6CLASS-FIX] 原实现：Dense(1, activation='sigmoid')。
        # 但 RQ1 标签已是六分类 0..5 ⇒ 该头与训练目标不匹配（见 docs/07 §3）。
        # 最小修复：输出维改为 6、激活改为 softmax，与六分类标签一致。
        # 注意：下游 LightGBM 消费的是 peak_feat/energy_feat（各 64 维，共 128 维），
        #      本头仅用于给特征提取分支提供训练信号，**不改变** LightGBM 的输入宽度。
        output = Dense(6, activation='softmax', name='cls_head')(concatenated)

        model = Model(inputs=[peak_input, energy_input], outputs=output)

        return model, peak_input, energy_input, x1, x2

    def _prepare_input_data(self, X, Ncpt=7, Ncwin=100):
        """Prepare data for MFCNN input"""
        n_samples, n_features = X.shape

        peak_data = np.zeros((n_samples, Ncpt, Ncwin))
        energy_data = np.zeros((n_samples, Ncpt, Ncwin))

        for i in range(n_samples):
            sample = X[i]
            # [MFF-13D-COVERAGE-FIX] 修复前为 `n_features // Ncpt`（下取整）：
            # 13 维输入下得 1，于是只有特征 0..6 进入 7 个通道，**特征 7..12 被整段丢弃**，
            # 网络从未见过 ACCIDENT_PSIG / MOP_PSIG / INTERNAL_INSPECTION_IND /
            # PIPE_AGE_AT_ACCUMENT / MANUFACTURE_AGE_AT_ACCIDENT / PRESSURE_RATIO。
            # 修复为上取整 `ceil(n_features / Ncpt)`：通道 j 取连续特征块
            # [j*fpt, min((j+1)*fpt, n_features))，末块按 n_features 截断；
            # 由于 ceil(n/f)*f >= n，**全部特征都被分配，无一丢弃**，且每个通道仍是连续特征块，
            # 与原设计的"每个点位对应一段连续特征"语义一致。
            # 输入形状仍为 (n, Ncpt=7, Ncwin=100, 1)，网络结构/训练预算/其它配置一律不变。
            features_per_pt = max(1, -(-n_features // Ncpt))  # ceil division

            for j in range(Ncpt):
                feat_start = j * features_per_pt
                feat_end = min(feat_start + features_per_pt, n_features)
                feature_slice = sample[feat_start:feat_end]

                if len(feature_slice) < Ncwin:
                    padded = np.pad(feature_slice, (0, Ncwin - len(feature_slice)), mode='constant')
                    peak_data[i, j, :] = padded
                    energy_data[i, j, :] = padded
                else:
                    peak_data[i, j, :] = feature_slice[:Ncwin]
                    energy_data[i, j, :] = feature_slice[:Ncwin]

        peak_data = peak_data.reshape(n_samples, Ncpt, Ncwin, 1)
        energy_data = energy_data.reshape(n_samples, Ncpt, Ncwin, 1)

        return peak_data, energy_data

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=5, batch_size=256):
        """Train MFCNN-LightGBM"""
        logger.info("Starting MFF-CNN-LightGBM training")
        logger.info(f"Training data: X={X_train.shape}, y={y_train.shape}")

        # Prepare data
        train_peak, train_energy = self._prepare_input_data(X_train)
        logger.info(f"Input shapes: peak={train_peak.shape}, energy={train_energy.shape}")

        val_peak, val_energy = None, None
        if X_val is not None:
            val_peak, val_energy = self._prepare_input_data(X_val)

        # Build model
        self.keras_model, peak_input, energy_input, peak_feat, energy_feat = self._build_lightweight_mfcnn()

        from tensorflow.keras.optimizers import Adam
        self.keras_model.compile(
            optimizer=Adam(learning_rate=0.002),
            # [MFF-6CLASS-FIX] 二分类 BCE -> 六分类稀疏交叉熵（标签为整数 0..5）。
            # 训练预算（Adam lr=0.002 / epochs=3 / batch_size=512）保持不变。
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train MFCNN
        val_data = None
        if X_val is not None:
            val_data = ([val_peak, val_energy], y_val)

        logger.info("Training MFCNN...")
        self.keras_model.fit(
            [train_peak, train_energy],
            y_train,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        # Extract features
        logger.info("Extracting MFCNN features...")
        from tensorflow.keras.models import Model
        self.feature_extractor = Model(
            inputs=[peak_input, energy_input],
            outputs=[peak_feat, energy_feat]
        )

        train_feat = self.feature_extractor.predict(
            [train_peak, train_energy], verbose=0
        )

        # Handle different output formats
        if isinstance(train_feat, list):
            mfcnn_features = np.hstack(train_feat)
        elif isinstance(train_feat, tuple):
            mfcnn_features = np.hstack(train_feat)
        else:
            mfcnn_features = train_feat

        logger.info(f"MFCNN features shape: {mfcnn_features.shape}")

        # Train LightGBM
        logger.info("Training LightGBM...")
        import lightgbm as lgb

        train_data = lgb.Dataset(mfcnn_features, label=y_train)
        params = {
            'objective': 'multiclass',
            'num_class': 6,  # [RQ1-6CLASS] 二分类 -> 多分类
            'metric': 'multi_logloss',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'verbose': -1,
            'seed': 42
        }

        self.lgb_model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            callbacks=[lgb.log_evaluation(50)]
        )

        logger.info("Training completed")
        return self

    def predict(self, X):
        """Generate predictions"""
        test_peak, test_energy = self._prepare_input_data(X)

        feat = self.feature_extractor.predict([test_peak, test_energy], verbose=0)

        # Handle different output formats
        if isinstance(feat, list):
            mfcnn_features = np.hstack(feat)
        elif isinstance(feat, tuple):
            mfcnn_features = np.hstack(feat)
        else:
            mfcnn_features = feat

        y_pred_proba = self.lgb_model.predict(mfcnn_features)
        y_pred = y_pred_proba.argmax(1)  # [RQ1-6CLASS] 六分类 argmax

        return y_pred, y_pred_proba


def main():
    logger.info("=" * 60)
    logger.info("MFF-CNN-LightGBM Pipeline Safety Early Warning")
    logger.info("RQ1: Binary Classification - Anomaly Detection")
    logger.info("=" * 60)

    # Paths
    data_dir = _resolve_data_dir()
    result_dir = RESULT_DIR

    # Load data
    logger.info("Loading data...")
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    X_val = pd.read_csv(os.path.join(data_dir, 'X_val.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    y_val = pd.read_csv(os.path.join(data_dir, 'y_val.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))

    # [RQ1-6CLASS] 二分类标签 -> 六类标签（0..5）
    y_train_binary = y_train['ANOMALY_LABEL_ID'].values
    y_val_binary = y_val['ANOMALY_LABEL_ID'].values
    y_test_binary = y_test['ANOMALY_LABEL_ID'].values

    logger.info(f"Train: {X_train.shape}, class distribution: {np.bincount(y_train_binary, minlength=6).tolist()}")

    # [MFF-6CLASS-FIX / H3 可复现性] 原实现无任何随机种子 ⇒ 同一数据两次运行结果不同。
    # 与其余基线（SART-1DCNN / 1DCNN-DBO-LSTM / AUTDL 均设 seed=42）对齐；
    # 仅固定随机性，不改变方法、预算或划分；该 hunk 可独立回退。
    import random as _rq1f_random
    _rq1f_random.seed(42)
    np.random.seed(42)
    os.environ.setdefault('PYTHONHASHSEED', '42')
    import tensorflow as _rq1f_tf
    _rq1f_tf.random.set_seed(42)

    # Handle missing values
    # [RQ1-FAIR H1] 原实现用常数哨兵 -999 填补缺失值；改为**训练折中位数**，
    # 与内部模型（pilot_common.Preprocessor）的填补口径一致，且不泄漏任何折外信息。
    _medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(_medians)
    X_val = X_val.fillna(_medians)
    X_test = X_test.fillna(_medians)

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    logger.info(f"Numeric features: {len(numeric_cols)}")
    logger.info(f"[RQ1-FAIR F6] 实际输入维度 = {len(numeric_cols)}（13 维输入下类别列已序数编码并计入）")

    X_train_numeric = X_train[numeric_cols].values
    X_val_numeric = X_val[numeric_cols].values
    X_test_numeric = X_test[numeric_cols].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_numeric)
    X_val_scaled = scaler.transform(X_val_numeric)
    X_test_scaled = scaler.transform(X_test_numeric)

    # Train
    model = MFCNN_LightGBM()
    _rq1f_t0 = _rq1f_mark()
    model.fit(X_train_scaled, y_train_binary, X_val_scaled, y_val_binary, epochs=3, batch_size=512)
    _rq1f_train_s = _rq1f_mark() - _rq1f_t0

    # Predict
    logger.info("Generating predictions...")
    _rq1f_t1 = _rq1f_mark()
    y_pred, y_pred_proba = model.predict(X_test_scaled)
    _rq1f_pred_s = _rq1f_mark() - _rq1f_t1
    _rq1f_json.dump({"train_seconds": round(_rq1f_train_s, 3),
                     "predict_seconds": round(_rq1f_pred_s, 3),
                     "total_seconds": round(_rq1f_pred_s + _rq1f_train_s, 3)},
                    open(os.path.join(result_dir, "timing.json"), "w"))

    # Report
    logger.info("\n" + "=" * 60)
    report = classification_report(y_test_binary, y_pred, target_names=['CORROSION', 'EQUIPMENT_FAILURE', 'EXTERNAL_DAMAGE', 'MATERIAL_DEFECT', 'OPERATIONAL', 'OTHER'], digits=4)
    logger.info("\n" + report)

    # Calculate additional metrics
    mcc = matthews_corrcoef(y_test_binary, y_pred)
    auc = roc_auc_score(y_test_binary, y_pred_proba, multi_class='ovr', average='macro')

    logger.info(f"\nAdditional Metrics:")
    logger.info(f"MCC: {mcc:.4f}")
    logger.info(f"AUC: {auc:.4f}")

    # Save
    report_path = os.path.join(result_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("MFF-CNN-LightGBM Binary Classification Report\n")
        f.write("RQ1 - Pipeline Anomaly Detection\n")
        f.write("=" * 60 + "\n\n")
        f.write("Method: Multi-Feature-Fusion CNN + LightGBM\n")
        f.write("Based on paper: Pipeline Safety Early Warning\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        f.write("\nAdditional Metrics:\n")
        f.write(f"MCC: {mcc:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")

    # [RQ1-6CLASS] 导出统一格式的逐样本预测（test 行序）
    np.save(os.path.join(result_dir, 'y_pred.npy'), np.asarray(y_pred))
    np.save(os.path.join(result_dir, 'y_pred_proba.npy'), np.asarray(y_pred_proba))

    logger.info(f"Report saved to {report_path}")


if __name__ == '__main__':
    main()