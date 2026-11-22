"""
DeepPipe - Hybrid LSTM + ConvLSTM for pipeline condition recognition
RQ1: Binary classification - detect if anomaly exists (IS_ANOMALY)

Based on paper:
"Deeppipe: A hybrid model for multi-product pipeline condition recognition
based on process and data coupling"
Computers and Chemical Engineering 160 (2022) 107733

Method: LSTM (variable-length series processing) + ConvLSTM (time-space features)
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
        logging.FileHandler(os.path.join(LOG_DIR, 'deeppipe_train.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DeepPipe_HybridModel:
    """
    Hybrid LSTM + ConvLSTM model based on DeepPipe paper
    - LSTM: process variable-length time series, extract temporal features
    - ConvLSTM: extract time-space features of pressure series data
    """

    def __init__(self, n_features=14, num_classes=2):
        self.n_features = n_features
        self.num_classes = num_classes
        self.keras_model = None
        self.feature_extractor = None
        self.lgb_model = None

    def _build_lstm_branch(self, input_seq):
        """Build LSTM branch for temporal feature extraction"""
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        # LSTM layer to process sequence and extract temporal features
        x = LSTM(64, return_sequences=True)(input_seq)
        x = Dropout(0.3)(x)
        x = LSTM(32, return_sequences=False)(x)
        x = Dropout(0.3)(x)
        lstm_feat = Dense(64, activation='relu', name='lstm_feat')(x)

        return lstm_feat

    def _build_convlstm_branch(self, input_seq):
        """Build ConvLSTM branch for time-space feature extraction"""
        from tensorflow.keras.layers import Conv1D, ConvLSTM1D, GlobalAveragePooling1D, Dropout, Dense, Reshape

        # Reshape for ConvLSTM: (samples, timesteps, features) -> (samples, timesteps, features, 1)
        x = Reshape((-1, self.n_features, 1))(input_seq)

        # ConvLSTM layers to capture time-space features
        x = ConvLSTM1D(64, kernel_size=3, padding='same', return_sequences=True)(x)
        x = Dropout(0.3)(x)
        x = ConvLSTM1D(32, kernel_size=3, padding='same', return_sequences=True)(x)
        x = Dropout(0.3)(x)
        x = ConvLSTM1D(32, kernel_size=3, padding='same', return_sequences=False)(x)
        x = Dropout(0.3)(x)

        # Global pooling and dense
        x = GlobalAveragePooling1D()(x)
        convlstm_feat = Dense(64, activation='relu', name='convlstm_feat')(x)

        return convlstm_feat

    def _build_model(self):
        """Build the hybrid LSTM + ConvLSTM model"""
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Concatenate, Dense

        # Input: sequence of features (timesteps=1 for tabular data)
        input_seq = Input(shape=(1, self.n_features), name='input_sequence')

        # LSTM branch for temporal features
        lstm_feat = self._build_lstm_branch(input_seq)

        # ConvLSTM branch for time-space features
        convlstm_feat = self._build_convlstm_branch(input_seq)

        # Concatenate features
        concatenated = Concatenate()([lstm_feat, convlstm_feat])

        # Output layer
        if self.num_classes == 2:
            output = Dense(1, activation='sigmoid', name='output')(concatenated)
        else:
            output = Dense(self.num_classes, activation='softmax', name='output')(concatenated)

        model = Model(inputs=input_seq, outputs=output)

        return model, input_seq, lstm_feat, convlstm_feat

    def _prepare_input_data(self, X):
        """Prepare data for the model - reshape to sequence format (timesteps=1, features)"""
        n_samples, n_features = X.shape

        # Reshape to (samples, timesteps=1, features)
        X_seq = X.reshape(n_samples, 1, n_features)

        return X_seq

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=64):
        """Train DeepPipe hybrid model"""
        logger.info("Starting DeepPipe Hybrid LSTM + ConvLSTM training")
        logger.info(f"Training data: X={X_train.shape}, y={y_train.shape}")

        # Prepare sequence data
        train_seq = self._prepare_input_data(X_train)
        logger.info(f"Sequence input shape: {train_seq.shape}")

        val_seq = None
        if X_val is not None:
            val_seq = self._prepare_input_data(X_val)

        # Build model
        self.keras_model, input_seq, lstm_feat, convlstm_feat = self._build_model()

        from tensorflow.keras.optimizers import Adam
        if self.num_classes == 2:
            self.keras_model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            from tensorflow.keras.losses import SparseCategoricalCrossentropy
            self.keras_model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss=SparseCategoricalCrossentropy(),
                metrics=['accuracy']
            )

        logger.info(self.keras_model.summary())

        # Train
        val_data = None
        if X_val is not None:
            val_data = (val_seq, y_val)

        logger.info("Training DeepPipe model...")
        self.keras_model.fit(
            train_seq,
            y_train,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        # Build feature extractor for LightGBM
        logger.info("Extracting features with LSTM + ConvLSTM branches...")
        from tensorflow.keras.models import Model
        self.feature_extractor = Model(
            inputs=input_seq,
            outputs=[lstm_feat, convlstm_feat]
        )

        train_feat = self.feature_extractor.predict(train_seq, verbose=0)

        if isinstance(train_feat, list):
            deeppipe_features = np.hstack(train_feat)
        else:
            deeppipe_features = train_feat

        logger.info(f"DeepPipe features shape: {deeppipe_features.shape}")

        # Train LightGBM
        logger.info("Training LightGBM with extracted features...")
        import lightgbm as lgb

        train_data = lgb.Dataset(deeppipe_features, label=y_train)

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
        test_seq = self._prepare_input_data(X)

        feat = self.feature_extractor.predict(test_seq, verbose=0)

        if isinstance(feat, list):
            deeppipe_features = np.hstack(feat)
        else:
            deeppipe_features = feat

        y_pred_proba = self.lgb_model.predict(deeppipe_features)
        y_pred = y_pred_proba.argmax(1)  # [RQ1-6CLASS] 六分类 argmax

        return y_pred, y_pred_proba


def main():
    logger.info("=" * 60)
    logger.info("DeepPipe Hybrid LSTM + ConvLSTM Pipeline Condition Recognition")
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

    # Binary classification target
    y_train_binary = y_train['ANOMALY_LABEL_ID'].values  # [RQ1-6CLASS] 六类标签
    y_val_binary = y_val['ANOMALY_LABEL_ID'].values  # [RQ1-6CLASS] 六类标签
    y_test_binary = y_test['ANOMALY_LABEL_ID'].values  # [RQ1-6CLASS] 六类标签

    logger.info(f"Train: {X_train.shape}, positive rate: {y_train_binary.mean():.4f}")
    logger.info(f"Val: {X_val.shape}, positive rate: {y_val_binary.mean():.4f}")
    logger.info(f"Test: {X_test.shape}, positive rate: {y_test_binary.mean():.4f}")

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

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_numeric)
    X_val_scaled = scaler.transform(X_val_numeric)
    X_test_scaled = scaler.transform(X_test_numeric)

    # Train DeepPipe model
    n_features = len(numeric_cols)
    model = DeepPipe_HybridModel(n_features=n_features, num_classes=6)  # [RQ1-6CLASS] 六类输出
    _rq1f_t0 = _rq1f_mark()
    model.fit(X_train_scaled, y_train_binary, X_val_scaled, y_val_binary, epochs=10, batch_size=64)
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

    # Calculate MCC and AUC
    mcc = matthews_corrcoef(y_test_binary, y_pred)
    auc = roc_auc_score(y_test_binary, y_pred_proba, multi_class='ovr', average='macro')
    logger.info(f"MCC: {mcc:.4f}")
    logger.info(f"AUC: {auc:.4f}")

    # Report
    logger.info("\n" + "=" * 60)
    report = classification_report(y_test_binary, y_pred, target_names=['CORROSION', 'EQUIPMENT_FAILURE', 'EXTERNAL_DAMAGE', 'MATERIAL_DEFECT', 'OPERATIONAL', 'OTHER'], digits=4)
    logger.info("\n" + report)

    # Save report with MCC and AUC
    report_path = os.path.join(result_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("DeepPipe Hybrid LSTM + ConvLSTM Binary Classification Report\n")
        f.write("RQ1 - Pipeline Anomaly Detection\n")
        f.write("=" * 60 + "\n\n")
        f.write("Method: LSTM (temporal) + ConvLSTM (time-space) + LightGBM\n")
        f.write("Based on paper: DeepPipe (Computers and Chemical Engineering 2022)\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        f.write(f"\nAdditional Metrics:\n")
        f.write(f"MCC: {mcc:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")

    # Save predictions for future use
    np.save(os.path.join(result_dir, 'y_pred.npy'), y_pred)
    np.save(os.path.join(result_dir, 'y_pred_proba.npy'), y_pred_proba)

    logger.info(f"Report saved to {report_path}")


if __name__ == '__main__':
    main()