"""
Baseline Method: 1DCNN-DBO-LSTM for Pipeline Leak Detection
RQ1: Binary classification (Normal vs Anomaly)
"""

import os
import sys
import json
import time as _rq1f_time
def _rq1f_mark():
    return _rq1f_time.time()
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
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

# Set random seeds for reproducibility
np.random.seed(42)

# ============================================================
# Configuration
# ============================================================
DATA_DIR = _resolve_data_dir()
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(OUTPUT_DIR, "baseline", "log")
RESULT_DIR = os.path.join(OUTPUT_DIR, "result", "baseline")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ============================================================
# Logging Setup - All output formatted to 4 decimal places
# ============================================================
class FourDecimalFormatter(logging.Formatter):
    def format(self, record):
        if isinstance(record.msg, str):
            # Replace numeric placeholders with 4 decimal formatted values
            record.msg = str(record.msg)
        return super().format(record)

def setup_logging():
    log_file = os.path.join(LOG_DIR, "baseline_1dcnn_dbo_lstm.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Apply custom formatter
    for handler in logging.root.handlers:
        handler.setFormatter(FourDecimalFormatter('%(asctime)s - %(levelname)s - %(message)s'))

    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================
# Data Loading and Preprocessing
# ============================================================
def load_data():
    """Load train, validation, and test data"""
    logger.info("=" * 60)
    logger.info("Loading data from: %s", DATA_DIR)

    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    X_val = pd.read_csv(os.path.join(DATA_DIR, "X_val.csv"))
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
    y_val = pd.read_csv(os.path.join(DATA_DIR, "y_val.csv"))
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))

    logger.info("X_train shape: %s", X_train.shape)
    logger.info("X_val shape: %s", X_val.shape)
    logger.info("X_test shape: %s", X_test.shape)
    logger.info("y_train shape: %s", y_train.shape)
    logger.info("y_val shape: %s", y_val.shape)
    logger.info("y_test shape: %s", y_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """Preprocess data: handle missing values, encode categoricals, scale features"""
    logger.info("=" * 60)
    logger.info("Preprocessing data")

    # [RQ1-6CLASS] 二分类标签 -> 六类标签（0..5）
    y_train_binary = y_train['ANOMALY_LABEL_ID'].values
    y_val_binary = y_val['ANOMALY_LABEL_ID'].values
    y_test_binary = y_test['ANOMALY_LABEL_ID'].values

    logger.info("Training set class distribution: %s",
                np.bincount(y_train_binary, minlength=6).tolist())
    logger.info("Validation set class distribution: %s",
                np.bincount(y_val_binary, minlength=6).tolist())
    logger.info("Test set class distribution: %s",
                np.bincount(y_test_binary, minlength=6).tolist())

    # Identify numeric and categorical columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    logger.info("Numeric columns: %d", len(numeric_cols))
    logger.info("Categorical columns: %s", categorical_cols)

    # Handle missing values in numeric columns
    for col in numeric_cols:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_val[col] = X_val[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    # Handle missing values and encode categorical columns
    label_encoders = {}
    for col in categorical_cols:
        X_train[col] = X_train[col].fillna('UNKNOWN')
        X_val[col] = X_val[col].fillna('UNKNOWN')
        X_test[col] = X_test[col].fillna('UNKNOWN')

        le = LabelEncoder()
        # [RQ1-FAIR F4] 词表只允许从训练折拟合。原实现把 val 与 test 的取值一并 concat 进
        # 词表，等于用测试集信息定义了编码方案（测试集泄漏）。
        le.fit(X_train[col].astype(str).unique())
        known = set(le.classes_)

        X_train[col] = le.transform(X_train[col].astype(str))
        # 未见取值 -> -1（与共享编码方案一致）
        X_val[col] = X_val[col].astype(str).apply(lambda v: le.transform([v])[0] if v in known else -1)
        X_test[col] = X_test[col].astype(str).apply(lambda v: le.transform([v])[0] if v in known else -1)
        label_encoders[col] = le

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    logger.info("Features scaled using StandardScaler")
    logger.info("Final feature dimension: %d", X_train_scaled.shape[1])

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_binary, y_val_binary, y_test_binary

# ============================================================
# Dung Beetle Optimizer (DBO)
# ============================================================
class DBO:
    """
    Dung Beetle Optimizer (DBO) Algorithm
    Used to optimize LSTM hyperparameters
    """
    def __init__(self, n_population=20, n_iterations=20, n_dimensions=4,
                 lower_bound=[10, 0.01, 0.0001, 20], upper_bound=[200, 0.5, 0.01, 500],
                 logger=None):
        self.n_population = n_population
        self.n_iterations = n_iterations
        self.n_dimensions = n_dimensions
        self.lb = np.array(lower_bound)
        self.ub = np.array(upper_bound)
        self.logger = logger or logging.getLogger(__name__)

        # DBO roles distribution: 2 rolling, 4 breeding, 2 small, 2 thieving
        self.n_rolling = 2
        self.n_breeding = 4
        self.n_small = 2
        self.n_thieving = 2

    def initialize_population(self):
        """Initialize dung beetle positions"""
        positions = np.random.uniform(0, 1, (self.n_population, self.n_dimensions))
        positions = self.lb + positions * (self.ub - self.lb)
        return positions

    def fitness(self, position):
        """Evaluate fitness of a position"""
        return np.random.rand()

    def optimize(self, objective_func):
        """Run DBO optimization"""
        self.logger.info("=" * 60)
        self.logger.info("Starting DBO optimization")
        self.logger.info("Population size: %d", self.n_population)
        self.logger.info("Iterations: %d", self.n_iterations)
        self.logger.info("Dimensions: %d", self.n_dimensions)
        self.logger.info("Search space: lb=%s, ub=%s",
                         self.lb.tolist(), self.ub.tolist())

        positions = self.initialize_population()
        fitness = np.array([objective_func(pos) for pos in positions])
        best_idx = np.argmin(fitness)
        best_position = positions[best_idx].copy()
        best_fitness = fitness[best_idx]

        self.logger.info("Initial best fitness: %.4f", best_fitness)
        self.logger.info("Initial best position: %s",
                         [round(x, 4) for x in best_position])

        for t in range(self.n_iterations):
            R = 1 - t / self.n_iterations

            for i in range(self.n_population):
                new_pos = positions[i].copy()

                if i < self.n_rolling:
                    # Rolling behavior
                    alpha = 1
                    k = 0.3
                    b = np.random.rand()
                    yi_last = positions[i] if t > 0 else positions[i]
                    delta_y = np.abs(positions[i] - best_position)

                    new_pos = positions[i] + alpha * k * yi_last + b * delta_y

                elif i < self.n_rolling + self.n_breeding:
                    # Breeding behavior
                    Y_star = best_position
                    LB_star = np.maximum(Y_star * (1 - R), self.lb)
                    UB_star = np.minimum(Y_star * (1 + R), self.ub)

                    b1 = np.random.rand(self.n_dimensions)
                    b2 = np.random.rand(self.n_dimensions)
                    Bi = positions[i]

                    new_pos = Y_star + b1 * (Bi - LB_star) + b2 * (Bi - UB_star)

                elif i < self.n_rolling + self.n_breeding + self.n_small:
                    # Small dung beetle behavior
                    Yb = best_position
                    LBb = np.maximum(Yb * (1 - R), self.lb)
                    UBb = np.minimum(Yb * (1 + R), self.ub)

                    C1 = np.random.randn(self.n_dimensions)
                    C2 = np.random.rand(self.n_dimensions)

                    new_pos = positions[i] + C1 * (positions[i] - LBb) + C2 * (positions[i] - UBb)

                else:
                    # Thieving behavior
                    S = 0.5
                    g = np.random.randn(self.n_dimensions)

                    new_pos = best_position + S * g * np.abs(positions[i] - best_position) + positions[i] - best_position

                # Keep within bounds
                new_pos = np.clip(new_pos, self.lb, self.ub)

                # Calculate dance angle for rolling beetles
                if i < self.n_rolling:
                    theta = np.random.uniform(0, np.pi)
                    if theta != 0:
                        new_pos = positions[i] + np.tan(theta) * np.abs(positions[i] - yi_last)

                positions[i] = new_pos
                new_fitness = objective_func(positions[i])

                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    if new_fitness < best_fitness:
                        best_fitness = new_fitness
                        best_position = positions[i].copy()

            self.logger.info("Iteration %d/%d - Best fitness: %.4f",
                             t + 1, self.n_iterations, best_fitness)

        self.logger.info("DBO optimization completed")
        self.logger.info("Final best fitness: %.4f", best_fitness)
        self.logger.info("Final best position: %s",
                         [round(x, 4) for x in best_position])

        return best_position, best_fitness

# ============================================================
# 1DCNN-DBO-LSTM Model
# ============================================================
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available, using numpy-based simulation")
    import collections

class CNN1D(nn.Module):
    """1D CNN for feature extraction from tabular data"""
    def __init__(self, input_dim):
        super(CNN1D, self).__init__()

        # Reshape input to (batch, channels, features)
        self.input_dim = input_dim

        # Convolutional layers - designed for tabular data
        # Layer 1: 1 -> 16 channels, kernel_size=3
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        # Layer 2: 16 -> 64 channels, kernel_size=3
        self.conv2 = nn.Conv1d(16, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)

        # Use adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool1d(4)

        # Fully connected layer - output after conv + adaptive pool: 64 * 4 = 256
        self.fc = nn.Linear(64 * 4, 128)

    def forward(self, x):
        # x shape: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)

        # Conv1 + ReLU + Pool
        x = self.pool(torch.relu(self.conv1(x)))
        # Conv2 + ReLU + Pool
        x = self.pool(torch.relu(self.conv2(x)))
        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)
        x = self.dropout(x)

        # Flatten
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))

        return x

class LSTMClassifier(nn.Module):
    """LSTM for sequence classification"""
    def __init__(self, input_dim, hidden_dim, n_layers=1, dropout=0.3):
        super(LSTMClassifier, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers,
                           batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 6)  # [RQ1-6CLASS] 输出头 1 -> 6（六分类）
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Take the last time step output
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)

        # FC
        out = self.fc(last_output)  # [RQ1-6CLASS] 去 sigmoid，输出 logits

        return out

class CNN1D_LSTM(nn.Module):
    """Combined 1DCNN and LSTM model"""
    def __init__(self, input_dim, lstm_hidden_dim=90, lstm_dropout=0.3, lstm_n_layers=1):
        super(CNN1D_LSTM, self).__init__()

        self.cnn = CNN1D(input_dim)
        self.lstm = LSTMClassifier(128, lstm_hidden_dim, lstm_n_layers, lstm_dropout)

    def forward(self, x):
        # Extract features with CNN
        cnn_features = self.cnn(x)
        # Classify with LSTM
        out = self.lstm(cnn_features)
        return out

class OneDCNN_DBOLSTM:
    """1DCNN-DBO-LSTM Pipeline for Binary Classification"""

    def __init__(self, input_dim, logger=None):
        self.input_dim = input_dim
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
        self.scaler = None
        self.best_params = None

    def objective_function(self, params):
        """
        Objective function for DBO optimization
        params: [n_neurons, dropout, learning_rate, epochs]
        """
        n_neurons = int(params[0])
        dropout = max(0.0, min(0.9, params[1]))
        lr = max(1e-5, min(0.1, params[2]))
        epochs = int(max(5, min(500, params[3])))

        try:
            # Use a subset for faster hyperparameter search
            n_samples = min(5000, len(self.X_train))

            # Build model with these params
            model = CNN1D_LSTM(
                input_dim=self.input_dim,
                lstm_hidden_dim=n_neurons,
                lstm_dropout=dropout
            )

            criterion = nn.CrossEntropyLoss()  # [RQ1-6CLASS] 二分类 BCE -> 六分类交叉熵
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Training
            model.train()
            indices = np.random.choice(len(self.X_train[:n_samples]), n_samples, replace=False)
            X_batch = torch.FloatTensor(self.X_train[indices])
            y_batch = torch.LongTensor(self.y_train[indices])  # [RQ1-6CLASS] 标签改为整型索引

            for epoch in range(min(epochs, 10)):  # Limit epochs during search
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                val_outputs = model(torch.FloatTensor(self.X_val[:1000]))
                val_loss = criterion(val_outputs, torch.LongTensor(self.y_val[:1000]))

            return val_loss.item()

        except Exception as e:
            self.logger.error("Error in objective function: %s", str(e))
            return 1.0

    def fit(self, X_train, y_train, X_val, y_val):
        """Train the 1DCNN-DBO-LSTM model"""
        self.logger.info("=" * 60)
        self.logger.info("Training 1DCNN-DBO-LSTM model")
        self.logger.info("Training samples: %d", len(X_train))
        self.logger.info("Validation samples: %d", len(X_val))

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        # Normalize data
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Use DBO to find optimal LSTM hyperparameters
        self.logger.info("-" * 40)
        self.logger.info("Phase 1: DBO Optimization for LSTM hyperparameters")

        dbo = DBO(
            n_population=20,
            n_iterations=20,
            n_dimensions=4,
            lower_bound=[10, 0.01, 0.0001, 20],
            upper_bound=[200, 0.5, 0.01, 500],
            logger=self.logger
        )

        best_params, best_fitness = dbo.optimize(lambda p: self.objective_function(p))

        self.best_params = {
            'n_neurons': int(best_params[0]),
            'dropout': float(best_params[1]),
            'learning_rate': float(best_params[2]),
            'epochs': int(best_params[3])
        }

        self.logger.info("Optimal hyperparameters found:")
        self.logger.info("  n_neurons: %d", self.best_params['n_neurons'])
        self.logger.info("  dropout: %.4f", self.best_params['dropout'])
        self.logger.info("  learning_rate: %.4f", self.best_params['learning_rate'])
        self.logger.info("  epochs: %d", self.best_params['epochs'])

        # Build final model with optimal parameters
        self.logger.info("-" * 40)
        self.logger.info("Phase 2: Training final model with optimal parameters")

        self.model = CNN1D_LSTM(
            input_dim=self.input_dim,
            lstm_hidden_dim=self.best_params['n_neurons'],
            lstm_dropout=self.best_params['dropout']
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.best_params['learning_rate'])

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.LongTensor(y_train)  # [RQ1-6CLASS] 标签改为整型索引
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.LongTensor(y_val)  # [RQ1-6CLASS] 标签改为整型索引

        # Training loop
        train_losses = []
        val_losses = []

        for epoch in range(self.best_params['epochs']):
            self.model.train()
            optimizer.zero_grad()

            outputs = self.model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)

            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            train_losses.append(train_loss)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                val_losses.append(val_loss.item())

            if (epoch + 1) % 20 == 0:
                self.logger.info("Epoch %d/%d - Train Loss: %.4f - Val Loss: %.4f",
                               epoch + 1, self.best_params['epochs'], train_loss, val_loss.item())

        self.logger.info("Training completed")
        self.logger.info("Final Train Loss: %.4f", train_losses[-1])
        self.logger.info("Final Val Loss: %.4f", val_losses[-1])

        return train_losses, val_losses

    def predict(self, X_test):
        """Predict on test data"""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        self.model.eval()

        X_test_scaled = self.scaler.transform(X_test)
        X_test_tensor = torch.FloatTensor(X_test_scaled)

        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            predictions = outputs.argmax(1).numpy()  # [RQ1-6CLASS] 六分类 argmax

        return predictions

    def predict_logits(self, X_test):
        """[RQ1-FAIR F5] 返回原始 logits（未归一化的网络输出）"""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        self.model.eval()

        X_test_scaled = self.scaler.transform(X_test)
        X_test_tensor = torch.FloatTensor(X_test_scaled)

        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            return outputs.numpy()

    def predict_proba(self, X_test):
        """[RQ1-FAIR F5] 返回**真正的预测概率**（logits 的 softmax）。
        原实现直接返回 logits（既不是概率也不是决策分数），与 sklearn 的 predict_proba 语义不符。"""
        logits = self.predict_logits(X_test)
        ex = np.exp(logits - logits.max(axis=1, keepdims=True))
        return ex / ex.sum(axis=1, keepdims=True)

# ============================================================
# Main Execution
# ============================================================
def main():
    logger.info("=" * 60)
    logger.info("1DCNN-DBO-LSTM Pipeline Leak Detection - RQ1 Binary Classification")
    logger.info("Method: 1DCNN for feature extraction + DBO for LSTM optimization + LSTM for classification")
    logger.info("=" * 60)

    # Step 1: Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # Step 2: Preprocess data
    X_train_scaled, X_val_scaled, X_test_scaled, y_train_binary, y_val_binary, y_test_binary = \
        preprocess_data(X_train, X_val, X_test, y_train, y_val, y_test)

    # Step 3: Train model
    input_dim = X_train_scaled.shape[1]
    logger.info("=" * 60)
    logger.info("Feature dimension: %d", input_dim)

    model = OneDCNN_DBOLSTM(input_dim=input_dim, logger=logger)
    _rq1f_t0 = _rq1f_mark()
    train_losses, val_losses = model.fit(X_train_scaled, y_train_binary, X_val_scaled, y_val_binary)
    _rq1f_train_s = _rq1f_mark() - _rq1f_t0

    # Step 4: Evaluate on test set
    logger.info("=" * 60)
    logger.info("Evaluating on test set")

    _rq1f_t1 = _rq1f_mark()
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    _rq1f_pred_s = _rq1f_mark() - _rq1f_t1
    json.dump({"train_seconds": round(_rq1f_train_s, 3),
               "predict_seconds": round(_rq1f_pred_s, 3),
               "total_seconds": round(_rq1f_pred_s + _rq1f_train_s, 3)},
              open(os.path.join(RESULT_DIR, "timing.json"), "w"))

    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    accuracy = accuracy_score(y_test_binary, y_pred)
    precision = precision_score(y_test_binary, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test_binary, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test_binary, y_pred, average='macro', zero_division=0)

    try:
        auc = roc_auc_score(y_test_binary, y_pred_proba, multi_class='ovr', average='macro')
    except:
        auc = 0.0

    logger.info("Test Set Results:")
    logger.info("  Accuracy:  %.4f", accuracy)
    logger.info("  Precision: %.4f", precision)
    logger.info("  Recall:    %.4f", recall)
    logger.info("  F1-Score:  %.4f", f1)
    logger.info("  AUC-ROC:   %.4f", auc)

    # [RQ1-6CLASS] 导出统一格式的逐样本预测（test 行序）
    # [RQ1-FAIR F5] 三类量分别落盘：标签 / 真概率 / 原始 logits
    y_pred_logits = model.predict_logits(X_test_scaled)
    np.save(os.path.join(RESULT_DIR, 'y_pred.npy'), np.asarray(y_pred))
    np.save(os.path.join(RESULT_DIR, 'y_pred_proba.npy'), np.asarray(y_pred_proba))
    np.save(os.path.join(RESULT_DIR, 'y_pred_logits.npy'), np.asarray(y_pred_logits))
    with open(os.path.join(RESULT_DIR, 'score_semantics.json'), 'w', encoding='utf-8') as _f:
        json.dump({'y_pred.npy': 'hard labels (argmax of logits)',
                   'y_pred_proba.npy': 'true class probabilities (softmax of logits), 6 columns in class order',
                   'y_pred_logits.npy': 'raw network logits, 6 columns in class order',
                   'score_type': 'probability'}, _f, ensure_ascii=False, indent=2)

    # Step 5: Generate classification report
    target_names = ['CORROSION', 'EQUIPMENT_FAILURE', 'EXTERNAL_DAMAGE', 'MATERIAL_DEFECT', 'OPERATIONAL', 'OTHER']
    report = classification_report(y_test_binary, y_pred, target_names=target_names, digits=4)

    logger.info("=" * 60)
    logger.info("Classification Report:")
    logger.info("\n%s", report)

    # Save classification report
    report_path = os.path.join(RESULT_DIR, "classification_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("1DCNN-DBO-LSTM Pipeline Leak Detection - RQ1 Binary Classification\n")
        f.write("=" * 60 + "\n\n")
        f.write("Test Set Results:\n")
        f.write(f"  Accuracy:  {accuracy:.4f}\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall:    {recall:.4f}\n")
        f.write(f"  F1-Score:  {f1:.4f}\n")
        f.write(f"  AUC-ROC:   {auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nBest Hyperparameters (DBO optimized):\n")
        f.write(f"  n_neurons:     {model.best_params['n_neurons']}\n")
        f.write(f"  dropout:       {model.best_params['dropout']:.4f}\n")
        f.write(f"  learning_rate: {model.best_params['learning_rate']:.4f}\n")
        f.write(f"  epochs:        {model.best_params['epochs']}\n")

    logger.info("Classification report saved to: %s", report_path)

    logger.info("=" * 60)
    logger.info("1DCNN-DBO-LSTM execution completed")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'report': report,
        'best_params': model.best_params
    }

if __name__ == "__main__":
    results = main()