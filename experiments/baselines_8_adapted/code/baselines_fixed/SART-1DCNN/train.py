"""
SART-1DCNN: Improved 1D Convolutional Neural Network for Pipeline Leakage Detection
Based on the paper: "An Improved Convolutional Neural Network for Pipe Leakage Identification Based on Acoustic Emission"
"""

import numpy as np
import json as _rq1f_json
import time as _rq1f_time
def _rq1f_mark():
    return _rq1f_time.time()
import pandas as pd
import logging
import os
from sklearn.metrics import classification_report, matthews_corrcoef, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'sart_1dcnn_train.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

np.random.seed(42)
torch.manual_seed(42)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + residual)


class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.query = nn.Conv1d(in_channels, in_channels // 4, 1)
        self.key = nn.Conv1d(in_channels, in_channels // 4, 1)
        self.value = nn.Conv1d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, length = x.size()
        query = self.query(x).view(batch_size, -1, length).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, length)
        value = self.value(x).view(batch_size, -1, length)
        attention = torch.bmm(query, key)
        attention = torch.softmax(attention / (C ** 0.5), dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, length)
        return self.gamma * out + x


class TransformerBlock(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.ReLU(),
            nn.Linear(in_channels * 2, in_channels)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size, channels, length = x.size()
        x_flat = x.permute(0, 2, 1)
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        x_flat = self.norm1(x_flat + self.dropout(attn_out))
        ffn_out = self.ffn(x_flat)
        x_flat = self.norm2(x_flat + self.dropout(ffn_out))
        return x_flat.permute(0, 2, 1)


class SART1DCNN(nn.Module):
    def __init__(self, input_dim=1, num_classes=1):
        super(SART1DCNN, self).__init__()
        self.conv_initial = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.res_block1 = ResidualBlock(32, 64)
        self.res_block2 = ResidualBlock(64, 128)
        self.attention1 = AttentionModule(128)
        self.transformer1 = TransformerBlock(128, num_heads=4)
        self.transformer2 = TransformerBlock(128, num_heads=4)
        self.attention2 = AttentionModule(128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv_initial(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.attention1(x)
        x = self.transformer1(x)
        x = self.transformer2(x)
        x = self.attention2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.classifier(x)


def load_data(data_dir):
    logger.info(f"Loading data from {data_dir}")

    X_train_df = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    X_val_df = pd.read_csv(os.path.join(data_dir, 'X_val.csv'))
    X_test_df = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))

    # [RQ1-FAIR F3] 原实现对全部列执行 to_numeric(errors='coerce').fillna(0)：
    # 4 个类别列因转换失败被**无条件填 0**，其取值信息全部丢失（等价于常数列）。
    # 修复为：数值列用**训练折中位数**填补；类别列用**训练折词表**做序数编码（未知取值 -> -1）。
    cat_cols = [c for c in X_train_df.columns if X_train_df[c].dtype == object]
    vocab = {}
    for c in cat_cols:
        vals = sorted(X_train_df[c].astype(str).unique())
        vocab[c] = {v: i for i, v in enumerate(vals)}

    def _to_numeric_frame(df):
        out = pd.DataFrame(index=df.index)
        for c in df.columns:
            if c in vocab:
                out[c] = df[c].astype(str).map(lambda v: vocab[c].get(v, -1)).astype(np.float64)
            else:
                out[c] = pd.to_numeric(df[c], errors='coerce')
        return out

    Xn_train = _to_numeric_frame(X_train_df)
    Xn_val = _to_numeric_frame(X_val_df)
    Xn_test = _to_numeric_frame(X_test_df)

    medians = Xn_train.median()  # [RQ1-FAIR F3/H1] 填补统计量只从训练折计算
    Xn_train = Xn_train.fillna(medians)
    Xn_val = Xn_val.fillna(medians)
    Xn_test = Xn_test.fillna(medians)

    X_train = Xn_train.values.astype(np.float64)
    X_val = Xn_val.values.astype(np.float64)
    X_test = Xn_test.values.astype(np.float64)

    # Handle inf values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e10, neginf=-1e10)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)

    # Normalize per feature using training data statistics
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = ((X_train - mean) / std).astype(np.float32)
    X_val = ((X_val - mean) / std).astype(np.float32)
    X_test = ((X_test - mean) / std).astype(np.float32)

    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    y_val_df = pd.read_csv(os.path.join(data_dir, 'y_val.csv'))
    y_test_df = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))

    y_train = y_train_df['ANOMALY_LABEL_ID'].astype(int).values  # [RQ1-6CLASS] 六类标签
    y_val = y_val_df['ANOMALY_LABEL_ID'].astype(int).values  # [RQ1-6CLASS] 六类标签
    y_test = y_test_df['ANOMALY_LABEL_ID'].astype(int).values  # [RQ1-6CLASS] 六类标签

    logger.info(f"Data loaded: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    logger.info(f"[RQ1-FAIR F6] 实际输入维度 = {X_train.shape[1]}（类别列已序数编码，不再被填 0）")
    logger.info(f"Labels distribution - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}, Test: {np.bincount(y_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64):
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)  # [RQ1-6CLASS] 标签改为整型索引
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)  # [RQ1-6CLASS] 标签改为整型索引
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)  # [RQ1-6CLASS] 标签改为整型索引

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, device, epochs=30, lr=0.001):
    logger.info("Starting training...")
    criterion = nn.CrossEntropyLoss()  # [RQ1-6CLASS] 二分类 BCE -> 六分类交叉熵
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            predictions = outputs.argmax(1)  # [RQ1-6CLASS] 六分类 argmax
            train_correct += (predictions == batch_y).sum().item()
            train_total += batch_y.size(0)

        train_acc = train_correct / train_total

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                predictions = outputs.argmax(1)  # [RQ1-6CLASS] 六分类 argmax
                val_correct += (predictions == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_acc = val_correct / val_total
        scheduler.step()

        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, "
                    f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            logger.info(f"Best model saved at epoch {epoch+1} with Val Acc: {val_acc:.4f}")

    model.load_state_dict(best_model_state)
    logger.info("Training completed")
    return model


def evaluate_model(model, test_loader, device):
    logger.info("Evaluating model on test set...")
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = probs.argmax(1)  # [RQ1-6CLASS] 六分类 softmax + argmax
            y_pred.extend(predictions)
            y_prob.extend(probs)
            y_true.extend(batch_y.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    # Result directory is not shipped with the repository; create it before the
    # first write. Previously an implicit os.makedirs() came only after these
    # np.save() calls, which broke every fresh clone.
    os.makedirs(os.path.join(os.path.dirname(__file__), 'result'), exist_ok=True)
    np.save(os.path.join(os.path.dirname(__file__), 'result', 'y_pred.npy'), np.asarray(y_pred))
    np.save(os.path.join(os.path.dirname(__file__), 'result', 'y_pred_proba.npy'), np.asarray(y_prob))
    accuracy = (y_pred == y_true).mean()
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test MCC: {mcc:.4f}")
    logger.info(f"Test AUC: {auc:.4f}")

    target_names = ['CORROSION', 'EQUIPMENT_FAILURE', 'EXTERNAL_DAMAGE', 'MATERIAL_DEFECT', 'OPERATIONAL', 'OTHER']
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4, output_dict=True)

    result_dir = os.path.join(os.path.dirname(__file__), 'result')
    os.makedirs(result_dir, exist_ok=True)
    report_path = os.path.join(result_dir, 'classification_report.txt')

    with open(report_path, 'w') as f:
        f.write("SART-1DCNN Pipeline Leak Detection Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Test MCC: {mcc:.4f}\n")
        f.write(f"Test AUC: {auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}\n")
        f.write("-" * 60 + "\n")
        for cls in target_names:
            if cls in report:
                f.write(f"{cls:<20} {report[cls]['precision']:>10.4f} {report[cls]['recall']:>10.4f} {report[cls]['f1-score']:>10.4f} {report[cls]['support']:>10.0f}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Accuracy':<20} {'':<10} {'':<10} {report['accuracy']:>10.4f} {report['macro avg']['support']:>10.0f}\n")
        f.write(f"{'Macro Avg':<20} {report['macro avg']['precision']:>10.4f} {report['macro avg']['recall']:>10.4f} {report['macro avg']['f1-score']:>10.4f} {report['macro avg']['support']:>10.0f}\n")
        f.write(f"{'Weighted Avg':<20} {report['weighted avg']['precision']:>10.4f} {report['weighted avg']['recall']:>10.4f} {report['weighted avg']['f1-score']:>10.4f} {report['weighted avg']['support']:>10.0f}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("Model: SART-1DCNN (Spatial Attention + Residual + Transformer)\n")
        f.write("Additional Metrics: MCC (Matthews Correlation Coefficient), AUC (Area Under ROC Curve)\n")

    logger.info(f"Classification report saved to {report_path}")
    return accuracy, mcc, auc


def main():
    logger.info("=" * 60)
    logger.info("SART-1DCNN Pipeline Leak Detection")
    logger.info("Based on paper: An Improved CNN for Pipe Leakage Identification")
    logger.info("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = _resolve_data_dir()
    batch_size = 128
    epochs = 30
    learning_rate = 0.001

    logger.info(f"Configuration: batch_size={batch_size}, epochs={epochs}, lr={learning_rate}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(data_dir)

    train_loader, val_loader, test_loader = prepare_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size
    )

    input_dim = X_train.shape[1]
    logger.info(f"Input dimension: {input_dim}")

    model = SART1DCNN(input_dim=1, num_classes=6).to(device)  # [RQ1-6CLASS] 六类输出
    logger.info(f"Model initialized: SART-1DCNN")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")

    _rq1f_t0 = _rq1f_mark()
    model = train_model(model, train_loader, val_loader, device, epochs, learning_rate)
    _rq1f_train_s = _rq1f_mark() - _rq1f_t0

    _rq1f_t1 = _rq1f_mark()
    accuracy, mcc, auc = evaluate_model(model, test_loader, device)
    _rq1f_pred_s = _rq1f_mark() - _rq1f_t1
    _rq1f_json.dump({"train_seconds": round(_rq1f_train_s, 3),
                     "predict_seconds": round(_rq1f_pred_s, 3),
                     "total_seconds": round(_rq1f_pred_s + _rq1f_train_s, 3)},
                    open(os.path.join(os.path.dirname(__file__), "result", "timing.json"), "w"))

    logger.info("=" * 60)
    logger.info("SART-1DCNN Pipeline Leak Detection Completed")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()