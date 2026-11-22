"""
DeepPipeNet简化版 - 管道异常二分类 (PyTorch实现)
主要思路: 混合CNN特征提取 + 注意力机制 + 分类器
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
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

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


# ============== 日志配置 ==============
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log')
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f'baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

DATA_DIR = _resolve_data_dir()


class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(channels // ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        attention = (avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        return x * attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        spatial = self.sigmoid(self.conv(concat))
        return x * spatial


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class VGGStyleBranch(nn.Module):
    """VGG风格分支 - 3x3卷积堆叠"""
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.att = AttentionBlock(64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.att(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.squeeze(-1).squeeze(-1)


class ResNetStyleBranch(nn.Module):
    """ResNet风格分支 - 残差连接"""
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.shortcut = nn.Conv2d(in_channels, 64, 1, bias=False)
        self.att = AttentionBlock(64)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + shortcut
        x = self.att(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.squeeze(-1).squeeze(-1)


class DenseNetStyleBranch(nn.Module):
    """DenseNet风格分支 - 密集连接"""
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.att = AttentionBlock(96)  # 32*3

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.att(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.squeeze(-1).squeeze(-1)


class DeepPipeNet(nn.Module):
    """DeepPipeNet模型"""
    def __init__(self, input_dim=14, num_classes=2):
        super().__init__()

        # 计算合适的2D矩阵大小
        side = int(np.ceil(np.sqrt(input_dim)))
        padded_dim = side * side
        self.pad_len = padded_dim - input_dim
        self.side = side

        # 三个CNN分支
        self.vgg_branch = VGGStyleBranch()
        self.res_branch = ResNetStyleBranch()
        self.dense_branch = DenseNetStyleBranch()

        # 特征维度: 64 + 64 + 96 = 224
        feature_dim = 64 + 64 + 96

        # 全连接分类器
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1 if num_classes == 2 else num_classes),
        )

        if num_classes == 2:
            self.output_activation = nn.Sigmoid()
        else:
            # [RQ1-FAIR F1] 原实现对多分类输出头接了 Softmax，又用 CrossEntropyLoss 训练，
            # 属于"softmax + 交叉熵"的重复组合。改为 Identity：网络输出 logits，
            # 交叉熵吃 logits，softmax 只在推理时施加。
            self.output_activation = nn.Identity()

    def forward(self, x):
        # 填充到完美平方
        if self.pad_len > 0:
            x = F.pad(x, (0, self.pad_len))
        # 转为2D矩阵 (batch, 1, side, side)
        x = x.view(-1, 1, self.side, self.side)

        # 三个分支
        vgg_out = self.vgg_branch(x)
        res_out = self.res_branch(x)
        dense_out = self.dense_branch(x)

        # 拼接
        merged = torch.cat([vgg_out, res_out, dense_out], dim=1)

        # 分类
        out = self.classifier(merged)
        out = self.output_activation(out)

        return out


def load_data():
    """加载数据"""
    logger.info("=" * 60)
    logger.info("加载数据...")
    logger.info("=" * 60)

    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    X_val = pd.read_csv(os.path.join(DATA_DIR, "X_val.csv"))
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
    y_val = pd.read_csv(os.path.join(DATA_DIR, "y_val.csv"))
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))

    logger.info(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")

    y_train_bin = y_train['ANOMALY_LABEL_ID'].values  # [RQ1-6CLASS] 六类标签
    y_val_bin = y_val['ANOMALY_LABEL_ID'].values  # [RQ1-6CLASS] 六类标签
    y_test_bin = y_test['ANOMALY_LABEL_ID'].values  # [RQ1-6CLASS] 六类标签

    logger.info(f"训练集标签分布: {np.bincount(y_train_bin, minlength=6).tolist()}")

    return X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin


def preprocess_data(X_train, X_val, X_test):
    """数据预处理"""
    logger.info("=" * 60)
    logger.info("数据预处理...")
    logger.info("=" * 60)

    # 只保留数值列
    numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    X_train = X_train[numeric_cols].fillna(X_train[numeric_cols].median())
    X_val = X_val[numeric_cols].fillna(X_train[numeric_cols].median())
    X_test = X_test[numeric_cols].fillna(X_train[numeric_cols].median())

    logger.info(f"使用特征数量: {X_train.shape[1]}")

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 保留小数点后4位
    X_train_scaled = np.round(X_train_scaled, 4)
    X_val_scaled = np.round(X_val_scaled, 4)
    X_test_scaled = np.round(X_test_scaled, 4)

    return X_train_scaled, X_val_scaled, X_test_scaled


def train_and_evaluate():
    """训练和评估"""
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 1. 加载数据
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # 2. 预处理
    X_train_scaled, X_val_scaled, X_test_scaled = preprocess_data(X_train, X_val, X_test)

    # 3. SMOTE处理类别不平衡
    logger.info("=" * 60)
    logger.info("SMOTE处理...")
    logger.info("=" * 60)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    # [RQ1-FAIR F2] SMOTE 只允许用于训练集；验证集必须保留原始样本分布
    # （原实现对验证集也做了 SMOTE，导致"验证分布"被过采样污染，模型选择失真）。
    X_val_res, y_val_res = X_val_scaled, y_val

    logger.info(f"SMOTE后训练集: {X_train_res.shape}, 0={np.sum(y_train_res==0)}, 1={np.sum(y_train_res==1)}")

    # 4. 转换为PyTorch张量
    X_train_t = torch.FloatTensor(X_train_res)
    y_train_t = torch.LongTensor(y_train_res)  # [RQ1-6CLASS] 标签改为整型索引
    X_val_t = torch.FloatTensor(X_val_res)
    y_val_t = torch.LongTensor(y_val_res)  # [RQ1-6CLASS] 标签改为整型索引
    X_test_t = torch.FloatTensor(X_test_scaled)
    y_test_t = torch.LongTensor(y_test)

    # DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 5. 构建模型
    logger.info("=" * 60)
    logger.info("构建模型...")
    logger.info("=" * 60)

    model = DeepPipeNet(input_dim=X_train_scaled.shape[1], num_classes=6).to(device)  # [RQ1-6CLASS] 六类输出
    logger.info(f"模型结构:\n{model}")

    # 6. 训练配置
    criterion = nn.CrossEntropyLoss()  # [RQ1-6CLASS] 二分类 BCE -> 六分类交叉熵
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # 7. 训练
    logger.info("=" * 60)
    logger.info("开始训练...")
    logger.info("=" * 60)

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10

    _rq1f_t0 = _rq1f_mark()
    for epoch in range(50):
        # 训练
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        logger.info(f"Epoch {epoch+1}/50 - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(RESULT_DIR, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"早停于 Epoch {epoch+1}")
                break

    _rq1f_train_s = _rq1f_mark() - _rq1f_t0
    # 8. 评估
    logger.info("=" * 60)
    logger.info("评估模型...")
    logger.info("=" * 60)

    model.load_state_dict(torch.load(os.path.join(RESULT_DIR, 'best_model.pth'),
                                     map_location='cpu', weights_only=True))
    model.eval()

    _rq1f_t1 = _rq1f_mark()
    with torch.no_grad():
        X_test_t = X_test_t.to(device)
        logits = model(X_test_t).cpu().numpy()  # [RQ1-FAIR F1] 原始 logits
        ex = np.exp(logits - logits.max(axis=1, keepdims=True))
        y_pred_prob = ex / ex.sum(axis=1, keepdims=True)  # [RQ1-FAIR F1] 真概率
        y_pred = logits.argmax(1)  # [RQ1-6CLASS] 六分类 argmax
        np.save(os.path.join(RESULT_DIR, 'y_pred.npy'), np.asarray(y_pred))
        _rq1f_pred_s = _rq1f_mark() - _rq1f_t1
        _rq1f_json.dump({'train_seconds': round(_rq1f_train_s, 3),
                        'predict_seconds': round(_rq1f_pred_s, 3),
                        'total_seconds': round(_rq1f_pred_s + _rq1f_train_s, 3)},
                       open(os.path.join(RESULT_DIR, 'timing.json'), 'w'))
        np.save(os.path.join(RESULT_DIR, 'y_pred_proba.npy'), np.asarray(y_pred_prob))
        np.save(os.path.join(RESULT_DIR, 'y_pred_logits.npy'), np.asarray(logits))

    report = classification_report(y_test, y_pred, target_names=['CORROSION', 'EQUIPMENT_FAILURE', 'EXTERNAL_DAMAGE', 'MATERIAL_DEFECT', 'OPERATIONAL', 'OTHER'], digits=4)
    logger.info("\n" + report)

    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\n混淆矩阵:\n{cm}")

    # 计算MCC和AUC
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr', average='macro')
    logger.info(f"\nMCC: {mcc:.4f}")
    logger.info(f"AUC: {auc:.4f}")

    # 保存报告
    report_file = os.path.join(RESULT_DIR, 'DeepPipeNet_classification_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"Model: DeepPipeNet Baseline (PyTorch)\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write(f"\n\nMCC: {mcc:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")

    logger.info(f"报告已保存: {report_file}")
    logger.info("=" * 60)
    logger.info("训练完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    train_and_evaluate()