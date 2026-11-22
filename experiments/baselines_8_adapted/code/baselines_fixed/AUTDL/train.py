"""
AUTDL-based method for pipeline anomaly binary classification
Based on the paper: "Defect identification for oil and gas pipeline safety based on autonomous deep learning network"

This script implements the AUTDL method adapted for tabular data.
Core principles from AUTDL:
- Simplified CNN architecture (inspired by modified AlexNet)
- Dropout for regularization (avoid overfitting)
- Softmax output for classification
- Data fusion concept (for tabular, we process features together)

Author: Claude
RQ1: Binary classification (Normal vs Anomaly)
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import json
import time as _rq1f_time

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

def _rq1f_mark():
    return _rq1f_time.time()

# ============== Configuration ==============
DATA_DIR = _resolve_data_dir()
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(OUTPUT_DIR, "log")
RESULT_DIR = os.path.join(OUTPUT_DIR, "result")

# Hyperparameters (based on AUTDL paper principles)
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.5
RANDOM_SEED = 42

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============== Logging Setup ==============
def setup_logging():
    """Setup logging to both file and console"""
    os.makedirs(LOG_DIR, exist_ok=True)

    log_file = os.path.join(LOG_DIR, f"autodl_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# ============== AUTDL Network Architecture ==============
class AUTDLNetwork(nn.Module):
    """
    Simplified CNN network based on AUTDL paper.
    Key features from AUTDL:
    - Simplified architecture (fewer layers than standard AlexNet)
    - Dropout for regularization (avoid overfitting)
    - ReLU activation
    - MaxPooling
    - Softmax output
    """
    def __init__(self, input_dim, num_features):
        super(AUTDLNetwork, self).__init__()

        # Reshape input: (batch, num_features) -> (batch, 1, num_features)
        # Apply 1D convolution to extract features

        # Conv layer (inspired by AUTDL's Conv_1)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        # MaxPooling
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Second conv layer
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Calculate feature dimension after conv layers
        self.feature_dim = num_features // 4 * 128

        # Dropout (key component from AUTDL to avoid overfitting)
        self.dropout1 = nn.Dropout(p=DROPOUT_RATE)

        # Fully connected layers (simplified from AUTDL: Fc_1, Fc_2, Fc_3)
        self.fc1 = nn.Linear(self.feature_dim, 256)
        self.dropout2 = nn.Dropout(p=DROPOUT_RATE)
        self.fc2 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(p=DROPOUT_RATE)
        self.fc3 = nn.Linear(128, 6)  # [RQ1-6CLASS] 输出头 2 -> 6（六分类）

        # [RQ1-FAIR F1] 原实现在 forward 内接了 Softmax，与 CrossEntropyLoss 重复施加，
        # 训练目标被二次压缩、推理概率被再次归一化。改为：forward 只输出 logits，
        # 交叉熵直接吃 logits，softmax 仅在推理阶段施加一次。

    def forward(self, x):
        # Reshape for 1D conv: (batch, num_features) -> (batch, 1, num_features)
        x = x.unsqueeze(1)

        # Conv block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Dropout
        x = self.dropout1(x)

        # Fully connected layers
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout3(x)

        x = self.fc3(x)

        return x  # [RQ1-FAIR F1] logits（softmax 在 predict() 内施加一次）


# ============== Data Loading & Preprocessing ==============
def load_data(logger):
    """Load and preprocess data"""
    logger.info("Loading data...")

    # Load data
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
    X_val = pd.read_csv(os.path.join(DATA_DIR, "X_val.csv"))
    y_val = pd.read_csv(os.path.join(DATA_DIR, "y_val.csv"))
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))

    logger.info(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")

    # Extract binary label (IS_ANOMALY: 0=Normal, 1=Anomaly)
    y_train_labels = y_train['ANOMALY_LABEL_ID'].values
    y_val_labels = y_val['ANOMALY_LABEL_ID'].values
    y_test_labels = y_test['ANOMALY_LABEL_ID'].values

    logger.info(f"Train class distribution: {np.bincount(y_train_labels, minlength=6).tolist()}")
    logger.info(f"Test class distribution: {np.bincount(y_test_labels, minlength=6).tolist()}")

    # Only keep numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train = X_train[numeric_cols]
    X_val = X_val[numeric_cols]
    X_test = X_test[numeric_cols]

    logger.info(f"Numeric features: {len(numeric_cols)}")

    # Handle missing values - fill with median
    X_train = X_train.fillna(X_train.median())
    X_val = X_val.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Convert to numpy
    X_train_np = X_train_scaled.astype(np.float32)
    X_val_np = X_val_scaled.astype(np.float32)
    X_test_np = X_test_scaled.astype(np.float32)

    num_features = X_train_np.shape[1]
    logger.info(f"Number of features: {num_features}")

    return (X_train_np, y_train_labels,
            X_val_np, y_val_labels,
            X_test_np, y_test_labels, num_features)


def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, logger):
    """Create PyTorch dataloaders"""
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    # Create datasets
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    logger.info(f"Created dataloaders with batch_size={BATCH_SIZE}")

    return train_loader, val_loader, test_loader


# ============== Training ==============
def train_epoch(model, dataloader, criterion, optimizer, device, logger):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device, logger):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def predict(model, dataloader, device):
    """Make predictions"""
    model.eval()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            output = model(data)
            logits = output  # [RQ1-FAIR F1] 网络输出为 logits
            probs = torch.softmax(logits, dim=1)  # [RQ1-FAIR F1] 推理阶段唯一的 softmax
            pred = logits.argmax(dim=1)  # [RQ1-FAIR F1] argmax(logits) 与 argmax(softmax) 等价
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_probs)


def save_classification_report(y_true, y_pred, output_path, logger):
    """Save classification report"""
    report = classification_report(y_true, y_pred, target_names=['CORROSION', 'EQUIPMENT_FAILURE', 'EXTERNAL_DAMAGE', 'MATERIAL_DEFECT', 'OPERATIONAL', 'OTHER'], digits=4)
    confusion = confusion_matrix(y_true, y_pred)

    logger.info("\n" + "="*50)
    logger.info("CLASSIFICATION REPORT")
    logger.info("="*50)
    logger.info("\n" + report)
    logger.info("\nCONFUSION MATRIX:")
    logger.info(confusion)

    # Save to file
    with open(output_path, 'w') as f:
        f.write("AUTDL Method - Binary Classification Report\n")
        f.write("="*50 + "\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(confusion))
        f.write("\n\nLabel Mapping: 0=Normal, 1=Anomaly\n")

    logger.info(f"Classification report saved to {output_path}")


# ============== Main ==============
def main():
    # Setup
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    logger = setup_logging()

    logger.info("="*60)
    logger.info("AUTDL Method for Pipeline Anomaly Binary Classification")
    logger.info("="*60)
    logger.info(f"Task: RQ1 - Binary Classification (Normal vs Anomaly)")
    logger.info(f"Based on paper: Defect identification for oil and gas pipeline...")
    logger.info("="*60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data
    (X_train, y_train,
     X_val, y_val,
     X_test, y_test, num_features) = load_data(logger)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, logger
    )

    # Create model
    model = AUTDLNetwork(input_dim=X_train.shape[1], num_features=num_features).to(device)
    logger.info(f"\nModel Architecture:\n{model}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

    logger.info(f"\nTraining settings:")
    logger.info(f"  - Epochs: {EPOCHS}")
    logger.info(f"  - Learning rate: {LEARNING_RATE:.4f}")
    logger.info(f"  - Momentum: {0.9:.4f}")
    logger.info(f"  - Weight decay: {0.0005:.4f}")
    logger.info(f"  - Dropout rate: {DROPOUT_RATE:.4f}")
    logger.info(f"  - Batch size: {BATCH_SIZE}")

    # Training loop
    best_val_acc = 0
    best_model_state = None

    logger.info("\n" + "="*60)
    logger.info("Starting Training...")
    logger.info("="*60)

    _rq1f_t0 = _rq1f_mark()
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, logger)
        val_loss, val_acc = validate(model, val_loader, criterion, device, logger)

        logger.info(f"Epoch [{epoch+1:3d}/{EPOCHS}] "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}% | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            logger.info(f"  -> New best model saved! Val Acc: {val_acc:.4f}%")

    _rq1f_train_s = _rq1f_mark() - _rq1f_t0
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"\nLoaded best model with Val Acc: {best_val_acc:.4f}%")

    # Test evaluation
    logger.info("\n" + "="*60)
    logger.info("Evaluating on Test Set...")
    logger.info("="*60)

    test_loss, test_acc = validate(model, test_loader, criterion, device, logger)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}%")

    # Predictions
    _rq1f_t1 = _rq1f_mark()
    y_pred, y_prob = predict(model, test_loader, device)
    _rq1f_pred_s = _rq1f_mark() - _rq1f_t1
    json.dump({"train_seconds": round(_rq1f_train_s, 3),
               "predict_seconds": round(_rq1f_pred_s, 3),
               "total_seconds": round(_rq1f_pred_s + _rq1f_train_s, 3)},
              open(os.path.join(RESULT_DIR, "timing.json"), "w"))

    # Save classification report
    report_path = os.path.join(RESULT_DIR, "classification_report.txt")
    save_classification_report(y_test, y_pred, report_path, logger)

    # Save predictions
    np.save(os.path.join(RESULT_DIR, 'y_pred.npy'), np.asarray(y_pred))
    np.save(os.path.join(RESULT_DIR, 'y_pred_proba.npy'), np.asarray(y_prob))
    pred_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred,
    })
    for _i, _c in enumerate(['CORROSION', 'EQUIPMENT_FAILURE', 'EXTERNAL_DAMAGE', 'MATERIAL_DEFECT', 'OPERATIONAL', 'OTHER']):
        pred_df[f'prob_{_c}'] = np.asarray(y_prob)[:, _i]  # [RQ1-6CLASS] 6 维概率逐列落地
    pred_path = os.path.join(RESULT_DIR, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"Predictions saved to {pred_path}")

    # Save metrics summary
    summary = {
        'method': 'AUTDL',
        'task': 'RQ1 Binary Classification',
        'best_val_accuracy': float(best_val_acc),
        'test_accuracy': float(test_acc),
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'dropout_rate': DROPOUT_RATE,
        'total_parameters': total_params
    }
    summary_path = os.path.join(RESULT_DIR, "metrics_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Metrics summary saved to {summary_path}")

    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)

    return model, test_acc


if __name__ == "__main__":
    main()