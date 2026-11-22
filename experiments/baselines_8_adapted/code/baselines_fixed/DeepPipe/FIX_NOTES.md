# DeepPipe Adaptation & Fairness Fixes

| ID | Fix | Before | After |
|---|---|---|---|
| H1 | Missing numeric imputation | Constant sentinel `-999` | Training-fold median (aligned with shared protocol) |
| F6 | Unified 13-D input | Numeric columns only (9-D) | Full 13-D input from `baseline_data_13d` (Keras input shape adapts dynamically) |

Unchanged: LSTM / ConvLSTM branches, LightGBM hyperparameters, sparse cross-entropy loss, random seed, and argmax decision rule.
