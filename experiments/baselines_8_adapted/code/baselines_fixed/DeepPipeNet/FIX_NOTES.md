# DeepPipeNet Adaptation & Fairness Fixes

| ID | Fix | Before | After |
|---|---|---|---|
| F1 | Softmax + CrossEntropy duplication | `nn.Softmax(dim=1)` head with `CrossEntropyLoss` | `nn.Identity()` logit head; softmax applied once at inference; `y_pred_logits.npy` saved |
| F2 | SMOTE scope | Applied to both train and validation splits | Applied strictly to training split; validation split retains original distribution |
| F6 | Unified 13-D input | Numeric columns only (9-D) | Full 13-D input from `baseline_data_13d` (padded to 4x4 grid) |

Unchanged: network architecture, hyperparameters, random seed, label mapping, and argmax decision rule.
