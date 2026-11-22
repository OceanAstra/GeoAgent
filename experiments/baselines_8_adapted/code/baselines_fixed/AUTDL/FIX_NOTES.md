# AUTDL Adaptation & Fairness Fixes

| ID | Fix | Before | After |
|---|---|---|---|
| F1 | Softmax + CrossEntropy duplication | `self.softmax(x)` inside `forward` with `CrossEntropyLoss` | `forward` outputs logits; softmax applied once in `predict()` |
| F6 | Unified 13-D input | Numeric columns only (9-D) | Full 13-D input from `baseline_data_13d` (train-fitted ordinal encoding for categoricals) |

Unchanged: network architecture, hyperparameters, random seed, epochs, and argmax decision rule.
