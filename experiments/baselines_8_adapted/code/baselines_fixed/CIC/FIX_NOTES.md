# CIC Adaptation & Fairness Fixes

| ID | Fix | Before | After |
|---|---|---|---|
| F4 | Categorical vocabulary scope | Fitted on `concat([train, val])` | Fitted strictly on `X_train` |
| F4 | Unseen category sentinel | `-999` | `-1` (aligned with shared encoder) |
| F6 | Unified 13-D input | 13 columns | 13 columns (train-fitted ordinal encoding, unseen -> `-1`) |

Intentionally preserved CIC algorithm components: 3-cluster MiniBatchKMeans, per-cluster MatrixImputation (`-999` missing sentinel), StandardScaler, RUSBoost + AdaBoost (SAMME, 30 trees), hyperparameters, random seed, and argmax decision rule.
