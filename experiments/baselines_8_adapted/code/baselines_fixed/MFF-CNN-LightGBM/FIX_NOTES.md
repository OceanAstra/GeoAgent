# MFF-CNN-LightGBM Adaptation & Fairness Fixes

| ID | Fix | Before | After |
|---|---|---|---|
| H1 | Missing numeric imputation | Constant sentinel `-999` | Training-fold median (aligned with shared protocol) |
| F6 | Unified 13-D input | Numeric columns only (9-D) | Full 13-D input from `baseline_data_13d` |

Unchanged: MFF-CNN feature extractor, LightGBM multiclass parameters (`num_class=6`), random seed, and argmax decision rule.
