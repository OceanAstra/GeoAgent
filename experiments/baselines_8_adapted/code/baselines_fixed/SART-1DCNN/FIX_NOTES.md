# SART-1DCNN Adaptation & Fairness Fixes

| ID | Fix | Before | After |
|---|---|---|---|
| F3 | Categorical columns coerced to 0 | `to_numeric(..., errors='coerce').fillna(0)` zeroed all 4 categorical fields | Numeric fields imputed with training-fold medians; categoricals ordinal-encoded on training vocabulary (unseen -> `-1`) |
| H1 | Missing numeric imputation | Filled with 0 | Unified training-fold median imputation |
| F6 | Unified 13-D input | 13 columns with zeroed categoricals | Full 13 columns preserving ordinal categorical values |

Unchanged: attention/residual blocks, 30 epochs, learning rate, cross-entropy on logits, softmax inference, and argmax decision rule.
