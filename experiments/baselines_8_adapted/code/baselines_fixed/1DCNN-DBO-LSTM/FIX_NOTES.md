# 1DCNN-DBO-LSTM Adaptation & Fairness Fixes

| ID | Fix | Before | After |
|---|---|---|---|
| F4 | Categorical vocabulary scope | Fitted on `concat([train, val, test])` | Fitted strictly on `X_train`; unseen values -> `-1` |
| F5 | Probability / logit separation | `predict_proba` returned raw logits | `predict_proba` returns `softmax(logits)`; logits saved separately with `score_semantics.json` |
| F6 | Unified 13-D input | 13 columns | 13 columns (train-fitted ordinal categorical encoding) |

Note: `argmax(logits) == argmax(softmax(logits))`, so F5 preserves argmax metrics while aligning exported probability semantics. Unchanged: 1DCNN + DBO + LSTM architecture, DBO hyperparameters, cross-entropy loss, random seed, and argmax decision rule.
