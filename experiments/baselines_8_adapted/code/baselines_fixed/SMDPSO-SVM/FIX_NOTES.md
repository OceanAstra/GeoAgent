# SMDPSO-SVM Adaptation, Fairness Fixes & Cost-Control Profile

## 1. Computational Bottleneck

SMDPSO executes 20 particles x 30 iterations = 600 fitness evaluations. Each evaluation fits a 6-class precomputed-kernel SVC (15 OvO subproblems) and scores a validation subsample. At 1,500 fit / 300 val rows per evaluation, full search exceeds the 5,400 s compute budget by ~4x.

## 2. Budget Profiles (Algorithm Preserved)

| Profile | Particles x Iters | Fitness Fit / Val Subsample | Note |
|---|---|---|---|
| `full` | 20 x 30 | 1,500 / 300 | Original subsample size (exceeds compute budget) |
| `reduced` (used) | 20 x 30 | 400 / 150 | Identical search space, particle/iteration count, and hybrid kernel; reduced fitness subsample resolution |
| `smoke` | 2 x 2 | 150 / 100 | Fast pipeline verification |

The SMDPSO search dynamics, search space, hybrid kernel (`alpha * RBF + (1 - alpha) * Poly`), and final model re-fit remain unchanged (final training cap controlled by `RQ1_SMDPSO_TRAIN_CAP`, default 5,000 rows).

## 3. Fairness Fixes

| ID | Fix | Before | After |
|---|---|---|---|
| F5 | Decision score semantics | Default OvO `decision_function` returned 5 columns | `decision_function_shape='ovr'` returns 6 columns aligned with `model.classes_`; documented in `score_semantics.json` |
| F6 | Unified 13-D input | Numeric columns only (9-D) | Full 13-D input from `baseline_data_13d` |
