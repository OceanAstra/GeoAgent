# Algorithm Description

## Task

Given one structured PHMSA incident record, predict which of six reported cause classes the incident belongs to. Because the 13 input features include `ACCIDENT_PSIG`, `MOP_PSIG`, and `PRESSURE_RATIO`, this is a **post-incident cause diagnosis** task rather than real-time sensor leak detection or early warning.

## Components

### 1. CatBoost Backbone (3-Model Ensemble)

- Three `CatBoostClassifier` instances (seeds `42`, `123`, `456`; 500 iterations; learning rate `0.1`; depth `6`).
- Native categorical handling for `PIPE_SPECIFICATION`, `PIPE_SEAM_TYPE`, `PIPE_COATING_TYPE`, and `PIPE_TYPE`; numeric missing values handled by CatBoost.
- **Aggregation is a probability mean**, $P_{\text{CB}} = (p_1 + p_2 + p_3)/3$ (not a logit mean).
- Out-of-fold predictions for the fit split use `StratifiedGroupKFold(3, shuffle=True, random_state=42)` keyed on `accident_group`, preventing accident-group leakage across folds.

### 2. Geometry-Aware Dual-View Residual Correction

Two complementary views are projected into a shared 32-dimensional embedding space:

- **Rule view (`rule4`):** Four fixed binary engineering indicators (coating absent/unrecorded, pipe age $\le 10$, manufacturing age $\le 5$, pressure ratio $> 1.2$), projected by `W_rule`.
- **Neural view:** The six CatBoost ensemble class probabilities, projected by `W_neural`.

Distance-aware fusion anchors on the rule representation:

```text
t = exp(log_temp) + 0.1
a = exp(-||r - h||_2 / t)
z = (r + a * h) / (1 + a + 1e-8)
```

When the two views align closely, `a` is larger and the neural view receives higher weight; when they diverge, the fused representation shifts conservatively toward the rule anchor. `log_temp` is a single learned scalar initialized to `1.0`.

### 3. Residual Scoring and Probability Averaging

```text
delta_s = W_delta(z)                     # W_delta zero-initialized
ell_s   = log(clamp(P_CB, 1e-8)) + delta_s
p_s     = softmax(ell_s)
p       = (p_42 + p_123 + p_456) / 3
```

Three independent residual heads apply their own correction and softmax before averaging probabilities across seeds.

### 4. Training Objective

```text
loss = CE(ell_s, y) + 1e-3 * mean(delta_s^2) + 1e-2 * align
```

- `align` is the mean squared distance between the retained views' embeddings (`rule` vs `neural`).
- Optimizer: Adam (`lr = 0.01`), up to 80 full-batch updates.
- Checkpoint selection: highest validation (`calib`) Macro-F1, breaking ties by earliest epoch.
- `W_proto` remains in the state dictionary purely for schema compatibility with the multi-view ablation module (`drop='proto'`) and is unused in the forward pass.

## Explainability

TreeSHAP is computed on the three-model CatBoost backbone to attribute initial class raw scores to the 13 recorded engineering features, complementing the agent's structured per-record decision traces.
