# Model Card — GeoAgent Dual-View Classifier (`P_no_proto`)

## Overview

| Item | Value |
|---|---|
| Model | GeoAgent Dual-View Classifier (3-seed CatBoost ensemble + geometry-aware dual-view residual correction) |
| Version tag | `Tier1-2026-10-28` (frozen) |
| Task | Six-class post-incident cause classification on structured PHMSA pipeline incident records |
| Input | 13 structured engineering features (9 numeric, 4 categorical) |
| Output | 6 class probabilities (softmax) and argmax cause label |
| Intended use | Research on retrospective pipeline accident-cause attribution, explainability, and controlled repair |
| Out of scope | Continuous real-time sensor leak detection or physical localization |

## Exact Architecture

```text
x (13) ──► CatBoost x3 (seeds 42/123/456, 500 iters, lr 0.1, depth 6)
              └─► P_CB = mean of 3 predict_proba outputs (probability mean)

rule4 = [coat, age<=10, mfg<=5, pr>1.2]        (4 binary engineering indicators)
r = W_rule(rule4)      in R^32
h = W_neural(P_CB)     in R^32
t = exp(log_temp) + 0.1
a = exp(-||r - h||_2 / t)
z = (r + a * h) / (1 + a + 1e-8)
delta_s = W_delta(z)   in R^6                  (per head seed s)
ell_s   = log(clamp(P_CB, 1e-8)) + delta_s
p       = (softmax(ell_42) + softmax(ell_123) + softmax(ell_456)) / 3
```

- `W_proto` is retained in the state dict for compatibility with the multi-view ablation module (`drop='proto'`) and is inactive during forward inference.
- Active head parameters: **583** (903 including inactive `W_proto`).

## Shipped Checkpoints

| File | Size (B) | SHA-256 (first 16) |
|---|---:|---|
| `weights/cb_seed42.cbm` | 2,714,868 | `f6e3e61e7abf3016` |
| `weights/cb_seed123.cbm` | 2,927,812 | `cd523fa4e384dc78` |
| `weights/cb_seed456.cbm` | 2,772,000 | `38cf1be438328010` |
| `weights/P_no_proto_seed42.pt` | 7,166 | `dc5cb76d70618cec` |
| `weights/P_no_proto_seed123.pt` | 7,179 | `6e161514e87e9003` |
| `weights/P_no_proto_seed456.pt` | 7,179 | `e0fe50ab8e001c46` |

## Loading Example

```python
from catboost import CatBoostClassifier
import torch

m = CatBoostClassifier()
m.load_model("models_and_configs/geoad_materials/weights/cb_seed42.cbm")

# DualViewHead is defined in experiments/geoad_code/inference/run_inference.py
net.load_state_dict(
    torch.load("models_and_configs/geoad_materials/weights/P_no_proto_seed42.pt",
               map_location="cpu", weights_only=True)
)
```

## Performance (Development-Confirmation Split, $n = 6,417$ / 6,373 Accident Groups)

| Metric | GeoAgent (`P_no_proto`) | CatBoost Ensemble (`A_ensemble`) |
|---|---:|---:|
| Accuracy | **71.5911%** | 71.4508% |
| Macro-F1 | **58.6332%** | 57.8349% |
| Weighted-F1 | **69.4082%** | 69.0039% |
| Correct records | **4,594** | 4,585 |
| $\Delta$Macro-F1 | **+0.7983 pp** (Holm-adjusted $p = 0.0009$) | — |
| $\Delta$Accuracy | +0.1403 pp (McNemar $p = 0.3356$) | — |

## Scope & Data Characteristics

- Class imbalance ratio is 9.45:1 (`EXTERNAL_DAMAGE` 40.54% vs `OPERATIONAL` 3.92%); natural class priors are preserved without synthetic oversampling to maintain probability calibration for downstream agent repair.
- Several structural fields exhibit high historical missingness (`PIPE_TYPE` 95.11%, `INTERNAL_INSPECTION_IND` 93.28%, `MANUFACTURE_AGE_AT_ACCIDENT` 89.15%, `PIPE_SEAM_TYPE` 88.92%), documented in `DATA_DICTIONARY.csv`.
- Outputs represent predictive cause attribution from reported records rather than direct measurements of physical causation.
