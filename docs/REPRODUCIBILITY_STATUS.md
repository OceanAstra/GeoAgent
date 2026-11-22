# Reproducibility Status and Scope

This document summarizes the verified reproduction level for each component included in GeoAgent `v1.0.0`.

## Component Verification Matrix

| # | Component | Shipped Artifacts | Verification Level | Observed Result |
|---|---|---|---|---|
| 1 | Six-class dual-view classifier (6,417 dev-confirm rows) | 3 `.cbm` + 3 `.pt` checkpoints, `run_inference.py` | **Frozen Inference Verified** | Acc **71.5911%**, Macro-F1 **58.6332%**, Weighted-F1 **69.4082%**; 100.00% argmax agreement; max $\|\Delta p\| = 1.788 \times 10^{-7}$ |
| 2 | Label-free demo forward pass (24 rows) | `predict_demo.py`, `dev_confirm_24rows_no_label.csv` | **Frozen Inference Verified** | Exact probability match ($\max|\Delta p| = 0.0$) |
| 3 | 2025 historical feedback replay (542 rows) | `P23_INPUT_GROUPKEY_REPAIRED.csv`, 3 `.joblib` heads | **Replay Verified** | KEEP 61.81% → Full 64.02% Acc; Macro-F1 35.64% → 39.27%; 12/12 correct, 0 harms |
| 4 | 2026 historical feedback replay (40 rows) | Same as above | **Replay Verified** | KEEP 55.00% → Full 60.00% Acc; Macro-F1 28.71% → 35.66%; 2/2 correct, 0 harms |
| 5 | Row-level 582-record policy audit (8 P23 columns + 2 P24 columns) | `PREDICTIONS_582.csv`, `export_p23_rowlevel.py`, `export_p24_rowlevel.py` | **Replay Verified** | **582/582** exact match across all 10 policy columns (see [P24_ARMS.md](P24_ARMS.md)) |
| 6 | Equal-$K$ controls ($K=12$ / $K=2$) | `results/equal_k_corrected_20261120/` | **Replay Verified** | 582/582 exact match with `PREDICTIONS_582.csv` |
| 7 | Eight adapted external baselines (archived predictions) | `results/per_row_predictions/baselines_8/` | **Metrics Verified (8/8)** | `reproduce/verify_baselines.py` passes 8/8 |
| 8 | Baseline tabular dataset builder (13-D / 9-D) | `rq1f_build_baseline_data.py` | **Deterministic Rebuild** | Exact 29,941 / 6,411 / 6,417 split reconstruction |
| 9 | Baseline fresh re-training | `baselines_fixed/*/train.py` | **Partial Fresh Retrain** | 4/8 exact row-level match (CIC, AUTDL, SART-1DCNN, SMDPSO-SVM); 2/8 stochastic variance (DeepPipe, DeepPipeNet); 2/8 resource-bounded (1DCNN-DBO-LSTM, MFF-CNN-LightGBM); see [BASELINES_AND_PROTOCOL.md](BASELINES_AND_PROTOCOL.md) |
| 10 | TreeSHAP figures & additivity verification | `per_sample_shap_6417.npz` (`allow_pickle=False`), `generate_shap_figures.py` | **Verified** | Max summary diff $1.392 \times 10^{-9}$; ensemble additivity residual $1.332 \times 10^{-14}$ |
| 11 | Full classifier re-training from scratch | `experiments/geoad_code/training/t1_train_arms.py` | **Source Provided** | GPU training script included; not required to reproduce manuscript metrics |
| 12 | Raw PHMSA download to curated manifest | `data/development_manifest/DATA_CONSTRUCTION_SUMMARY.md` | **Provenance Documented** | Curated development and replay manifests are provided; complete raw upstream PHMSA dumps are not redistributed |

## Evaluation Protocol Scope

- **Development-confirmation split (`n = 6,417`, 1968–2023):** Used for six-class classifier evaluation (RQ2) and TreeSHAP feature attribution (RQ3). Because this split also informed earlier model development, it is reported as development-confirmation evidence rather than an unseen blind test.
- **Historical feedback replay (`n = 1,160`, 2024–2026):** Strictly temporally disjoint from the 1968–2023 development pool (0 shared `row_id`, 0 shared `accident_group`). Because exact historical public-release dates of every cause label are unavailable in the PHMSA archive, `--mode strict` refuses execution without `event_date` and `label_available_at`, and the study reports controlled retrospective annual-window replay.
- **Equal-$K$ alignment:** Authoritative Equal-$K$ ranking controls use $K=12$ (2025) and $K=2$ (2026) matching the full agent's accepted revision counts. Earlier pre-alignment tables are archived under `results/archived_rq4_tables_20261116/` for provenance only.
