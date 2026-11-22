# Software License, Baseline Provenance, and Data Rights

This document clarifies the licensing and provenance boundaries across original software code, adapted baseline implementations, curated PHMSA-derived data, and frozen model weights.

## 1. Project Software and MIT License

Original contributor-authored source code in this repository—including the GeoAgent dual-view classifier, inference utilities, P23/P24 feedback replay modules, verification tooling, plotting scripts, and the eight independently implemented baseline adaptations—is released under the [MIT License](../LICENSE) (`Copyright (c) 2026 GeoAgent contributors`). Individual manuscript authors are listed in [CITATION.cff](../CITATION.cff).

Third-party Python dependencies (CatBoost, PyTorch, scikit-learn, TensorFlow, LightGBM, SHAP, etc.) remain governed by their respective open-source licenses.

Serialized model checkpoints (`.cbm`, `.pt`, and `.joblib`) are included to support independent scientific reproducibility. They are provided for research reproduction and evaluation rather than certified operational deployment.

## 2. Eight Independently Implemented Baseline Adaptations

The eight baseline scripts under `experiments/baselines_8_adapted/code/baselines_fixed/*/train.py` cover:

| Method | Code Location |
|---|---|
| AUTDL | `experiments/baselines_8_adapted/code/baselines_fixed/AUTDL/` |
| CIC | `experiments/baselines_8_adapted/code/baselines_fixed/CIC/` |
| DeepPipe | `experiments/baselines_8_adapted/code/baselines_fixed/DeepPipe/` |
| DeepPipeNet | `experiments/baselines_8_adapted/code/baselines_fixed/DeepPipeNet/` |
| SART-1DCNN | `experiments/baselines_8_adapted/code/baselines_fixed/SART-1DCNN/` |
| MFF-CNN-LightGBM | `experiments/baselines_8_adapted/code/baselines_fixed/MFF-CNN-LightGBM/` |
| 1DCNN-DBO-LSTM | `experiments/baselines_8_adapted/code/baselines_fixed/1DCNN-DBO-LSTM/` |
| SMDPSO-SVM | `experiments/baselines_8_adapted/code/baselines_fixed/SMDPSO-SVM/` |

**Implementation provenance:** These eight baselines were independently implemented from the method descriptions in their original publications, with AI-assisted coding and author validation, and adapted to the unified 13-feature six-class tabular incident-record protocol. No upstream author repositories were copied, and these scripts are not represented as official upstream implementations. Please credit and cite the original method papers when using or referencing these baselines.

## 3. PHMSA Source Data and Curated Research Artifacts

- **Canonical public source:** [U.S. Pipeline and Hazardous Materials Safety Administration (PHMSA) Source Data](https://www.phmsa.dot.gov/data-and-statistics/pipeline/source-data).
- The project does not claim copyright ownership over original U.S. government PHMSA incident records and does not relicense raw PHMSA source files under MIT.
- The curated research datasets included for verification are:
  - `models_and_configs/geoad_materials/data/data_row_manifest_development_only.csv` — 42,769 harmonized development records (1968–2023), six-class labels, accident-group IDs, and split assignments.
  - `data/feedback_replay/P23_INPUT_GROUPKEY_REPAIRED.csv` — 1,160 temporally disjoint post-2023 historical replay records (2024–2026) with frozen classifier probabilities and labels.
  - `data/feedback_replay/EVALUATION_582.csv` and `results/per_row_predictions/` — evaluation labels and archived per-row predictions.
  - `data/demo_sample/dev_confirm_24rows_no_label.csv` — label-free 24-row inference sample.
- The 1,065 earlier held-out post-2023 master-table records are not redistributed. Field-level scans confirmed no personal-name, email, phone, operator-name, or street-address columns are present in the included CSV headers.

## 4. Attribution and Scope

TreeSHAP figures and plotting scripts are project-created research artifacts. This repository supports offline retrospective pipeline incident-cause diagnosis and controlled repair research, rather than continuous real-time sensor leak detection.
