# GeoAgent

**Code, frozen weights, and reproducibility archive for the manuscript:**

> **Towards Interpretable Agent-Based Oil Pipeline Anomaly Detection**

GeoAgent is a unified, deterministic, interpretable algorithmic agent for **six-class pipeline accident-cause classification and controlled repair** on structured PHMSA incident records. Its classification module combines a three-model CatBoost ensemble with geometry-aware dual-view residual correction (engineering-rule indicators anchored against learned class probabilities). On top of frozen class probabilities, the agent reviews each prediction, generates candidate alternatives, scores repairs using historical feedback, and applies a conservative **KEEP-or-REPAIR** gate. TreeSHAP explains the CatBoost classification backbone, while structured decision logs trace every candidate evaluation and final action.

---

## Key Results

| Evaluation | Cohort | Key Metrics |
|---|---|---|
| **Dual-View Classifier** | 6,417 PHMSA development-confirmation records (1968–2023) | Accuracy **71.59%**, Macro-F1 **58.63%**, Weighted-F1 **69.41%**; leads all 8 adapted external baselines across all three overall metrics and 5 of 6 cause classes |
| **Full GeoAgent Workflow** | 2025 historical replay (542 records) | Accuracy **61.81% → 64.02%** (+2.21 pp, McNemar $p = 4.88 \times 10^{-4}$), Macro-F1 **35.64% → 39.27%** (+3.64 pp); **12/12 revisions correct (100.0% repair precision), 0 harmful revisions**, changing only **2.21%** of records |
| **Follow-Up Replay** | 2026 historical replay (40 records) | Accuracy **55.00% → 60.00%**, Macro-F1 **28.71% → 35.66%**; **2/2 revisions correct, 0 harms** (pooled 2025–2026: **14/14 correct, 0 harms**) |

> **Protocol note:** The 6,417-record classification evaluation (1968–2023) and the 1,160-record historical feedback replay (2024–2026; 578 fit / 542 main / 40 follow-up) are temporally disjoint cohorts with zero shared record IDs or accident groups; their metrics are reported under their respective protocols.

---

## Quick Start (Linux x86_64, Python 3.12)

### 1. Create an isolated environment

```bash
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -r reproduce/requirements-pinned.txt
```

On CPU-only machines, install the CPU build of PyTorch 2.5.1 first to avoid downloading CUDA libraries:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1+cpu"
```

### 2. Verify archive integrity and archived metrics (standard library only, ~1 s)

```bash
python reproduce/verify_manifest.py
python reproduce/verify_release_metrics.py
python reproduce/verify_baselines.py
python reproduce/check_doc_paths.py
```

### 3. Run label-free inference demo (24 records, ~5 s)

```bash
python predict_demo.py \
  --input data/demo_sample/dev_confirm_24rows_no_label.csv \
  --weights models_and_configs/geoad_materials/weights \
  --out /tmp/demo_predictions.csv
```

### 4. Reproduce full 6,417-record frozen classifier inference (~3 s on CPU)

```bash
python experiments/geoad_code/inference/run_inference.py \
  --manifest models_and_configs/geoad_materials/data/data_row_manifest_development_only.csv \
  --weights models_and_configs/geoad_materials/weights \
  --split dev_confirm \
  --compare results/per_row_predictions/geoad/P_no_proto.csv \
  --out /tmp/preds_6417.csv
# Expected: argmax accuracy = 71.5911%, max|dp| = 1.788e-07, argmax agreement = 100.0000%
```

### 5. Reproduce historical feedback replay & row-level audit (582 records, ~10 s)

```bash
python agent/feedback/p23_feedback_replay.py \
  --input data/feedback_replay/P23_INPUT_GROUPKEY_REPAIRED.csv \
  --models-dir models_and_configs/p22_feedback \
  --out /tmp/geoagent_p23 \
  --mode retrospective \
  --expected-input-sha256 64b9937b7e949e6656aa2532243846c214f0a35c3a32bb9d6c8ef59cce0ff27b

python agent/feedback/export_p23_rowlevel.py \
  --input data/feedback_replay/P23_INPUT_GROUPKEY_REPAIRED.csv \
  --models-dir models_and_configs/p22_feedback \
  --out /tmp/p23_rowlevel \
  --compare results/per_row_predictions/PREDICTIONS_582.csv
# Expected: 582/582 exact row-level match across all 8 P23 replay policy columns

python -m unittest discover -s agent/feedback -p "test_p23.py" -v
```

*(The two additional `P24_*` policy columns in `PREDICTIONS_582.csv` are reproduced 582/582 by `agent/feedback/p24_conditional_repair.py` and `agent/feedback/export_p24_rowlevel.py`; see [docs/P24_ARMS.md](docs/P24_ARMS.md).)*

---

## Repository Map

| Folder | Description |
|---|---|
| `experiments/geoad_code/` | Dual-view classifier training, residual architecture, evaluation, inference, and statistical analysis scripts (internal code prefix `geoad` preserved for reproducibility). |
| `agent/feedback/` | Paper feedback replay (`p23_feedback_replay.py`, `p24_conditional_repair.py`, frozen P22 reference helpers, row-level exporters, and unit tests). |
| `models_and_configs/` | Feature definitions, label crosswalk, protocol lock, data dictionary, frozen CatBoost (`.cbm`) and residual (`.pt`) weights, and frozen P22 `.joblib` heads. |
| `data/` | Curated PHMSA-derived development provenance, retrospective feedback replay inputs, evaluation labels, and label-free 24-row demo sample. |
| `experiments/baselines_8_adapted/` | Eight independently implemented adaptations of published pipeline methods under the unified 13-feature six-class protocol (see [docs/BASELINES_AND_PROTOCOL.md](docs/BASELINES_AND_PROTOCOL.md)). |
| `results/` | Archived per-row predictions, baseline comparisons, ablation tables, TreeSHAP summaries, RQ4 policy tables, and figure source data. |
| `figures/shap_reproduction/` | Cached per-row TreeSHAP tensor (`.npz`, numeric-only) and plotting/verification scripts reproducing Figures 3 and 4. |
| `reproduce/` | Pinned environment files and lightweight integrity/metric verification scripts. |
| `provenance/` | Archived pre-update Equal-K prediction file preserved for transparent auditability. |

---

## Reproducibility Summary

| Verification Item | Status | Details |
|---|---|---|
| Archive SHA-256 integrity & archived metrics (6,417 classifier, 582 replay, 8 baselines) | **Verified** | `reproduce/verify_manifest.py`, `verify_release_metrics.py`, `verify_baselines.py` |
| Frozen classifier forward pass (`.cbm` + `.pt`) | **Verified** | 6,417/6,417 rows, 100.00% argmax agreement, max $\|\Delta p\| = 1.79 \times 10^{-7}$ |
| Historical feedback replay (`P23` & `P24`) | **Verified** | 582/582 exact row-level match across all 10 policy columns in `PREDICTIONS_582.csv` |
| TreeSHAP figures & additivity check | **Verified** | Max summary diff $1.39 \times 10^{-9}$, additivity residual $1.33 \times 10^{-14}$ |
| Baseline fresh re-training | **Documented** | 4/8 exact row-level reproduction; 2/8 stochastic variance; 2/8 resource-bounded (see [docs/BASELINES_AND_PROTOCOL.md](docs/BASELINES_AND_PROTOCOL.md)) |

Full environment details and compatibility notes (including the `scikit-learn==1.6.1` pin for `.joblib` heads and `weights_only=True` for `.pt` checkpoints) are documented in [docs/DEPENDENCY_NOTES.md](docs/DEPENDENCY_NOTES.md) and [docs/REPRODUCIBILITY_STATUS.md](docs/REPRODUCIBILITY_STATUS.md).

---

## Equal-K Alignment Note

The authoritative Equal-$K$ ranking controls in `results/per_row_predictions/PREDICTIONS_582.csv` and `results/rq4_feedback_ablation/TABLE_RQ4_2025_PAPER_CORRECTED.csv` use **$K=12$ (2025)** and **$K=2$ (2026)**, matching the full agent's accepted revision count. Earlier pre-alignment files are preserved under `provenance/` and `results/archived_rq4_tables_20261116/` for full auditability.

---

## License, Data Provenance, and Citation

- **Software License:** Original GeoAgent source code and contributor-authored scripts are released under the [MIT License](LICENSE) (Copyright © 2026 GeoAgent contributors).
- **Data & Baseline Provenance:** Source incident records originate from public [PHMSA Source Data](https://www.phmsa.dot.gov/data-and-statistics/pipeline/source-data). The eight adapted baseline scripts are AI-assisted independent implementations from the published method descriptions, not official upstream repositories. Complete provenance and scope notes are provided in [docs/DATA_AND_BASELINE_RIGHTS.md](docs/DATA_AND_BASELINE_RIGHTS.md).
- **Citation:** Please cite this repository using [CITATION.cff](CITATION.cff) and cite the original method papers when using any adapted baseline.
