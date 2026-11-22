# Eight Adapted External Baselines: Protocol and Reproduction Status

The eight baseline implementations under `experiments/baselines_8_adapted/code/baselines_fixed/*/train.py` were independently implemented with AI assistance and author validation from the published method descriptions, and adapted to the shared 13-feature six-class incident-record protocol. They are not official upstream implementations; please cite the original publications for each method.

## Shared Data Protocol

| Development Split | Baseline Role | Records |
|---|---|---:|
| `fit` | `train` | 29,941 |
| `calib` | `val` | 6,411 |
| `dev_confirm` | `test` | 6,417 |

- **13-D input:** 9 numeric features + 4 train-fitted ordinal categorical features (unseen categories mapped to `-1`).
- **9-D input:** 9 numeric features only.
- All imputers, scalers, and categorical vocabularies are fitted strictly on the training split.
- Archived per-row predictions are stored in `results/per_row_predictions/baselines_8/`.

## Metric Verification (No Training Required)

```bash
python reproduce/verify_baselines.py
```

Re-scores all eight archived prediction CSV files against the 6,417 `dev_confirm` ground-truth labels (8/8 exact match with manuscript Tables 5–6).

## Re-Running Baseline Training

```bash
python -m pip install -r reproduce/requirements-baselines.txt
python experiments/baselines_8_adapted/code/rq1f_build_baseline_data.py
python experiments/baselines_8_adapted/code/baselines_fixed/CIC/train.py
```

`rq1f_build_baseline_data.py` builds `experiments/baselines_8_adapted/data/baseline_data_13d/` and `baseline_data_9d/` from the included development manifest. Each `train.py` resolves that default path automatically (overridable via `RQ1_DATA_DIR`).

## Observed Fresh Re-Training Status

Verified on Linux x86_64 with the pinned baseline stack (Python 3.12.3, PyTorch 2.5.1+cu124, TensorFlow 2.21.0, LightGBM 4.6.0):

| Baseline | Fresh Retrain | Outcome |
|---|---|---|
| **CIC** | Complete | Exact metric and **6,417/6,417** row-level match (Acc 55.9140%) |
| **AUTDL** | Complete | Exact metric and **6,417/6,417** row-level match (Acc 59.5917%) |
| **SART-1DCNN** | Complete | Exact metric and **6,417/6,417** row-level match (Acc 69.0821%) |
| **SMDPSO-SVM** | Complete | Exact metric and **6,417/6,417** row-level match (Acc 63.8772%) under its documented reduced fit cap |
| **DeepPipe** | Complete | Stochastic variance on CPU/cuDNN path (69.16% vs archived 69.49%) |
| **DeepPipeNet** | Complete | Stochastic variance (64.30% vs archived 65.44%); uses training-set oversampling |
| **1DCNN-DBO-LSTM** | Resource-bounded | 20-particle × 20-iteration DBO search requires extended compute budget |
| **MFF-CNN-LightGBM** | Resource-bounded | Sensitive to local TensorFlow/cuDNN runtime compatibility on GPU |

## Training Budget Notes

- **DeepPipeNet** applies oversampling within its training fold following its adapted training procedure.
- **SMDPSO-SVM** uses a reduced training cap of 5,000 rows (`RQ1_SMDPSO_TRAIN_CAP=5000`) and a reduced PSO fitness subsample (`400 fit / 150 val`, `RQ1_SMDPSO_PROFILE=reduced`) to bound SVM optimization cost.
