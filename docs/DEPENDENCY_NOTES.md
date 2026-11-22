# Environment and Dependency Notes

## Requirement Files

| File | Scope |
|---|---|
| `reproduce/requirements-pinned.txt` | Core verification scripts, frozen classifier inference, demo script, and P23/P24 historical replay |
| `reproduce/requirements-baselines.txt` | Additional backends required only for re-training external baselines (TensorFlow/Keras, LightGBM, imbalanced-learn) |

## Verified Reference Stack (Linux x86_64)

| Package | Pinned Version |
|---|---|
| Python | 3.12.3 |
| numpy | 2.0.2 |
| pandas | 2.2.3 |
| scikit-learn | **1.6.1** (required for P22 `.joblib` deserialization) |
| scipy | 1.13.1 |
| joblib | 1.4.2 |
| catboost | 1.2.10 |
| torch | 2.5.1+cu124 (GPU) or 2.5.1+cpu (CPU-only) |
| shap | 0.51.0 |
| matplotlib | 3.9.0 |
| tensorflow / keras *(baselines only)* | 2.21.0 / 3.14.1 |
| lightgbm *(baselines only)* | 4.6.0 |
| imbalanced-learn *(baselines only)* | 0.14.1 |

## Fresh-Environment Verification

All core non-training entry points were verified in a newly created isolated Python 3.12.3 virtual environment (`python -I`, CPU-only `torch==2.5.1+cpu`):

- `reproduce/verify_manifest.py`, `verify_release_metrics.py`, `verify_baselines.py`, `check_doc_paths.py`: **PASS**
- `python -m unittest discover -s agent/feedback -p test_p23.py`: **9/9 PASS**
- `run_inference.py` on 6,417 `dev_confirm` rows: **71.5911% Acc**, **100.0000%** argmax agreement, $\max|\Delta p| = 1.788 \times 10^{-7}$
- `predict_demo.py` on 24 label-free rows: **PASS** ($\max|\Delta p| = 0.0$)
- `p23_feedback_replay.py` + `export_p23_rowlevel.py` + `export_p24_rowlevel.py`: **582/582** exact row-level agreement across all 10 policy columns

## Compatibility Notes

1. **`scikit-learn==1.6.1` is required for P22 `.joblib` models.** Newer versions (e.g., 1.8.x) change internal `ColumnTransformer` serialization and cannot load the frozen 1.6.1 heads.
2. **PyTorch checkpoints load safely with `weights_only=True`.** On CPU-only hosts, installing `torch==2.5.1+cpu` from the PyTorch CPU wheel index avoids a ~2 GB CUDA download with identical frozen inference outputs.
3. **Platform coverage:** Validated on Linux x86_64 (glibc 2.35). Standard-library checks (`verify_manifest.py`, `verify_release_metrics.py`, `verify_baselines.py`, `check_doc_paths.py`) are platform-independent.
4. **TensorFlow and PyTorch isolation:** Classifier scripts and TensorFlow-based baseline scripts run as separate processes and are never imported together.
