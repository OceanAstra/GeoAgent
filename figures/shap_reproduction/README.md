# Paper SHAP figures: self-contained frozen reproduction inputs

Inputs shipped here:

- `inputs/style_reference/color_palette.json` — color-palette specification for Figure 3 (aggregated class-feature TreeSHAP heatmap) and Figure 4 (instance-level TreeSHAP panels).
- `inputs/02_VERIFIED_LATEST_SHAP_SUMMARIES/*.csv` — archive-derived per-class and global summary tables; exact copies of `results/rq3_interpretability/`.
- `figures/shap_reproduction/data/per_sample_shap_6417.npz` — numeric-only archived signed SHAP values, no pickle loading required (`allow_pickle=False`).
- `scripts/generate_shap_figures.py` — original plotting calculations, with portable default data paths.
- `scripts/verify_reproduction.py` — more expensive full CatBoost/TreeSHAP verification (requires compatible packages).

Regenerate **new** PDFs (does not overwrite paper's existing figure PDFs; `figures/shap_reproduction/generated/` is Git-ignored and created on demand):

```bash
python figures/shap_reproduction/scripts/generate_shap_figures.py --outdir figures/shap_reproduction/generated --dpi 300
```

To rerun full frozen CatBoost/TreeSHAP verification (also writes into the Git-ignored
`generated/` directory, which the script creates):

```bash
python figures/shap_reproduction/scripts/verify_reproduction.py --out figures/shap_reproduction/generated/full_verification.json
```

This result applies to the frozen classifier backbone, not an explanation of the feedback gate. The provided originals are `figures/SHAP_class_feature_importance_heatmap.pdf` and `figures/SHAP_instance_panels.pdf`; publication figure captions and exact typesetting remain the author's responsibility.
