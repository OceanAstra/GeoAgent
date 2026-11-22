# Source code map

This directory holds the classifier code that produced the manuscript's diagnostic
results. Only files that are actually present in this release are listed here.

## Files

| File | Role |
|---|---|
| `data_processing/s2_common.py` | protocol loading, `MAIN_13`, sealed-row drop, prediction saving |
| `catboost_backbone/g2_common.py` | constants, CatBoost backbone + grouped OOF, `build_views` (proto/rule), `GeoNet`, `FResidual` |
| `dual_view_residual/rq2_ablation_net.py` | `GeoNetViewMasked` — **the forward pass of the official model** |
| `dual_view_residual/s2_heads.py` | head forward/score helpers |
| `training/t1_train_arms.py` | trains A_single / A_ensemble / F_concat / P_no_proto |
| `training/t2_train_abl_b.py` | single-backbone ablation arm (ABL_B) |
| `evaluation/t1_score.py` | metrics, confusion matrix, decision changes |
| `statistics/t1_macro_perm.py` | accident-group sign-flip permutation, Holm (Tier-1 comparisons) |
| `statistics/t2_macro_perm.py` | the same procedure for the Tier-2 ablation comparisons |
| `configurations/t1_arm_configs_tier1.json` | arm definitions for `t1_train_arms.py` |
| `configurations/protocol_lock.json` | locked protocol record (split sizes, seeds, feature list) |
| `inference/run_inference.py` | **supported entry point**: portable frozen-weight inference |
| `feature_engineering/README.md` | where the rule view is built (`g2_common.build_views`) |

## Not shipped

The original research tree also contained helper modules that only ran on the
authors' build machine and are **not** part of this release: `t1_verify.py`,
`t1_compare_repro.py`, `t1_package.py`, `t2_score.py`, `t2_verify.py`,
`t2_package.py`, and a byte-identical `original_verbatim/` snapshot of the modules
listed above. Their functions are covered by `evaluation/t1_score.py`,
`reproduce/verify_release_metrics.py` and `reproduce/verify_manifest.py`; nothing in
the documented workflow refers to them.

## Entry points

| Purpose | Entry |
|---|---|
| **Inference (supported)** | `inference/run_inference.py` |
| Label-free demo forward pass | `predict_demo.py` (repository root) |
| Training all four arms | `training/t1_train_arms.py` (needs GPU) |
| Training the single-backbone ablation | `training/t2_train_abl_b.py` |
| Scoring a set of predictions | `evaluation/t1_score.py` |
| Permutation / Holm statistics | `statistics/t1_macro_perm.py`, `statistics/t2_macro_perm.py` |
| SHAP + figures | `figures/shap_reproduction/scripts/` |
| Archived-metric verification (no training) | `reproduce/verify_release_metrics.py`, `reproduce/verify_baselines.py` |

`training/` and `statistics/` scripts keep the original experiment layout and are
retained for provenance; the supported public entry points are
`inference/run_inference.py`, `predict_demo.py` and the `reproduce/` verifiers, all of
which take explicit paths.

## Where the official forward pass lives

`dual_view_residual/rq2_ablation_net.py`, class `GeoNetViewMasked`, method `forward`
with `drop='proto'`:

```python
retained = ['rule', 'neural']; anchor = 'rule'
ea = e['rule']
t  = torch.exp(self.log_temp) + 0.1
a  = torch.exp(-torch.norm(ea - e['neural'], dim=1, keepdim=True) / t)
z  = (ea + a * e['neural']) / (1 + den_const + a + 1e-8)
delta = self.W_delta(z)
```

`inference/run_inference.py` re-implements exactly this and is verified to reproduce the
archived probabilities to `1.8e-07` (argmax agreement 100.00 % on all 6,417 rows).
Checkpoints are loaded with `weights_only=True`.
