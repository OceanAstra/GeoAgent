# Figures Source Data

All CSV and JSON files in this directory contain verified, recomputed values used for manuscript tables and figures.

| File | Scope | Description |
|---|---|---|
| `fig_diagnosis_external_6417.csv` | Classifier external benchmark ($n=6,417$) | GeoAgent dual-view classifier + 8 adapted baselines under the shared 6-class protocol. |
| `fig_risk_benefit_2025.csv` | 2025 repair risk–benefit ($n=542$) | Accepted revisions (`changed`), `net_correct`, `harms`, and `harm_per_100_rows`. |
| `fig_per_class_f1_2025.csv` | 2025 per-class F1 | 5 policy arms × 6 classes on the 2025 historical replay. |
| `fig_per_class_recall_delta_2025.csv` | 2025 per-class recall shifts | Per-class recall delta relative to KEEP (Full GeoAgent: 0.00 pp drop across all classes; `NO_MEMORY`: -3.03 pp on `MATERIAL_DEFECT`). |
| `fig_shap_global_importance_6417.csv` | TreeSHAP global importance | Cross-seed mean absolute TreeSHAP values for the CatBoost classification backbone. |
| `fig_case_decision_trace.csv` | Per-record decision traces | Complete probability and KEEP/REPAIR traces for all 14 accepted repairs (2025–2026). |
| `fig_gate_separation_2025.csv` | 2025 gate separation | The 20 records initially predicted as `OPERATIONAL` sorted by frozen `EXTERNAL_DAMAGE` probability (12 true `EXTERNAL_DAMAGE` in 0.1786–0.2754; 8 other records in 0.0095–0.0365). |
| `fig_gate_behaviour_by_window.json` | Window gate summary | Per-window counts of `OPERATIONAL` base predictions, true `EXTERNAL_DAMAGE` cases, and full-agent accepted revisions. |
| `fig_rq2_diagnosis_ablation_6417.csv` | RQ2 architecture ablation | `P_no_proto`, `A_ensemble`, `F_concat`, `ABL_B`, and `A_single` on 6,417 records. |
| `fig_rq2_per_class_f1_6417.csv` | RQ2 per-class F1 | 4 classifier arms × 6 classes. |
| `fig_rq2_paired_differences_6417.csv` | RQ2 paired tests | Accident-group bootstrap CIs and paired McNemar / permutation tests. |
| `fig_rq2_diagnostic_swap_windows.csv` | Agent-level backbone swap | Comparing dual-view vs CatBoost ensemble probabilities as frozen input to the feedback replay. |
| `fig_rq4_change_set_overlap_582.csv` | Revision set nesting | Set overlap between full-agent revisions (14) and larger-intervention policies across 582 records. |
| `fig_rq4_zero_parameter_audit_582.json` | Selection audit (JSON) | Post-hoc top-$K$ probability ranking comparison and unconstrained 20-record rule (4 harms). |
| `fig_rq4_zero_parameter_and_swap_582.csv` | Selection audit (CSV) | Per-window metrics for the selection audit and backbone-swap checks. |
| `fig_ablation_pooled_582.csv` | Pooled 582-record ablation | Policy revisions, net gains, and efficiency across 2025 + 2026. |
| `fig_feedback_windows.csv` | Feedback window sizes | Window counts and roles (2024: 578, 2025: 542, 2026: 40). |

## Reporting Guidelines

1. Always report harmful revisions alongside net gains when plotting repair policy trade-offs.
2. Keep the 6,417-record classifier evaluation (1968–2023) and the 542/582-record historical replay (2025–2026) under their separate cohort axes.
3. TreeSHAP explains the CatBoost classification backbone, while decision traces document the repair policy.
