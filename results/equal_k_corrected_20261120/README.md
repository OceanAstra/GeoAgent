# Equal-K correction, 2026-11-20

The 2025 accepted-action budget is **K=12**, the 2026 accepted-action budget is **K=2**, based on the complete agent's archived decisions. Both confidence and margin rules use these same counts. The 582-row `EQUAL_K_ROW_LEVEL.csv` is the independent source for the corrected predictions; `EQUAL_K_NEW_METRICS.csv` reports results. In `../per_row_predictions/PREDICTIONS_582.csv`, only the two Equal-K columns were adjusted to match this new archive; the KEEP and full GeoAgent output columns are preserved exactly.

The original reference package and audit record include author server paths. Those paths are not needed to understand the computations and were not copied into this GitHub candidate.
