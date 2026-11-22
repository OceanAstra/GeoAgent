#!/usr/bin/env python3
"""Verification for the GeoAgent six-class TreeSHAP figures.

Checks, in order:
  1. handoff CSVs (6 classes x 13 features x 3 seeds x n=6417, no NaN)
  2. frozen CatBoost weights reproduce archived per-row probabilities exactly
  3. TreeSHAP recomputed from the frozen model reproduces the archived aggregate CSV
  4. SHAP additivity  sum_j phi_ij + base = raw margin
Writes verify_report.json next to the script.  No model is trained.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

MAIN_13 = ['PIPE_DIAMETER','PIPE_WALL_THICKNESS','PIPE_SMYS','PIPE_SPECIFICATION',
           'PIPE_SEAM_TYPE','PIPE_COATING_TYPE','PIPE_TYPE','ACCIDENT_PSIG','MOP_PSIG',
           'INTERNAL_INSPECTION_IND','PIPE_AGE_AT_ACCIDENT','MANUFACTURE_AGE_AT_ACCIDENT',
           'PRESSURE_RATIO']
CATEGORICAL = ['PIPE_SPECIFICATION','PIPE_SEAM_TYPE','PIPE_COATING_TYPE','PIPE_TYPE']
SIX = ['CORROSION','EQUIPMENT_FAILURE','EXTERNAL_DAMAGE','MATERIAL_DEFECT','OPERATIONAL','OTHER']
SEEDS = (42, 123, 456)


def catboost_frame(df):
    """Identical to experiments/geoad_code/catboost_backbone/g2_common.feature_frame.

    Categorical -> NaN filled with the fixed token NOT_RECORDED, cast to str.
    Numeric     -> to_numeric(errors='coerce'); NaN is passed to CatBoost as-is
                   (the frozen models were fit on NaN, not on imputed values).
    """
    X = df[MAIN_13].copy()
    for c in CATEGORICAL:
        X[c] = X[c].fillna('NOT_RECORDED').astype(str)
    for c in MAIN_13:
        if c not in CATEGORICAL:
            X[c] = pd.to_numeric(X[c], errors='coerce')
    return X


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve()
    ap.add_argument('--handoff', default=str(here.parents[1] / 'inputs'))
    ap.add_argument('--repro', default=str(here.parents[3]))
    ap.add_argument('--out', default=str(here.parents[1] / 'verification' / 'verify_report_regenerated.json'))
    ap.add_argument('--save-shap', default='')
    a = ap.parse_args()

    rep = {}
    from catboost import CatBoostClassifier, Pool
    import shap

    # ---------------- 1. handoff CSVs ----------------
    pcf = pd.read_csv(f'{a.handoff}/02_VERIFIED_LATEST_SHAP_SUMMARIES/shap_per_class_feature_importance.csv')
    glo = pd.read_csv(f'{a.handoff}/02_VERIFIED_LATEST_SHAP_SUMMARIES/shap_global_importance.csv')
    assert len(pcf) == 234, len(pcf)
    assert pcf.cls.nunique() == 6 and pcf.feature.nunique() == 13
    assert set(pcf.seed.unique()) == set(SEEDS)
    assert set(pcf.n.unique()) == {6417}
    assert int(pcf.isna().any(axis=1).sum()) == 0
    # the shipped global CSV must equal the class x record mean of the per-class CSV
    m = pcf.groupby(['feature', 'seed']).mean_abs_shap.mean().unstack()
    g = glo.set_index('feature')
    gdiff = max(float((m[s] - g[f'mean_abs_shap_seed{s}']).abs().max()) for s in SEEDS)
    assert gdiff < 1e-12, gdiff
    rep['csv_check'] = dict(rows=len(pcf), classes=6, features=13, seeds=list(SEEDS), n=6417,
                            global_vs_perclass_maxdiff=gdiff)

    # ---------------- 2. manifest + frozen weights ----------------
    man = pd.read_csv(f'{a.repro}/models_and_configs/geoad_materials/data/data_row_manifest_development_only.csv',
                      low_memory=False)
    assert len(man) == 42769
    assert man.split.value_counts().to_dict() == {'fit': 29941, 'calib': 6411, 'dev_confirm': 6417}
    dc = man[man.split == 'dev_confirm'].reset_index(drop=True)
    X = catboost_frame(dc)
    cat_idx = [MAIN_13.index(c) for c in CATEGORICAL]
    pool = Pool(X, cat_features=cat_idx)

    W = f'{a.repro}/models_and_configs/geoad_materials/weights'
    models = {}
    for s in SEEDS:
        mm = CatBoostClassifier()
        mm.load_model(os.path.join(W, f'cb_seed{s}.cbm'))
        models[s] = mm

    ref_single = pd.read_csv(f'{a.repro}/results/per_row_predictions/geoad/A_single.csv').set_index('row_id')
    ref_ens    = pd.read_csv(f'{a.repro}/results/per_row_predictions/geoad/A_ensemble.csv').set_index('row_id')
    order = dc.row_id.to_numpy()
    P = {s: models[s].predict_proba(pool) for s in SEEDS}
    Ens = np.mean([P[s] for s in SEEDS], axis=0)
    d_single = float(np.abs(P[42] - ref_single.loc[order, [f'prob_{c}' for c in SIX]].to_numpy()).max())
    d_ens    = float(np.abs(Ens   - ref_ens.loc[order,    [f'prob_{c}' for c in SIX]].to_numpy()).max())
    assert d_single < 1e-12 and d_ens < 1e-12, (d_single, d_ens)
    rep['prob_reproduction'] = dict(seed42_vs_A_single_max_abs_dp=d_single,
                                    ensemble_vs_A_ensemble_max_abs_dp=d_ens,
                                    n=len(dc))
    ref_probs = Ens   # archived ensemble used for the f(x) overlays

    # ---------------- 3. TreeSHAP vs archived aggregate ----------------
    arch = pcf.pivot_table(index=['seed', 'cls'], columns='feature', values='mean_abs_shap')
    shap_store, bases = {}, {}
    maxdiff = 0.0
    for s in SEEDS:
        ex = shap.TreeExplainer(models[s])
        sv = np.asarray(ex.shap_values(pool), dtype=np.float64)   # (n, 13, 6)
        assert sv.shape == (len(dc), 13, 6), sv.shape
        shap_store[s] = sv
        # additivity: sum_j phi + base == raw margin
        raw = models[s].predict(pool, prediction_type='RawFormulaVal')
        ev = np.ravel(np.asarray(ex.expected_value, dtype=np.float64))
        bases[s] = ev
        resid = np.abs(sv.sum(axis=1) + ev - raw).max()
        rep.setdefault('additivity', {})[str(s)] = dict(
            max_abs_residual=float(resid), base_values=ev.tolist())
        assert resid < 1e-6, (s, resid)   # float32 tournament margins
        absmean = np.abs(sv).mean(axis=0).T          # (6, 13)
        got = pd.DataFrame(absmean, index=SIX, columns=MAIN_13)
        want = arch.loc[s].loc[SIX, MAIN_13]
        maxdiff = max(maxdiff, float((got - want).abs().max().max()))
    rep['shap_reproduction'] = dict(max_abs_diff_vs_archived_csv=maxdiff,
                                    shape=[len(dc), 13, 6])

    # additivity of the seed-averaged quantity that figure B actually draws
    shap_mean = np.mean([shap_store[s] for s in SEEDS], axis=0)
    base_mean = np.mean([bases[s] for s in SEEDS], axis=0)
    raw_mean = np.mean([np.asarray(models[s].predict(pool, prediction_type='RawFormulaVal'),
                                    dtype=np.float64) for s in SEEDS], axis=0)
    ens_resid = float(np.abs(shap_mean.sum(axis=1) + base_mean - raw_mean).max())
    rep['additivity_ensemble_mean'] = dict(max_abs_residual=ens_resid)
    assert ens_resid < 1e-9, ens_resid

    if a.save_shap:
        # Ensemble-mean signed SHAP. Linear in the three models, so it is still
        # additive to the mean margin. Stored in float64 so a cached run is
        # bit-identical to an uncached run; per-seed arrays stay float32 to keep
        # the archive small (they are only used for the seed-level audit).
        np.savez_compressed(a.save_shap,
                            shap_mean=np.mean([shap_store[s] for s in SEEDS], axis=0),
                            shap_seed42=shap_store[42].astype(np.float32),
                            shap_seed123=shap_store[123].astype(np.float32),
                            shap_seed456=shap_store[456].astype(np.float32),
                            base_mean=np.mean([bases[s] for s in SEEDS], axis=0),
                            y=dc.y.to_numpy().astype(np.int16),
                            prob_mean=ref_probs.astype(np.float32),
                            row_id=order.astype(str),
                            features=np.array(MAIN_13), classes=np.array(SIX))
        rep['shap_archive'] = a.save_shap

    print(json.dumps(rep, indent=2, ensure_ascii=False))
    out_path = __import__('pathlib').Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print('\n[ok] all assertions passed')


if __name__ == '__main__':
    main()
