"""
run_inference.py -- Portable inference entry point for GeoAgent (P_no_proto).

Given the development manifest (`models_and_configs/geoad_materials/data/data_row_manifest_development_only.csv`)
and frozen weights under `models_and_configs/geoad_materials/weights/`, runs the exact forward pass:

    P_CB   = mean_over_seeds( catboost_seed.predict_proba(X13) )
    r      = W_rule(rule4)
    h      = W_neural(P_CB)
    t      = exp(log_temp) + 0.1
    a      = exp(-||r - h|| / t)
    z      = (r + a*h) / (1 + a + 1e-8)
    delta  = W_delta(z)
    logits = log(clamp(P_CB, 1e-8)) + delta
    p      = mean_over_seeds( softmax(logits) )

Matches `GeoNetViewMasked.forward(drop='proto')` in `experiments/geoad_code/dual_view_residual/rq2_ablation_net.py`.
No model training or parameter modification is performed.

Usage
-----
    python -I experiments/geoad_code/inference/run_inference.py --split dev_confirm --limit 0 \
        --manifest models_and_configs/geoad_materials/data/data_row_manifest_development_only.csv \
        --weights models_and_configs/geoad_materials/weights --out preds_replay.csv

    --limit 200  Run on the first 200 rows (quick smoke test)
    --compare results/per_row_predictions/geoad/P_no_proto.csv   Row-level comparison vs archived predictions
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SIX = ['CORROSION', 'EQUIPMENT_FAILURE', 'EXTERNAL_DAMAGE',
       'MATERIAL_DEFECT', 'OPERATIONAL', 'OTHER']
MAIN_13 = ['PIPE_DIAMETER', 'PIPE_WALL_THICKNESS', 'PIPE_SMYS',
           'PIPE_SPECIFICATION', 'PIPE_SEAM_TYPE', 'PIPE_COATING_TYPE', 'PIPE_TYPE',
           'ACCIDENT_PSIG', 'MOP_PSIG', 'INTERNAL_INSPECTION_IND',
           'PIPE_AGE_AT_ACCIDENT', 'MANUFACTURE_AGE_AT_ACCIDENT', 'PRESSURE_RATIO']
CATEGORICAL = ['PIPE_SPECIFICATION', 'PIPE_SEAM_TYPE', 'PIPE_COATING_TYPE', 'PIPE_TYPE']
SEEDS = (42, 123, 456)


class DualViewHead(nn.Module):
    """Matches parameter shapes and layer ordering of rq2_ablation_net.GeoNetViewMasked."""

    def __init__(self, proto_dim=9, rule_dim=4, emb_dim=32, n_classes=6):
        super().__init__()
        self.W_proto = nn.Linear(proto_dim, emb_dim)     # Unused (proto view disabled)
        self.W_rule = nn.Linear(rule_dim, emb_dim)
        self.W_neural = nn.Linear(n_classes, emb_dim)
        self.log_temp = nn.Parameter(torch.tensor([1.0]))
        self.W_delta = nn.Linear(emb_dim, n_classes)

    def forward(self, rule, cb):
        r = self.W_rule(rule)
        h = self.W_neural(cb)
        t = torch.exp(self.log_temp) + 0.1
        a = torch.exp(-torch.norm(r - h, dim=1, keepdim=True) / t)
        z = (r + a * h) / (1.0 + a + 1e-8)
        return self.W_delta(z)


def rule_view(df):
    coat = df['PIPE_COATING_TYPE'].astype(str).str.upper().isin(['NONE', 'UNKNOWN', 'N/A'])
    age = pd.to_numeric(df['PIPE_AGE_AT_ACCIDENT'], errors='coerce') <= 10
    mfg = pd.to_numeric(df['MANUFACTURE_AGE_AT_ACCIDENT'], errors='coerce') <= 5
    pr = pd.to_numeric(df['PRESSURE_RATIO'], errors='coerce') > 1.2
    return np.stack([coat.fillna(False).to_numpy(float),
                     age.fillna(False).to_numpy(float),
                     mfg.fillna(False).to_numpy(float),
                     pr.fillna(False).to_numpy(float)], axis=1).astype(np.float32)


def cb_frame(df):
    X = df[MAIN_13].copy()
    for c in CATEGORICAL:
        X[c] = X[c].fillna('NOT_RECORDED').astype(str)
    for c in MAIN_13:
        if c not in CATEGORICAL:
            X[c] = pd.to_numeric(X[c], errors='coerce')
    return X


def predict(df, weights_dir):
    from catboost import CatBoostClassifier, Pool
    Xf = cb_frame(df)
    cat_idx = [MAIN_13.index(c) for c in CATEGORICAL]

    probs = []
    for s in SEEDS:
        m = CatBoostClassifier()
        m.load_model(os.path.join(weights_dir, f'cb_seed{s}.cbm'))
        probs.append(m.predict_proba(Pool(Xf, cat_features=cat_idx)))
    P_CB = np.mean(probs, axis=0)

    rule = torch.as_tensor(rule_view(df), dtype=torch.float32)
    cb_t = torch.as_tensor(P_CB, dtype=torch.float32)
    outs = []
    for s in SEEDS:
        net = DualViewHead()
        net.load_state_dict(torch.load(os.path.join(weights_dir, f'P_no_proto_seed{s}.pt'),
                                       map_location='cpu', weights_only=True))
        net.eval()
        with torch.no_grad():
            d = net(rule, cb_t)
            lg = torch.log(cb_t.clamp_min(1e-8)) + d
            outs.append(torch.softmax(lg, 1).numpy())
    return np.mean(outs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--split', default='dev_confirm')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='')
    ap.add_argument('--compare', default='')
    a = ap.parse_args()

    man = pd.read_csv(a.manifest, low_memory=False)
    df = man[man.split == a.split].reset_index(drop=True)
    if a.limit:
        df = df.iloc[:a.limit].reset_index(drop=True)
    print(f'[infer] split={a.split} rows={len(df)}')

    p = predict(df, a.weights)
    assert p.shape == (len(df), 6), p.shape
    assert np.allclose(p.sum(1), 1.0, atol=1e-5), 'probability rows must sum to 1'
    pred = p.argmax(1)
    y = df['y'].to_numpy()
    acc = float((pred == y).mean())
    print(f'[infer] argmax accuracy = {acc*100:.4f}%')

    out = pd.DataFrame({'row_id': df.row_id})
    for i, c in enumerate(SIX):
        out[f'prob_{c}'] = p[:, i]
    out['pred'] = pred
    if a.out:
        out.to_csv(a.out, index=False)
        print(f'[infer] wrote {a.out}')

    if a.compare:
        ref = pd.read_csv(a.compare)
        ref = ref[ref.row_id.isin(out.row_id)].set_index('row_id')
        o = out.set_index('row_id')
        d = np.abs(o[[f'prob_{c}' for c in SIX]].to_numpy()
                   - ref.loc[o.index, [f'prob_{c}' for c in SIX]].to_numpy())
        print(f'[compare] max|dp| = {d.max():.3e}  mean|dp| = {d.mean():.3e}')
        print(f'[compare] argmax agreement = '
              f'{(o["pred"].to_numpy() == ref.loc[o.index, "pred"].to_numpy()).mean()*100:.4f}%')


if __name__ == '__main__':
    main()
