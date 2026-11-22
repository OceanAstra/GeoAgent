#!/usr/bin/env python3
"""Label-free forward pass for the frozen GeoAgent classifier.

This is a thin, read-only wrapper around `run_inference.predict()`. It exists
because the archived inference entry (`run_inference.py`) selects rows with a
`split` column and reports accuracy against a `y` column, so it cannot be used
on a label-free input such as `data/demo_sample/dev_confirm_24rows_no_label.csv`.

No model weights, features, or decision rules are changed here. The wrapper
only re-shapes the input frame and writes probabilities.

Usage
-----
    python predictions only, no labels required:
        python predict_demo.py \
            --input data/demo_sample/dev_confirm_24rows_no_label.csv \
            --weights models_and_configs/geoad_materials/weights \
            --out demo_predictions.csv

The input needs the 13 model fields from `FEATURE_DEFINITIONS.md`. Columns
`split` and `y`/`label` are optional and are never required.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'experiments', 'geoad_code', 'inference'))
from run_inference import MAIN_13, SIX, predict  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True,
                    help='CSV without labels; needs the 13 model input fields')
    ap.add_argument('--weights', required=True, help='directory with 3 .cbm + 3 .pt')
    ap.add_argument('--out', default='', help='optional output CSV path')
    a = ap.parse_args()

    df = pd.read_csv(a.input, low_memory=False)
    missing = [c for c in MAIN_13 if c not in df.columns]
    if missing:
        raise SystemExit(f'MISSING_FEATURE_COLUMNS: {",".join(missing)}')
    if 'row_id' not in df.columns:
        raise SystemExit('MISSING_COLUMN: row_id')

    print(f'[demo] rows={len(df)} label_columns_present='
          f'{[c for c in ("label", "y") if c in df.columns] or "none"}')

    p = predict(df, a.weights)
    assert p.shape == (len(df), 6), p.shape
    assert np.allclose(p.sum(1), 1.0, atol=1e-5), 'probability rows must sum to 1'

    out = pd.DataFrame({'row_id': df['row_id']})
    for i, c in enumerate(SIX):
        out[f'prob_{c}'] = p[:, i]
    out['pred_label'] = [SIX[i] for i in p.argmax(1)]
    out['confidence'] = p.max(1)

    print(out.head(10).to_string(index=False))
    if a.out:
        out.to_csv(a.out, index=False)
        print(f'[demo] wrote {a.out}')


if __name__ == '__main__':
    main()
