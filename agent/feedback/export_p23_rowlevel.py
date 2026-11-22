#!/usr/bin/env python3
"""Export per-row P23 decisions without changing any decision.

`p23_feedback_replay.py` writes only aggregate reports (`P23_RESULT.json`,
`P23_PROVENANCE.json`, `P23_DECISION.md`). Auditing a replay against the
archived `results/per_row_predictions/PREDICTIONS_582.csv` requires row-level
output, so this tool re-runs the *same* functions in the *same* order and
exports every policy's per-row decision plus an optional paired comparison.

Nothing is re-fitted differently and no threshold is touched: the module
imports `evaluate_one()` verbatim from the replay script, so the exported
predictions are the predictions the replay made.

Usage
-----
    python agent/feedback/export_p23_rowlevel.py \
        --input data/feedback_replay/P23_INPUT_GROUPKEY_REPAIRED.csv \
        --models-dir models_and_configs/p22_feedback \
        --out /tmp/p23_rowlevel \
        [--compare results/per_row_predictions/PREDICTIONS_582.csv]

Outputs
-------
    P23_ROWLEVEL.csv   one row per evaluated row: row_id, window, label,
                       KEEP, and every policy column
    P23_COMPARE.json   (only with --compare) paired equality per policy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p23_feedback_replay as P  # noqa: E402


def export(inp: Path, models_dir: Path) -> pd.DataFrame:
    models = {n: joblib.load(models_dir / n) for n in P.MODEL_SHA}
    d = pd.read_csv(inp, dtype={'row_id': str, 'accident_group': str}, low_memory=False)
    P.check_data(d)
    d = d.sort_values(['incident_year', 'accident_group', 'row_id'],
                      kind='stable').reset_index(drop=True)
    years = [int(v) for v in sorted(d.incident_year.astype(int).unique())]
    rows = []
    for it in (1, 2):
        train = d[d.incident_year.astype(int).isin(years[:it])].reset_index(drop=True)
        ev = d[d.incident_year.astype(int) == years[it]].reset_index(drop=True)
        _, preds, base = P.evaluate_one(train, ev, models, it)
        for i, rid in enumerate(ev.row_id):
            rec = {'row_id': rid, 'window': years[it], 'label': ev.label.iloc[i],
                   'KEEP': P.CLASSES[int(base[i])]}
            for k, v in preds.items():
                rec[k] = P.CLASSES[int(v[i])]
            rows.append(rec)
    return pd.DataFrame(rows)


def compare(cur: pd.DataFrame, archive: Path) -> dict:
    """Pair on row_id; report identical-row counts, never rewrite the archive."""
    arc = pd.read_csv(archive, dtype={'row_id': str})
    out = {'archive': str(archive), 'rows_exported': int(len(cur)),
           'rows_archive': int(len(arc)), 'policies': {}}
    pairs = [('KEEP', 'KEEP'), ('AGENT1_FEEDBACK_MEMORY', 'P23_FEEDBACK_MEMORY'),
             ('ABLATION_NO_ERROR_MEMORY', 'P23_NO_MEMORY'),
             ('ORDINARY_FEEDBACK_LOGREG', 'ORDINARY_LOGREG'),
             ('STATIC_P22_CVG', 'STATIC_P22_CVG'),
             ('P21_FIXED_RAW', 'P21_FIXED_RAW'),
             ('B0_CONFIDENCE_EQUAL_K', 'B0_CONFIDENCE_EQUAL_K'),
             ('B1_MARGIN_EQUAL_K', 'B1_MARGIN_EQUAL_K')]
    a = cur.set_index('row_id')
    b = arc.set_index('row_id')
    common = a.index.intersection(b.index)
    out['paired_rows'] = int(len(common))
    for mine, theirs in pairs:
        if mine not in a.columns or theirs not in b.columns:
            out['policies'][mine] = {'status': 'COLUMN_ABSENT'}
            continue
        x, y = a.loc[common, mine], b.loc[common, theirs]
        diff = common[(x != y).to_numpy()]
        out['policies'][mine] = {
            'archive_column': theirs,
            'identical': int((x == y).sum()),
            'mismatch': int(len(diff)),
            'mismatch_row_ids': diff.tolist()[:50],
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True)
    ap.add_argument('--models-dir', required=True)
    ap.add_argument('--out', required=True, help='new output directory; never overwritten')
    ap.add_argument('--compare', default='',
                    help='archived PREDICTIONS_582.csv for a read-only paired comparison')
    a = ap.parse_args()

    dst = Path(a.out)
    if dst.exists():
        raise SystemExit(f'OUTPUT_ALREADY_EXISTS: {dst}')
    cur = export(Path(a.input), Path(a.models_dir))
    dst.mkdir(parents=True, exist_ok=False)
    cur.to_csv(dst / 'P23_ROWLEVEL.csv', index=False)
    print(f'[export] wrote {dst / "P23_ROWLEVEL.csv"} rows={len(cur)}')
    if a.compare:
        rep = compare(cur, Path(a.compare))
        (dst / 'P23_COMPARE.json').write_text(
            json.dumps(rep, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        for k, v in rep['policies'].items():
            if v.get('status') == 'COLUMN_ABSENT':
                print(f'[compare] {k}: column absent in archive')
            else:
                flag = 'OK' if v['mismatch'] == 0 else f"MISMATCH {v['mismatch']}"
                print(f'[compare] {k}: {v["identical"]}/{rep["paired_rows"]} identical {flag}')


if __name__ == '__main__':
    main()
