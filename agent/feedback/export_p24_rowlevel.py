#!/usr/bin/env python3
"""Export the P24 arms per row without changing any decision.

`p24_conditional_repair.py` writes only `P24_RESULT.json` / `P24_DECISION.md`.
Two columns of `results/per_row_predictions/PREDICTIONS_582.csv`
(`P24_CONDITIONAL`, `P24_GLOBAL_ABLATION`) are produced by that module rather
than by `p23_feedback_replay.py`, so auditing the archive needs a row-level dump
from the same code path.

This tool imports `evaluate()` from `p24_conditional_repair.py` verbatim and
writes every policy's per-row decision plus a paired comparison against an
archive CSV. Nothing is re-fitted differently and no threshold is touched.

Usage
-----
    python agent/feedback/export_p24_rowlevel.py \
        --input data/feedback_replay/P23_INPUT_GROUPKEY_REPAIRED.csv \
        --models-dir models_and_configs/p22_feedback \
        --out /tmp/p24_rowlevel \
        [--compare results/per_row_predictions/PREDICTIONS_582.csv]

Outputs
-------
    P24_ROWLEVEL.csv    row_id, window, label, KEEP, P24_CONDITIONAL,
                        ABLATION_GLOBAL_PRIOR and the P23 reference arms
    P24_COMPARE.json    (with --compare) per-policy identical/mismatch counts
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
import p24_conditional_repair as R  # noqa: E402


def export(inp: Path, models_dir: Path) -> pd.DataFrame:
    models = {n: joblib.load(models_dir / n) for n in R.MODEL_SHA}
    d = pd.read_csv(inp, dtype={"row_id": str, "accident_group": str}, low_memory=False)
    P.check_data(d)
    d = d.sort_values(["incident_year", "row_id"], kind="stable").reset_index(drop=True)
    windows = {int(y): d[d.incident_year == y].reset_index(drop=True)
               for y in sorted(d.incident_year.astype(int).unique())}
    rows = []
    for step, target in enumerate([2025, 2026], start=1):
        feedback = d[d.incident_year < target].reset_index(drop=True)
        _, preds, base, y = R.evaluate(feedback, windows[target], models, step)
        ev = windows[target]
        for i, rid in enumerate(ev.row_id):
            rec = {"row_id": rid, "window": target,
                   "label": P.CLASSES[int(y[i])], "KEEP": P.CLASSES[int(base[i])]}
            for k, v in preds.items():
                rec[k] = P.CLASSES[int(v[i])]
            rows.append(rec)
    return pd.DataFrame(rows)


def compare(cur: pd.DataFrame, archive: Path) -> dict:
    arc = pd.read_csv(archive, dtype={"row_id": str})
    pairs = [("KEEP", "KEEP"),
             ("P24_CONDITIONAL", "P24_CONDITIONAL"),
             ("ABLATION_GLOBAL_PRIOR", "P24_GLOBAL_ABLATION"),
             ("P23_FEEDBACK_MEMORY", "P23_FEEDBACK_MEMORY"),
             ("P23_NO_MEMORY", "P23_NO_MEMORY"),
             ("ORDINARY_FEEDBACK_LOGREG", "ORDINARY_LOGREG"),
             ("B0_CONFIDENCE_EQUAL_K", "B0_CONFIDENCE_EQUAL_K"),
             ("B1_MARGIN_EQUAL_K", "B1_MARGIN_EQUAL_K")]
    a = cur.set_index("row_id")
    b = arc.set_index("row_id")
    common = a.index.intersection(b.index)
    out = {"archive": str(archive), "rows_exported": int(len(cur)),
           "rows_archive": int(len(arc)), "paired_rows": int(len(common)), "policies": {}}
    for mine, theirs in pairs:
        if mine not in a.columns or theirs not in b.columns:
            out["policies"][mine] = {"status": "COLUMN_ABSENT"}
            continue
        bad = sorted(common[(a.loc[common, mine].to_numpy()
                             != b.loc[common, theirs].to_numpy())].tolist())
        out["policies"][mine] = {"archive_column": theirs,
                                 "identical": int(len(common) - len(bad)),
                                 "mismatch": len(bad),
                                 "mismatch_row_ids": bad[:50]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default=None)
    a = ap.parse_args()

    inp, models_dir, out = Path(a.input), Path(a.models_dir), Path(a.out)
    if R.file_sha(inp) != R.EXPECTED_INPUT_SHA:
        raise SystemExit("INPUT_SHA256_MISMATCH")
    for name, digest in R.MODEL_SHA.items():
        if R.file_sha(models_dir / name) != digest:
            raise SystemExit("FROZEN_MODEL_SHA_MISMATCH:" + name)

    cur = export(inp, models_dir)
    out.mkdir(parents=True, exist_ok=True)
    cur.to_csv(out / "P24_ROWLEVEL.csv", index=False)
    print(f"[export] wrote {out / 'P24_ROWLEVEL.csv'} rows={len(cur)}")

    if a.compare:
        res = compare(cur, Path(a.compare))
        (out / "P24_COMPARE.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        for k, v in res["policies"].items():
            if v.get("status"):
                print(f"[compare] {k}: {v['status']}")
            else:
                print(f"[compare] {k}: {v['identical']}/{res['paired_rows']} identical")


if __name__ == "__main__":
    main()
