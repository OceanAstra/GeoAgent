#!/usr/bin/env python3
"""Check that repository-relative paths quoted in docs actually exist.

Scans Markdown / JSON / CFF / TXT files for tokens that look like a path inside
this repository (they start with one of the known top-level directories) and
verifies that each token resolves — either from the repository root or relative
to the directory of the file that mentions it.

Paths under a directory the release creates at runtime (`generated/`) and
`__pycache__` are ignored. Exit code is 1 when anything is missing.

Usage
-----
    python reproduce/check_doc_paths.py [--root .] [--json OUT.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TOPS = (
    "agent", "data", "docs", "experiments", "figures", "models_and_configs",
    "provenance", "reproduce", "results",
)
PAT = re.compile(
    r"(?<![\w/.-])((?:" + "|".join(re.escape(t) for t in TOPS) + r")[\w./-]*?\.[A-Za-z0-9]{1,6})(?![\w/])"
)
SUFFIXES = {".md", ".json", ".cff", ".txt"}
RUNTIME_PREFIXES = ("generated/",)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="repository root (default: cwd)")
    ap.add_argument("--json", default="", help="optional JSON report path")
    a = ap.parse_args()

    root = Path(a.root).resolve()
    missing: list[dict] = []
    seen: set[tuple[str, str]] = set()
    checked = 0

    for doc in sorted(root.rglob("*")):
        if not doc.is_file() or doc.suffix.lower() not in SUFFIXES:
            continue
        if "__pycache__" in doc.parts:
            continue
        try:
            text = doc.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel_doc = doc.relative_to(root).as_posix()
        for m in PAT.finditer(text):
            token = m.group(1).rstrip(".,;:)")
            if any(token.startswith(p) for p in RUNTIME_PREFIXES):
                continue
            if "generated/" in token:
                continue
            checked += 1
            if (root / token).exists() or (doc.parent / token).exists():
                continue
            if (root / token.split("*")[0]).exists():
                continue
            key = (rel_doc, token)
            if key in seen:
                continue
            seen.add(key)
            missing.append({
                "file": rel_doc,
                "line": text[: m.start()].count("\n") + 1,
                "token": token,
            })

    report = {"references_checked": checked, "missing_count": len(missing),
              "missing": missing}
    print(f"references checked: {checked}; missing: {len(missing)}")
    for item in missing:
        print(f"  MISSING {item['file']}:{item['line']} -> {item['token']}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
