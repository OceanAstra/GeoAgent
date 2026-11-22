#!/usr/bin/env python3
"""Read-only SHA-256 verification; release manifest excludes itself."""
from pathlib import Path
import csv,hashlib,sys
ROOT=Path(__file__).resolve().parents[1]
manifest=ROOT/'FILE_MANIFEST_SHA256.csv'
with manifest.open(newline='',encoding='utf-8') as f: records=list(csv.DictReader(f))
errors=[]
if len({r['path'] for r in records}) != len(records): errors.append('duplicate manifest paths')
for r in records:
    path=ROOT/r['path']
    if not path.is_file(): errors.append(f'MISSING: {r["path"]}'); continue
    if path.stat().st_size!=int(r['bytes']):errors.append(f'SIZE: {r["path"]}');continue
    if hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']:errors.append(f'HASH: {r["path"]}')
actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and not any(part in {'.venv','__pycache__','.pytest_cache'} for part in p.parts)}
expected={r['path'] for r in records}|{'FILE_MANIFEST_SHA256.csv'}
for extra in sorted(actual-expected):errors.append(f'UNLISTED: {extra}')
for ghost in sorted(expected-actual):errors.append(f'GHOST: {ghost}')
if errors:
    for e in errors[:30]:print('FAIL',e)
    sys.exit(1)
print(f'PASS SHA-256 manifest: {len(records)} checks; no unexpected files')
