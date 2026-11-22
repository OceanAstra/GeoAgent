"""rq1f_package.py — 组装 RQ1 正式验证交付包（zip + sha256）

用法：python3 code/rq1f_package.py
输出：" + ARCH_ROOT + "/GeOAD_RQ1_RQ1Formal_20261025_完整交付包.zip（及 .sha256）
"""
import os
import sys
import json
import shutil
import hashlib
import zipfile
import datetime

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = ARCH_ROOT + ''
DELIV_NAME = 'GeOAD_RQ1_FormalValidation_20261025_完整交付包'
DELIV_DIR = os.path.join(OUT_ROOT, DELIV_NAME)

EXCLUDE_DIRS = {'__pycache__', '.ipynb_checkpoints'}
EXCLUDE_SUFFIX = {'.zip', '.sha256'}


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def iter_files(root):
    for r, ds, fs in os.walk(root):
        ds[:] = [d for d in ds if d not in EXCLUDE_DIRS]
        for fn in sorted(fs):
            p = os.path.join(r, fn)
            if os.path.splitext(fn)[1] in EXCLUDE_SUFFIX:
                continue
            yield p


def main():
    if os.path.isdir(DELIV_DIR):
        shutil.rmtree(DELIV_DIR)
    shutil.copytree(HERE, DELIV_DIR,
                    ignore=shutil.ignore_patterns(*EXCLUDE_DIRS, '*.zip', '*.sha256'))

    sums = []
    for p in sorted(iter_files(DELIV_DIR)):
        sums.append((sha256(p), os.path.relpath(p, DELIV_DIR), os.path.getsize(p)))
    with open(os.path.join(DELIV_DIR, 'SHA256SUMS.txt'), 'w', encoding='utf-8') as f:
        for h, rel, sz in sums:
            f.write(f'{h}  {rel}\n')

    zp = os.path.join(OUT_ROOT, DELIV_NAME + '.zip')
    if os.path.exists(zp):
        os.remove(zp)
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(iter_files(DELIV_DIR)):
            z.write(p, os.path.join(DELIV_NAME, os.path.relpath(p, DELIV_DIR)))
    zh = sha256(zp)
    with open(zp + '.sha256', 'w', encoding='utf-8') as f:
        f.write(f'{zh}  {os.path.basename(zp)}\n')

    print(f'交付目录: {DELIV_DIR}')
    print(f'文件数  : {len(sums)}')
    print(f'ZIP     : {zp}  ({os.path.getsize(zp)/1e6:.1f} MB)')
    print(f'SHA256  : {zh}')
    print(f'时间    : {datetime.datetime.now().isoformat()}')


if __name__ == '__main__':
    main()
