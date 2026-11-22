"""rq1f_collect_env.py — 收集环境、代码版本与文件哈希（供论文可复现性章节）

输出：results/rq1f_environment.json、results/rq1f_code_hashes.csv
"""
import os
import sys
import json
import hashlib
import platform
import subprocess
import datetime

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, 'results')
RQ2 = ARCH_ROOT + '/GeOAD_RQ2_SecondStage_LargeDevConfirm_20261025'


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def main():
    env = {
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'python': sys.version,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'cpu_count': os.cpu_count(),
        'loadavg': os.getloadavg(),
        'packages': {},
    }
    for m in ['numpy', 'pandas', 'sklearn', 'scipy', 'catboost', 'torch', 'imblearn']:
        try:
            mod = __import__(m)
            env['packages'][m] = getattr(mod, '__version__', 'n/a')
        except Exception as e:
            env['packages'][m] = f'IMPORT_FAIL: {e}'
    try:
        env['torch_cuda_available'] = __import__('torch').cuda.is_available()
        env['torch_threads'] = __import__('torch').get_num_threads()
    except Exception:
        pass
    try:
        env['git_head'] = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                                         text=True).stdout.strip() or 'not_a_git_repo'
    except Exception:
        env['git_head'] = 'n/a'

    # 冻结协议与内部对照的哈希
    frozen = {}
    for p in [os.path.join(RQ2, 'protocol', 'protocol_lock.json'),
              os.path.join(RQ2, 'protocol', 'data_row_manifest.csv'),
              os.path.join(RQ2, 'code', 'g2_common.py'),
              os.path.join(RQ2, 'predictions_s2', 'P_full.csv'),
              os.path.join(RQ2, 'predictions_s2', 'A_ensemble.csv'),
              os.path.join(RQ2, 'predictions_s2', 'A_single.csv'),
              os.path.join(RQ2, 'predictions_s2', 'F_concat.csv')]:
        if os.path.isfile(p):
            frozen[os.path.relpath(p, ARCH_ROOT + '')] = sha256(p)
    env['frozen_artifact_sha256'] = frozen

    with open(os.path.join(RES, 'rq1f_environment.json'), 'w', encoding='utf-8') as f:
        json.dump(env, f, ensure_ascii=False, indent=2)

    rows = []
    for root, dirs, files in os.walk(HERE):
        dirs[:] = [d for d in dirs if d not in ('__pycache__', 'baseline_out')]
        for fn in sorted(files):
            p = os.path.join(root, fn)
            if os.path.getsize(p) > 200 * 1024 * 1024:
                continue
            rel = os.path.relpath(p, HERE)
            if rel.startswith(('predictions/', 'data/')) or rel.endswith(('.zip', '.sha256')):
                continue
            rows.append({'path': rel, 'bytes': os.path.getsize(p), 'sha256': sha256(p)})
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(RES, 'rq1f_code_hashes.csv'), index=False)
    print(json.dumps(env, ensure_ascii=False, indent=2))
    print(f'\n[OK] {len(rows)} 个文件已记录哈希 -> results/rq1f_code_hashes.csv')


if __name__ == '__main__':
    main()
