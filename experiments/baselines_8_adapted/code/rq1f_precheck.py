"""rq1f_precheck.py — RQ1 正式执行前的强制核查（数据哈希/行ID/分组/列序/标签映射/预处理拟合范围/
                                        事故后可得性审计）

输出：results/rq1f_precheck.json, results/rq1f_precheck_readme.md, logs/rq1f_precheck_log.txt

任何一项 FAIL 都必须在启动训练前解决；本脚本不做任何拟合，只做核对。
"""
import os
import json
import hashlib
import platform
import re
import subprocess
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RQ2 = ARCH_ROOT + '/GeOAD_RQ2_SecondStage_LargeDevConfirm_20261025'
SUB = ARCH_ROOT + '/GeOAD_Submission_Final_20261024'
PILOT = ARCH_ROOT + '/GeOAD_RQ1_EightBaselines_Pilot_20261025'
MANIFEST = os.path.join(RQ2, 'protocol', 'data_row_manifest.csv')

SIX = ['CORROSION', 'EQUIPMENT_FAILURE', 'EXTERNAL_DAMAGE',
       'MATERIAL_DEFECT', 'OPERATIONAL', 'OTHER']
MAIN_13 = ['PIPE_DIAMETER', 'PIPE_WALL_THICKNESS', 'PIPE_SMYS',
           'PIPE_SPECIFICATION', 'PIPE_SEAM_TYPE', 'PIPE_COATING_TYPE', 'PIPE_TYPE',
           'ACCIDENT_PSIG', 'MOP_PSIG', 'INTERNAL_INSPECTION_IND',
           'PIPE_AGE_AT_ACCIDENT', 'MANUFACTURE_AGE_AT_ACCIDENT', 'PRESSURE_RATIO']
CATEGORICAL = ['PIPE_SPECIFICATION', 'PIPE_SEAM_TYPE', 'PIPE_COATING_TYPE', 'PIPE_TYPE']
NUMERIC_9 = [c for c in MAIN_13 if c not in CATEGORICAL]
SPLIT_MAP = {'train': 'fit', 'val': 'calib', 'test': 'dev_confirm'}

RESULTS = os.path.join(HERE, 'results')
LOGS = os.path.join(HERE, 'logs')
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)
_LOG = open(os.path.join(LOGS, 'rq1f_precheck_log.txt'), 'w', encoding='utf-8')


def log(m):
    print(m, flush=True)
    _LOG.write(str(m) + '\n')
    _LOG.flush()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    out = {'checks': [], 'env': {}, 'availability_audit': {}}

    def ck(name, ok, detail=''):
        out['checks'].append({'check': name, 'status': 'PASS' if ok else 'FAIL',
                              'detail': str(detail)})
        log(f'[{ "PASS" if ok else "FAIL"}] {name}: {detail}')
        return ok

    # ---------- 0. 环境与版本
    try:
        import sklearn, scipy, catboost, torch
        out['env'] = {
            'python': platform.python_version(),
            'platform': platform.platform(),
            'numpy': np.__version__, 'pandas': pd.__version__,
            'sklearn': sklearn.__version__, 'scipy': scipy.__version__,
            'catboost': catboost.__version__, 'torch': torch.__version__,
            'torch_cuda': torch.cuda.is_available(),
            'n_cpu': os.cpu_count(),
        }
    except Exception as e:  # pragma: no cover
        out['env'] = {'error': str(e)}
    log(f'[env] {out["env"]}')

    man = pd.read_csv(MANIFEST)
    out['manifest_sha256'] = sha256_file(MANIFEST)
    ck('manifest 行数 == 43,834', len(man) == 43834, len(man))
    ck('manifest row_id 唯一', man.row_id.is_unique, int(man.row_id.nunique()))
    ck('manifest 划分计数 fit/calib/dev_confirm/sealed',
       dict(man.split.value_counts()) == {'fit': 29941, 'calib': 6411,
                                          'dev_confirm': 6417, 'sealed': 1065},
       dict(man.split.value_counts()))

    # ---------- 1. sealed 集绝不使用
    n_sealed = int((man.split == 'sealed').sum())
    ck('sealed 行数 == 1,065（标签不在本轮任何流程中被读取）', n_sealed == 1065, n_sealed)
    ck('sealed 行的 label 全为 SEALED / y 全为 -1',
       bool((man.loc[man.split == 'sealed', 'label'] == 'SEALED').all()
            and (man.loc[man.split == 'sealed', 'y'] == -1).all()), 'ok')

    # ---------- 2. 基线输入目录核对
    for variant, cols in [('baseline_data_13d', MAIN_13), ('baseline_data_9d', NUMERIC_9)]:
        d = os.path.join(HERE, 'data', variant)
        ck(f'{variant} 目录存在', os.path.isdir(d), d)
        fm = json.load(open(os.path.join(d, 'fair_manifest.json'), encoding='utf-8'))
        ck(f'{variant} 列序 == 预期', fm['columns'] == cols, fm['columns'])
        ck(f'{variant} 词表拟合范围 == train(=fit)', fm['vocab_fit_split'].startswith('train'), fm['vocab_fit_split'])
        for sp in SPLIT_MAP:
            X = pd.read_csv(os.path.join(d, f'X_{sp}.csv'))
            rm = pd.read_csv(os.path.join(d, f'rowmap_{sp}.csv'))
            y = pd.read_csv(os.path.join(d, f'y_{sp}.csv'))
            src = man[man.split == SPLIT_MAP[sp]]
            ck(f'{variant}/{sp} 行数 == {len(src)}', len(X) == len(src) == len(rm) == len(y),
               f'{len(X)}/{len(rm)}/{len(y)} vs {len(src)}')
            ck(f'{variant}/{sp} row_id 逐行一致', bool((rm.row_id.values == src.row_id.values).all()), '')
            ck(f'{variant}/{sp} 标签与 manifest 逐行一致',
               bool((rm.y_true.values == src.y.to_numpy().astype(int)).all()), '')
            ck(f'{variant}/{sp} 标签名与 SIX 序一致',
               bool(all(SIX[int(a)] == b for a, b in zip(rm.y_true, rm.y_true_label))), '')
            ck(f'{variant}/{sp} y_*.csv 与 rowmap 一致',
               bool((y.ANOMALY_LABEL_ID.values == rm.y_true.values).all()), '')
            # 数值列逐值核对
            bad = []
            for c in NUMERIC_9:
                a = pd.to_numeric(X[c], errors='coerce').to_numpy(dtype=float)
                b = pd.to_numeric(src[c], errors='coerce').to_numpy(dtype=float)
                if not np.array_equal(a, b, equal_nan=True):
                    bad.append(c)
            ck(f'{variant}/{sp} 9 个数值列与 manifest 逐值一致（含 NaN 位置）', not bad, bad)
            if variant == 'baseline_data_13d':
                # 类别列：未知 -> -1 的政策正确性（用 manifest 原值重算）
                mism = []
                for c in CATEGORICAL:
                    v = fm['vocab'][c]
                    exp = src[c].astype(str).map(lambda t: v.get(t, -1)).to_numpy()
                    if not np.array_equal(exp, X[c].to_numpy()):
                        mism.append(c)
                ck(f'{variant}/{sp} 4 个类别列词表编码正确（未知→-1）', not mism, mism)
            # 文件哈希
            for f in [f'X_{sp}.csv', f'y_{sp}.csv', f'rowmap_{sp}.csv']:
                h = sha256_file(os.path.join(d, f))
                ck(f'{variant}/{sp}/{f} sha256 与 fair_manifest 一致',
                   h == fm['files'][sp]['sha256'][f], h[:16])

    # ---------- 3. 分组隔离：train/val/test 的 accident_group 不相交
    gmap = man.set_index('row_id')['accident_group'].to_dict()
    gs = {}
    for sp in SPLIT_MAP:
        rm = pd.read_csv(os.path.join(HERE, 'data', 'baseline_data_13d', f'rowmap_{sp}.csv'))
        gs[sp] = set(gmap[r] for r in rm.row_id)
    ck('train ∩ val 事故组为空', not (gs['train'] & gs['val']), len(gs['train'] & gs['val']))
    ck('train ∩ test 事故组为空', not (gs['train'] & gs['test']), len(gs['train'] & gs['test']))
    ck('val ∩ test 事故组为空', not (gs['val'] & gs['test']), len(gs['val'] & gs['test']))
    out['n_groups'] = {k: len(v) for k, v in gs.items()}

    # ---------- 4. 内部对照预测可复用性（协议/行ID/标签一致）
    for arm in ['A_single', 'A_ensemble', 'F_concat', 'P_full']:
        p = os.path.join(RQ2, 'predictions_s2', f'{arm}.csv')
        if not os.path.isfile(p):
            ck(f'内部对照 {arm} 存在', False, p)
            continue
        d = pd.read_csv(p)
        src = man[man.split == 'dev_confirm']
        ck(f'内部对照 {arm} 行数 == dev_confirm 6,417', len(d) == 6417, len(d))
        ck(f'内部对照 {arm} row_id 集合 == dev_confirm',
           set(d.row_id) == set(src.row_id), '')
        ck(f'内部对照 {arm} 标签与 manifest 一致',
           bool((d.y_true.values == src.set_index('row_id').loc[d.row_id, 'y'].to_numpy().astype(int)).all()), '')
        ck(f'内部对照 {arm} accident_group 与 manifest 一致',
           bool((d.accident_group.values == src.set_index('row_id').loc[d.row_id, 'accident_group'].to_numpy()).all()), '')
        out.setdefault('internal_controls', {})[arm] = {
            'acc': float(d.correct.mean()), 'n': len(d),
            'path': os.path.relpath(p, ARCH_ROOT + ''),
        }

    # ---------- 5. 预处理拟合范围抽查（公平性修复版基线源码）
    bl = os.path.join(PILOT, 'code', 'baselines_fixed')
    fit_scope = {}
    for slug in sorted(os.listdir(bl)):
        tp = os.path.join(bl, slug, 'train.py')
        if not os.path.isfile(tp):
            continue
        src = open(tp, encoding='utf-8').read()
        fit_scope[slug] = {
            'reads_X_train': 'X_train.csv' in src,
            'reads_X_val': 'X_val.csv' in src,
            'reads_X_test': 'X_test.csv' in src,
            'n_test_mentions': src.count('X_test'),
            'n_val_mentions': src.count('X_val'),
        }
        # 关键泄漏模式：直接从 X_test 计算统计量（fit/mean/scaler 拟合）
        leaks = []
        for ln, line in enumerate(src.splitlines(), 1):
            s = line.strip()
            if s.startswith('#') or 'logger.info' in s or 'print(' in s:
                continue  # 纯日志行不构成拟合；仅记录，不算泄漏
            if 'X_test' in s and any(k in s for k in ['.fit(', 'fit_transform', 'SimpleImputer(', 'StandardScaler(']):
                leaks.append(f'{slug}:{ln}: {s[:120]}')
            # 统计量（均值/标准差/极值）若直接由 X_test 计算，同样是拟合越界
            if 'X_test' in s and re.search(r'X_test\w*\.(mean|std|min|max)\(\)', s):
                leaks.append(f'{slug}:{ln}: {s[:120]}')
        fit_scope[slug]['suspected_test_fit'] = leaks
        ck(f'{slug} 未见"用 X_test 拟合统计量"模式', not leaks, leaks[:3])
    out['preprocessing_fit_scope'] = fit_scope

    # ---------- 6. 事故后可得性审计（ACCIDENT_PSIG / PRESSURE_RATIO / INTERNAL_INSPECTION_IND）
    fd = pd.read_csv(os.path.join(SUB, 'feature_dictionary.csv'))
    for f in ['ACCIDENT_PSIG', 'PRESSURE_RATIO', 'INTERNAL_INSPECTION_IND', 'MOP_PSIG']:
        r = fd[fd.feature == f].iloc[0]
        out['availability_audit'][f] = {
            'decision_time': r.decision_time,
            'definition': r.definition,
            'in_main13': bool(r.in_main13),
            'in_strict_pre11': bool(r.in_strict_pre11),
            'source_columns': str(r.source_columns),
            'note': str(r.note),
        }
    ck('ACCIDENT_PSIG 决策时点 == POST（事故后诊断时点可得，符合任务定义）',
       out['availability_audit']['ACCIDENT_PSIG']['decision_time'] == 'POST',
       out['availability_audit']['ACCIDENT_PSIG']['decision_time'])
    ck('PRESSURE_RATIO 决策时点 == POST（派生自 ACCIDENT_PSIG）',
       out['availability_audit']['PRESSURE_RATIO']['decision_time'] == 'POST',
       out['availability_audit']['PRESSURE_RATIO']['decision_time'])
    ck('INTERNAL_INSPECTION_IND 决策时点已记录为 UNCLEAR（不确定性已登记，未静默改变任务）',
       out['availability_audit']['INTERNAL_INSPECTION_IND']['decision_time'] == 'UNCLEAR',
       out['availability_audit']['INTERNAL_INSPECTION_IND']['decision_time'])

    # ---------- 7. 正式实现与公平性修复版实现哈希（冻结）
    code_hashes = {}
    for p in [os.path.join(SUB, 'code', 'g2_common.py'),
              os.path.join(RQ2, 'code', 's2_common.py'),
              os.path.join(RQ2, 'protocol', 'protocol_lock.json')]:
        if os.path.isfile(p):
            code_hashes[os.path.relpath(p, ARCH_ROOT + '')] = sha256_file(p)
    for slug in sorted(os.listdir(bl)):
        tp = os.path.join(bl, slug, 'train.py')
        if os.path.isfile(tp):
            code_hashes[f'baselines_fixed/{slug}/train.py'] = sha256_file(tp)
    out['code_hashes'] = code_hashes
    ck('g2_common.py sha256 == 842217739ba315b85b6a5a9aa88d4cbde13c9871487a71be61424bcf4589abfa',
       code_hashes.get('GeOAD_Submission_Final_20261024/code/g2_common.py')
       == '842217739ba315b85b6a5a9aa88d4cbde13c9871487a71be61424bcf4589abfa',
       code_hashes.get('GeOAD_Submission_Final_20261024/code/g2_common.py', '')[:16])
    ck('protocol_lock.json sha256 == c90cd2ab44b28d8a1c444cdd79b3641ec4c250ed79b62252c225a2848653ef6c',
       code_hashes.get('GeOAD_RQ2_SecondStage_LargeDevConfirm_20261025/protocol/protocol_lock.json')
       == 'c90cd2ab44b28d8a1c444cdd79b3641ec4c250ed79b62252c225a2848653ef6c',
       code_hashes.get('GeOAD_RQ2_SecondStage_LargeDevConfirm_20261025/protocol/protocol_lock.json', '')[:16])

    n_fail = sum(1 for c in out['checks'] if c['status'] == 'FAIL')
    out['n_checks'] = len(out['checks'])
    out['n_fail'] = n_fail
    out['verdict'] = 'ALL_PASS' if n_fail == 0 else f'FAIL_{n_fail}'
    with open(os.path.join(RESULTS, 'rq1f_precheck.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f'\n[verdict] {out["verdict"]}  ({len(out["checks"])} 项检查, {n_fail} 项 FAIL)')
    _LOG.close()
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
