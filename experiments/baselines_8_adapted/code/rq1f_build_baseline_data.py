"""rq1f_build_baseline_data.py — RQ1 正式验证：从 RQ2 锁定的 manifest 重建全量基线输入

设计要点（严格继承 RQ2-S2-LARGE-DEV-CONFIRM-v1 + RQ1 公平性修复轮的编码规则）：

1. 划分映射：fit(29,941) -> train；calib(6,411) -> val；dev_confirm(6,417) -> test。
   **sealed(1,065) 行完全不进入本脚本**（不读其 label、不写入任何输出文件）。
2. 13 维 = 9 数值列（保留原始缺失）+ 4 类别列（序数编码）。
   词表 **只从 train(=fit) 拟合**，未知取值 -> -1。数值列此处不做任何插补/标准化，
   交由各基线自身的"训练折拟合"流程处理（与公平性修复轮一致）。
3. 9 维对照 = 仅 9 个数值列，数值逐值与 13 维相同。
4. 逐行断言：row_id 与 manifest 一致、y 与 manifest 的 label 映射一致、
   数值列与 manifest 逐值一致、类别列编码在 train 上可逆。

输出：data/baseline_data_13d/{X,y,rowmap}_{train,val,test}.csv + fair_manifest.json
      data/baseline_data_9d/...
"""
import os
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = Path(__file__).resolve().parents[3]
ARCH_ROOT = str(REPO_ROOT)
# Default to the curated, already unsealed manifest shipped with the release.
# An authorized alternate manifest can be supplied without editing source.
MANIFEST = os.environ.get('GEOAGENT_DEVELOPMENT_MANIFEST', str(
    REPO_ROOT / 'models_and_configs/geoad_materials/data/data_row_manifest_development_only.csv'))

MAIN_13 = ['PIPE_DIAMETER', 'PIPE_WALL_THICKNESS', 'PIPE_SMYS',
           'PIPE_SPECIFICATION', 'PIPE_SEAM_TYPE', 'PIPE_COATING_TYPE', 'PIPE_TYPE',
           'ACCIDENT_PSIG', 'MOP_PSIG', 'INTERNAL_INSPECTION_IND',
           'PIPE_AGE_AT_ACCIDENT', 'MANUFACTURE_AGE_AT_ACCIDENT', 'PRESSURE_RATIO']
CATEGORICAL = ['PIPE_SPECIFICATION', 'PIPE_SEAM_TYPE', 'PIPE_COATING_TYPE', 'PIPE_TYPE']
NUMERIC_9 = [c for c in MAIN_13 if c not in CATEGORICAL]

SIX = ['CORROSION', 'EQUIPMENT_FAILURE', 'EXTERNAL_DAMAGE',
       'MATERIAL_DEFECT', 'OPERATIONAL', 'OTHER']

SPLIT_MAP = {'train': 'fit', 'val': 'calib', 'test': 'dev_confirm'}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    man = pd.read_csv(MANIFEST)
    assert man.row_id.is_unique
    # The provided public-candidate manifest is already development-only (42,769).
    # The historical full manifest additionally had 1,065 sealed rows; these
    # must never be read into training or published as baseline inputs.
    n_sealed = int((man.split == 'sealed').sum())
    assert (len(man), n_sealed) in ((42769, 0), (43834, 1065)), (len(man), n_sealed)
    expected = {'fit': 29941, 'calib': 6411, 'dev_confirm': 6417}
    counts = man.loc[man.split.isin(expected)].split.value_counts().to_dict()
    assert counts == expected, counts

    parts = {sp: man[man.split == role].copy() for sp, role in SPLIT_MAP.items()}
    for sp, df in parts.items():
        assert (df.label != 'SEALED').all(), sp

    out_root = os.environ.get('GEOAGENT_BASELINE_OUT', os.path.join(HERE, 'data'))

    # ---- 词表只从 train(=fit) 拟合
    Xtr = parts['train'][MAIN_13].copy()
    vocab = {}
    for c in CATEGORICAL:
        vals = sorted(Xtr[c].astype(str).unique())
        vocab[c] = {v: i for i, v in enumerate(vals)}

    def build(out_name, columns, encode_cats):
        out = os.path.join(out_root, out_name)
        os.makedirs(out, exist_ok=True)
        manifest = {
            'source': os.path.relpath(MANIFEST, ARCH_ROOT + ''),
            'source_sha256': sha256_file(MANIFEST),
            'protocol_id': 'RQ2-S2-LARGE-DEV-CONFIRM-v1',
            'output': out_name,
            'columns': columns,
            'categorical_encoding': encode_cats,
            'vocab_fit_split': 'train (= fit, 29941 rows)',
            'unknown_policy': -1,
            'sealed_rows_in_input': n_sealed,
            'sealed_rows_excluded_from_output': n_sealed,
            'note': 'Development-only manifest omits 1065 historical sealed rows' if n_sealed == 0 else 'Sealed rows present but excluded',
            'vocab': vocab,
            'files': {},
        }
        for sp, df in parts.items():
            X = df[MAIN_13].copy()
            if encode_cats:
                Xo = pd.DataFrame(index=X.index)
                for c in columns:
                    if c in CATEGORICAL:
                        Xo[c] = X[c].astype(str).map(lambda v: vocab[c].get(v, -1)).astype(np.int64)
                    else:
                        Xo[c] = pd.to_numeric(X[c], errors='coerce')
                for c in CATEGORICAL:
                    assert Xo[c].isna().sum() == 0, c
            else:
                Xo = X[columns].copy()
            Xo = Xo.reset_index(drop=True)

            rowmap = pd.DataFrame({
                'row_id': df.row_id.to_numpy(),
                'y_true': df.y.to_numpy().astype(int),
                'y_true_label': df.label.to_numpy(),
            })
            y = pd.DataFrame({
                'ANOMALY_TYPE': df.label.to_numpy(),
                'ANOMALY_LABEL_ID': df.y.to_numpy().astype(int),
                'IS_ANOMALY': 1,
            })
            assert len(Xo) == len(rowmap) == len(y)
            # 标签映射核对：manifest 的 label <-> y 必须与 SIX 顺序一致
            for lab, yy in zip(rowmap.y_true_label, rowmap.y_true):
                assert SIX[int(yy)] == lab, (lab, yy)
            Xo.to_csv(os.path.join(out, f'X_{sp}.csv'), index=False)
            rowmap.to_csv(os.path.join(out, f'rowmap_{sp}.csv'), index=False)
            y.to_csv(os.path.join(out, f'y_{sp}.csv'), index=False)
            manifest['files'][sp] = {
                'n': int(len(Xo)),
                'sha256': {f: sha256_file(os.path.join(out, f))
                           for f in [f'X_{sp}.csv', f'y_{sp}.csv', f'rowmap_{sp}.csv']},
            }
        with open(os.path.join(out, 'fair_manifest.json'), 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f'[OK] {out_name}: {len(columns)} 列 -> {out}')
        return manifest

    m13 = build('baseline_data_13d', MAIN_13, True)
    m9 = build('baseline_data_9d', NUMERIC_9, False)

    # ---- 自检
    for sp in SPLIT_MAP:
        a = pd.read_csv(os.path.join(out_root, 'baseline_data_13d', f'X_{sp}.csv'))
        b = pd.read_csv(os.path.join(out_root, 'baseline_data_9d', f'X_{sp}.csv'))
        src = parts[sp]
        for col in NUMERIC_9:
            x = pd.to_numeric(src[col], errors='coerce').to_numpy(dtype=float)
            assert np.array_equal(x, a[col].to_numpy(dtype=float), equal_nan=True), (sp, col, '13d')
            assert np.array_equal(x, b[col].to_numpy(dtype=float), equal_nan=True), (sp, col, '9d')
    # 13d 类别编码在 train 上可逆
    src_tr = parts['train']
    Xtr13 = pd.read_csv(os.path.join(out_root, 'baseline_data_13d', 'X_train.csv'))
    for col in CATEGORICAL:
        inv = {i: s for s, i in m13['vocab'][col].items()}
        for code, raw in zip(Xtr13[col].to_numpy(), src_tr[col].astype(str).to_numpy()):
            assert inv[int(code)] == raw, (col, code, raw)
    # 覆盖统计
    for sp in ['val', 'test']:
        for col in CATEGORICAL:
            unseen = sorted(set(parts[sp][col].astype(str).unique()) - set(m13['vocab'][col]))
            if unseen:
                print(f'  [信息] {sp}/{col} 训练折未见取值 {len(unseen)} 个 -> -1')
    print('[自检通过] 13d/9d 数值列与 manifest 逐值一致；13d 类别编码在 train 上可逆；'
          'sealed 行已排除。')


if __name__ == '__main__':
    main()
