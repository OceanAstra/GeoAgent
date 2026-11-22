"""
s2_common.py — RQ2 第二小阶段：大样本开发确认协议的数据/特征适配层

============================================================================
**不修改**任何历史工程：
  * 正式 P 的源码 `GeOAD_Submission_Final_20261024/code/g2_common.py` 只读引用；
  * 消融实现 `GeOAD_RQ1_EightBaselines_Pilot_20261025/code/rq2_ablation_net.py` 只读引用；
  * 本模块只负责"把冻结协议（protocol/）读进来 + 预处理"。

数据来源：`protocol/data_row_manifest.csv`（门槛 A 生成，43,834 行）
  * 开发池 = split ∈ {fit, calib, dev_confirm}，42,769 行；
  * 封存集 = split == 'sealed'（incident_year ≥ 2024，1,065 行），**本轮不读其标签**。

角色
------------------------------------------------------------------------------
  * fit        : 训练（CatBoost 骨干 + 残差头）
  * calib      : 仅用于检查点选择（calib Macro-F1 argmax）
  * dev_confirm: 开发确认评价（开发性验证集，**不是**盲测）

预处理统计量（中位数 / 缩放 / 类别词表）**全部只在 fit 行拟合**。
"""
import os
import json
import hashlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTO = os.path.join(HERE, 'protocol')

SIX = ['CORROSION', 'EQUIPMENT_FAILURE', 'EXTERNAL_DAMAGE',
       'MATERIAL_DEFECT', 'OPERATIONAL', 'OTHER']
CLS2IDX = {c: i for i, c in enumerate(SIX)}

MAIN_13 = ['PIPE_DIAMETER', 'PIPE_WALL_THICKNESS', 'PIPE_SMYS',
           'PIPE_SPECIFICATION', 'PIPE_SEAM_TYPE', 'PIPE_COATING_TYPE', 'PIPE_TYPE',
           'ACCIDENT_PSIG', 'MOP_PSIG', 'INTERNAL_INSPECTION_IND',
           'PIPE_AGE_AT_ACCIDENT', 'MANUFACTURE_AGE_AT_ACCIDENT', 'PRESSURE_RATIO']
CATEGORICAL = ['PIPE_SPECIFICATION', 'PIPE_SEAM_TYPE', 'PIPE_COATING_TYPE', 'PIPE_TYPE']
NUMERIC = [c for c in MAIN_13 if c not in CATEGORICAL]

# 5 个标签邻接泄漏字段：本轮明确排除，且不进入任何视图
EXCLUDED_LEAK_ADJACENT = ['UNDER_CATHODIC_PROTECTION_IND', 'CP_ANNUAL_SURVEY_IND',
                          'CLOSE_INTERVAL_SURVEY_IND', 'EXTERNALLY_COATED', 'PRIOR_DAMAGE']

ROLES = ['fit', 'calib', 'dev_confirm']


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def load_lock():
    with open(os.path.join(PROTO, 'protocol_lock.json'), encoding='utf-8') as f:
        return json.load(f)


def load_manifest():
    p = os.path.join(PROTO, 'data_row_manifest.csv')
    df = pd.read_csv(p, low_memory=False)
    return df


def load_protocol(limit=None):
    """返回 dict：role -> dict(X, y, groups, row_id, label, pattern13, incident_date,
    data_source, operator_id, identity_certainty)。只含开发池三种角色。

    limit 仅供**冒烟测试**使用（各角色取前 N 行，不改结构/超参/规则）。"""
    df = load_manifest()
    lock = load_lock()
    assert list(lock['class_order']) == SIX, '协议锁类别顺序与本模块不一致'
    assert lock['features']['main_13'] == MAIN_13, '协议锁特征集与本模块不一致'

    dev = df[df.split.isin(ROLES)].copy().reset_index(drop=True)
    assert (dev.y >= 0).all(), '开发池存在无标签行'
    assert set(dev.split.unique()) == set(ROLES), '角色集合异常'
    for c in CATEGORICAL:
        dev[c] = dev[c].fillna('NOT_RECORDED').astype(str)
    for c in NUMERIC:
        dev[c] = pd.to_numeric(dev[c], errors='coerce')

    out = {}
    for r in ROLES:
        d = dev[dev.split == r].reset_index(drop=True)
        if limit:
            d = d.iloc[:limit].reset_index(drop=True)
        out[r] = dict(
            X=d[MAIN_13].copy(),
            y=d['y'].to_numpy().astype(int),
            groups=d['accident_group'].astype(str).to_numpy(),
            row_id=d['row_id'].astype(str).to_numpy(),
            label=d['label'].astype(str).to_numpy(),
            pattern13=d['pattern13'].astype(str).to_numpy(),
            data_source=d['data_source'].astype(str).to_numpy(),
            operator_id=d['operator_id'].astype(str).to_numpy(),
            identity_certainty=d['identity_certainty'].astype(str).to_numpy(),
            incident_date=d['incident_date'].astype(str).to_numpy(),
        )
    out['manifest'] = df
    out['dev'] = dev
    return out


def catboost_frame(X):
    """CatBoost 输入：类别列转字符串（缺失固定 token），数值列转数值（保留 NaN）。"""
    X = X.copy()
    for c in CATEGORICAL:
        X[c] = X[c].fillna('NOT_RECORDED').astype(str)
    for c in NUMERIC:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    return X[MAIN_13]


def cat_feature_index():
    return [MAIN_13.index(c) for c in CATEGORICAL]


def save_predictions(name, y_true, probs, part, outdir, extra=None):
    """统一逐样本结果文件：row_id / accident_group / 真值 / 预测 / 6 维概率。"""
    os.makedirs(outdir, exist_ok=True)
    probs = np.asarray(probs, dtype=float)
    pred = probs.argmax(1)
    df = pd.DataFrame({
        'row_id': part['row_id'],
        'accident_group': part['groups'],
        'y_true': np.asarray(y_true, dtype=int),
        'y_true_label': part['label'],
        'pred': pred,
        'pred_label': [SIX[i] for i in pred],
        'correct': (pred == np.asarray(y_true, dtype=int)).astype(int),
    })
    for i, c in enumerate(SIX):
        df[f'prob_{c}'] = probs[:, i]
    if extra:
        for k, v in extra.items():
            df[k] = v
    p = os.path.join(outdir, f'{name}.csv')
    df.to_csv(p, index=False)
    return p
