"""
g2_common.py — G2 公平对照的公共模块（新数据契约 v1）
=====================================================
设计约束（来自任务书 G2 与不可变约束 ②③④）：
  * **不新增架构海选**：臂集合固定为 A（三种子 CatBoost）/ B（平衡 CatBoost）/
    F（非几何残差）/ P（几何头，按新特征规格重训）。不引入 TabM/TabR/TabPFN，
    不调几何/门控/GMM，不做融合阈值扫描。
  * **所有臂共用同一合法特征集**（MAIN_13），同一划分协议，同一预处理；
    预处理/超参/检查点选择**只看 fit + calib**，eval 封存到最后一刻。
  * **类别权重只由训练折统计**（w_c = n/(K·n_c)）。
  * **缺失填补只拟合在 fit 行**（数值中位数；分类为固定 token）。
  * P/F 的主干概率在 fit 行用**组隔离 OOF** 产生，calib/eval 用全 fit 模型，
    避免主干"背答案"导致残差头学到失配的输入分布。

本模块不写任何既有目录，只读 `source_trace.csv` / `split_manifest.json`。
"""
import os
import json
import hashlib
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix)

# 归档源目录（可选）。打包件不附带该目录；load_main() 需要它读取 source_trace.csv。
ROOT = os.environ.get('ZHL_GEOAD_SUBMISSION_DIR', '')
ART = os.path.join(ROOT, 'artifacts')
G2ART = os.path.join(ART, 'g2')

SIX_CLASSES = ['CORROSION', 'EQUIPMENT_FAILURE', 'EXTERNAL_DAMAGE',
               'MATERIAL_DEFECT', 'OPERATIONAL', 'OTHER']
MINORITY = ['MATERIAL_DEFECT', 'OPERATIONAL']

MAIN_13 = ['PIPE_DIAMETER', 'PIPE_WALL_THICKNESS', 'PIPE_SMYS',
           'PIPE_SPECIFICATION', 'PIPE_SEAM_TYPE', 'PIPE_COATING_TYPE', 'PIPE_TYPE',
           'ACCIDENT_PSIG', 'MOP_PSIG', 'INTERNAL_INSPECTION_IND',
           'PIPE_AGE_AT_ACCIDENT', 'MANUFACTURE_AGE_AT_ACCIDENT', 'PRESSURE_RATIO']
STRICT_PRE_11 = [f for f in MAIN_13 if f not in ('ACCIDENT_PSIG', 'PRESSURE_RATIO')]
LEGACY_18 = MAIN_13 + ['UNDER_CATHODIC_PROTECTION_IND', 'CP_ANNUAL_SURVEY_IND',
                        'CLOSE_INTERVAL_SURVEY_IND', 'EXTERNALLY_COATED', 'PRIOR_DAMAGE']

CATEGORICAL = ['PIPE_SPECIFICATION', 'PIPE_SEAM_TYPE', 'PIPE_COATING_TYPE', 'PIPE_TYPE']

# 冻结的训练预算（沿用历史 GeOAD 口径，不再调参）
CB_ITERATIONS = 500
CB_LR = 0.1
CB_DEPTH = 6
CB_SEEDS = (42, 123, 456)
GEOM_SEEDS = (42, 123, 456)
GEO_EPOCHS = 80
GEO_LR = 0.01
LAM_REG = 1e-3
MU_ALIGN = 1e-2
OOF_FOLDS = 3
THREADS = 16


# ------------------------------------------------------------------ 工具
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def log_open(name):
    os.makedirs(G2ART, exist_ok=True)
    path = os.path.join(G2ART, name)

    def log(msg):
        line = f'[{pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
        print(line, flush=True)
        with open(path, 'a') as f:
            f.write(line + '\n')
    return log


# ------------------------------------------------------------------ 数据装载
_CACHE = {}


def load_main():
    """装载主集（in_main_6class==1）全部 18 维特征 + 标签 + 两个协议的划分。"""
    if 'main' in _CACHE:
        return _CACHE['main']
    st = pd.read_csv(os.path.join(ROOT, 'source_trace.csv'), low_memory=False)
    st = st[st.in_main_6class == 1].reset_index(drop=True)
    st['label'] = st['label'].astype(str)
    st = st[st.label.isin(SIX_CLASSES)].reset_index(drop=True)
    st['y'] = st['label'].map({c: i for i, c in enumerate(SIX_CLASSES)}).astype(int)
    _CACHE['main'] = st
    return st


def feature_frame(df, spec):
    """按规格取特征子集，返回 (X_cb DataFrame, numeric 列名, categorical 列名)。"""
    num = [c for c in spec if c not in CATEGORICAL]
    cat = [c for c in spec if c in CATEGORICAL]
    X = df[spec].copy()
    for c in cat:
        X[c] = X[c].fillna('NOT_RECORDED').astype(str)
    for c in num:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    return X, num, cat


def fit_imputer(X_fit, num_cols):
    """数值中位数，只在 fit 行拟合。返回 dict col->median。"""
    med = {}
    for c in num_cols:
        v = X_fit[c]
        med[c] = float(v.median()) if v.notna().any() else 0.0
    return med


def apply_imputer(X, num_cols, med):
    X = X.copy()
    for c in num_cols:
        X[c] = X[c].fillna(med[c])
    return X


def build_views(X_fit, X_calib, X_test, num_cols, spec, med, scaler=None):
    """构造 P/F 的两路输入：
       proto = 数值视角（填补后 StandardScaler，只在 fit 拟合）
       rule  = 规则视角（4~5 维，按规格中存在的字段决定）
    返回 dict(proto={split:ndarray}, rule={split:ndarray}, scaler, med)"""
    raw = {}
    for k, v in [('fit', X_fit), ('calib', X_calib), ('test', X_test)]:
        raw[k] = v[num_cols].to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(raw['fit'])
    proto = {k: scaler.transform(v).astype(np.float32) for k, v in raw.items()}

    rules = []
    if 'PIPE_COATING_TYPE' in spec:
        rules.append(('coat', lambda d: d['PIPE_COATING_TYPE'].astype(str).str.upper()
                      .isin(['NONE', 'UNKNOWN', 'N/A']).to_numpy(dtype=float)))
    if 'PRIOR_DAMAGE' in spec:
        rules.append(('prior', lambda d: d['PRIOR_DAMAGE'].astype(str).str.upper()
                      .isin(['1', 'YES', 'TRUE', 'Y']).to_numpy(dtype=float)))
    if 'PIPE_AGE_AT_ACCIDENT' in spec:
        rules.append(('age', lambda d: (pd.to_numeric(d['PIPE_AGE_AT_ACCIDENT'],
                                                      errors='coerce') <= 10)
                      .fillna(False).to_numpy(dtype=float)))
    if 'MANUFACTURE_AGE_AT_ACCIDENT' in spec:
        rules.append(('mfg', lambda d: (pd.to_numeric(d['MANUFACTURE_AGE_AT_ACCIDENT'],
                                                       errors='coerce') <= 5)
                      .fillna(False).to_numpy(dtype=float)))
    if 'PRESSURE_RATIO' in spec:
        rules.append(('pr', lambda d: (pd.to_numeric(d['PRESSURE_RATIO'],
                                                      errors='coerce') > 1.2)
                      .fillna(False).to_numpy(dtype=float)))
    rule = {}
    for k, v in [('fit', X_fit), ('calib', X_calib), ('test', X_test)]:
        rule[k] = np.stack([f(v) for _, f in rules], axis=1).astype(np.float32)
    return dict(proto=proto, rule=rule, scaler=scaler, med=med,
                proto_dim=len(num_cols), rule_dim=len(rules))


# ------------------------------------------------------------------ CatBoost 骨干
def _catboost(X, y, cat_idx, seed, balanced=False, iterations=CB_ITERATIONS):
    from catboost import CatBoostClassifier
    kw = dict(iterations=iterations, learning_rate=CB_LR, depth=CB_DEPTH,
              random_seed=seed, verbose=0, cat_features=cat_idx,
              thread_count=THREADS, allow_writing_files=False)
    if balanced:
        kw['auto_class_weights'] = 'Balanced'
    m = CatBoostClassifier(**kw)
    m.fit(X, y)
    return m


def train_backbone(X_fit, y_fit, Xs, cat_idx, seeds=CB_SEEDS, balanced=False,
                   groups=None, oof_folds=OOF_FOLDS, log=None):
    """训练 CatBoost 骨干。

    返回 dict：
      models        : 全 fit 模型列表（3 种子）
      probs         : {split: (n,6)} 全 fit 模型在 calib/eval 的等权平均
      oof_fit       : (n_fit,6) fit 行的**组隔离 OOF** 概率（残差头用）
      class_weights : 平衡臂的类别权重（只由 fit 统计）
    """
    t0 = time.time()
    models = []
    for s in seeds:
        models.append(_catboost(X_fit, y_fit, cat_idx, s, balanced=balanced))
    probs = {}
    for k, Xq in Xs.items():
        p = np.zeros((len(Xq), len(SIX_CLASSES)))
        for m in models:
            p += m.predict_proba(Xq) / len(models)
        probs[k] = p

    # fit 行 OOF（组隔离）
    oof = np.zeros((len(X_fit), len(SIX_CLASSES)))
    if groups is not None:
        sgkf = StratifiedGroupKFold(n_splits=oof_folds, shuffle=True, random_state=42)
        for tr, te in sgkf.split(X_fit, y_fit, groups):
            oof[te] = np.mean([_catboost(X_fit.iloc[tr], y_fit[tr], cat_idx, s,
                                         balanced=balanced).predict_proba(X_fit.iloc[te])
                               for s in seeds], axis=0)
    else:
        for s in seeds:
            oof += models[seeds.index(s)].predict_proba(X_fit) / len(seeds)

    cnt = np.bincount(y_fit, minlength=len(SIX_CLASSES)).astype(float)
    w = len(y_fit) / (len(SIX_CLASSES) * np.maximum(cnt, 1))
    if log:
        log(f'  backbone balanced={balanced} seeds={seeds} '
            f'fit={len(y_fit)} 用时 {time.time() - t0:.1f}s 类别权重={np.round(w, 4).tolist()}')
    return dict(models=models, probs=probs, oof_fit=oof, class_weights=w,
                class_counts=cnt.tolist(), seconds=time.time() - t0)


# ------------------------------------------------------------------ F / P 头
class FResidual(nn.Module):
    """非几何残差头：logits = log(clamp(p_CB,1e-8)) + δ（与历史 F 完全同构）。"""

    def __init__(self, in_dim, hidden=(32,), out_dim=len(SIX_CLASSES)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(d, out_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feats, cb):
        z = self.body(feats)
        delta = self.out(z)
        return torch.log(cb.clamp_min(1e-8)) + delta, delta


class GeoNet(nn.Module):
    """几何头（与历史 GeOAD 同构）：Proto/Rule/Neural 三视角 + 距离注意力 + align。"""

    def __init__(self, proto_dim, rule_dim, emb_dim=32, n_classes=len(SIX_CLASSES)):
        super().__init__()
        self.W_proto = nn.Linear(proto_dim, emb_dim)
        self.W_rule = nn.Linear(rule_dim, emb_dim)
        self.W_neural = nn.Linear(n_classes, emb_dim)
        self.log_temp = nn.Parameter(torch.tensor([1.0]))
        self.W_delta = nn.Linear(emb_dim, n_classes)
        nn.init.zeros_(self.W_delta.weight)
        nn.init.zeros_(self.W_delta.bias)

    def forward(self, proto, rule, cb):
        p = self.W_proto(proto)
        r = self.W_rule(rule)
        h = self.W_neural(cb)
        t = torch.exp(self.log_temp) + 0.1
        ar = torch.exp(-torch.norm(p - r, dim=1, keepdim=True) / t)
        ah = torch.exp(-torch.norm(p - h, dim=1, keepdim=True) / t)
        total = 1.0 + ar + ah + 1e-8
        z = (p + ar * r + ah * h) / total
        delta = self.W_delta(z)
        return delta, z

    @staticmethod
    def align_loss(p, r, h):
        return (((p - r) ** 2).mean() + ((p - h) ** 2).mean() + ((r - h) ** 2).mean()) / 3


def _fit_loop(net, step_fn, calib_eval, epochs, lr):
    """检查点规则与 P 一致：calib Macro-F1 argmax（并列取最早）。"""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best = (-1.0, None, -1)
    for ep in range(epochs):
        net.train()
        opt.zero_grad()
        loss = step_fn()
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            f1 = calib_eval()
        if f1 > best[0]:
            best = (f1, {k: v.detach().clone() for k, v in net.state_dict().items()}, ep)
    net.load_state_dict(best[1])
    return net, best[0], best[2]


def train_f_arm(views, cb, y_fit, y_calib, seed, hidden=(32,)):
    """非几何残差臂：输入 = [proto ; rule ; 主干概率]，与 P 同一份输入。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    feats_fit = np.concatenate([views['proto']['fit'], views['rule']['fit'],
                                cb['oof_fit']], axis=1)
    feats_c = np.concatenate([views['proto']['calib'], views['rule']['calib'],
                              cb['probs']['calib']], axis=1)
    feats_t = np.concatenate([views['proto']['test'], views['rule']['test'],
                              cb['probs']['test']], axis=1)
    ff = torch.as_tensor(feats_fit, dtype=torch.float32)
    fc = torch.as_tensor(feats_c, dtype=torch.float32)
    ft = torch.as_tensor(feats_t, dtype=torch.float32)
    cbf = torch.as_tensor(cb['oof_fit'], dtype=torch.float32)
    cbc = torch.as_tensor(cb['probs']['calib'], dtype=torch.float32)
    cbt = torch.as_tensor(cb['probs']['test'], dtype=torch.float32)
    yf = torch.tensor(y_fit, dtype=torch.long)

    net = FResidual(ff.shape[1], hidden=hidden)

    def step_fn():
        logits, delta = net(ff, cbf)
        return F.cross_entropy(logits, yf) + LAM_REG * delta.square().mean()

    def calib_eval():
        with torch.no_grad():
            lc, _ = net(fc, cbc)
        return float(f1_score(y_calib, lc.argmax(1).numpy(), average='macro',
                              zero_division=0))

    net, f1, ep = _fit_loop(net, step_fn, calib_eval, GEO_EPOCHS, GEO_LR)
    net.eval()
    with torch.no_grad():
        pc, _ = net(fc, cbc)
        pt, _ = net(ft, cbt)
    return dict(probs_calib=F.softmax(pc, 1).numpy(), probs_test=F.softmax(pt, 1).numpy(),
                sel_epoch=ep, sel_f1=float(f1),
                n_params=int(sum(p.numel() for p in net.parameters())))


def train_p_arm(views, cb, y_fit, y_calib, seed):
    """几何臂 P（三视角 + 距离注意力 + align loss），预算与历史 GeOAD 完全一致。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    pf = torch.as_tensor(views['proto']['fit'], dtype=torch.float32)
    rf = torch.as_tensor(views['rule']['fit'], dtype=torch.float32)
    pc_ = torch.as_tensor(views['proto']['calib'], dtype=torch.float32)
    rc_ = torch.as_tensor(views['rule']['calib'], dtype=torch.float32)
    pt_ = torch.as_tensor(views['proto']['test'], dtype=torch.float32)
    rt_ = torch.as_tensor(views['rule']['test'], dtype=torch.float32)
    cbf = torch.as_tensor(cb['oof_fit'], dtype=torch.float32)
    cbc = torch.as_tensor(cb['probs']['calib'], dtype=torch.float32)
    cbt = torch.as_tensor(cb['probs']['test'], dtype=torch.float32)
    yf = torch.tensor(y_fit, dtype=torch.long)

    net = GeoNet(pf.shape[1], rf.shape[1])
    torch.nn.init.zeros_(net.W_delta.weight)
    torch.nn.init.zeros_(net.W_delta.bias)

    def step_fn():
        d, _ = net(pf, rf, cbf)
        p, r, h = net.W_proto(pf), net.W_rule(rf), net.W_neural(cbf)
        logits = torch.log(cbf.clamp_min(1e-8)) + d
        return (F.cross_entropy(logits, yf)
                + LAM_REG * d.square().mean()
                + MU_ALIGN * GeoNet.align_loss(p, r, h))

    def calib_eval():
        with torch.no_grad():
            d, _ = net(pc_, rc_, cbc)
            lc = torch.log(cbc.clamp_min(1e-8)) + d
        return float(f1_score(y_calib, lc.argmax(1).numpy(), average='macro',
                              zero_division=0))

    net, f1, ep = _fit_loop(net, step_fn, calib_eval, GEO_EPOCHS, GEO_LR)
    net.eval()
    out = {}
    with torch.no_grad():
        for nm, (pr, ru, cbp) in {'calib': (pc_, rc_, cbc),
                                  'test': (pt_, rt_, cbt)}.items():
            d, _ = net(pr, ru, cbp)
            out['probs_' + nm] = F.softmax(torch.log(cbp.clamp_min(1e-8)) + d, 1).numpy()
    return dict(probs_calib=out['probs_calib'], probs_test=out['probs_test'],
                sel_epoch=ep, sel_f1=float(f1),
                n_params=int(sum(p.numel() for p in net.parameters())))


# ------------------------------------------------------------------ 指标
def metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(6),
                                                 zero_division=0)
    return dict(acc=float(acc), macro_f1=float(macro), weighted_f1=float(weighted),
                per_class={SIX_CLASSES[i]: dict(precision=float(p[i]), recall=float(r[i]),
                                                f1=float(f[i]), support=int(s[i]))
                           for i in range(6)},
                minority_f1=float(np.mean([f[SIX_CLASSES.index(c)] for c in MINORITY])),
                confusion=confusion_matrix(y_true, y_pred, labels=range(6)).tolist())


def paired_corrections(y_true, pred_base, pred_new):
    """修正/破坏计数：以 base 为基准，new 修正了 base 的错误 / 破坏了 base 的正确。"""
    base_ok = (pred_base == y_true)
    new_ok = (pred_new == y_true)
    fixed = int((~base_ok & new_ok).sum())
    broken = int((base_ok & ~new_ok).sum())
    return dict(n=len(y_true), base_correct=int(base_ok.sum()), new_correct=int(new_ok.sum()),
                corrections=fixed, breakages=broken, net=fixed - broken,
                changed=int((pred_base != pred_new).sum()))


def mcnemar_p(b, c):
    """McNemar 精确二项检验 p 值（b/c 为不一致计数），无 scipy 依赖。"""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return float(min(1.0, 2 * tail))


def _group_confusion(y, pred, groups):
    """按组聚合的混淆矩阵 (n_groups, 6, 6) + 组键（用于快速 bootstrap）。"""
    y = np.asarray(y); pred = np.asarray(pred); groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    out = np.zeros((len(uniq), 6, 6), dtype=np.int64)
    np.add.at(out, (inv, y, pred), 1)
    return uniq, out


def _f1_from_conf(conf):
    """由 (6,6) 混淆矩阵算 accuracy / macro-F1（与 sklearn 口径一致：全 6 类平均，
    分母为 0 的类 F1 记 0）。"""
    tp = np.diag(conf).astype(float)
    support = conf.sum(axis=1).astype(float)
    pred_pos = conf.sum(axis=0).astype(float)
    prec = np.where(pred_pos > 0, tp / np.maximum(pred_pos, 1), 0.0)
    rec = np.where(support > 0, tp / np.maximum(support, 1), 0.0)
    den = prec + rec
    f1 = np.where(den > 0, 2 * prec * rec / np.maximum(den, 1e-12), 0.0)
    n = conf.sum()
    acc = tp.sum() / n if n else 0.0
    return float(acc), float(f1.mean())


def group_bootstrap_ci(y_true, pred, groups, n_boot=2000, seed=42, stat='acc'):
    """按组（事故/模式组）重抽样的 bootstrap CI（对组混淆矩阵做聚合，精确且快）。"""
    rng = np.random.default_rng(seed)
    _, conf = _group_confusion(y_true, pred, groups)
    n_groups = conf.shape[0]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_groups, n_groups)
        acc, f1 = _f1_from_conf(conf[pick].sum(axis=0))
        vals.append(acc if stat == 'acc' else f1)
    return dict(lo=float(np.percentile(vals, 2.5)), hi=float(np.percentile(vals, 97.5)),
                n_boot=n_boot)


def paired_group_bootstrap(y_true, pred_a, pred_b, groups, n_boot=2000, seed=42):
    """配对差值（P−A 的 Macro-F1 差）的组 bootstrap CI。"""
    rng = np.random.default_rng(seed)
    _, ca = _group_confusion(y_true, pred_a, groups)
    _, cb = _group_confusion(y_true, pred_b, groups)
    n_groups = ca.shape[0]
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_groups, n_groups)
        _, fa = _f1_from_conf(ca[pick].sum(axis=0))
        _, fb = _f1_from_conf(cb[pick].sum(axis=0))
        diffs.append(fb - fa)
    diffs = np.array(diffs)
    return dict(lo=float(np.percentile(diffs, 2.5)), hi=float(np.percentile(diffs, 97.5)),
                mean=float(diffs.mean()), n_boot=n_boot)
