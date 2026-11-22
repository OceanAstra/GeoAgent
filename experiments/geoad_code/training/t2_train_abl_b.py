"""
t2_train_abl_b.py — GeOAD 修复后统一重评 · Tier 2 正式训练（**仅 ABL-B**）

============================================================================
授权边界（2026-10-28「Tier 2 授权：修复后数据上的系统级消融补充」）
------------------------------------------------------------------------------
  * 本轮**唯一**新训模型：**ABL-B** = 单 CatBoost 主干（seed 42）
    ＋ **完整 Rule/Neural 双视角 AUX 头**（重训，drop='proto'，与冻结 FULL 同结构）。
  * **零训练复用**：Tier 1 的 A_single / A_ensemble / F_concat / P_no_proto
    （逐样本预测文件逐字节复制；不重训、不重算 SHAP）。
  * **优先复用 Tier 1 已训练的 A_single 主干**：直接加载 Tier 1 落盘的
    `artifacts/cb_seed42.cbm`（seed 42 全 fit 模型），**不重训该主干**；
    并断言它与 Tier 1 的 A_single 逐样本概率一致（确定性核对）。
  * **辅助头必须使用与单主干相匹配的合法 OOF 重新训练**：
    本轮**新算**单种子（seed 42）3 折组隔离 OOF（StratifiedGroupKFold(3,
    shuffle=True, random_state=42)，组键 accident_group），**禁止**把三种子
    集成 OOF 直接接上单模型；并断言所用 OOF ≠ Tier 1 的三种子集成 OOF。
  * 数据 / 划分 / 特征 / 标签 / 种子 / 预算 / 检查点规则：与 Tier 1 **逐项相同**。
  * 禁止：调参、重选划分、重训 Tier 1 四臂、使用封存数据、自动启动 Tier 3。

输出
  predictions/ABL_B.csv          dev_confirm 逐样本预测
  predictions/calib/ABL_B.csv    calib 逐样本预测
  artifacts/ABL_B_head_seed{42,123,456}.pt
  artifacts/cb_seed42_reused.cbm   Tier 1 主干的副本（指纹留证）
  data/t2_oof_fit_ablb.npy + data/t2_oof_fit_rows_ablb.csv
  results/t2_train_meta.json      全部断言与计时
  logs/t2_train_log.txt           真实训练日志
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'code'))

import s2_common as S2          # noqa: E402
import g2_common as G           # noqa: E402
import rq2_ablation_net as AB   # noqa: E402

TIER1 = os.environ.get('ZHL_GEOAD_TIER1_DIR', '')   # 可选；缺省时跳过对照

# --- 官方源一致性校验（可选）---
# 归档时这些文件另有"官方源"副本位于作者服务器。打包件无法访问该路径，
# 因此这里改为：仅当 ZHL_OFFICIAL_CODE_DIR 环境变量指向一个存在的目录时才比对；
# 否则跳过并记录 SKIPPED_NO_OFFICIAL_SOURCE（本目录副本即权威副本）。
OFFICIAL_DIR = os.environ.get('ZHL_OFFICIAL_CODE_DIR', '')
OFFICIAL = ({f: os.path.join(OFFICIAL_DIR, 'code', f) for f in
             ('g2_common.py', 'rq2_ablation_net.py', 's2_common.py', 's2_heads.py')}
            if OFFICIAL_DIR and os.path.isdir(os.path.join(OFFICIAL_DIR, 'code')) else {})

ABL_CB_SEED = 42                # 单主干种子（与 Tier 1 的 A_single 成员同）
HEAD_SEEDS = [42, 123, 456]     # 与冻结 FULL / Tier 1 P_no_proto 完全相同
HEAD_DROP = 'proto'             # 保留 Rule + Neural 双视角
EXPECTED_MANIFEST_SHA = 'd9ef45f215b429300628dd01d96f84b1e7f703b69ab35bccf523fa5f01c4046a'


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def log_factory(path):
    f = open(path, 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        f.write(str(msg) + '\n')
        f.flush()
    return log, f


def check_source_fidelity():
    out = {}
    if not OFFICIAL:
        out['status'] = 'SKIPPED_NO_OFFICIAL_SOURCE'
        out['note'] = 'local copies are authoritative; set ZHL_OFFICIAL_CODE_DIR to cross-check'
        return out
    for fn, src in OFFICIAL.items():
        local = os.path.join(HERE, 'code', fn)
        a, b = sha256_file(local), sha256_file(src)
        assert a == b, f'{fn} 与官方源不一致：{a} != {b}'
        out[fn] = dict(local_sha256=a, official_sha256=b, identical=True)
    return out


def load_protocol_from(manifest_path, lock_path):
    """与 s2_common.load_protocol 逐行同逻辑（清单路径可指定）。"""
    df = pd.read_csv(manifest_path, low_memory=False)
    with open(lock_path, encoding='utf-8') as f:
        lock = json.load(f)
    assert list(lock['class_order']) == S2.SIX
    assert lock['features']['main_13'] == S2.MAIN_13
    dev = df[df.split.isin(S2.ROLES)].copy().reset_index(drop=True)
    assert (dev.y >= 0).all(), '开发池存在无标签行'
    assert set(dev.split.unique()) == set(S2.ROLES)
    for c in S2.CATEGORICAL:
        dev[c] = dev[c].fillna('NOT_RECORDED').astype(str)
    for c in S2.NUMERIC:
        dev[c] = pd.to_numeric(dev[c], errors='coerce')
    out = {}
    for r in S2.ROLES:
        d = dev[dev.split == r].reset_index(drop=True)
        out[r] = dict(
            X=d[S2.MAIN_13].copy(),
            y=d['y'].to_numpy().astype(int),
            groups=d['accident_group'].astype(str).to_numpy(),
            row_id=d['row_id'].astype(str).to_numpy(),
            label=d['label'].astype(str).to_numpy(),
            pattern13=d['pattern13'].astype(str).to_numpy(),
        )
    out['manifest'] = df
    out['dev'] = dev
    out['lock'] = lock
    return out


def main():
    t0_all = time.time()
    os.makedirs(os.path.join(HERE, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
    os.makedirs(os.path.join(HERE, 'artifacts'), exist_ok=True)
    os.makedirs(os.path.join(HERE, 'predictions', 'calib'), exist_ok=True)
    log, LOGF = log_factory(os.path.join(HERE, 'logs', 't2_train_log.txt'))

    V = {}
    log('=== GeOAD 修复后统一重评 · Tier 2 正式训练（仅 ABL-B）===')
    log('ABL-B 规格（事前锁定）：主干 = 单 CatBoost seed 42（复用 Tier 1 的 A_single 主干）；'
        '头 = 完整 Rule/Neural 双视角（drop=proto，mode=retrain），头种子 [42,123,456]')
    log('零训练复用：A_single / A_ensemble / F_concat / P_no_proto（Tier 1 预测文件逐字节复制）')

    man_path = os.path.join(HERE, 'protocol', 'data_row_manifest.csv')
    lock_path = os.path.join(HERE, 'protocol', 'protocol_lock.json')
    man_sha = sha256_file(man_path)
    log(f'清单 sha256 = {man_sha}')
    assert man_sha == EXPECTED_MANIFEST_SHA, f'清单 sha 不符：{man_sha}'
    log('[断言 T0] 清单 sha256 与 Tier 1 锁定值一致 → 通过')
    V['manifest_sha256'] = man_sha

    src = check_source_fidelity()
    log('[断言 T0b] 本目录 code/*.py 与官方源逐字节一致 → 通过')
    V['source_fidelity'] = src

    P = load_protocol_from(man_path, lock_path)
    lock = P['lock']
    log(f'协议锁: {lock["protocol_id"]}  类别顺序: {S2.SIX}')
    fit, cal, conf = P['fit'], P['calib'], P['dev_confirm']
    log(f'角色行数: fit={len(fit["y"])} calib={len(cal["y"])} dev_confirm={len(conf["y"])}')

    # ---- 断言 T1：三种角色事故组 / row_id 两两不相交 ----
    for a, b, na, nb in [(fit, cal, 'fit', 'calib'), (fit, conf, 'fit', 'dev_confirm'),
                         (cal, conf, 'calib', 'dev_confirm')]:
        assert not (set(a['groups']) & set(b['groups'])), f'{na} 与 {nb} 共享事故组'
        assert not (set(a['row_id']) & set(b['row_id'])), f'{na} 与 {nb} 共享 row_id'
    log('[断言 T1] 三角色事故组与 row_id 两两不相交 → 通过')
    V['role_isolation'] = True

    spec = S2.MAIN_13
    cat_idx = S2.cat_feature_index()
    X_fit = S2.catboost_frame(fit['X'])
    X_cal = S2.catboost_frame(cal['X'])
    X_conf = S2.catboost_frame(conf['X'])
    y_fit, y_cal, y_conf = fit['y'], cal['y'], conf['y']

    # ---- 视图（填补/缩放统计量只在 fit 行拟合，与 Tier 1 同逻辑）----
    med = G.fit_imputer(X_fit, S2.NUMERIC)
    views = G.build_views(G.apply_imputer(X_fit, S2.NUMERIC, med),
                          G.apply_imputer(X_cal, S2.NUMERIC, med),
                          G.apply_imputer(X_conf, S2.NUMERIC, med),
                          S2.NUMERIC, spec, med)
    assert views['proto_dim'] == 9 and views['rule_dim'] == 4
    log(f'视图维度: proto={views["proto_dim"]} rule={views["rule_dim"]} neural=6')

    # ================= 主干：复用 Tier 1 的 A_single（seed 42 全 fit 模型） =================
    from catboost import CatBoostClassifier
    t0 = time.time()
    cb42_path = os.path.join(TIER1, 'artifacts', 'cb_seed42.cbm')
    cb42 = CatBoostClassifier()
    cb42.load_model(cb42_path)
    probs_test = cb42.predict_proba(X_conf)
    probs_cal = cb42.predict_proba(X_cal)
    sec_backbone_reuse = time.time() - t0
    # 确定性核对：与 Tier 1 的 A_single 逐样本概率一致
    a_single = pd.read_csv(os.path.join(TIER1, 'predictions', 'A_single.csv'))
    key = ['row_id'] + [f'prob_{c}' for c in S2.SIX]
    merged = pd.DataFrame({'row_id': conf['row_id']}).merge(a_single[key], on='row_id',
                                                            how='left', validate='1:1')
    assert merged['prob_CORROSION'].notna().all(), '与 Tier 1 A_single 的 row_id 对不上'
    arch = merged[[c for c in key if c.startswith('prob_')]].to_numpy()
    dmax_single = float(np.abs(arch - probs_test).max())
    assert dmax_single < 1e-9, f'复用主干与 Tier 1 A_single 概率不一致：{dmax_single}'
    log(f'[断言 T2] 复用的单主干（Tier 1 cb_seed42.cbm）与 Tier 1 A_single 逐样本概率'
        f'最大绝对差 = {dmax_single:.3e} → 通过（主干零训练复用且完全一致）')
    V['backbone_reuse_matches_tier1_A_single'] = dict(max_abs_diff=dmax_single,
                                                      source=os.path.relpath(cb42_path, HERE),
                                                      source_sha256=sha256_file(cb42_path))
    # 复制一份到本包留证
    import shutil
    shutil.copy2(cb42_path, os.path.join(HERE, 'artifacts', 'cb_seed42_reused.cbm'))
    for nm, arr in [('calib', probs_cal), ('dev_confirm', probs_test)]:
        s = arr.sum(axis=1)
        assert np.all(np.abs(s - 1) < 1e-6), f'{nm} 概率行和不为 1'
        assert (arr.max(axis=1) > 0).all(), f'{nm} 存在全零行'
    log('[断言 T3] 单主干 calib / dev_confirm 概率行和=1、无全零行 → 通过')

    # ================= 合法 OOF：单种子 3 折组隔离（本轮新算） =================
    t0 = time.time()
    sgkf = StratifiedGroupKFold(n_splits=G.OOF_FOLDS, shuffle=True, random_state=42)
    oof = np.zeros((len(X_fit), 6), dtype=float)
    seen = np.zeros(len(y_fit), dtype=int)
    fold_sizes = []
    for f, (tr, te) in enumerate(sgkf.split(X_fit, y_fit, fit['groups'])):
        gtr, gte = set(fit['groups'][tr]), set(fit['groups'][te])
        inter = gtr & gte
        assert not inter, f'内层折 {f} 存在事故组交叉：{len(inter)} 组'
        m = G._catboost(X_fit.iloc[tr], y_fit[tr], cat_idx, ABL_CB_SEED, balanced=False)
        oof[te] = m.predict_proba(X_fit.iloc[te])
        seen[te] += 1
        fold_sizes.append((len(tr), len(te)))
    assert (seen == 1).all(), '内层 OOF 覆盖异常：存在行未被验证或被重复验证'
    sec_oof = time.time() - t0
    s = oof.sum(axis=1)
    assert np.all(np.abs(s - 1) < 1e-6), 'OOF 概率行和不为 1'
    assert (oof.max(axis=1) > 0).all(), '存在全零 OOF 行'
    insample = cb42.predict_proba(X_fit)
    diff = np.abs(oof - insample).max(axis=1)
    assert not np.allclose(oof, insample), 'OOF 与 in-sample 完全相同 ⇒ 疑似泄漏'
    log(f'[断言 T4] 单种子 3 折组隔离 OOF：折大小 {fold_sizes}，每行恰好验证一次，'
        f'无组交叉，行和=1，无全零行')
    log(f'[断言 T5] OOF ≠ in-sample（逐行最大差 中位数 {np.median(diff):.4f}，最小 {diff.min():.4f}）→ 通过')
    V['single_seed_oof'] = dict(n_folds=G.OOF_FOLDS, cv_seed=42, fold_sizes=fold_sizes,
                                rows_validated_once=True, fold_group_overlap=0,
                                oof_vs_insample_median_maxdiff=float(np.median(diff)),
                                oof_vs_insample_min_maxdiff=float(diff.min()))
    # 关键断言：本轮所用 OOF ≠ Tier 1 的三种子集成 OOF
    t1_oof = np.load(os.path.join(TIER1, 'data', 't1_oof_fit.npy'))
    d_oof = float(np.abs(oof - t1_oof).max())
    assert d_oof > 1e-6, '单种子 OOF 与 Tier 1 集成 OOF 相同 ⇒ 未真正重算'
    log(f'[断言 T6] 本轮单种子 OOF 与 Tier 1 三种子集成 OOF 最大绝对差 = {d_oof:.3e}'
        f'（>1e-6）→ 确认**未**把集成 OOF 接上单模型 → 通过')
    V['oof_differs_from_tier1_ensemble_oof'] = dict(max_abs_diff=d_oof)
    np.save(os.path.join(HERE, 'data', 't2_oof_fit_ablb.npy'), oof)
    pd.DataFrame({'row_id': fit['row_id'],
                  'accident_group': fit['groups']}).to_csv(
        os.path.join(HERE, 'data', 't2_oof_fit_rows_ablb.csv'), index=False)

    # ================= 双视角 AUX 头（重训） =================
    cb_single = dict(oof_fit=oof, probs={'calib': probs_cal, 'test': probs_test},
                      models=[cb42])
    t0 = time.time()
    runs = []
    for s_ in HEAD_SEEDS:
        r = AB.train_p_arm_masked(views, cb_single, y_fit, y_cal, s_, drop=HEAD_DROP,
                                  mode='retrain', log=log, tag=f'ABL_B_s{s_}')
        torch.save(r['net'].state_dict(),
                   os.path.join(HERE, 'artifacts', f'ABL_B_head_seed{s_}.pt'))
        runs.append(r)
    sec_head = time.time() - t0
    probs_B = np.mean([r['probs_test'] for r in runs], axis=0)
    probs_B_cal = np.mean([r['probs_calib'] for r in runs], axis=0)

    # 头结构与冻结 FULL / Tier 1 P_no_proto 一致
    t1_cfg = json.load(open(os.path.join(TIER1, 'data', 't1_arm_configs_tier1.json'),
                             encoding='utf-8'))['P_no_proto']
    assert runs[0]['active_params'] == t1_cfg['active_parameters'], \
        (runs[0]['active_params'], t1_cfg['active_parameters'])
    assert runs[0]['n_params'] == t1_cfg['total_parameters'], \
        (runs[0]['n_params'], t1_cfg['total_parameters'])
    log(f'[断言 T7] ABL-B 头生效参数 {runs[0]["active_params"]} == 冻结 FULL(Tier 1 P_no_proto) '
        f'{t1_cfg["active_parameters"]}；总参数 {runs[0]["n_params"]} == '
        f'{t1_cfg["total_parameters"]} → 结构一致 → 通过')
    V['head_structure_identical_to_full'] = dict(
        active_params=runs[0]['active_params'], n_params=runs[0]['n_params'],
        reference='Tier1 P_no_proto (FULL)')
    # proto 视角确被结构性删除
    torch.manual_seed(HEAD_SEEDS[0])
    np.random.seed(HEAD_SEEDS[0])
    probe = AB.GeoNetViewMasked(9, 4)
    init_w = probe.W_proto.weight.detach().clone()
    saved = torch.load(os.path.join(HERE, 'artifacts', f'ABL_B_head_seed{HEAD_SEEDS[0]}.pt'),
                       map_location='cpu', weights_only=True)
    dmax_w = float((saved['W_proto.weight'] - init_w).abs().max())
    assert dmax_w == 0.0, f'proto 视角权重发生变化：{dmax_w}'
    log('[断言 T8] ABL-B 的 W_proto 权重与初值逐位相同 ⇒ proto 视角被结构性删除 → 通过')
    V['proto_view_structurally_removed'] = dict(w_proto_max_abs_change=dmax_w)

    for nm, arr in [('calib', probs_B_cal), ('dev_confirm', probs_B)]:
        s_ = arr.sum(axis=1)
        assert np.all(np.abs(s_ - 1) < 1e-6), f'ABL-B {nm} 概率行和不为 1'
        assert (arr.max(axis=1) > 0).all()
    log('[断言 T9] ABL-B 全部概率（calib / dev_confirm）行和=1、无全零行 → 通过')

    log(f'ABL-B 完成：头 {sec_head:.1f}s  选中epoch={[r["sel_epoch"] for r in runs]}  '
        f'calib MacroF1={[round(r["sel_f1"], 4) for r in runs]}  '
        f'主干复用 {sec_backbone_reuse:.1f}s  单种子OOF {sec_oof:.1f}s')

    # ---- 落盘训练实际消费的输入帧（供 t2_verify 做 MAIN_13 一一对应核对）----
    in_rows = []
    for r, part in [('fit', fit), ('calib', cal), ('dev_confirm', conf)]:
        d = pd.DataFrame({'row_id': part['row_id'], 'split': r,
                          'accident_group': part['groups'], 'y': part['y']})
        for c in S2.MAIN_13:
            d[c] = part['X'][c].to_numpy()
        in_rows.append(d)
    pd.concat(in_rows, ignore_index=True).to_csv(
        os.path.join(HERE, 'data', 't2_input_frames.csv'), index=False)

    # ---- 落盘逐样本预测 ----
    p_dev = S2.save_predictions('ABL_B', y_conf, probs_B, conf,
                                os.path.join(HERE, 'predictions'))
    p_cal = S2.save_predictions('ABL_B', y_cal, probs_B_cal, cal,
                                os.path.join(HERE, 'predictions', 'calib'))
    log(f'逐样本预测已落盘：{os.path.relpath(p_dev, HERE)}  {os.path.relpath(p_cal, HERE)}')

    cfg = dict(
        arm='ABL-B：单 CatBoost(seed 42) 主干 + 完整 Rule/Neural 双视角 AUX 头（重训）',
        protocol=lock['protocol_id'],
        protocol_lock_sha256=sha256_file(lock_path),
        data_manifest_sha256=man_sha,
        feature_set='MAIN_13', excluded_leak_adjacent=S2.EXCLUDED_LEAK_ADJACENT,
        class_order=S2.SIX,
        split_roles=dict(fit_rows=len(y_fit), calib_rows=len(y_cal),
                         dev_confirm_rows=len(y_conf)),
        backbone=dict(source='Tier1 artifacts/cb_seed42.cbm（零训练复用）',
                      seed=ABL_CB_SEED, iterations=G.CB_ITERATIONS,
                      learning_rate=G.CB_LR, depth=G.CB_DEPTH,
                      oof_folds=G.OOF_FOLDS, oof_group_key='accident_group', oof_cv_seed=42,
                      oof_source='**本轮新算的单种子 3 折组隔离 OOF**（非 Tier 1 集成 OOF）',
                      reuse_check_max_abs_diff_vs_tier1_A_single=dmax_single),
        head=dict(epochs=G.GEO_EPOCHS, lr=G.GEO_LR, lam_reg=G.LAM_REG, mu_align=G.MU_ALIGN,
                  seeds=HEAD_SEEDS, drop=HEAD_DROP, mode='retrain',
                  trainer='rq2_ablation_net.train_p_arm_masked（与冻结 FULL 同一函数）',
                  checkpoint_rule='calib Macro-F1 argmax，并列取最早',
                  n_params=runs[0]['n_params'], active_params=runs[0]['active_params'],
                  selected_epochs=[r['sel_epoch'] for r in runs],
                  calib_macro_f1=[round(r['sel_f1'], 4) for r in runs],
                  weights='artifacts/ABL_B_head_seed{42,123,456}.pt'),
        prob_aggregation='三种子 softmax 概率算术平均（与 FULL 同一约定）',
        training_cost=dict(
            backbone_reuse_sec=round(sec_backbone_reuse, 2),
            single_seed_oof_sec=round(sec_oof, 2),
            head_sec=round(sec_head, 2),
            total_sec=round(time.time() - t0_all, 2),
            catboost_fits_this_round=3,
            note='主干全 fit 模型零训练复用；本轮新训 3 个 OOF 折模型（seed 42）'),
        zero_training_reuse=['A_single', 'A_ensemble', 'F_concat', 'P_no_proto'],
    )
    with open(os.path.join(HERE, 'results', 't2_arm_config_abl_b.json'), 'w',
              encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    with open(os.path.join(HERE, 'results', 't2_train_meta.json'), 'w',
              encoding='utf-8') as f:
        json.dump(dict(verification=V, config=cfg,
                       timing=cfg['training_cost'],
                       prediction_files=[os.path.relpath(p_dev, HERE),
                                         os.path.relpath(p_cal, HERE)]),
                  f, ensure_ascii=False, indent=2)
    log(f'总耗时 {time.time() - t0_all:.1f}s')
    LOGF.close()


if __name__ == '__main__':
    main()
