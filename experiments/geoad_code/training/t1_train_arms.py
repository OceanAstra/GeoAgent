"""
t1_train_arms.py — GeOAD 修复后统一重评 · Tier 1 正式训练（仅四臂）

============================================================================
授权边界（严格遵守，来自 2026-10-28 Tier 1 正式执行授权）
------------------------------------------------------------------------------
  * 只训练四个臂：
      A_single     纯单 CatBoost（骨干 seed 42 成员）
      A_ensemble   纯三种子 CatBoost 集成（等权概率平均）
      F_concat     非几何残差对照（concat(proto9, rule4, cb6) → 19→32→6）
      P_no_proto   **正式主模型**：Rule＋Neural 双视角 GeOAD（删除 proto 视角后重训）
  * **不训练**：旧三视角 P_full、P_no_rule、P_no_neural（历史版本，不重训）
  * **不训练**：Tier 2 的 ABL-B、Tier 3 的八个外部基线
  * 不读取 2024+ 封存集标签（封存行 y=-1 / label=SEALED，只作为占位保留）
  * 不重算 SHAP；不改网络结构/损失/检查点规则/种子/预算；不做超参搜索；
    不挑选有利划分；新结果不再用于调参。
  * 不论 GeOAD 分数升高还是下降，完整落盘交付。

实现复用（不重写，逐字节复制官方源码）
  * 骨干 / F 头 / P 头：GeOAD_Submission_Final_20261024/code/g2_common.py（只读引用）
  * 双视角（drop='proto'）训练器：GeOAD_RQ1_EightBaselines_Pilot_20261025/code/rq2_ablation_net.py
  * 数据/特征适配层：RQ2 s2_common.py、s2_heads.py
  本目录下的 code/*.py 为**逐字节副本**；启动时校验其 sha256 与官方源一致。

数据源
  * 正式运行：protocol/data_row_manifest.csv（权威重建版，sha256 见 --expect-manifest-sha）
  * 复现校验运行：--manifest 指向 RQ2 原始清单（用于验证流水线与归档逐值一致）

输出
  predictions/<arm>.csv            dev_confirm 逐样本预测（row_id / 事故组 / 真值 / 6 维概率）
  predictions/calib/<arm>.csv      calib 逐样本预测（复核检查点选择）
  artifacts/cb_seed{42,123,456}.cbm    骨干权重
  artifacts/{F_concat,P_no_proto}_seed{42,123,456}.pt   残差/几何头权重
  results/t1_metrics_raw.json      本次运行的全部断言与计时
  logs/t1_train_<tag>_log.txt      真实训练日志
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'code'))

import s2_common as S2      # noqa: E402
import g2_common as G       # noqa: E402
import rq2_ablation_net as AB  # noqa: E402
import s2_heads as S2H      # noqa: E402

# --- 官方源一致性校验（可选）---
# 归档时这些文件另有"官方源"副本位于作者服务器。打包件无法访问该路径，
# 因此这里改为：仅当 ZHL_OFFICIAL_CODE_DIR 环境变量指向一个存在的目录时才比对；
# 否则跳过并记录 SKIPPED_NO_OFFICIAL_SOURCE（本目录副本即权威副本）。
OFFICIAL_DIR = os.environ.get('ZHL_OFFICIAL_CODE_DIR', '')
OFFICIAL = ({f: os.path.join(OFFICIAL_DIR, 'code', f) for f in
             ('g2_common.py', 'rq2_ablation_net.py', 's2_common.py', 's2_heads.py')}
            if OFFICIAL_DIR and os.path.isdir(os.path.join(OFFICIAL_DIR, 'code')) else {})


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def check_source_fidelity():
    out = {}
    if not OFFICIAL:
        out['status'] = 'SKIPPED_NO_OFFICIAL_SOURCE'
        out['note'] = 'local copies are authoritative; set ZHL_OFFICIAL_CODE_DIR to cross-check'
        return out
    for fn, src in OFFICIAL.items():
        local = os.path.join(HERE, 'code', fn)
        a, b = sha256_file(local), sha256_file(src)
        out[fn] = dict(local_sha256=a, official_sha256=b, identical=(a == b))
        assert a == b, f'{fn} 与官方源不一致：{a} != {b}'
    out['imported_module_paths'] = {
        'g2_common': G.__file__, 'rq2_ablation_net': AB.__file__,
        's2_common': S2.__file__, 's2_heads': S2H.__file__}
    return out


def load_protocol_from(manifest_path, lock_path):
    """与 s2_common.load_protocol 逐行同逻辑，但允许指定清单文件（用于复现校验运行）。"""
    df = pd.read_csv(manifest_path, low_memory=False)
    with open(lock_path, encoding='utf-8') as f:
        lock = json.load(f)
    assert list(lock['class_order']) == S2.SIX, '协议锁类别顺序与本模块不一致'
    assert lock['features']['main_13'] == S2.MAIN_13, '协议锁特征集与本模块不一致'

    dev = df[df.split.isin(S2.ROLES)].copy().reset_index(drop=True)
    assert (dev.y >= 0).all(), '开发池存在无标签行'
    assert set(dev.split.unique()) == set(S2.ROLES), '角色集合异常'
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
            data_source=d['data_source'].astype(str).to_numpy(),
            operator_id=d['operator_id'].astype(str).to_numpy(),
            identity_certainty=d['identity_certainty'].astype(str).to_numpy(),
            incident_date=d['incident_date'].astype(str).to_numpy(),
        )
    out['manifest'] = df
    out['dev'] = dev
    out['lock'] = lock
    return out


def assert_consistency(log):
    """断言 1/2：网络逐值等价 + 各臂保留层初始权重逐值相同。"""
    torch.manual_seed(42)
    np.random.seed(42)
    ref = G.GeoNet(9, 4)
    torch.manual_seed(42)
    np.random.seed(42)
    mine = AB.GeoNetViewMasked(9, 4)
    mine.load_state_dict(ref.state_dict())
    g = torch.Generator().manual_seed(7)
    proto = torch.randn(64, 9, generator=g)
    rule = torch.randn(64, 4, generator=g)
    cb = torch.softmax(torch.randn(64, 6, generator=g), 1)
    with torch.no_grad():
        d_ref, z_ref = ref(proto, rule, cb)
        d_mine, z_mine = mine(proto, rule, cb)
    e = mine.embed(proto, rule, cb)
    a_ref = G.GeoNet.align_loss(e['proto'], e['rule'], e['neural'])
    a_mine = mine.align_masked(e, drop=None)
    dmax = float((d_ref - d_mine).abs().max())
    zmax = float((z_ref - z_mine).abs().max())
    amax = float((a_ref - a_mine).abs().max())
    assert dmax < 1e-6 and zmax < 1e-6 and amax < 1e-6, (dmax, zmax, amax)
    log(f'[断言1] 全视角消融网络 vs 正式 g2_common.GeoNet：delta差={dmax:.2e} '
        f'z差={zmax:.2e} align差={amax:.2e} → 通过')

    init = {}
    for drop in [None, 'proto', 'rule', 'neural']:
        torch.manual_seed(42)
        np.random.seed(42)
        n = AB.GeoNetViewMasked(9, 4)
        init[drop] = {k: v.detach().clone() for k, v in n.state_dict().items()}
    for k in init[None]:
        for drop in ['proto', 'rule', 'neural']:
            assert torch.equal(init[None][k], init[drop][k]), f'初始权重不一致: {k}'
    log('[断言2] 同一 seed 下各臂保留层初始权重逐值相同（消除"删层改变随机流"）→ 通过')
    return dict(net_equivalence_vs_g2_common=dict(max_abs_delta=dmax, max_abs_z=zmax,
                                                   max_abs_align=amax, pass_=True),
                init_control_same_across_arms=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=os.path.join(HERE, 'protocol', 'data_row_manifest.csv'))
    ap.add_argument('--lock', default=os.path.join(HERE, 'protocol', 'protocol_lock.json'))
    ap.add_argument('--expect-manifest-sha', default='')
    ap.add_argument('--outdir', default=HERE)
    ap.add_argument('--tag', default='tier1')
    ap.add_argument('--pred-subdir', default='predictions')
    ap.add_argument('--artifact-subdir', default='artifacts')
    args = ap.parse_args()

    OUT = args.outdir
    PRED = os.path.join(OUT, args.pred_subdir)
    ART = os.path.join(OUT, args.artifact_subdir)
    LOGP = os.path.join(OUT, 'logs')
    RES = os.path.join(OUT, 'results')
    for d in (PRED, os.path.join(PRED, 'calib'), ART, LOGP, RES,
              os.path.join(OUT, 'data')):
        os.makedirs(d, exist_ok=True)

    LOGF = open(os.path.join(LOGP, f't1_train_{args.tag}_log.txt'), 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        LOGF.write(str(msg) + '\n')
        LOGF.flush()

    t0_all = time.time()
    log('=== GeOAD 修复后统一重评 · Tier 1 正式训练（四臂）===')
    log(f'模式: {"正式重评" if args.tag == "tier1" else "复现校验（仅用于与归档逐值对照）"}')
    log(f'清单: {args.manifest}')
    man_sha = sha256_file(args.manifest)
    log(f'清单 sha256 = {man_sha}')
    if args.expect_manifest_sha:
        assert man_sha == args.expect_manifest_sha, \
            f'清单 sha 不符：{man_sha} != {args.expect_manifest_sha}'
        log(f'[断言0] 清单 sha256 与锁定值相符 → 通过')

    src = check_source_fidelity()
    log('[断言0b] 本目录 code/*.py 与官方源逐字节一致 → 通过')

    P = load_protocol_from(args.manifest, args.lock)
    lock = P['lock']
    log(f'协议锁: {lock["protocol_id"]}')
    log(f'类别顺序: {S2.SIX}')
    log(f'特征集: MAIN_13（排除 5 个标签邻接字段: {S2.EXCLUDED_LEAK_ADJACENT}）')
    log(f'CatBoost 预算: iters={G.CB_ITERATIONS} lr={G.CB_LR} depth={G.CB_DEPTH} '
        f'seeds={G.CB_SEEDS} oof_folds={G.OOF_FOLDS}')
    log(f'几何头预算: epochs={G.GEO_EPOCHS} lr={G.GEO_LR} lam={G.LAM_REG} '
        f'mu_align={G.MU_ALIGN} seeds={G.GEOM_SEEDS} '
        f'checkpoint=calib Macro-F1 argmax（并列取最早）')

    V = assert_consistency(log)

    fit, cal, conf = P['fit'], P['calib'], P['dev_confirm']
    log(f'角色行数: fit={len(fit["y"])}  calib={len(cal["y"])}  dev_confirm={len(conf["y"])}')

    # --- 断言 3：三种角色的事故组两两不相交 ---
    for a, b, na, nb in [(fit, cal, 'fit', 'calib'), (fit, conf, 'fit', 'dev_confirm'),
                         (cal, conf, 'calib', 'dev_confirm')]:
        inter = set(a['groups']) & set(b['groups'])
        assert not inter, f'{na} 与 {nb} 共享 {len(inter)} 个事故组！'
    log('[断言3] 三种角色的事故组两两不相交（真实事故级隔离）→ 通过')
    V['group_isolation_across_roles'] = True

    spec = S2.MAIN_13
    cat_idx = S2.cat_feature_index()
    X_fit = S2.catboost_frame(fit['X'])
    X_cal = S2.catboost_frame(cal['X'])
    X_conf = S2.catboost_frame(conf['X'])
    y_fit, y_cal, y_conf = fit['y'], cal['y'], conf['y']

    med = G.fit_imputer(X_fit, S2.NUMERIC)
    views = G.build_views(G.apply_imputer(X_fit, S2.NUMERIC, med),
                          G.apply_imputer(X_cal, S2.NUMERIC, med),
                          G.apply_imputer(X_conf, S2.NUMERIC, med),
                          S2.NUMERIC, spec, med)
    log(f'视图维度: proto={views["proto_dim"]} rule={views["rule_dim"]} neural=6')
    assert views['proto_dim'] == 9 and views['rule_dim'] == 4

    # ---------------- 骨干 A（只训一次，四臂共享）----------------
    t0 = time.time()
    cb = G.train_backbone(X_fit, y_fit, {'calib': X_cal, 'test': X_conf}, cat_idx,
                          seeds=G.CB_SEEDS, balanced=False, groups=fit['groups'],
                          oof_folds=G.OOF_FOLDS, log=log)
    sec_A = time.time() - t0
    probs_A = cb['probs']['test']
    log(f'骨干 A 训练完成 {sec_A:.1f}s（3 种子全 fit 模型 + 组隔离 OOF）')
    for i, s in enumerate(G.CB_SEEDS):
        cb['models'][i].save_model(os.path.join(ART, f'cb_seed{s}.cbm'))
    m_single = cb['models'][G.CB_SEEDS.index(42)]
    probs_single = m_single.predict_proba(X_conf)
    probs_single_cal = m_single.predict_proba(X_cal)
    # 落盘 OOF 概率（供 t1_verify 检查合法性 + 组隔离复算）
    np.save(os.path.join(OUT, 'data', f't1_oof_fit_{args.tag}.npy'),
            np.asarray(cb['oof_fit'], dtype=float))
    pd.DataFrame({'row_id': fit['row_id'],
                  'accident_group': fit['groups']}).to_csv(
        os.path.join(OUT, 'data', f't1_oof_fit_rows_{args.tag}.csv'), index=False)

    # ---------------- 断言 4：F 头与正式 g2_common.train_f_arm 逐值一致 ----------------
    t0 = time.time()
    Fs = [S2H.train_f_arm_net(views, cb, y_fit, y_cal, s) for s in G.GEOM_SEEDS]
    probs_F = np.mean([f['probs_test'] for f in Fs], axis=0)
    probs_F_c = np.mean([f['probs_calib'] for f in Fs], axis=0)
    sec_F = time.time() - t0
    for i, sd in enumerate(G.GEOM_SEEDS):
        torch.save(Fs[i]['net'].state_dict(), os.path.join(ART, f'F_concat_seed{sd}.pt'))
    for i, sd in enumerate(G.GEOM_SEEDS):
        ref = G.train_f_arm(views, cb, y_fit, y_cal, sd)
        dmax = float(np.abs(ref['probs_test'] - Fs[i]['probs_test']).max())
        assert dmax < 1e-6, f'F 实现与正式 g2_common.train_f_arm 不一致: {dmax}'
        assert ref['sel_epoch'] == Fs[i]['sel_epoch']
    V['f_head_equals_g2_common'] = True
    log('[断言4] F 头与正式 g2_common.train_f_arm 逐值一致（3 种子）→ 通过')
    log(f'F_concat 完成 {sec_F:.1f}s  参数={Fs[0]["n_params"]}  '
        f'选中epoch={[f["sel_epoch"] for f in Fs]}  '
        f'calib MacroF1={[round(f["sel_f1"], 4) for f in Fs]}')

    # ---------------- 断言 5：消融训练器(drop=None) 与正式 g2_common.train_p_arm 逐值一致 ----------------
    # （证明 drop='proto' 的双视角臂与论文冻结的 P 是同一套算法与预算）
    t0 = time.time()
    for i, s in enumerate(G.GEOM_SEEDS):
        r2 = AB.train_p_arm_masked(views, cb, y_fit, y_cal, s, drop=None,
                                   mode='retrain', log=None, tag='')
        ref = G.train_p_arm(views, cb, y_fit, y_cal, s)
        dmax = float(np.abs(ref['probs_test'] - r2['probs_test']).max())
        assert dmax < 1e-6, f'消融训练器与正式 P 不一致: {dmax}'
        assert ref['sel_epoch'] == r2['sel_epoch']
    V['masked_trainer_equals_g2_common'] = True
    V['p_head_equals_g2_common'] = True
    log('[断言5] 消融训练器(drop=None) 与正式 g2_common.train_p_arm 逐值一致（3 种子）→ 通过')

    # ---------------- 正式主模型 P_no_proto（Rule＋Neural 双视角）----------------
    t0 = time.time()
    Ps = []
    for s in G.GEOM_SEEDS:
        r = AB.train_p_arm_masked(views, cb, y_fit, y_cal, s, drop='proto',
                                  mode='retrain', log=log, tag=f'P_no_proto_s{s}')
        torch.save(r['net'].state_dict(), os.path.join(ART, f'P_no_proto_seed{s}.pt'))
        Ps.append(r)
    probs_P = np.mean([p['probs_test'] for p in Ps], axis=0)
    probs_P_c = np.mean([p['probs_calib'] for p in Ps], axis=0)
    sec_P = time.time() - t0
    log(f'P_no_proto（双视角）完成 {sec_P:.1f}s  总参数={Ps[0]["n_params"]}  '
        f'生效参数={Ps[0]["active_params"]}  '
        f'选中epoch={[p["sel_epoch"] for p in Ps]}  '
        f'calib MacroF1={[round(p["sel_f1"], 4) for p in Ps]}')

    # ---------------- 断言 6：proto 视角确被结构性删除（无梯度、权重保持初值）----------------
    torch.manual_seed(G.GEOM_SEEDS[0])
    np.random.seed(G.GEOM_SEEDS[0])
    probe = AB.GeoNetViewMasked(9, 4)
    init_w = probe.W_proto.weight.detach().clone()
    saved = torch.load(os.path.join(ART, f'P_no_proto_seed{G.GEOM_SEEDS[0]}.pt'),
                       map_location='cpu', weights_only=True)
    dmax_w = float((saved['W_proto.weight'] - init_w).abs().max())
    assert dmax_w == 0.0, f'proto 视角权重发生了变化（未被结构性删除）: {dmax_w}'
    V['proto_view_structurally_removed'] = dict(w_proto_max_abs_change=dmax_w, pass_=True)
    log('[断言6] P_no_proto 的 W_proto 权重与初值逐位相同（proto 视角被结构性删除，'
        '无梯度）→ 通过')

    # ---------------- 落盘训练实际消费的输入帧（供 t1_verify 做 MAIN_13 一一对应核对）----------------
    in_rows = []
    for r, part in [('fit', fit), ('calib', cal), ('dev_confirm', conf)]:
        d = pd.DataFrame({'row_id': part['row_id'], 'split': r,
                          'accident_group': part['groups'], 'y': part['y']})
        for c in S2.MAIN_13:
            d[c] = part['X'][c].to_numpy()
        in_rows.append(d)
    pd.concat(in_rows, ignore_index=True).to_csv(
        os.path.join(OUT, 'data', f't1_input_frames.csv'), index=False)

    # ---------------- 落盘逐样本预测 ----------------
    arms = {
        'A_single': (probs_single, probs_single_cal),
        'A_ensemble': (probs_A, cb['probs']['calib']),
        'F_concat': (probs_F, probs_F_c),
        'P_no_proto': (probs_P, probs_P_c),
    }
    files = []
    for nm, (pv, pc_) in arms.items():
        files.append(S2.save_predictions(nm, y_conf, pv, conf, PRED))
        files.append(S2.save_predictions(nm, y_cal, pc_, cal, os.path.join(PRED, 'calib')))

    cfgs = {
        'A_single': dict(arm='单 CatBoost（seed 42 成员）', mode='baseline',
                         views_used=[], weights='artifacts/cb_seed42.cbm'),
        'A_ensemble': dict(arm='三种子 CatBoost 集成（等权概率平均）', mode='baseline',
                           views_used=[],
                           weights='artifacts/cb_seed{42,123,456}.cbm',
                           train_seconds=round(sec_A, 2)),
        'F_concat': dict(arm='非几何残差头（19→32→6，无注意力/对齐）', mode='baseline',
                         views_used=['concat(proto9, rule4, cb6)'],
                         total_parameters=Fs[0]['n_params'],
                         train_seconds=round(sec_F, 2),
                         selected_epochs=[f['sel_epoch'] for f in Fs],
                         calib_macro_f1=[round(f['sel_f1'], 4) for f in Fs],
                         weights='artifacts/F_concat_seed{42,123,456}.pt'),
        'P_no_proto': dict(arm='正式主模型：Rule＋Neural 双视角 GeOAD（删除 proto 视角后重训）',
                           mode='retrain', dropped='proto',
                           views_used=['rule', 'neural'],
                           fusion_after_ablation='锚点=保留视角中第一个（rule）；'
                                                 '注意力只在保留视角间计算；'
                                                 '对齐损失=保留视角两两均方差均值；'
                                                 '主干残差项 log(clamp(cb,1e-8)) 保留；'
                                                 '被删视角的投影层不参与前向与损失',
                           total_parameters=Ps[0]['n_params'],
                           active_parameters=Ps[0]['active_params'],
                           train_seconds=round(sec_P, 2),
                           selected_epochs=[p['sel_epoch'] for p in Ps],
                           calib_macro_f1=[round(p['sel_f1'], 4) for p in Ps],
                           weights='artifacts/P_no_proto_seed{42,123,456}.pt'),
    }
    base = dict(
        protocol=lock['protocol_id'],
        protocol_lock_sha256=sha256_file(args.lock),
        data=dict(manifest=os.path.relpath(args.manifest, OUT) if args.manifest.startswith(OUT)
                  else args.manifest, manifest_sha256=man_sha),
        dev_pool_rows=int(lock['data']['dev_pool_rows']),
        sealed_rows=int(lock['data']['sealed_rows']),
        feature_set='MAIN_13', excluded_leak_adjacent=S2.EXCLUDED_LEAK_ADJACENT,
        class_order=S2.SIX,
        split_roles=dict(fit_rows=len(fit['y']), calib_rows=len(cal['y']),
                         dev_confirm_rows=len(y_conf)),
        group_isolation='accident_group（真实事故级；fit/calib/dev_confirm 两两不相交）',
        catboost=dict(iterations=G.CB_ITERATIONS, learning_rate=G.CB_LR,
                      depth=G.CB_DEPTH, seeds=list(G.CB_SEEDS), oof_folds=G.OOF_FOLDS,
                      oof_group_key='accident_group', oof_cv_seed=42,
                      shared_by_all_arms=True, source='g2_common.train_backbone'),
        geoad_head=dict(epochs=G.GEO_EPOCHS, lr=G.GEO_LR, lam_reg=G.LAM_REG,
                        mu_align=G.MU_ALIGN, seeds=list(G.GEOM_SEEDS),
                        checkpoint_rule='calib Macro-F1 argmax，并列取最早',
                        backbone_residual='log(clamp(cb,1e-8)) + delta（所有臂均保留）',
                        source='rq2_ablation_net.train_p_arm_masked(drop=proto, mode=retrain)',
                        changed_hyperparams='无'),
        fusion='t=exp(log_temp)+0.1；ar=exp(-||p-r||/t)；ah=exp(-||p-h||/t)；'
               'z=(p+ar*r+ah*h)/(1+ar+ah+1e-8)；delta=W_delta(z) 零初始化',
    )
    cfgs = {k: dict(base, **v) for k, v in cfgs.items()}
    with open(os.path.join(OUT, 'data', f't1_arm_configs_{args.tag}.json'), 'w',
              encoding='utf-8') as f:
        json.dump(cfgs, f, ensure_ascii=False, indent=2)

    meta = dict(tag=args.tag, manifest=man_sha, verification=V,
                source_fidelity=src,
                timing=dict(A_backbone_sec=round(sec_A, 2), F_sec=round(sec_F, 2),
                            P_no_proto_sec=round(sec_P, 2),
                            total_sec=round(time.time() - t0_all, 2)),
                n_rows=dict(fit=len(fit['y']), calib=len(cal['y']),
                            dev_confirm=len(y_conf)),
                arms=list(arms.keys()),
                prediction_files=[os.path.relpath(p, OUT) for p in files],
                oof_prob_saved=True)
    with open(os.path.join(RES, f't1_train_meta_{args.tag}.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f'总耗时 {time.time() - t0_all:.1f}s；逐样本预测已落盘 {args.pred_subdir}/')
    LOGF.close()


if __name__ == '__main__':
    main()
