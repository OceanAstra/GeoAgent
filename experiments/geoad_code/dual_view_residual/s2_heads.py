"""
s2_heads.py — F / P 残差头的"带权重返回"实现（与正式 g2_common 逐值同构）

============================================================================
**为什么不直接用 g2_common.train_f_arm / train_p_arm**：
正式实现不返回网络对象，交付要求里"七臂 manifest 需含模型权重"，因此这里
**逐行复制**正式实现的训练逻辑并额外返回 net 状态；随后在 s2_train_arms.py 中
用**同种子、同骨干**的正式实现做逐值对照断言（差必须 < 1e-6），
从而证明本文件与被论文冻结的正式 P / F 是同一算法。

**不改**：网络结构、损失函数、优化器、学习率、epoch 数、检查点规则。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

import g2_common as G


def _fit_loop(net, step_fn, calib_eval, epochs, lr):
    """与 g2_common._fit_loop 逐行同逻辑：calib Macro-F1 argmax（并列取最早）。"""
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


def train_f_arm_net(views, cb, y_fit, y_calib, seed, hidden=(32,)):
    """与 g2_common.train_f_arm 同构；额外返回 net 与 state_dict。"""
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

    net = G.FResidual(ff.shape[1], hidden=hidden)

    def step_fn():
        logits, delta = net(ff, cbf)
        return F.cross_entropy(logits, yf) + G.LAM_REG * delta.square().mean()

    def calib_eval():
        with torch.no_grad():
            lc, _ = net(fc, cbc)
        return float(f1_score(y_calib, lc.argmax(1).numpy(), average='macro',
                              zero_division=0))

    net, f1, ep = _fit_loop(net, step_fn, calib_eval, G.GEO_EPOCHS, G.GEO_LR)
    net.eval()
    with torch.no_grad():
        pc, _ = net(fc, cbc)
        pt, _ = net(ft, cbt)
    return dict(probs_calib=F.softmax(pc, 1).numpy(), probs_test=F.softmax(pt, 1).numpy(),
                sel_epoch=ep, sel_f1=float(f1),
                n_params=int(sum(p.numel() for p in net.parameters())),
                net=net)


def train_p_arm_net(views, cb, y_fit, y_calib, seed):
    """与 g2_common.train_p_arm 同构（正式 GeOAD P）；额外返回 net 与 state_dict。"""
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

    net = G.GeoNet(pf.shape[1], rf.shape[1])
    torch.nn.init.zeros_(net.W_delta.weight)
    torch.nn.init.zeros_(net.W_delta.bias)

    def step_fn():
        d, _ = net(pf, rf, cbf)
        p, r, h = net.W_proto(pf), net.W_rule(rf), net.W_neural(cbf)
        logits = torch.log(cbf.clamp_min(1e-8)) + d
        return (F.cross_entropy(logits, yf)
                + G.LAM_REG * d.square().mean()
                + G.MU_ALIGN * G.GeoNet.align_loss(p, r, h))

    def calib_eval():
        with torch.no_grad():
            d, _ = net(pc_, rc_, cbc)
            lc = torch.log(cbc.clamp_min(1e-8)) + d
        return float(f1_score(y_calib, lc.argmax(1).numpy(), average='macro',
                              zero_division=0))

    net, f1, ep = _fit_loop(net, step_fn, calib_eval, G.GEO_EPOCHS, G.GEO_LR)
    net.eval()
    out = {}
    with torch.no_grad():
        for nm, (pr, ru, cbp) in {'calib': (pc_, rc_, cbc),
                                  'test': (pt_, rt_, cbt)}.items():
            d, _ = net(pr, ru, cbp)
            out['probs_' + nm] = F.softmax(torch.log(cbp.clamp_min(1e-8)) + d, 1).numpy()
    return dict(probs_calib=out['probs_calib'], probs_test=out['probs_test'],
                sel_epoch=ep, sel_f1=float(f1),
                n_params=int(sum(p.numel() for p in net.parameters())),
                net=net)
