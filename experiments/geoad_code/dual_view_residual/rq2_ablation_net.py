"""
rq2_ablation_net.py — RQ2 第一小阶段：GeOAD P 的"三视角"消融实现

============================================================================
**不修改** `GeOAD_Submission_Final_20261024/code/g2_common.py`（那是论文正式 P 的源码）。
本模块在其**外部**复制一份同构实现，仅为支持两种消融并加训练日志：

  1) 删除后重训（retrain，主证据）
     结构性删除某个视角的投影（该视角不再参与前向、注意力与对齐损失），
     用**与完整 P 完全相同**的预算重新训练（种子、epoch、lr、λ、μ、检查点规则）。
  2) 冻结置零（freeze，辅证）
     训练不变（仍用完整 P 的权重与已选检查点），只在推理时把该视角的嵌入置零，
     再按原公式重算注意力与融合。

视角定义（与 g2_common.build_views 一致）
  proto   : 9 维数值视角（pilot_fit 上拟合的中位数填补 + StandardScaler）
  rule    : 4 维规则视角（coat / age / mfg / pr；PRIOR_DAMAGE 因标签邻接被排除）
  neural  : 6 维主干概率视角（h = W_neural(catboost 概率)）

融合与对齐（完整 P 的原始公式）
  p = W_proto(proto), r = W_rule(rule), h = W_neural(cb)
  t = exp(log_temp) + 0.1
  ar = exp(-||p - r|| / t),  ah = exp(-||p - h|| / t)
  z  = (p + ar*r + ah*h) / (1 + ar + ah + 1e-8)
  delta = W_delta(z)
  logits = log(clamp(cb, 1e-8)) + delta          <-- 主干残差项，**所有臂都保留**
  loss  = CE(logits, y) + LAM_REG*mean(delta^2) + MU_ALIGN*align_loss

消融后的融合约定（写死并可审计）
  * retrain：保留视角集合 = 其余两个；**锚点 = 保留视角中第一个**（proto → rule → neural 顺序）；
    注意力只在保留视角之间计算；融合分子分母同原公式；
    对齐损失 = 保留视角两两之间的均方差的**均值**（完整 P 时为 3 对的均值 = 原公式）。
  * freeze：三个视角照常计算，被删视角的嵌入置零，注意力与融合按**原公式**重算
    （分母中的 1 保留，即"该视角的槽位仍在、但不再提供信息"）。
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE + '/code')
_SUB = os.environ.get('ZHL_GEOAD_SUBMISSION_DIR', '')
if _SUB and os.path.isdir(os.path.join(_SUB, 'code')):
    sys.path.insert(0, os.path.join(_SUB, 'code'))

import g2_common as G  # noqa: E402

VIEWS = ['proto', 'rule', 'neural']


class GeoNetViewMasked(nn.Module):
    """GeoNet 的视角消融版：全视角时与 g2_common.GeoNet **逐值等价**。

    参数量、层顺序、初始化顺序与 g2_common.GeoNet 完全一致，
    以保证同一种子下"保留下来的层"在所有臂中拿到**相同的初始权重**
    （避免"删层导致随机数流变化"这一混淆因素）。
    """

    def __init__(self, proto_dim, rule_dim, emb_dim=32, n_classes=len(G.SIX_CLASSES)):
        super().__init__()
        self.W_proto = nn.Linear(proto_dim, emb_dim)
        self.W_rule = nn.Linear(rule_dim, emb_dim)
        self.W_neural = nn.Linear(n_classes, emb_dim)
        self.log_temp = nn.Parameter(torch.tensor([1.0]))
        self.W_delta = nn.Linear(emb_dim, n_classes)

    # ---- 视角嵌入
    def embed(self, proto, rule, cb):
        return dict(proto=self.W_proto(proto), rule=self.W_rule(rule), neural=self.W_neural(cb))

    # ---- 对齐损失：保留视角两两之间均方差的均值（完整 P 时 == 原 align_loss）
    def align_masked(self, e, drop=None):
        retained = VIEWS if drop is None else [v for v in VIEWS if v != drop]
        pairs = [(retained[i], retained[j]) for i in range(len(retained))
                 for j in range(i + 1, len(retained))]
        if not pairs:
            return torch.zeros((), dtype=e[VIEWS[0]].dtype)
        s = sum(((e[a] - e[b]) ** 2).mean() for a, b in pairs)
        return s / len(pairs)

    # ---- 前向
    def forward(self, proto, rule, cb, drop=None, mode='retrain'):
        """mode='retrain'：结构性删除 drop 视角；mode='freeze'：置零 drop 视角。"""
        e = self.embed(proto, rule, cb)

        if mode == 'freeze':
            if drop is not None:
                e[drop] = torch.zeros_like(e[drop])
            p, r, h = e['proto'], e['rule'], e['neural']
            t = torch.exp(self.log_temp) + 0.1
            ar = torch.exp(-torch.norm(p - r, dim=1, keepdim=True) / t)
            ah = torch.exp(-torch.norm(p - h, dim=1, keepdim=True) / t)
            total = 1.0 + ar + ah + 1e-8
            z = (p + ar * r + ah * h) / total
            return self.W_delta(z), z

        retained = VIEWS if drop is None else [v for v in VIEWS if v != drop]
        anchor = retained[0]
        ea = e[anchor]
        t = torch.exp(self.log_temp) + 0.1
        num = ea
        den = torch.ones(ea.shape[0], 1, dtype=ea.dtype, device=ea.device)
        for v in retained[1:]:
            a = torch.exp(-torch.norm(ea - e[v], dim=1, keepdim=True) / t)
            num = num + a * e[v]
            den = den + a
        z = num / (den + 1e-8)
        return self.W_delta(z), z

    # ---- 生效参数（被删视角的投影不计）
    def active_params(self, drop=None):
        tot = 0
        for name, mod in [('W_proto', self.W_proto), ('W_rule', self.W_rule),
                          ('W_neural', self.W_neural), ('W_delta', self.W_delta)]:
            if drop is not None and name == {'proto': 'W_proto', 'rule': 'W_rule',
                                             'neural': 'W_neural'}[drop]:
                continue
            tot += sum(p.numel() for p in mod.parameters())
        return tot + self.log_temp.numel()


def _fit_loop_logged(net, step_fn, calib_eval, epochs, lr, log=None, tag=''):
    """与 g2_common._fit_loop **逐行同逻辑**的检查点规则（calib Macro-F1 argmax，并列取最早），
    仅额外记录逐 epoch 日志。"""
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
        if log is not None and (ep % 10 == 0 or ep == epochs - 1):
            log(f'    [{tag}] epoch {ep:3d}  loss={float(loss):.6f}  calib MacroF1={f1:.4f}')
        if f1 > best[0]:
            best = (f1, {k: v.detach().clone() for k, v in net.state_dict().items()}, ep)
    net.load_state_dict(best[1])
    return net, best[0], best[2]


def train_p_arm_masked(views, cb, y_fit, y_calib, seed, drop=None, mode='retrain',
                        log=None, tag=''):
    """P 臂（含消融）。预算与检查点规则与 g2_common.train_p_arm 完全一致。

    返回 dict：probs_calib / probs_test / sel_epoch / sel_f1 / n_params / net / active_params
    """
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

    net = GeoNetViewMasked(pf.shape[1], rf.shape[1])
    torch.nn.init.zeros_(net.W_delta.weight)
    torch.nn.init.zeros_(net.W_delta.bias)

    if mode == 'freeze':
        # 冻结消融：先按完整 P 的方式训练（不改任何超参），再在推理时置零
        assert drop is not None, 'freeze 模式必须指定 drop'
        drop_eff = None
    else:
        drop_eff = drop

    def step_fn():
        d, _ = net(pf, rf, cbf, drop=drop_eff, mode='retrain')
        e = net.embed(pf, rf, cbf)
        logits = torch.log(cbf.clamp_min(1e-8)) + d
        return (F.cross_entropy(logits, yf)
                + G.LAM_REG * d.square().mean()
                + G.MU_ALIGN * net.align_masked(e, drop=drop_eff))

    def calib_eval():
        with torch.no_grad():
            d, _ = net(pc_, rc_, cbc, drop=drop_eff, mode='retrain')
            lc = torch.log(cbc.clamp_min(1e-8)) + d
        return float(f1_score(y_calib, lc.argmax(1).numpy(), average='macro',
                              zero_division=0))

    net, f1, ep = _fit_loop_logged(net, step_fn, calib_eval, G.GEO_EPOCHS, G.GEO_LR,
                                   log=log, tag=tag)
    net.eval()
    out = {}
    with torch.no_grad():
        for nm, (pr, ru, cbp) in {'calib': (pc_, rc_, cbc),
                                  'test': (pt_, rt_, cbt)}.items():
            if mode == 'retrain':
                d, _ = net(pr, ru, cbp, drop=drop_eff, mode='retrain')
            else:                      # freeze：权重不变，推理时置零被删视角
                d, _ = net(pr, ru, cbp, drop=drop, mode='freeze')
            out['probs_' + nm] = F.softmax(torch.log(cbp.clamp_min(1e-8)) + d, 1).numpy()
    return dict(probs_calib=out['probs_calib'], probs_test=out['probs_test'],
                sel_epoch=ep, sel_f1=float(f1),
                n_params=int(sum(p.numel() for p in net.parameters())),
                active_params=int(net.active_params(drop=drop_eff)),
                net=net)
