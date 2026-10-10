#!/usr/bin/env python3
"""TRAM-X v7: Trajectory-Verified Signed Evidence Memory (TV-SEM).

Core Scientific Mechanism:
  - Disagreement between spatial and motion-compensated geometries indicates hard false-positive
    hypotheses when motion confidence is high:
      agreement a_i = 0.5 * (1 + cos(norm(hS), norm(hT)))
      tilde_a_i = (1 - q_m) + q_m * a_i
      u_i^+ = w_i * tilde_a_i        (support evidence)
      u_i^- = w_i * q_m * (1 - a_i)   (contradiction evidence; u^- != 1 - p)
  - DualFastEvidenceMemorySG maintains scalar positive & negative evidence accumulators
    (E^+, E^-) in the fused chronological block loop.
  - Signed evidence verification ratio:
      rho_i = (e_i^+ - kappa * e_i^-) / (e_i^+ + kappa * e_i^- + eps) in [-1, 1]
  - Residual logit calibration:
      z_i^{fg} = z_{i,base}^{fg} + gamma_e * tanh(eta * rho_i)
      gamma_e = gamma_max * sigmoid(theta_e) (init ~0.10)
    rho > 0 promotes true target persistence (FN -> TP; Recall up),
    rho < 0 suppresses self-reinforcing background clutter (FP -> TN; Precision preserved).
  - Clean V6-C baseline loss: CE + 0.15 Lovasz + 0.10 pre-CE + 0.02 motion (fg_weight=1.0).
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import r2_address as r2

SCOPE = 1024
CIN = 7
DATA = Path(os.environ.get('MOVER_DATA', 'data/EV-UAV'))
VMAX, N_BLOCKS = 8.0, 32
S_FINE, S_COARSE = 4, 16
W_IMG, H_IMG, T_TAU = 352.0, 260.0, 125.0

NX_FINE = int(math.ceil(W_IMG / S_FINE))      # 88
NY_FINE = int(math.ceil(H_IMG / S_FINE))      # 65
N_CF = NX_FINE * NY_FINE                      # 5720

NX_COARSE = int(math.ceil(W_IMG / S_COARSE))  # 22
NY_COARSE = int(math.ceil(H_IMG / S_COARSE))  # 17
N_CC = NX_COARSE * NY_COARSE                  # 374


def morton_key(qx, qy, qt, bits=10):
    k = torch.zeros_like(qx)
    for b in range(bits):
        k |= ((qx >> b) & 1) << (3 * b)
        k |= ((qy >> b) & 1) << (3 * b + 1)
        k |= ((qt >> b) & 1) << (3 * b + 2)
    return k


def layout_of(x, y, tau, S=4):
    qx = torch.clamp((x / S).floor().long(), 0, 1023)
    qy = torch.clamp((y / S).floor().long(), 0, 1023)
    qt = torch.clamp(tau.floor().long(), 0, 1023)
    key = morton_key(qx, qy, qt)
    order = torch.argsort(key)
    n = len(x)
    rows = int(math.ceil(n / SCOPE)) * SCOPE
    row_all = torch.empty(n, dtype=torch.long, device=x.device)
    row_all[order] = torch.arange(n, device=x.device)
    return row_all, rows, n


def place(v, row, rows):
    o = torch.zeros((rows,) + v.shape[1:], dtype=v.dtype, device=v.device)
    o[row] = v
    return o


def vrows(row, rows):
    vr = torch.zeros(rows, dtype=torch.bool, device=row.device)
    vr[row] = True
    return vr


# ---------------------------------------------------------------- Modules
class NormalizedStem(nn.Module):
    def __init__(self, c_stem=64):
        super().__init__()
        self.w_c = nn.Linear(3, c_stem, bias=False)
        self.w_e = nn.Linear(CIN, c_stem, bias=False)
        self.norm = nn.LayerNorm(c_stem)
        self.act = nn.GELU()

    def forward(self, c, f):
        return self.act(self.norm(self.w_c(c) + self.w_e(f)))


class LocalSDPABlock(nn.Module):
    def __init__(self, channels=64, num_heads=4, scope=SCOPE, mlp_ratio=4.0):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scope = scope
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels, bias=True)
        self.proj = nn.Linear(channels, channels, bias=True)
        self.gamma1 = nn.Parameter(torch.full((channels,), 0.1))
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(channels * mlp_ratio), channels)
        )
        self.gamma2 = nn.Parameter(torch.full((channels,), 0.1))
        self.post_norm = nn.LayerNorm(channels)

    def forward(self, x, valid_mask):
        device_type = x.device.type
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=device_type == 'cuda',
        ):
            N, C = x.shape
            num_scopes = N // self.scope
            x_norm = self.norm1(x)
            qkv = self.qkv(x_norm).view(num_scopes, self.scope, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            attn_out = F.scaled_dot_product_attention(q, k, v)
            attn_out = attn_out.transpose(1, 2).reshape(N, C)
            x = x + self.gamma1 * self.proj(attn_out)
            x = x * valid_mask.unsqueeze(1)
            x = x + self.gamma2 * self.mlp(self.norm2(x))
            x = x * valid_mask.unsqueeze(1)
            return self.post_norm(x).float()


class VelocityHead(nn.Module):
    def __init__(self, c_stem=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c_stem, 32), nn.GELU(), nn.Linear(32, 2))

    def forward(self, h):
        return VMAX * torch.tanh(self.mlp(h.float()))


class ConfidenceHead(nn.Module):
    def __init__(self, c_stem=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c_stem, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, h):
        return torch.sigmoid(self.mlp(h.float())).squeeze(1)


class DualFastEvidenceMemorySG(nn.Module):
    """Dual Spatial & Trajectory Memory with Trajectory-Verified Signed Evidence Accumulation.

    Modes:
      - mode='clean_base': zero evidence computation (standard DualFastMemorySG)
      - mode='pos_only'  : support evidence E^+ only (u^- = 0)
      - mode='signed'    : full signed evidence E^+ and E^- (contradiction evidence from cross-geometry disagreement)
    """
    def __init__(self, c_stem=64, lam=0.9, mode='signed', kappa=1.0, eps=1e-4):
        super().__init__()
        self.c_stem = c_stem
        self.lam = lam
        self.mode = mode
        self.kappa = kappa
        self.eps = eps

    def forward(self, h, hS, hT, cf_S, cc_S, cf_T, cc_T, blk, w, qm):
        dev = h.device
        N = h.shape[0]
        r_f_S = torch.zeros_like(h); r_c_S = torch.zeros_like(h)
        r_f_T = torch.zeros_like(h); r_c_T = torch.zeros_like(h)
        rho_out = torch.zeros((N, 1), dtype=torch.float32, device=dev)

        Mf_S = torch.zeros((N_CF, self.c_stem), dtype=torch.float32, device=dev)
        Mc_S = torch.zeros((N_CC, self.c_stem), dtype=torch.float32, device=dev)
        Wf_S = torch.zeros((N_CF, 1), dtype=torch.float32, device=dev)
        Wc_S = torch.zeros((N_CC, 1), dtype=torch.float32, device=dev)
        Ep_f_S = torch.zeros((N_CF, 1), dtype=torch.float32, device=dev)
        En_f_S = torch.zeros((N_CF, 1), dtype=torch.float32, device=dev)

        Mf_T = torch.zeros((N_CF, self.c_stem), dtype=torch.float32, device=dev)
        Mc_T = torch.zeros((N_CC, self.c_stem), dtype=torch.float32, device=dev)
        Wf_T = torch.zeros((N_CF, 1), dtype=torch.float32, device=dev)
        Wc_T = torch.zeros((N_CC, 1), dtype=torch.float32, device=dev)
        Ep_f_T = torch.zeros((N_CF, 1), dtype=torch.float32, device=dev)
        En_f_T = torch.zeros((N_CF, 1), dtype=torch.float32, device=dev)

        counts = torch.bincount(blk, minlength=N_BLOCKS)
        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(counts, 0)[:-1]])
        ends = starts + counts

        starts_cpu = starts.tolist()
        ends_cpu = ends.tolist()

        f_all = h.float()
        ww_all = w.unsqueeze(1)
        qm_all = qm.unsqueeze(1)

        compute_ev = (self.mode in ('pos_only', 'signed'))
        use_neg = (self.mode == 'signed')

        if compute_ev:
            norm_S = F.normalize(hS.detach().float(), dim=-1)
            norm_T = F.normalize(hT.detach().float(), dim=-1)
            cos_agree = (norm_S * norm_T).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
            agree = 0.5 * (1.0 + cos_agree)
            agree_tilde = (1.0 - qm_all) + qm_all * agree
            u_pos_all = ww_all * agree_tilde
            u_neg_all = (ww_all * qm_all * (1.0 - agree)) if use_neg else torch.zeros_like(u_pos_all)
        else:
            u_pos_all = None; u_neg_all = None

        for st, en in zip(starts_cpu, ends_cpu):
            if st >= en:
                continue
            cf_b_S = cf_S[st:en]; cc_b_S = cc_S[st:en]
            cf_b_T = cf_T[st:en]; cc_b_T = cc_T[st:en]

            # Read-before-write memory features
            r_f_S[st:en] = Mf_S[cf_b_S] / Wf_S[cf_b_S].clamp_min(1e-6)
            r_c_S[st:en] = Mc_S[cc_b_S] / Wc_S[cc_b_S].clamp_min(1e-6)
            r_f_T[st:en] = Mf_T[cf_b_T] / Wf_T[cf_b_T].clamp_min(1e-6)
            r_c_T[st:en] = Mc_T[cc_b_T] / Wc_T[cc_b_T].clamp_min(1e-6)

            # Read signed verification evidence
            if compute_ev:
                ep_S = Ep_f_S[cf_b_S]; en_S = En_f_S[cf_b_S]
                ep_T = Ep_f_T[cf_b_T]; en_T = En_f_T[cf_b_T]
                rho_S = (ep_S - self.kappa * en_S) / (ep_S + self.kappa * en_S + self.eps)
                rho_T = (ep_T - self.kappa * en_T) / (ep_T + self.kappa * en_T + self.eps)
                qm_b = qm_all[st:en]
                rho_out[st:en] = (1.0 - qm_b) * rho_S + qm_b * rho_T

            f = f_all[st:en]; ww = ww_all[st:en]; f_ww = f * ww

            # Decay memory
            Mf_S = self.lam * Mf_S.detach(); Wf_S = self.lam * Wf_S.detach()
            Mc_S = self.lam * Mc_S.detach(); Wc_S = self.lam * Wc_S.detach()
            Mf_T = self.lam * Mf_T.detach(); Wf_T = self.lam * Wf_T.detach()
            Mc_T = self.lam * Mc_T.detach(); Wc_T = self.lam * Wc_T.detach()

            Mf_S.index_add_(0, cf_b_S, f_ww); Wf_S.index_add_(0, cf_b_S, ww)
            Mc_S.index_add_(0, cc_b_S, f_ww); Wc_S.index_add_(0, cc_b_S, ww)
            Mf_T.index_add_(0, cf_b_T, f_ww); Wf_T.index_add_(0, cf_b_T, ww)
            Mc_T.index_add_(0, cc_b_T, f_ww); Wc_T.index_add_(0, cc_b_T, ww)

            # Accumulate evidence
            if compute_ev and u_pos_all is not None and u_neg_all is not None:
                up = u_pos_all[st:en]; un = u_neg_all[st:en]
                Ep_f_S = self.lam * Ep_f_S.detach(); En_f_S = self.lam * En_f_S.detach()
                Ep_f_T = self.lam * Ep_f_T.detach(); En_f_T = self.lam * En_f_T.detach()
                Ep_f_S.index_add_(0, cf_b_S, up); En_f_S.index_add_(0, cf_b_S, un)
                Ep_f_T.index_add_(0, cf_b_T, up); En_f_T.index_add_(0, cf_b_T, un)

        return r_f_S, r_c_S, r_f_T, r_c_T, rho_out


class TRAMXv7SEM(nn.Module):
    def __init__(self, c_stem=64, mode='signed', bound_mem=True, mem_scale_init=-1.1,
                 evidence_scale_init=-1.386, kappa=1.0):
        super().__init__()
        self.c_stem = c_stem
        self.mode = mode
        self.bound_mem = bound_mem
        num_heads = max(2, c_stem // 16)
        self.stem = NormalizedStem(c_stem=c_stem)
        self.local_S = LocalSDPABlock(channels=c_stem, num_heads=num_heads, scope=SCOPE)
        self.local_T = LocalSDPABlock(channels=c_stem, num_heads=num_heads, scope=SCOPE)
        self.velocity = VelocityHead(c_stem=c_stem)
        self.motion_conf = ConfidenceHead(c_stem=c_stem)
        self.mem_scale_logit = nn.Parameter(torch.tensor(float(mem_scale_init)))

        # Evidence calibration: gamma_e = 0.5 * sigmoid(logit); init -1.386 -> gamma_e ~= 0.10
        self.evidence_scale_logit = nn.Parameter(torch.tensor(float(evidence_scale_init)))

        self.fuse = nn.Sequential(nn.Linear(3 * c_stem, 32), nn.GELU(), nn.Linear(32, 1))
        self.dual_mem = DualFastEvidenceMemorySG(c_stem=c_stem, mode=mode, kappa=kappa)
        self.mem_fuse = nn.Sequential(nn.Linear(4 * c_stem, 32), nn.GELU(), nn.Linear(32, 1))
        self.mem_proj = nn.Linear(2 * c_stem, c_stem, bias=False)
        self.cons = nn.Sequential(nn.Linear(3 * c_stem, 32), nn.GELU(), nn.Linear(32, 1))
        self.res = nn.Linear(c_stem, c_stem, bias=False)
        nn.init.zeros_(self.res.weight)
        self.pre = nn.Sequential(nn.Linear(2 * c_stem, 64), nn.GELU(), nn.Linear(64, 2))
        self.head = nn.Sequential(nn.Linear(2 * c_stem, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, b):
        h0_point = self.stem(b['coords_point'], b['feat_point'])

        # Spatial Pass
        h0_S = place(h0_point, b['row_S'], b['rows_S'])
        hS = self.local_S(h0_S, b['vr_S'])[b['row_S']]

        # Motion Isolation
        v = self.velocity(hS.detach())
        qm = self.motion_conf(hS)

        # Detached Routing
        v_route = v.detach()
        xw = (b['x'] - v_route[:, 0] * b['dt']).clamp(0, W_IMG - 1)
        yw = (b['y'] - v_route[:, 1] * b['dt']).clamp(0, H_IMG - 1)
        row_T, rows_T, _ = layout_of(xw, yw, b['tau_raw'])
        coords_point_T = torch.stack([2 * xw / W_IMG - 1, 2 * yw / H_IMG - 1, b['tau_n']], 1)
        h0_point_T = self.stem(coords_point_T, b['feat_point'])
        h0_T = place(h0_point_T, row_T, rows_T)

        # Trajectory Pass
        hT = self.local_T(h0_T, vrows(row_T, rows_T))[row_T]

        # Confidence Fallback Local Fusion
        alpha_gate = torch.sigmoid(self.fuse(torch.cat([hS, hT, (hS - hT).abs()], dim=1)))
        alpha = qm.unsqueeze(1) * alpha_gate
        h = (1 - alpha) * hS + alpha * hT

        # Pre-Head
        pre_feat = torch.cat([h0_point, h], 1)
        pre_logits = self.pre(pre_feat)
        with torch.no_grad():
            w = torch.softmax(pre_logits.detach(), -1)[:, 1]

        # Fused Dual Fast Memory with Signed Evidence Verification
        cf_S = ((b['x'] / S_FINE).long().clamp(0, NX_FINE - 1) * NY_FINE +
                (b['y'] / S_FINE).long().clamp(0, NY_FINE - 1))
        cc_S = ((b['x'] / S_COARSE).long().clamp(0, NX_COARSE - 1) * NY_COARSE +
                (b['y'] / S_COARSE).long().clamp(0, NY_COARSE - 1))
        cf_T = ((xw / S_FINE).long().clamp(0, NX_FINE - 1) * NY_FINE +
                (yw / S_FINE).long().clamp(0, NY_FINE - 1))
        cc_T = ((xw / S_COARSE).long().clamp(0, NX_COARSE - 1) * NY_COARSE +
                (yw / S_COARSE).long().clamp(0, NY_COARSE - 1))

        r_f_S, r_c_S, r_f_T, r_c_T, rho = self.dual_mem(h, hS, hT, cf_S, cc_S, cf_T, cc_T, b['blk'], w, qm)

        beta_gate = torch.sigmoid(self.mem_fuse(torch.cat([h, r_f_S, r_f_T, (r_f_S - r_f_T).abs()], dim=1)))
        beta = qm.unsqueeze(1) * beta_gate
        r_fused = (1 - beta) * r_f_S + beta * r_f_T
        r = r_fused + self.mem_proj(torch.cat([torch.max(r_f_S, r_c_S), torch.max(r_f_T, r_c_T)], 1)).float()

        # Bounded Memory Consistency Residual
        c = torch.sigmoid(self.cons(torch.cat([h, r, (h - r).abs()], 1)))
        mem_res_raw = c * self.res(r)
        gamma_mem = (0.25 * torch.sigmoid(self.mem_scale_logit)) if self.bound_mem else torch.tensor(1.0, device=h.device)
        mem_res = gamma_mem * mem_res_raw
        h_out = h + mem_res

        with torch.no_grad():
            res_ratio = (mem_res.norm(dim=1).mean() / h.norm(dim=1).mean().clamp_min(1e-6)).item()
            res_ratio_raw = (mem_res_raw.norm(dim=1).mean() / h.norm(dim=1).mean().clamp_min(1e-6)).item()

        # Final Head with Direct Pointwise Stem Skip
        final_feat = torch.cat([h0_point, h_out], 1)
        base_logits = self.head(final_feat)

        # Evidence Residual Logit Calibration (when mode != 'clean_base')
        if self.mode != 'clean_base':
            gamma_e = 0.5 * torch.sigmoid(self.evidence_scale_logit)
            rho_calib = gamma_e * torch.tanh(2.0 * rho)
            final_logits = torch.stack([
                base_logits[:, 0],
                base_logits[:, 1] + rho_calib.squeeze(1)
            ], dim=1)
        else:
            final_logits = base_logits
            gamma_e = torch.tensor(0.0, device=h.device)

        stats = {
            'res_ratio': float(res_ratio),
            'res_ratio_raw': float(res_ratio_raw),
            'gamma_mem': float(gamma_mem.detach().item()) if self.bound_mem else 1.0,
            'gamma_e': float(gamma_e.detach().item()),
            'rho_mean': float(rho.mean().item()),
            'rho_min': float(rho.min().item()),
            'rho_max': float(rho.max().item()),
            'w_mean': float(w.mean().item()),
            'qm_mean': float(qm.mean().item()),
            'w_res_norm': float(self.res.weight.norm().item())
        }
        return final_logits, pre_logits, v, qm, stats


# ---------------------------------------------------------------- Data Preparation
def load_clips(path):
    z = np.load(path); locs, seg, feats = z['locs'], z['seg'], z['feats']
    n = min(len(locs), len(seg), len(feats))
    locs, seg, feats = np.asarray(locs)[:n], np.asarray(seg)[:n], np.asarray(feats)[:n]
    return [dict(name=f'{Path(path).stem}#{c}', locs=locs[locs[:, 0] == c], seg=seg[locs[:, 0] == c],
                 feats=feats[locs[:, 0] == c]) for c in np.unique(locs[:, 0])]


def prep(clip, dev, mu, sd, tau_scale=64.0, n_blocks=N_BLOCKS):
    locs, seg, fz = clip['locs'], clip['seg'], clip['feats']
    if int(n_blocks) != N_BLOCKS:
        raise ValueError(f"MoVeR requires n_blocks={N_BLOCKS} for the fixed memory layout")
    t = locs[:, 3].astype(np.float64); t0 = float(t.min()); tau = (t - t0) / float(tau_scale)
    x = torch.as_tensor(locs[:, 1], dtype=torch.float32, device=dev)
    y = torch.as_tensor(locs[:, 2], dtype=torch.float32, device=dev)
    tau_t = torch.as_tensor(tau, dtype=torch.float32, device=dev)
    F_feat = (r2.stem_features(locs, fz, {'tau': tau, 't0': t0}) - mu) / sd
    row_S, rows_S, n = layout_of(x, y, tau_t)
    coords_point = torch.stack([2 * x / W_IMG - 1, 2 * y / H_IMG - 1, tau_t / T_TAU], 1)
    feat_point = torch.as_tensor(F_feat, dtype=torch.float32, device=dev)
    tmax = max(1e-6, float(tau.max()))
    blk = torch.clamp((tau_t / (tmax / N_BLOCKS)).floor().long(), 0, N_BLOCKS - 1)

    fg_mask = (seg == 1)
    v_gt = torch.zeros((n, 2), dtype=torch.float32, device=dev)
    has_target_vel = False
    if fg_mask.sum() >= 8:
        x_fg = locs[fg_mask, 1].astype(np.float64)
        y_fg = locs[fg_mask, 2].astype(np.float64)
        t_fg = tau[fg_mask]
        dt_fg = t_fg - t_fg.mean()
        den = float((dt_fg ** 2).sum())
        if den > 1e-4:
            vx_gt = float(((x_fg - x_fg.mean()) * dt_fg).sum() / den)
            vy_gt = float(((y_fg - y_fg.mean()) * dt_fg).sum() / den)
            vx_gt = max(-VMAX, min(VMAX, vx_gt))
            vy_gt = max(-VMAX, min(VMAX, vy_gt))
            v_gt[fg_mask, 0] = vx_gt
            v_gt[fg_mask, 1] = vy_gt
            has_target_vel = True

    return {'name': clip['name'], 'x': x, 'y': y, 'tau_raw': tau_t, 'tau_n': tau_t / T_TAU,
            'dt': tau_t - tau_t.mean(), 'coords_point': coords_point, 'feat_point': feat_point,
            'row_S': row_S, 'rows_S': rows_S, 'vr_S': vrows(row_S, rows_S),
            'blk': blk, 'lab': torch.as_tensor(seg, dtype=torch.long, device=dev),
            'v_gt': v_gt, 'has_target_vel': has_target_vel, 'n': n}


# ---------------------------------------------------------------- Optimization Utilities
@torch.no_grad()
def agc(params, clip=0.02, eps=1e-3):
    for p in params:
        if p.grad is None or p.ndim <= 1:
            continue
        p_norm = p.norm().clamp_min(eps)
        g_norm = p.grad.norm()
        max_norm = clip * p_norm
        if g_norm > max_norm:
            p.grad.mul_(max_norm / (g_norm + 1e-6))


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for s_p, m_p in zip(self.shadow.parameters(), model.parameters()):
            s_p.data.mul_(self.decay).add_(m_p.data, alpha=1.0 - self.decay)


def _lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        out = jaccard.clone()
        out[1:p] = jaccard[1:p] - jaccard[0:p - 1]
        return out
    return jaccard


def lovasz_softmax_binary(logits, labels):
    probs = torch.softmax(logits.float(), dim=-1)
    losses = []
    for c in (1, 0):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        pc = probs[:, c].clamp(1e-6, 1.0 - 1e-6)
        errors = (fg - pc).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg[perm])))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def evaluate_set(model, batches):
    model.eval()
    P, T = [], []
    with torch.no_grad():
        for b in batches:
            final, _, _, _, _ = model(b)
            P.append(torch.softmax(final.float(), -1)[:, 1].cpu().numpy())
            T.append(b['lab'].cpu().numpy())
    prob, truth = np.concatenate(P), np.concatenate(T)
    pred = (prob >= 0.5).astype(np.int64)
    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())
    tot = len(truth)
    return {
        'IoU': round(tp / max(1, tp + fp + fn), 4),
        'P': round(tp / max(1, tp + fp), 4),
        'R_Pd': round(tp / max(1, tp + fn), 4),
        'Acc': round((tp + tn) / tot, 4),
        'Fa_10k': round((fp / tot) * 10000.0, 2),
        'total_events': tot
    }


def train(a, dev):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    if getattr(a, 'deterministic', True):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
        print(f'[{a.arm}] deterministic algorithms ON (CUBLAS_WORKSPACE_CONFIG=:4096:8)', flush=True)

    tr_files = sorted(DATA.glob('train_*.npz'))
    va_files = sorted(DATA.glob('val_*.npz'))
    te_files = sorted(DATA.glob('test_*.npz'))

    json.dump({'git_sha': a.git_sha, 'config': vars(a),
               'architecture': f'TRAM-X v7 SEM ({a.mode}) C={a.c_stem} with DualFastEvidenceMemory + Lovasz ({a.lovasz_w})',
               'splits': {'train': len(tr_files), 'val': len(va_files), 'test': len(te_files)},
               'test_sealed': not a.eval_test,
               'deterministic_algorithms': bool(getattr(a, 'deterministic', True))},
              open(out / 'receipt.json', 'w'), indent=1)

    torch.manual_seed(a.seed); np.random.seed(a.seed)

    print(f'[{a.arm}] Pre-caching RAM dataset...', flush=True)
    t_c0 = time.time()
    Fs = np.concatenate([r2.stem_features(c['locs'], c['feats'],
                                          {'tau': (c['locs'][:, 3] - c['locs'][:, 3].min()) / 64.0,
                                           't0': float(c['locs'][:, 3].min())})
                         for f in tr_files for c in load_clips(str(f))])
    mu, sd = Fs.mean(0), Fs.std(0) + 1e-6

    tr_batches = [prep(cl, dev, mu, sd) for f in tr_files for cl in load_clips(str(f))]
    va_batches = [prep(cl, dev, mu, sd) for f in va_files for cl in load_clips(str(f))]
    te_batches = [prep(cl, dev, mu, sd) for f in te_files for cl in load_clips(str(f))]
    print(f'[{a.arm}] Cached {len(tr_batches)} train / {len(va_batches)} val / {len(te_batches)} test clips in {time.time()-t_c0:.1f}s', flush=True)

    m = TRAMXv7SEM(c_stem=a.c_stem, mode=a.mode, bound_mem=a.bound_mem,
                   mem_scale_init=a.mem_scale_init,
                   evidence_scale_init=a.evidence_scale_init,
                   kappa=a.kappa).to(dev)
    ema = ModelEMA(m, decay=0.999)

    # Differential parameter groups
    local_params = list(m.local_S.parameters()) + list(m.local_T.parameters())
    stem_params = list(m.stem.parameters())
    other_params = (list(m.velocity.parameters()) + list(m.motion_conf.parameters()) +
                    [m.mem_scale_logit, m.evidence_scale_logit] +
                    list(m.fuse.parameters()) +
                    list(m.dual_mem.parameters()) + list(m.mem_fuse.parameters()) +
                    list(m.mem_proj.parameters()) + list(m.cons.parameters()) +
                    list(m.res.parameters()) + list(m.pre.parameters()) +
                    list(m.head.parameters()))

    param_groups = [
        {'params': [p for p in local_params if p.ndim > 1], 'lr': 5e-5, 'weight_decay': 1e-2},
        {'params': [p for p in local_params if p.ndim <= 1], 'lr': 5e-5, 'weight_decay': 0.0},
        {'params': [p for p in stem_params if p.ndim > 1], 'lr': 7.5e-5, 'weight_decay': 1e-2},
        {'params': [p for p in stem_params if p.ndim <= 1], 'lr': 7.5e-5, 'weight_decay': 0.0},
        {'params': [p for p in other_params if p.ndim > 1], 'lr': 2e-4, 'weight_decay': 1e-2},
        {'params': [p for p in other_params if p.ndim <= 1], 'lr': 2e-4, 'weight_decay': 0.0},
    ]

    opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-6)
    total_steps = a.epochs * len(tr_batches)
    warmup_steps = 5 * len(tr_batches)

    def lr_factor(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * p)))

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f'[{a.arm}] Total Params: {total_params:,} (~{total_params/1e6:.3f}M) | C: {a.c_stem} | Mode: {a.mode} | Epochs: {a.epochs}', flush=True)

    rec = []; best_val_iou = -1.0; best_kind = 'raw'
    stats = {'qm_mean': 0.0, 'res_ratio': 0.0, 'res_ratio_raw': 0.0, 'gamma_mem': 0.0,
             'gamma_e': 0.0, 'rho_mean': 0.0}

    for ep in range(a.epochs):
        t0 = time.time(); tl = 0.0
        g_tot_list, g_mot_list, g_loc_list = [], [], []

        # Deterministic cross-seed epoch shuffle
        g_rng = torch.Generator()
        g_rng.manual_seed(10000 + ep)
        order = torch.randperm(len(tr_batches), generator=g_rng).tolist()

        m.train()
        for idx in order:
            b = tr_batches[idx]
            opt.zero_grad(set_to_none=True)
            final, pre, v, qm, stats = m(b)
            loss_final = nn.functional.cross_entropy(final, b['lab'])
            loss_pre = nn.functional.cross_entropy(pre, b['lab'])
            loss = loss_final + a.pre_weight * loss_pre
            if a.lovasz_w > 0:
                loss = loss + a.lovasz_w * lovasz_softmax_binary(final, b['lab'])

            if b['has_target_vel']:
                fg_mask = (b['lab'] == 1)
                if fg_mask.any():
                    l_motion = nn.functional.smooth_l1_loss(v[fg_mask], b['v_gt'][fg_mask])
                    loss = loss + a.lambda_v * l_motion

            loss.backward()

            def pnorm(params):
                v_ = [p.grad.norm().item()**2 for p in params if p.grad is not None]
                return math.sqrt(sum(v_)) if v_ else 0.0

            g_tot = pnorm(m.parameters())
            g_mot = pnorm(list(m.velocity.parameters()) + list(m.motion_conf.parameters()))
            g_loc = pnorm(local_params)
            g_tot_list.append(g_tot); g_mot_list.append(g_mot); g_loc_list.append(g_loc)

            # AGC + group clipping
            agc(local_params, clip=0.02)
            agc(stem_params, clip=0.02)
            torch.nn.utils.clip_grad_norm_(local_params, 0.5)
            torch.nn.utils.clip_grad_norm_(stem_params, 0.5)
            torch.nn.utils.clip_grad_norm_(other_params, 1.0)

            opt.step()
            sch.step()
            ema.update(m)
            tl += float(loss.detach())

        raw_val = evaluate_set(m, va_batches)
        ema_val = evaluate_set(ema.shadow, va_batches)

        ep_stat = {
            'epoch': ep,
            'train_loss': round(tl / len(tr_batches), 5),
            'val_IoU': raw_val['IoU'], 'val_P': raw_val['P'], 'val_R': raw_val['R_Pd'],
            'ema_IoU': ema_val['IoU'], 'ema_P': ema_val['P'], 'ema_R': ema_val['R_Pd'],
            'G_total': round(float(np.mean(g_tot_list)), 3),
            'G_motion': round(float(np.mean(g_mot_list)), 3),
            'G_local': round(float(np.mean(g_loc_list)), 3),
            'Qm_mean': round(stats['qm_mean'], 3),
            'R_res': round(stats['res_ratio'], 3),
            'gamma_mem': round(stats['gamma_mem'], 4),
            'gamma_e': round(stats['gamma_e'], 4),
            'rho_mean': round(stats['rho_mean'], 3),
            'epoch_s': round(time.time() - t0, 1)
        }
        rec.append(ep_stat)
        print(f"[{a.arm} ep{ep:02d}] loss={ep_stat['train_loss']} VAL_IoU={raw_val['IoU']:.4f} "
              f"EMA_IoU={ema_val['IoU']:.4f} "
              f"P={raw_val['P']:.3f} R={raw_val['R_Pd']:.3f} G_tot={ep_stat['G_total']:.2f} "
              f"G_loc={ep_stat['G_local']:.2f} G_mot={ep_stat['G_motion']:.2f} "
              f"Qm={ep_stat['Qm_mean']:.3f} R_res={ep_stat['R_res']:.3f} "
              f"rho={ep_stat['rho_mean']:.3f} {ep_stat['epoch_s']}s", flush=True)
        json.dump(rec, open(out / 'per_epoch.json', 'w'), indent=1)

        cands = [('raw', raw_val['IoU'], m.state_dict()), ('ema', ema_val['IoU'], ema.shadow.state_dict())]
        kind, iou_v, weights = max(cands, key=lambda t: t[1])
        if iou_v > best_val_iou:
            best_val_iou = iou_v
            best_kind = kind
            torch.save({'model': weights, 'epoch': ep, 'kind': kind, 'is_ema': kind == 'ema',
                        'val': ep_stat, 'config': vars(a)}, out / 'best.pt')
            print(f"[{a.arm}] new best by VAL: {kind} IoU={iou_v:.4f} @ep{ep}", flush=True)

    best_ck = torch.load(out / 'best.pt', map_location=dev)
    print(f"\n[{a.arm}] Training complete. best-by-VAL = {best_kind} IoU={best_val_iou:.4f} "
          f"@ep{best_ck['epoch']}", flush=True)

    val_summary = {
        'arm': a.arm, 'seed': a.seed, 'c_stem': a.c_stem, 'mode': a.mode,
        'epochs': a.epochs, 'lambda_v': a.lambda_v, 'pre_weight': a.pre_weight,
        'lovasz_w': a.lovasz_w, 'bound_mem': a.bound_mem,
        'best_kind': best_kind, 'best_val_epoch': best_ck['epoch'],
        'best_val_IoU': best_val_iou,
        'raw_best_val_IoU': max(r['val_IoU'] for r in rec),
        'ema_best_val_IoU': max(r['ema_IoU'] for r in rec),
        'params': total_params,
        'final_R_res': rec[-1]['R_res'], 'final_gamma_mem': rec[-1]['gamma_mem'],
        'final_gamma_e': rec[-1]['gamma_e'],
        'final_rho_mean': rec[-1]['rho_mean'],
        'test_evaluated': bool(a.eval_test),
    }
    json.dump(val_summary, open(out / 'val_summary.json', 'w'), indent=2)
    print(f"[{a.arm}] VAL SUMMARY: {json.dumps(val_summary, indent=2)}", flush=True)

    if not a.eval_test:
        print(f"[{a.arm}] TEST remains SEALED (pass --eval-test only after the recipe is frozen).",
              flush=True)
        return

    eval_model = copy.deepcopy(m)
    eval_model.load_state_dict(best_ck['model'])
    test_metrics = evaluate_set(eval_model, te_batches)
    test_results = {
        'arm': a.arm, 'seed': a.seed, 'c_stem': a.c_stem, 'mode': a.mode,
        'best_val_epoch': best_ck['epoch'], 'best_val_IoU': best_val_iou,
        'selection': 'VAL-only, TEST read once after freeze',
        'official_test': test_metrics,
    }
    json.dump(test_results, open(out / 'test_evaluation.json', 'w'), indent=2)
    print(f"[{a.arm}] OFFICIAL TEST:", json.dumps(test_metrics, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='v7-sem')
    ap.add_argument('--mode', choices=['clean_base', 'pos_only', 'signed'], default='signed')
    ap.add_argument('--c-stem', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--lambda-v', type=float, default=0.02)
    ap.add_argument('--pre-weight', type=float, default=0.10)
    ap.add_argument('--lovasz-w', type=float, default=0.15)
    ap.add_argument('--kappa', type=float, default=1.0)
    ap.add_argument('--no-bound-mem', dest='bound_mem', action='store_false')
    ap.add_argument('--mem-scale-init', type=float, default=-1.1)
    ap.add_argument('--evidence-scale-init', type=float, default=-1.386)
    ap.add_argument('--eval-test', action='store_true',
                    help='unseal the official TEST split (only after the recipe is frozen)')
    ap.add_argument('--deterministic', dest='deterministic', action='store_true', default=True)
    ap.add_argument('--no-deterministic', dest='deterministic', action='store_false')
    ap.add_argument('--seed', type=int, default=37)
    ap.add_argument('--git-sha', default='')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = 'cuda:0'
    train(a, dev)
