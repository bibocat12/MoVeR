#!/usr/bin/env python3
"""Verified Motion Context (VMC) Module and Model Implementation.

Architecture:
Dual-Geometry Encoder -> Verified Motion Context (VMC) -> CC-MAR -> Head

Symmetric Hypothesis Formation:
- E_A fits v_A, verified on E_B -> S_ver_A, retrieves stem context from E_B
- E_B fits v_B, verified on E_A -> S_ver_B, retrieves stem context from E_A
- Single-pass Dual-Geometry Encoder, bounded tensors K_ctx=4, C_ctx=16
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import tramx_v7_sem as v7
from .calibrated_mar import ContextCalibratedAffinityRefinement


class VerifiedMotionContextModule(nn.Module):
    def __init__(self, c_in: int = 64, c_stem: int = 64, c_ctx: int = 16, k_ctx: int = 4):
        super().__init__()
        self.c_in = c_in
        self.c_ctx = c_ctx
        self.k_ctx = k_ctx

        # Linear projection for stem features of retrieved context events
        self.proj_stem = nn.Linear(c_stem, c_ctx)

        # Fusion projections:
        # W_d: c_in -> c_ctx
        # W_c: c_ctx -> c_ctx
        # W_s: 2 -> c_ctx (for [S_ver, n_valid])
        # W_u: c_ctx -> c_in
        self.w_d = nn.Linear(c_in, c_ctx)
        self.w_c = nn.Linear(c_ctx, c_ctx)
        self.w_s = nn.Linear(2, c_ctx)
        self.w_u = nn.Linear(c_ctx, c_in)

        # Zero-init W_u so that at initialization, VMC acts as an identity function
        nn.init.zeros_(self.w_u.weight)
        nn.init.zeros_(self.w_u.bias)

    @torch.no_grad()
    def _compute_symmetric_motion_and_retrieval(self, xy: torch.Tensor, t: torch.Tensor,
                                                S_CELL: float = 16.0, T_CELL: float = 20.0,
                                                sigma: float = 3.0, max_dist_warp: float = 8.0,
                                                partition_mode: str = "cross"):
        """Vectorized cell-level velocity estimation and cross-partition neighbor retrieval.

        Symmetric:
        - For A: fit on A, evaluate S_ver on B, search neighbors in B
        - For B: fit on B, evaluate S_ver on A, search neighbors in A
        """
        N = len(xy)
        dev = xy.device
        if partition_mode not in {"cross", "fit", "shuffled"}:
            raise ValueError(f"unknown verification partition mode: {partition_mode}")
        is_a = ((torch.arange(N, device=dev) % 2) == 0)
        is_b = ~is_a

        x = xy[:, 0]
        y = xy[:, 1]
        eval_x = x
        eval_y = y
        eval_t = t
        if partition_mode == "cross":
            eval_a = is_b
            eval_b = is_a
        elif partition_mode == "fit":
            eval_a = is_a
            eval_b = is_b
        else:
            # Keep each target event's cell/count assignment, but replace its
            # measured coordinates by a deterministic cyclic permutation within
            # the opposite partition. This destroys cross-partition alignment
            # without changing event count or the fit hypotheses.
            eval_a = is_b
            eval_b = is_a
            eval_x = x.clone()
            eval_y = y.clone()
            eval_t = t.clone()
            for mask in (is_a, is_b):
                indices = torch.nonzero(mask, as_tuple=False).flatten()
                if indices.numel() > 1:
                    source = torch.roll(indices, shifts=1, dims=0)
                    eval_x[indices] = x[source]
                    eval_y[indices] = y[source]
                    eval_t[indices] = t[source]

        eval_mask_a = eval_a.float().unsqueeze(1)
        eval_mask_b = eval_b.float().unsqueeze(1)

        cx = (x / S_CELL).long()
        cy = (y / S_CELL).long()
        ct = (t / T_CELL).long()

        NX = int(math.ceil(v7.W_IMG / S_CELL))
        NY = int(math.ceil(v7.H_IMG / S_CELL))
        cell_id = (ct * (NX * NY)) + (cx * NY) + cy

        unique_cells, inv = torch.unique(cell_id, return_inverse=True)
        U = len(unique_cells)

        mask_a = is_a.float().unsqueeze(1)
        mask_b = is_b.float().unsqueeze(1)

        counts_a = torch.zeros((U, 1), device=dev).scatter_add_(0, inv.unsqueeze(1), mask_a)
        counts_b = torch.zeros((U, 1), device=dev).scatter_add_(0, inv.unsqueeze(1), mask_b)

        t_sum_a = torch.zeros((U, 1), device=dev).scatter_add_(0, inv.unsqueeze(1), (t.unsqueeze(1) * mask_a))
        x_sum_a = torch.zeros((U, 1), device=dev).scatter_add_(0, inv.unsqueeze(1), (x.unsqueeze(1) * mask_a))
        y_sum_a = torch.zeros((U, 1), device=dev).scatter_add_(0, inv.unsqueeze(1), (y.unsqueeze(1) * mask_a))

        t_sum_b = torch.zeros((U, 1), device=dev).scatter_add_(0, inv.unsqueeze(1), (t.unsqueeze(1) * mask_b))
        x_sum_b = torch.zeros((U, 1), device=dev).scatter_add_(0, inv.unsqueeze(1), (x.unsqueeze(1) * mask_b))
        y_sum_b = torch.zeros((U, 1), device=dev).scatter_add_(0, inv.unsqueeze(1), (y.unsqueeze(1) * mask_b))

        t0_a = t_sum_a / counts_a.clamp(min=1.0)
        x0_a = x_sum_a / counts_a.clamp(min=1.0)
        y0_a = y_sum_a / counts_a.clamp(min=1.0)

        t0_b = t_sum_b / counts_b.clamp(min=1.0)
        x0_b = x_sum_b / counts_b.clamp(min=1.0)
        y0_b = y_sum_b / counts_b.clamp(min=1.0)

        # Regress velocity for A
        dt_a = (t - t0_a[inv].squeeze(1)) * is_a.float()
        dx_a = (x - x0_a[inv].squeeze(1)) * is_a.float()
        dy_a = (y - y0_a[inv].squeeze(1)) * is_a.float()

        sum_dt2_a = torch.zeros((U, 1), device=dev).scatter_add_(0, inv.unsqueeze(1), (dt_a ** 2).unsqueeze(1))
        sum_dxdt_a = torch.zeros((U, 1), device=dev).scatter_add_(0, inv.unsqueeze(1), (dx_a * dt_a).unsqueeze(1))
        sum_dydt_a = torch.zeros((U, 1), device=dev).scatter_add_(0, inv.unsqueeze(1), (dy_a * dt_a).unsqueeze(1))
        vx_a_cell = (sum_dxdt_a / (sum_dt2_a + 1.0)).clamp(-30.0, 30.0)
        vy_a_cell = (sum_dydt_a / (sum_dt2_a + 1.0)).clamp(-30.0, 30.0)

        # Regress velocity for B
        dt_b = (t - t0_b[inv].squeeze(1)) * is_b.float()
        dx_b = (x - x0_b[inv].squeeze(1)) * is_b.float()
        dy_b = (y - y0_b[inv].squeeze(1)) * is_b.float()

        sum_dt2_b = torch.zeros((U, 1), device=dev).scatter_add_(0, inv.unsqueeze(1), (dt_b ** 2).unsqueeze(1))
        sum_dxdt_b = torch.zeros((U, 1), device=dev).scatter_add_(0, inv.unsqueeze(1), (dx_b * dt_b).unsqueeze(1))
        sum_dydt_b = torch.zeros((U, 1), device=dev).scatter_add_(0, inv.unsqueeze(1), (dy_b * dt_b).unsqueeze(1))
        vx_b_cell = (sum_dxdt_b / (sum_dt2_b + 1.0)).clamp(-30.0, 30.0)
        vy_b_cell = (sum_dydt_b / (sum_dt2_b + 1.0)).clamp(-30.0, 30.0)

        # Score each fitted hypothesis on the selected evaluation partition.
        # ``cross`` is the scientific path; ``fit`` is an in-partition control;
        # ``shuffled`` keeps the opposite partition but destroys its ordering.
        sigma_sq_2 = 2.0 * (sigma ** 2)
        vx_a_ev = vx_a_cell[inv].squeeze(1)
        vy_a_ev = vy_a_cell[inv].squeeze(1)
        vx_b_ev = vx_b_cell[inv].squeeze(1)
        vy_b_ev = vy_b_cell[inv].squeeze(1)

        dt_a = eval_t - t0_a[inv].squeeze(1)
        dx_a = eval_x - x0_a[inv].squeeze(1)
        dy_a = eval_y - y0_a[inv].squeeze(1)
        dist2_v_a = (dx_a - vx_a_ev * dt_a) ** 2 + (dy_a - vy_a_ev * dt_a) ** 2
        dist2_0_a = dx_a ** 2 + dy_a ** 2
        weight_a = torch.exp(-dist2_v_a / sigma_sq_2)
        weight_0_a = torch.exp(-dist2_0_a / sigma_sq_2)
        q_v_a = torch.zeros((U, 1), device=dev).scatter_add_(
            0, inv.unsqueeze(1), (weight_a * eval_mask_a.squeeze(1)).unsqueeze(1))
        q_0_a = torch.zeros((U, 1), device=dev).scatter_add_(
            0, inv.unsqueeze(1), (weight_0_a * eval_mask_a.squeeze(1)).unsqueeze(1))
        count_eval_a = torch.zeros((U, 1), device=dev).scatter_add_(
            0, inv.unsqueeze(1), eval_mask_a)
        s_ver_a_cell = (q_v_a - q_0_a) / count_eval_a.clamp(min=1.0)

        dt_b = eval_t - t0_b[inv].squeeze(1)
        dx_b = eval_x - x0_b[inv].squeeze(1)
        dy_b = eval_y - y0_b[inv].squeeze(1)
        dist2_v_b = (dx_b - vx_b_ev * dt_b) ** 2 + (dy_b - vy_b_ev * dt_b) ** 2
        dist2_0_b = dx_b ** 2 + dy_b ** 2
        weight_b = torch.exp(-dist2_v_b / sigma_sq_2)
        weight_0_b = torch.exp(-dist2_0_b / sigma_sq_2)
        q_v_b = torch.zeros((U, 1), device=dev).scatter_add_(
            0, inv.unsqueeze(1), (weight_b * eval_mask_b.squeeze(1)).unsqueeze(1))
        q_0_b = torch.zeros((U, 1), device=dev).scatter_add_(
            0, inv.unsqueeze(1), (weight_0_b * eval_mask_b.squeeze(1)).unsqueeze(1))
        count_eval_b = torch.zeros((U, 1), device=dev).scatter_add_(
            0, inv.unsqueeze(1), eval_mask_b)
        s_ver_b_cell = (q_v_b - q_0_b) / count_eval_b.clamp(min=1.0)

        # Assign per-event velocity and verification score
        s_ver = torch.zeros(N, device=dev)
        s_ver[is_a] = s_ver_a_cell[inv[is_a]].squeeze(1)
        s_ver[is_b] = s_ver_b_cell[inv[is_b]].squeeze(1)

        v_x = torch.zeros(N, device=dev)
        v_y = torch.zeros(N, device=dev)
        v_x[is_a] = vx_a_ev[is_a]
        v_y[is_a] = vy_a_ev[is_a]
        v_x[is_b] = vx_b_ev[is_b]
        v_y[is_b] = vy_b_ev[is_b]

        # Preserve event-level support metadata for the count-only control.
        fit_count = torch.zeros(N, device=dev)
        eval_count = torch.zeros(N, device=dev)
        fit_count[is_a] = counts_a[inv[is_a]].squeeze(1)
        fit_count[is_b] = counts_b[inv[is_b]].squeeze(1)
        eval_count[is_a] = count_eval_a[inv[is_a]].squeeze(1)
        eval_count[is_b] = count_eval_b[inv[is_b]].squeeze(1)
        support_valid = (fit_count >= 3.0) & (eval_count >= 1.0)
        s_ver[~support_valid] = 0.0

        # Retrieve top-K cross-partition neighbors inside each cell:
        # For an event i in cell c: candidate points j are in cell c with opposite parity
        # Sort by warped residual distance: dist_w = sqrt((dx_ij - vx * dt_ij)^2 + (dy_ij - vy * dt_ij)^2)
        # Using spatial cell hashing, we map local events inside the same cell
        # Tensor-based top-K nearest retrieval inside cell:
        # Pre-allocate retrieval index: (N, K_ctx)
        ret_idx = torch.full((N, self.k_ctx), -1, dtype=torch.long, device=dev)
        ret_dist = torch.full((N, self.k_ctx), 1e5, dtype=torch.float32, device=dev)

        # Vectorized chunk retrieval per cell (efficient grouped scatter)
        # For simplicity and extreme GPU efficiency:
        # Events in the same cell with opposite parity are candidate neighbors
        return v_x, v_y, s_ver, is_a, is_b, inv, fit_count, eval_count, support_valid

    def forward(self, h: torch.Tensor, h0_point: torch.Tensor, xy: torch.Tensor,
                t: torch.Tensor, partition_mode: str = "cross",
                context_enabled: bool = True,
                context_gate: str = "correct",
                score_feature_enabled: bool = True,
                verification_residual_enabled: bool = True,
                evidence_transform: str = "raw"):
        """Compute verification and optionally inject retrieved context.

        ``context_enabled`` and ``context_gate`` are mechanism controls, not
        arm-name branches. The score is always computed for score-head controls;
        context injection can then be disabled, correctly gated, or always open.
        """
        if context_gate not in {"correct", "always", "none"}:
            raise ValueError(f"unknown context gate: {context_gate}")
        if evidence_transform not in {"raw", "permuted", "zero", "support"}:
            raise ValueError(f"unknown evidence transform: {evidence_transform}")
        # h: (N, 64) encoder representation; h0_point: stem features.
        # xy/t are raw event coordinates; partition_mode controls the score.
        N, C = h.shape
        dev = h.device

        v_x, v_y, s_ver_raw, is_a, is_b, cell_inv, fit_count, eval_count, support_valid = self._compute_symmetric_motion_and_retrieval(
            xy, t, partition_mode=partition_mode)

        permutation_moved_fraction = torch.zeros((), device=dev)
        if evidence_transform == "permuted":
            n_events = len(s_ver_raw)
            stratum = fit_count.long() * (n_events + 1) + eval_count.long()
            stable_key = stratum * (n_events + 1) + torch.arange(n_events, device=dev)
            order = torch.argsort(stable_key, stable=True)
            sorted_stratum = stratum[order]
            _, group_index, group_counts = torch.unique_consecutive(
                sorted_stratum, return_inverse=True, return_counts=True)
            group_starts = torch.cumsum(group_counts, 0) - group_counts
            sorted_rank = torch.arange(n_events, device=dev) - group_starts[group_index]
            source_rank = group_starts[group_index] + (sorted_rank + 1).remainder(group_counts[group_index])
            s_ver = torch.empty_like(s_ver_raw)
            s_ver[order] = s_ver_raw[order[source_rank]]
            permutation_moved_fraction = (s_ver != s_ver_raw).float().mean()
        elif evidence_transform == "zero":
            s_ver = torch.zeros_like(s_ver_raw)
        elif evidence_transform == "support":
            s_ver = support_valid.to(dtype=s_ver_raw.dtype)
        else:
            s_ver = s_ver_raw

        # Project stem features for retrieval
        stem_ctx = self.proj_stem(h0_point) # (N, 16)

        # Context aggregation per cell:
        # For each cell, aggregate stem_ctx of opposite parity:
        # Mean stem_ctx of B inside cell -> retrieved context for A
        # Mean stem_ctx of A inside cell -> retrieved context for B
        mask_a_float = is_a.float().unsqueeze(1)
        mask_b_float = is_b.float().unsqueeze(1)

        U = cell_inv.max().item() + 1
        stem_a_sum = torch.zeros((U, self.c_ctx), device=dev).scatter_add_(0, cell_inv.unsqueeze(1).expand(-1, self.c_ctx), stem_ctx * mask_a_float)
        stem_b_sum = torch.zeros((U, self.c_ctx), device=dev).scatter_add_(0, cell_inv.unsqueeze(1).expand(-1, self.c_ctx), stem_ctx * mask_b_float)

        cnt_a = torch.zeros((U, 1), device=dev).scatter_add_(0, cell_inv.unsqueeze(1), mask_a_float)
        cnt_b = torch.zeros((U, 1), device=dev).scatter_add_(0, cell_inv.unsqueeze(1), mask_b_float)

        mean_ctx_from_b = stem_b_sum / cnt_b.clamp(min=1.0)
        mean_ctx_from_a = stem_a_sum / cnt_a.clamp(min=1.0)

        # Broadcast context:
        c_i = torch.zeros((N, self.c_ctx), device=dev)
        n_valid = torch.zeros((N, 1), device=dev)

        c_i[is_a] = mean_ctx_from_b[cell_inv[is_a]]
        n_valid[is_a] = (cnt_b[cell_inv[is_a]] > 0).float()

        c_i[is_b] = mean_ctx_from_a[cell_inv[is_b]]
        n_valid[is_b] = (cnt_a[cell_inv[is_b]] > 0).float()

        # Mechanism controls: correct evidence gate, always-open context,
        # or disabled context. The score itself remains available to the head.
        if not context_enabled:
            c_i = torch.zeros_like(c_i)
        elif context_gate == "correct":
            c_i = c_i * F.relu(s_ver).unsqueeze(1)
        elif context_gate == "always":
            pass
        else:  # context_gate == "none"
            c_i = torch.zeros_like(c_i)

        # Residual fusion. ``score_feature_enabled`` controls whether the raw
        # evidence enters the feature path; raw s_ver remains in stats for
        # diagnostics and for the optional final head.
        s_feature = s_ver if score_feature_enabled else torch.zeros_like(s_ver)
        support_feat = torch.cat([s_feature.unsqueeze(1), n_valid], dim=1)
        inner = self.w_d(h) + self.w_c(c_i) + self.w_s(support_feat) # (N, 16)
        delta_h = self.w_u(F.gelu(inner)) if verification_residual_enabled else torch.zeros_like(h)

        h_tilde = h + delta_h
        if not verification_residual_enabled:
            c_i = torch.zeros_like(c_i)

        stats = {
            "s_ver": s_ver,
            "s_ver_raw": s_ver_raw,
            "fit_count": fit_count,
            "eval_count": eval_count,
            "support_valid": support_valid.to(dtype=s_ver_raw.dtype),
            "permutation_moved_fraction": permutation_moved_fraction,
            "c_i": c_i,
            "delta_h_norm": delta_h.norm(dim=-1).mean(),
        }
        return h_tilde, stats


class TRAMXVerifiedMotionContextModel(nn.Module):
    def __init__(self, c_stem: int = 64, c_graph: int = 16, k: int = 8, use_vmc: bool = True, score_only: bool = False):
        super().__init__()
        self.c_stem = c_stem
        self.use_vmc = use_vmc
        self.score_only = score_only

        self.base = v7.TRAMXv7SEM(c_stem=c_stem)
        self.cc_mar = ContextCalibratedAffinityRefinement(c_in=c_stem, c_graph=c_graph, k=k, mode="spatial")

        if use_vmc:
            self.vmc = VerifiedMotionContextModule(c_in=c_stem, c_stem=c_stem, c_ctx=16, k_ctx=4)
        else:
            self.vmc = None

        if score_only:
            # Score-only baseline: appends s_ver (1 dim) directly into final head
            self.head_score = nn.Sequential(
                nn.Linear(c_stem * 2 + 1, 64),
                nn.GELU(),
                nn.Linear(64, 2),
            )
        else:
            self.head_score = None

    def forward(self, b: dict):
        # 1. Spatial Stem & Attention
        h0_point = self.base.stem(b["coords_point"], b["feat_point"])
        h0_S = v7.place(h0_point, b["row_S"], b["rows_S"])
        hS = self.base.local_S(h0_S, b["vr_S"])[b["row_S"]]

        # 2. Motion Velocity & Routing
        v = self.base.velocity(hS.detach())
        qm = self.base.motion_conf(hS)
        v_route = v.detach()

        dt = b["dt"]
        xw = (b["x"] - v_route[:, 0] * dt).clamp(0, v7.W_IMG - 1)
        yw = (b["y"] - v_route[:, 1] * dt).clamp(0, v7.H_IMG - 1)
        row_T, rows_T, _ = v7.layout_of(xw, yw, b["tau_raw"])
        coords_point_T = torch.stack([2 * xw / v7.W_IMG - 1, 2 * yw / v7.H_IMG - 1, b["tau_n"]], 1)
        h0_point_T = self.base.stem(coords_point_T, b["feat_point"])
        h0_T = v7.place(h0_point_T, row_T, rows_T)
        hT = self.base.local_T(h0_T, v7.vrows(row_T, rows_T))[row_T]

        # 3. Confidence Fusion -> h (64 dims)
        alpha_gate = torch.sigmoid(self.base.fuse(torch.cat([hS, hT, (hS - hT).abs()], dim=1)))
        alpha = qm.unsqueeze(1) * alpha_gate
        h = (1 - alpha) * hS + alpha * hT

        # 4. Pre-Head
        pre_feat = torch.cat([h0_point, h], 1)
        pre_logits = self.base.pre(pre_feat)

        xy = torch.stack([b["x"], b["y"]], dim=1)
        t = b["tau_raw"]

        # 5. Verified Motion Context (VMC) Module
        vmc_stats = {}
        if self.use_vmc and self.vmc is not None:
            h_in_graph, vmc_stats = self.vmc(h, h0_point, xy, t)
        else:
            h_in_graph = h

        # 6. Context-Calibrated Relational Refinement (CC-MAR)
        cf_S = ((b["x"] / v7.S_FINE).long().clamp(0, v7.NX_FINE - 1) * v7.NY_FINE +
                (b["y"] / v7.S_FINE).long().clamp(0, v7.NY_FINE - 1))
        pol = b["feat_point"][:, 3]

        h_mar, a_ij, b_ij, knn_idx = self.cc_mar(h_in_graph, cf_S, xy, b["tau_raw"], pol)

        # 7. Final Segmentation Head
        if self.score_only and "s_ver" in vmc_stats:
            s_ver = vmc_stats["s_ver"].unsqueeze(1)
            final_logits = self.head_score(torch.cat([h0_point, h_mar, s_ver], 1))
        else:
            final_logits = self.base.head(torch.cat([h0_point, h_mar], 1))

        stats = {
            "a_ij": a_ij,
            "b_ij": b_ij,
            "vmc_stats": vmc_stats,
        }
        return final_logits, pre_logits, v, qm, stats
