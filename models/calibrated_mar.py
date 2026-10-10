"""Context-Calibrated Relational Refinement (CC-MAR).

Evaluates observed edge affinity ``a_ij`` relative to a learned reference
``b_ij``:
- ``a_ij`` uses projected feature difference/product and normalized geometry.
- ``b_ij`` uses normalized distance, normalized time difference, and the
  product of the fourth standardized event-history channel. It does not see
  learned semantic features ``z_i, z_j`` and is not a calibrated background
  probability.
- the selected operator converts the two scores into an edge coefficient;
  the fixed public MoVeR model uses ``relu(a_ij - b_ij)``.
- messages use normalized projected-feature differences, and ``up`` starts
  at zero for an identity refinement at initialization.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ContextCalibratedAffinityRefinement(nn.Module):
    def __init__(self, c_in: int = 64, c_graph: int = 16, k: int = 8,
                 mode: str = "spatial", operator: str = "signed"):
        super().__init__()
        self.c_in = c_in
        self.c_graph = c_graph
        self.k = k
        self.mode = mode
        self.operator = operator
        if operator not in {
            "nonnegative", "direct_signed", "signed", "clipped_signed",
            "clipped_global", "clipped_permuted", "clipped_count",
        }:
            raise ValueError(f"unknown refinement operator: {operator}")

        if self.mode != "none":
            self.down = nn.Linear(c_in, c_graph, bias=False)

            # Observed affinity MLP a_ij:
            # Inputs: |zi - zj| (16) + zi * zj (16) + norm_dist (1) + dt (1) = 34
            self.aff_mlp = nn.Sequential(
                nn.Linear(2 * c_graph + 2, 16),
                nn.GELU(),
                nn.Linear(16, 1),
            )

            # Learned reference MLP b_ij. It sees normalized distance, time
            # difference, and one standardized event-history scalar product;
            # it does not see learned semantic features z_i or z_j.
            self.ref_mlp = nn.Sequential(
                nn.Linear(3, 16),
                nn.GELU(),
                nn.Linear(16, 1),
            )

            self.up = nn.Linear(c_graph, c_in, bias=False)
            nn.init.zeros_(self.up.weight)

    def _effective_coefficients(self, a_ij: torch.Tensor, b_ij: torch.Tensor,
                                context_feat: torch.Tensor,
                                direct_signed: torch.Tensor | None = None,
                                count_features: torch.Tensor | None = None):
        """Apply one declared coefficient rule and return its reference tensor."""
        if self.operator == "nonnegative":
            return a_ij, b_ij
        if self.operator == "direct_signed":
            if direct_signed is None:
                raise ValueError("direct_signed requires its feature-derived coefficient")
            return direct_signed, b_ij
        if self.operator == "clipped_global":
            b_used = torch.sigmoid(self.ref_mlp(torch.zeros_like(context_feat)))
        elif self.operator == "clipped_permuted":
            b_used = torch.roll(b_ij, shifts=1, dims=1)
        elif self.operator == "clipped_count":
            if count_features is None:
                raise ValueError("clipped_count requires cell-count features")
            b_used = torch.sigmoid(self.ref_mlp(count_features))
        else:
            b_used = b_ij
        if self.operator == "signed":
            return a_ij - b_used, b_used
        return F.relu(a_ij - b_used), b_used

    def get_neighbors(self, cf_S, cf_T, N: int, dev: torch.device) -> torch.Tensor:
        """Extract candidate neighbor indices deterministically using spatial cell order."""
        perm_S = torch.argsort(cf_S)
        rev_S = torch.empty_like(perm_S)
        rev_S[perm_S] = torch.arange(N, device=dev)
        half_k = self.k // 2
        offsets = [off for off in range(-half_k, half_k + 1) if off != 0][:self.k]
        neighbors = [perm_S[(rev_S + off).clamp(0, N - 1)] for off in offsets]
        return torch.stack(neighbors, dim=-1)

    def forward(self, h: torch.Tensor, cf_S: torch.Tensor, xy: torch.Tensor, t: torch.Tensor, p_pol: torch.Tensor):
        if self.mode == "none":
            return h, None, None, None

        dev = h.device
        N = h.shape[0]
        knn_idx = self.get_neighbors(cf_S, None, N, dev)  # (N, K)

        count_features = None
        if self.operator == "clipped_count":
            _, cell_inverse, cell_counts = torch.unique(
                cf_S, sorted=True, return_inverse=True, return_counts=True)
            log_count = torch.log1p(cell_counts.to(dtype=h.dtype))[cell_inverse]
            count_i = log_count.unsqueeze(1).expand(-1, self.k)
            count_j = log_count[knn_idx]
            count_features = torch.stack(
                [count_i, count_j, torch.zeros_like(count_i)], dim=-1)

        # Coordinate diffs for geometry
        xy_i = xy.unsqueeze(1)          # (N, 1, 2)
        xy_j = xy[knn_idx]              # (N, K, 2)
        t_i = t.unsqueeze(1)            # (N, 1)
        t_j = t[knn_idx]                # (N, K)
        dt = (t_j - t_i).unsqueeze(-1)  # (N, K, 1)

        d_pos = xy_j - xy_i
        dist_norm = (d_pos.norm(dim=-1, keepdim=True) / 50.0).clamp(0.0, 5.0)
        dt_norm = (dt.abs() / 50.0).clamp(0.0, 5.0)

        # The fourth standardized event-history channel is passed through the
        # public model as p_pol. The reference uses its pairwise product; this
        # is not raw event-polarity agreement.
        hist_i = p_pol.unsqueeze(1).unsqueeze(-1)  # (N, 1, 1)
        hist_j = p_pol[knn_idx].unsqueeze(-1)      # (N, K, 1)
        hist_product = hist_i * hist_j              # (N, K, 1)

        geom_feat = torch.cat([dist_norm, dt_norm], dim=-1)  # (N, K, 2)
        context_feat = torch.cat([dist_norm, dt_norm, hist_product], dim=-1)  # (N, K, 3)

        device_type = h.device.type
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=device_type == "cuda",
        ):
            z = self.down(h)            # (N, 16)
            z_j = z[knn_idx]            # (N, K, 16)
            z_i = z.unsqueeze(1)        # (N, 1, 16)

            diff = (z_i - z_j).abs()
            prod = z_i * z_j
            edge_feat = torch.cat([diff, prod, geom_feat], dim=-1)  # (N, K, 34)

            # Observed affinity a_ij in [0, 1].  All operator variants keep
            # this projection and neighborhood identical; only the signed
            # coefficient construction changes.
            a_ij = torch.sigmoid(self.aff_mlp(edge_feat))  # (N, K, 1)

            # Context reference b_ij in [0, 1]
            b_ij = torch.sigmoid(self.ref_mlp(context_feat)) # (N, K, 1)

            direct_signed = torch.tanh(self.aff_mlp(edge_feat)) if self.operator == "direct_signed" else None
            d_ij, b_operator = self._effective_coefficients(
                a_ij, b_ij, context_feat, direct_signed=direct_signed,
                count_features=count_features)

            # Normalized difference message passing
            num = (d_ij * (z_j - z_i)).sum(dim=1)          # (N, 16)
            den = 1.0 + d_ij.abs().sum(dim=1)              # (N, 1)
            msg = num / den

            delta_h = self.up(msg)                        # (N, C)
            h_out = h + delta_h

        return h_out, a_ij.squeeze(-1), b_operator.squeeze(-1), knn_idx
