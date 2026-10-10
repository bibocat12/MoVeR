"""The fixed MoVeR full model.

The public repository intentionally exposes one architecture only. Component
switches and ablation orchestration are maintained in the private
``MoVeR_ablation`` repository.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import tramx_v7_sem as v7
from .calibrated_mar import ContextCalibratedAffinityRefinement
from .model_vmc import VerifiedMotionContextModule


class MoVeRModel(nn.Module):
    """Complete MoVeR with cross verification and clipped relational refinement."""

    def __init__(self, c_stem: int = 64, c_graph: int = 16, k: int = 8):
        super().__init__()
        self.c_stem = int(c_stem)
        self.base = v7.TRAMXv7SEM(c_stem=self.c_stem)
        self.verification = VerifiedMotionContextModule(
            c_in=self.c_stem, c_stem=self.c_stem, c_ctx=16, k_ctx=4
        )
        self.cc_mar = ContextCalibratedAffinityRefinement(
            c_in=self.c_stem,
            c_graph=int(c_graph),
            k=int(k),
            mode="spatial",
            operator="clipped_signed",
        )
        self.score_head = nn.Sequential(
            nn.Linear(self.c_stem * 2 + 1, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, batch: dict[str, torch.Tensor]):
        # Sensor-coordinate encoder branch.
        h0_point = self.base.stem(batch["coords_point"], batch["feat_point"])
        h0_sensor = v7.place(h0_point, batch["row_S"], batch["rows_S"])
        h_sensor = self.base.local_S(h0_sensor, batch["vr_S"])[batch["row_S"]]

        # Motion-compensated branch and learned fusion.
        velocity = self.base.velocity(h_sensor.detach())
        motion_confidence = self.base.motion_conf(h_sensor)
        v_route = velocity.detach()
        x_warp = (batch["x"] - v_route[:, 0] * batch["dt"]).clamp(0, v7.W_IMG - 1)
        y_warp = (batch["y"] - v_route[:, 1] * batch["dt"]).clamp(0, v7.H_IMG - 1)
        row_motion, rows_motion, _ = v7.layout_of(
            x_warp, y_warp, batch["tau_raw"]
        )
        coords_motion = torch.stack(
            [
                2 * x_warp / v7.W_IMG - 1,
                2 * y_warp / v7.H_IMG - 1,
                batch["tau_n"],
            ],
            dim=1,
        )
        h0_motion_point = self.base.stem(coords_motion, batch["feat_point"])
        h0_motion = v7.place(h0_motion_point, row_motion, rows_motion)
        h_motion = self.base.local_T(
            h0_motion, v7.vrows(row_motion, rows_motion)
        )[row_motion]
        alpha_gate = torch.sigmoid(
            self.base.fuse(
                torch.cat(
                    [h_sensor, h_motion, (h_sensor - h_motion).abs()], dim=1
                )
            )
        )
        alpha = motion_confidence.unsqueeze(1) * alpha_gate

        # Match the released full-model recipe: the sensor branch is the
        # prediction backbone, while the motion branch supplies verification.
        h = h_sensor
        h_verified, verification_stats = self.verification(
            h,
            h0_point,
            torch.stack([batch["x"], batch["y"]], dim=1),
            batch["tau_raw"],
            partition_mode="cross",
            evidence_transform="raw",
            context_enabled=True,
            context_gate="correct",
            score_feature_enabled=True,
            verification_residual_enabled=True,
        )

        pre_logits = self.base.pre(torch.cat([h0_point, h], dim=1))
        score_token = verification_stats["s_ver"]
        before_graph_logits = self.score_head(
            torch.cat([h0_point, h_verified, score_token.unsqueeze(1)], dim=1)
        )

        cell_ids = (
            (batch["x"] / v7.S_FINE)
            .long()
            .clamp(0, v7.NX_FINE - 1)
            * v7.NY_FINE
            + (batch["y"] / v7.S_FINE)
            .long()
            .clamp(0, v7.NY_FINE - 1)
        )
        # Channel 3 is the standardized elapsed time since an opposite-polarity
        # event at the same pixel (not raw polarity); refinement uses it only
        # inside the coordinate/history reference.
        history_reference = batch["feat_point"][:, 3]
        h_refined, affinity, reference, neighbor_idx = self.cc_mar(
            h_verified,
            cell_ids,
            torch.stack([batch["x"], batch["y"]], dim=1),
            batch["tau_raw"],
            history_reference,
        )
        final_logits = self.score_head(
            torch.cat([h0_point, h_refined, score_token.unsqueeze(1)], dim=1)
        )

        stats = {
            "a_ij": affinity,
            "b_ij": reference,
            "before_graph_logits": before_graph_logits,
            "verification": verification_stats,
            "route_scale": torch.ones_like(velocity[:, 0]),
            "route_score": torch.zeros_like(velocity[:, 0]),
            "route_support": torch.full_like(velocity[:, 0], -1.0),
            "encoder": "sensor",
            "graph": True,
            "verification_mode": "score",
            "routing_mode": "fixed",
            "backbone": "standard",
            "verification_partition": "cross",
            "backbone_gate": alpha,
            "neighbor_idx": neighbor_idx,
        }
        return final_logits, pre_logits, velocity, motion_confidence, stats


__all__ = ["MoVeRModel"]
