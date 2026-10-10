#!/usr/bin/env python3
"""Render 2-row Graph Dissection showing enhancement by refinement.

Uses the public MoVeRModel and the completed RTX 3090 full-model checkpoint
(seed 37).  Both rows illustrate how clipped relational refinement enhances
event-wise predictions; the degree of enhancement varies with local event
density and clutter.

Columns:
1. Before refinement (same head)
2. Local graph topology (uniform KNN)
3. Active relations w_ij = max(a_ij - b_ij, 0)
4. Prediction changes (transitions)
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

TAU = 0.90
COL = {
    "tp": "#2ca02c", "fp": "#d62728", "fn": "#ff7f0e", "tn": "#d3d3d3",
    "edge": "#9aa3aa", "pos": "#1f77b4", "neg": "#d62728", "focal": "#111111",
}


def transition(gt, pred):
    return np.where((~pred) & gt, "fn", np.where(pred & (~gt), "fp",
                     np.where(pred & gt, "tp", "tn")))


def local_crop(x, y, node_idx, r_span):
    cx, cy = x[node_idx], y[node_idx]
    mask = (np.abs(x - cx) <= r_span) & (np.abs(y - cy) <= r_span)
    return mask, (cx - r_span, cx + r_span, cy - r_span, cy + r_span)


def main():
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))

    from models.mover_model import MoVeRModel
    from models import tramx_v7_sem as v7
    from dataset.ev_uav import convert_official_arrays

    # Paths
    run_dir = repo / "results/ssh3_vast_mover_ablation/mover_full_s37"
    norm_path = repo / "results/mover_s37_rtx3090/normalization.json"
    data_root = Path("/home/phuc/Project/ev-tiny/EV-UAV-dataset")
    out_dir = Path("/home/phuc/Project/ev-tiny/paper_MoVeR/figures")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = MoVeRModel().to(device).eval()
    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    print("Model loaded.")

    norm = json.loads(norm_path.read_text())
    mean = np.asarray(norm["mean"], np.float32)
    std = np.asarray(norm["std"], np.float32)

    # Two representative enhancement cases (same clips as the original figure)
    cases = [
        ("val_022", 2738, 16.0),
        ("val_018", 11622, 22.0),
    ]

    selected = []
    for clip_stem, focal_node, r_span in cases:
        p = data_root / "val" / f"{clip_stem}.npz"
        clip = convert_official_arrays(np.load(str(p), allow_pickle=True), clip_stem + "#0")
        batch = v7.prep(clip, torch.device("cpu"), mean, std)

        inp = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
               for k, v in batch.items()}
        with torch.inference_mode():
            logits, _, _, _, stats = model(inp)

        before = stats["before_graph_logits"]
        pb = (torch.softmax(before.float(), -1)[:, 1].cpu().numpy() >= TAU)
        pa = (torch.softmax(logits.float(), -1)[:, 1].cpu().numpy() >= TAU)
        a = stats["a_ij"].detach().float().cpu().numpy()
        b = stats["b_ij"].detach().float().cpu().numpy()
        neigh = stats["neighbor_idx"].detach().cpu().numpy()
        selected.append((clip_stem, batch, pb, pa, a, b, neigh, focal_node, r_span))
        print(f"  {clip_stem}: {len(pb)} events, focal={focal_node}")

    # Render
    plt.rcParams.update({"font.family": "serif", "font.size": 8.5})
    fig, axes = plt.subplots(2, 4, figsize=(10.8, 5.0), facecolor="white")
    plt.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.06,
                        wspace=0.08, hspace=0.18)

    col_headers = [
        "(a) Before refinement",
        "(b) Local graph",
        r"(c) Active relations ($w_{ij}$)",
        "(d) Prediction changes",
    ]
    for col in range(4):
        axes[0, col].set_title(col_headers[col], fontsize=9.2, fontweight="bold", pad=6)

    for row_idx, (name, batch, pb, pa, a, b, neigh, focal_node, r_span) in enumerate(selected):
        x = batch["x"].numpy().astype(float)
        y = batch["y"].numpy().astype(float)
        gt = batch["lab"].numpy().astype(bool)
        tr_b = transition(gt, pb)
        tr_a = transition(gt, pa)

        crop_mask, (xmin, xmax, ymin, ymax) = local_crop(x, y, focal_node, r_span)
        idx_crop = np.flatnonzero(crop_mask)
        xc = x[crop_mask]
        yc = y[crop_mask]
        tr_b_c = tr_b[crop_mask]
        tr_a_c = tr_a[crop_mask]

        # Panel 1: Before refinement
        ax1 = axes[row_idx, 0]
        ax1.scatter(xc[tr_b_c == "tn"], yc[tr_b_c == "tn"], c="#e0e0e0", s=2.5, alpha=0.4, rasterized=True)
        ax1.scatter(xc[tr_b_c == "fn"], yc[tr_b_c == "fn"], c=COL["fn"], s=10.0, alpha=0.9, rasterized=True)
        ax1.scatter(xc[tr_b_c == "fp"], yc[tr_b_c == "fp"], c=COL["fp"], s=10.0, alpha=0.9, rasterized=True)
        ax1.scatter(xc[tr_b_c == "tp"], yc[tr_b_c == "tp"], c=COL["tp"], s=12.0, alpha=0.95, rasterized=True)
        ax1.scatter([x[focal_node]], [y[focal_node]], s=55, facecolors="none", edgecolors=COL["focal"], lw=1.5, zorder=5)
        ax1.set_xlim(xmin, xmax); ax1.set_ylim(ymax, ymin)
        ax1.set_xticks([]); ax1.set_yticks([])
        ax1.set_ylabel(f"{name}", fontsize=8.8, fontweight="bold")

        # Panel 2: Graph topology
        ax2 = axes[row_idx, 1]
        ax2.scatter(xc, yc, c="#b0b0b0", s=3.0, alpha=0.5, rasterized=True)
        for u in idx_crop:
            for v in neigh[u]:
                if v in idx_crop:
                    ax2.plot([x[u], x[v]], [y[u], y[v]], color="#7f7f7f", alpha=0.25, lw=0.6)
        ax2.scatter([x[focal_node]], [y[focal_node]], s=55, facecolors="none", edgecolors=COL["focal"], lw=1.5, zorder=5)
        ax2.set_xlim(xmin, xmax); ax2.set_ylim(ymax, ymin)
        ax2.set_xticks([]); ax2.set_yticks([])

        # Panel 3: Active relations
        ax3 = axes[row_idx, 2]
        w_mat = np.maximum(a - b, 0.0)
        node_active_weight = np.zeros(len(idx_crop))
        for i_c, u in enumerate(idx_crop):
            w_vals = w_mat[u]
            node_active_weight[i_c] = float(np.max(w_vals)) if len(w_vals) > 0 else 0.0
        pruned = node_active_weight <= 0.005
        active = node_active_weight > 0.005
        ax3.scatter(xc[pruned], yc[pruned], c="#d9d9d9", s=2.5, alpha=0.3, rasterized=True)
        ax3.scatter(xc[active], yc[active], c=node_active_weight[active],
                    cmap="Blues", vmin=0.0, vmax=max(0.2, float(np.percentile(node_active_weight[active], 95))),
                    s=9.0, alpha=0.9, rasterized=True, zorder=3)
        for u in idx_crop:
            for k_idx, v in enumerate(neigh[u]):
                if v in idx_crop:
                    w_val = float(w_mat[u, k_idx])
                    if w_val > 0.005:
                        ax3.plot([x[u], x[v]], [y[u], y[v]], color=COL["pos"], alpha=min(1.0, 0.4 + w_val * 4), lw=0.8 + w_val * 4, zorder=2)
                    else:
                        ax3.plot([x[u], x[v]], [y[u], y[v]], color="#e0e0e0", alpha=0.15, lw=0.4, linestyle=":", zorder=1)
        ax3.scatter([x[focal_node]], [y[focal_node]], s=55, facecolors="none", edgecolors=COL["focal"], lw=1.5, zorder=5)
        ax3.set_xlim(xmin, xmax); ax3.set_ylim(ymax, ymin)
        ax3.set_xticks([]); ax3.set_yticks([])

        # Panel 4: Prediction changes
        ax4 = axes[row_idx, 3]
        rescued = (tr_b_c == "fn") & (tr_a_c == "tp")
        degraded = (tr_b_c == "tp") & (tr_a_c == "fn")
        suppressed = (tr_b_c == "fp") & (tr_a_c == "tn")
        leakage = (tr_b_c == "tn") & (tr_a_c == "fp")
        unchanged_tp = (tr_b_c == "tp") & (tr_a_c == "tp")
        unchanged_tn = (tr_b_c == "tn") & (tr_a_c == "tn")
        ax4.scatter(xc[unchanged_tn], yc[unchanged_tn], c="#e8e8e8", s=1.5, alpha=0.35, rasterized=True)
        ax4.scatter(xc[unchanged_tp], yc[unchanged_tp], c="#a1d99b", s=6.0, alpha=0.6, rasterized=True)
        if rescued.any():
            ax4.scatter(xc[rescued], yc[rescued], c="#006d2c", s=16.0, marker="^", label="Rescued (FN->TP)", zorder=6)
        if suppressed.any():
            ax4.scatter(xc[suppressed], yc[suppressed], c="#08519c", s=16.0, marker="v", label="Suppressed (FP->TN)", zorder=6)
        if degraded.any():
            ax4.scatter(xc[degraded], yc[degraded], c="#cb181d", s=18.0, marker="x", label="Degraded (TP->FN)", zorder=6)
        if leakage.any():
            ax4.scatter(xc[leakage], yc[leakage], c="#d94701", s=18.0, marker="s", label="Leakage (TN->FP)", zorder=6)
        ax4.scatter([x[focal_node]], [y[focal_node]], s=55, facecolors="none", edgecolors=COL["focal"], lw=1.5, zorder=7)
        ax4.set_xlim(xmin, xmax); ax4.set_ylim(ymax, ymin)
        ax4.set_xticks([]); ax4.set_yticks([])

        for ax in (ax1, ax2, ax3, ax4):
            for s_line in ax.spines.values():
                s_line.set_color("#202020"); s_line.set_linewidth(0.65)

    # Legends
    handles_a = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL["tp"], markersize=4.5, label="Target (TP)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL["fn"], markersize=4.5, label="Missed (FN)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL["fp"], markersize=4.5, label="Clutter (FP)"),
    ]
    axes[1, 0].legend(handles=handles_a, loc="lower left", fontsize=6.8, frameon=True,
                      facecolor="white", framealpha=0.88, edgecolor="#cccccc", borderpad=0.25, handletextpad=0.3)

    handles_b = [
        Line2D([0], [0], color="#7f7f7f", lw=1.0, label="KNN edge"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#b0b0b0", markersize=4.5, label="Event node"),
    ]
    axes[1, 1].legend(handles=handles_b, loc="lower left", fontsize=6.8, frameon=True,
                      facecolor="white", framealpha=0.88, edgecolor="#cccccc", borderpad=0.25, handletextpad=0.3)

    handles_c = [
        Line2D([0], [0], color=COL["pos"], lw=1.6, label=r"Active ($w>0$)"),
        Line2D([0], [0], color="#e0e0e0", lw=1.0, linestyle=":", label=r"Pruned ($w=0$)"),
    ]
    axes[1, 2].legend(handles=handles_c, loc="lower left", fontsize=6.8, frameon=True,
                      facecolor="white", framealpha=0.88, edgecolor="#cccccc", borderpad=0.25, handletextpad=0.3)

    handles_d = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#006d2c", markersize=5.0, label=r"Rescued (FN $\to$ TP)"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#08519c", markersize=5.0, label=r"Suppressed (FP $\to$ TN)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#d94701", markersize=5.0, label=r"Added (TN $\to$ FP)"),
        Line2D([0], [0], marker="x", color="#cb181d", markeredgewidth=1.4, markersize=5.0, label=r"Degraded (TP $\to$ FN)"),
    ]
    axes[1, 3].legend(handles=handles_d, loc="lower left", fontsize=6.5, frameon=True,
                      facecolor="white", framealpha=0.88, edgecolor="#cccccc", borderpad=0.25, handletextpad=0.3)

    out_pdf = out_dir / "fig_graph_dissection.pdf"
    out_png = out_dir / "fig_graph_dissection.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
