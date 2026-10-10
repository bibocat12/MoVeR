from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _import_runtime():
    from models import mover_model  # type: ignore
    return mover_model


def _clip_to_campaign_format(clip: dict[str, Any]) -> dict[str, Any]:
    """Adapt the EV-UAV archive representation to the model's campaign input."""
    locs = np.asarray(clip["locs"], dtype=np.int64)
    seg = np.asarray(clip["seg"], dtype=np.int64)
    feats = np.asarray(clip["feats"], dtype=np.float32)
    n = min(len(locs), len(seg), len(feats))
    locs, seg, feats = locs[:n], seg[:n], feats[:n]
    return {"name": clip["name"], "locs": locs, "seg": seg, "feats": feats}


def prepare_clip(
    clip: dict[str, Any],
    device: torch.device,
    mean: np.ndarray,
    std: np.ndarray,
    tau_scale: float = 64.0,
    n_blocks: int = 32,
) -> dict[str, Any]:
    """Prepare one official EV-UAV clip using the same path as training."""
    model_module = _import_runtime()
    campaign_clip = _clip_to_campaign_format(clip)
    batch = model_module.v7.prep(
        campaign_clip, device, mean, std, tau_scale=tau_scale, n_blocks=n_blocks
    )
    if "idx" in clip:
        batch["idx"] = torch.as_tensor(clip["idx"], dtype=torch.long, device=device)
    if "ts" in clip:
        batch["ts"] = torch.as_tensor(clip["ts"], dtype=torch.float64, device=device)
    return batch


def build_model_from_config(config: Any, device: torch.device):
    model_module = _import_runtime()
    model = model_module.MoVeRModel(
        c_stem=int(config.c_stem),
        c_graph=int(config.c_graph),
        k=int(config.k),
    ).to(device)
    return model.eval()


def load_checkpoint(model: torch.nn.Module, checkpoint: str, device: torch.device) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
    return payload if isinstance(payload, dict) else {}


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard = jaccard.clone()
        jaccard[1:p] -= jaccard[:-1].clone()
    return jaccard


def lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if probas.numel() == 0:
        return probas.sum() * 0.0
    losses = []
    for cls in (1, 0):
        fg = (labels == cls).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probas[:, cls]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg[perm])))
    return torch.stack(losses).mean() if losses else probas.sum() * 0.0


def segmentation_metrics(probability: np.ndarray, labels: np.ndarray, threshold: float = 0.90) -> dict[str, float | int]:
    pred = probability >= threshold
    target = labels.astype(bool)
    tp = int(np.count_nonzero(pred & target))
    fp = int(np.count_nonzero(pred & ~target))
    fn = int(np.count_nonzero(~pred & target))
    tn = int(np.count_nonzero(~pred & ~target))
    return {
        "IoU": 100.0 * tp / max(1, tp + fp + fn),
        "SegAcc": 100.0 * tp / max(1, tp + fn),
        "Precision": 100.0 * tp / max(1, tp + fp),
        "Accuracy": 100.0 * (tp + tn) / max(1, len(labels)),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "total_events": int(len(labels)),
    }
