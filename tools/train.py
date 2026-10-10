#!/usr/bin/env python3
"""EV-UAV-style MoVeR training entry point.

The public EV-UAV baseline exposes a simple ``train.py`` script. MoVeR keeps
that operational shape while preserving its own model and loss: sectioned YAML,
train/VAL archive directories, one isolated clip per forward, VAL checkpoint
selection, and a saved ``best.pt`` payload.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from configs.configs import load_config
from dataset.ev_uav import EvUAV
from dataset.mover_runtime import (
    build_model_from_config,
    lovasz_softmax_flat,
    prepare_clip,
)
from utils.eval import evalute


REPO = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train MoVeR on EV-UAV-format archives")
    p.add_argument("--config", default="configs/mover_evuav.yaml")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default=None)
    p.add_argument("--resume", default=None)
    return p


def _set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)


def _normalization_path(config: SimpleNamespace) -> Path:
    path = Path(config.normalization_path)
    return path if path.is_absolute() else REPO / path


def _load_or_fit_normalization(config: SimpleNamespace, train: EvUAV, output: Path) -> tuple[np.ndarray, np.ndarray]:
    path = _normalization_path(config)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return np.asarray(payload["mean"], dtype=np.float32), np.asarray(payload["std"], dtype=np.float32)

    from models import r2_address as r2

    chunks = []
    for clip in train:
        t = clip["locs"][:, 3].astype(np.float64)
        tau = (t - t.min()) / float(config.tau_scale)
        chunks.append(r2.stem_features(clip["locs"], clip["feats"], {"tau": tau, "t0": float(t.min())}))
    features = np.concatenate(chunks, axis=0)
    mean = features.mean(axis=0).astype(np.float32)
    std = (features.std(axis=0) + 1e-6).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2) + "\n", encoding="utf-8")
    output.mkdir(parents=True, exist_ok=True)
    (output / "normalization.json").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return mean, std


def _prepare_dataset(
    dataset: EvUAV,
    device: torch.device,
    mean: np.ndarray,
    std: np.ndarray,
    config: SimpleNamespace,
) -> list[dict[str, Any]]:
    return [
        prepare_clip(
            dataset[i], device, mean, std,
            tau_scale=float(config.tau_scale), n_blocks=int(config.n_blocks),
        )
        for i in range(len(dataset))
    ]


def _evaluate(model: torch.nn.Module, batches: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    evaluator = evalute(threshold=threshold)
    with torch.inference_mode():
        for index, batch in enumerate(batches):
            probability = torch.softmax(model(batch)[0].float(), dim=-1)[:, 1].cpu().numpy()
            evaluator.matches[str(index)] = {
                "seg_pred": probability,
                "seg_gt": batch["lab"].cpu().numpy(),
            }
    return evaluator.event_metrics()


def _build_optimizer(model: torch.nn.Module, config: SimpleNamespace):
    """Reproduce the paper trainer's parameter-group and scheduler recipe."""
    decay, no_decay, stem, local, graph, verification = [], [], [], [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "verification" in name:
            verification.append(parameter)
        elif "cc_mar" in name or "positive_mar" in name:
            graph.append(parameter)
        elif "stem" in name:
            stem.append(parameter)
        elif "local" in name:
            local.append(parameter)
        elif parameter.dim() >= 2:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    groups = [
        {"params": decay, "weight_decay": float(config.weight_decay), "lr": float(config.lr)},
        {"params": no_decay, "weight_decay": 0.0, "lr": float(config.lr)},
        {"params": stem, "weight_decay": float(config.weight_decay), "lr": float(config.stem_lr)},
        {"params": local, "weight_decay": float(config.weight_decay), "lr": float(config.local_lr)},
    ]
    if graph:
        groups.append({"params": graph, "weight_decay": float(config.weight_decay), "lr": float(config.graph_lr)})
    if verification:
        groups.append({"params": verification, "weight_decay": float(config.weight_decay), "lr": float(config.verification_lr)})
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(config.epochs)), eta_min=float(config.eta_min)
    )
    return optimizer, scheduler


def _load_state(model: torch.nn.Module, path: str, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config.seed = args.seed
    if args.output is not None:
        config.model_save_root = args.output

    _set_seed(int(config.seed), bool(config.deterministic))
    if not torch.cuda.is_available():
        raise RuntimeError("MoVeR training requires CUDA")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("training device must be CUDA")

    output = Path(config.model_save_root)
    output.mkdir(parents=True, exist_ok=True)
    train_dataset = EvUAV(config.root, mode="train", max_events_num=config.max_events_num, sample_seed=int(config.seed))
    val_dataset = EvUAV(config.root, mode="val", max_events_num=None, sample_seed=int(config.seed))
    mean, std = _load_or_fit_normalization(config, train_dataset, output)
    train_batches = _prepare_dataset(train_dataset, device, mean, std, config)
    val_batches = _prepare_dataset(val_dataset, device, mean, std, config)

    model = build_model_from_config(config, device)
    if args.resume:
        _load_state(model, args.resume, device)

    optimizer, scheduler = _build_optimizer(model, config)
    best_iou = -1.0
    best_epoch = -1
    history: list[dict[str, Any]] = []
    start = time.perf_counter()

    for epoch in range(int(config.epochs)):
        model.train()
        order = np.random.permutation(len(train_batches))
        total_loss = 0.0
        for index in order:
            batch = train_batches[int(index)]
            optimizer.zero_grad(set_to_none=True)
            final_logits, pre_logits, velocity, _, _ = model(batch)
            labels = batch["lab"]
            final = final_logits.float()
            pre = pre_logits.float()
            ce = F.cross_entropy(final, labels)
            pre_ce = F.cross_entropy(pre, labels)
            probability = torch.softmax(final, dim=-1)
            margin = final[:, 1] - final[:, 0]
            shifted = torch.stack([torch.zeros_like(margin), margin - np.log(9.0)], dim=-1)
            lovasz = lovasz_softmax_flat(torch.softmax(shifted, dim=-1), labels)
            foreground = labels == 1
            if bool(foreground.sum() >= 8) and bool(batch["has_target_vel"]):
                if str(config.velocity_loss) == "squared_l2":
                    velocity_loss = ((velocity[foreground] - batch["v_gt"][foreground]) ** 2).sum(-1).mean()
                else:
                    velocity_loss = F.smooth_l1_loss(velocity[foreground], batch["v_gt"][foreground])
            else:
                velocity_loss = final.sum() * 0.0
            loss = ce + float(config.lovasz_w) * lovasz + float(config.pre_weight) * pre_ce
            loss = loss + float(config.lambda_v) * velocity_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())

        scheduler.step()
        metrics = _evaluate(model, val_batches, float(config.threshold))
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(1, len(train_batches)),
            "val": metrics,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(
            f"[MoVeR] epoch={epoch + 1}/{config.epochs} "
            f"loss={row['train_loss']:.5f} IoU={metrics['IoU']:.2f} "
            f"SegAcc={metrics['SegAcc']:.2f}",
            flush=True,
        )
        if float(metrics["IoU"]) > best_iou:
            best_iou = float(metrics["IoU"])
            best_epoch = epoch + 1
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": best_epoch,
                    "val": metrics,
                    "config": vars(config),
                },
                output / "best.pt",
            )

    summary = {
        "method": "MoVeR",
        "config": vars(config),
        "best_epoch": best_epoch,
        "best_val_IoU": best_iou,
        "history": history,
        "n_train_clips": len(train_dataset),
        "n_val_clips": len(val_dataset),
        "elapsed_s": time.perf_counter() - start,
        "checkpoint": str(output / "best.pt"),
    }
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("best_epoch", "best_val_IoU", "checkpoint")}, indent=2))


if __name__ == "__main__":
    main()
