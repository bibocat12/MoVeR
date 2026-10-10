#!/usr/bin/env python3
"""EV-UAV-style MoVeR inference/evaluation entry point.

The protocol mirrors the public EV-UAV runner:
- ``DATA.root/<split>/*.npz`` is the dataset layout;
- one archive/clip is one isolated forward sample;
- predictions are mapped back to event order;
- event metrics are computed after the loop;
- ``--measure-runtime`` times model-only forward, excluding file I/O,
  feature preparation, and host-to-device transfer.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from configs.configs import load_config
from dataset.ev_uav import EvUAV
from dataset.mover_runtime import build_model_from_config, load_checkpoint, prepare_clip
from utils.eval import evalute


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate MoVeR on EV-UAV-format archives")
    p.add_argument("--config", default="configs/mover_evuav.yaml")
    p.add_argument("--split", default=None, choices=["train", "val", "test"])
    p.add_argument("--model-path", default=None)
    p.add_argument("--root", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--measure-runtime", action="store_true")
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--runtime-split", default="train", choices=["train", "val", "test"],
                   help="split used for the optional model-only runtime fixture")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--output", default=None)
    p.add_argument("--limit", type=int, default=None)
    return p


def _load_normalization(config: SimpleNamespace, dataset: EvUAV) -> tuple[np.ndarray, np.ndarray]:
    path = Path(config.normalization_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return np.asarray(payload["mean"], dtype=np.float32), np.asarray(payload["std"], dtype=np.float32)

    # EV-UAV-style fallback: calculate TRAIN normalization once, never from VAL/TEST.
    train = EvUAV(config.root, mode="train", max_events_num=None)
    from models import r2_address as r2
    parts = []
    for clip in train:
        t = clip["locs"][:, 3].astype(np.float64)
        tau = (t - t.min()) / float(config.tau_scale)
        parts.append(r2.stem_features(clip["locs"], clip["feats"], {"tau": tau, "t0": float(t.min())}))
    features = np.concatenate(parts, axis=0)
    mean = features.mean(0).astype(np.float32)
    std = (features.std(0) + 1e-6).astype(np.float32)
    return mean, std


def _set_determinism(enabled: bool) -> None:
    if not enabled:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)


def _forward(model: torch.nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    logits = model(batch)[0]
    return torch.softmax(logits.float(), dim=-1)[:, 1]


def _runtime(model: torch.nn.Module, batch: dict[str, Any], warmup: int, repeats: int) -> dict[str, Any]:
    with torch.inference_mode():
        for _ in range(warmup):
            _forward(model, batch)
    torch.cuda.synchronize()
    values = []
    with torch.inference_mode():
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _forward(model, batch)
            torch.cuda.synchronize()
            values.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(values, dtype=np.float64)
    return {
        "warmup": warmup,
        "repeats": repeats,
        "events": int(batch["n"]),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(np.mean(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "scope": "model-only forward; preparation and host-to-device transfer excluded",
    }


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    if args.split is not None:
        config.split = args.split
    if args.root is not None:
        config.root = args.root
    if args.model_path is not None:
        config.model_path = args.model_path
    if args.threshold is not None:
        config.threshold = args.threshold
    if args.warmup is not None:
        config.warmup = args.warmup
    if args.repeats is not None:
        config.repeats = args.repeats
    _set_determinism(args.deterministic or bool(config.deterministic))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    dataset = EvUAV(config.root, mode=config.split, max_events_num=None)
    clips = [dataset[i] for i in range(len(dataset))]
    if args.limit is not None:
        clips = clips[:args.limit]
    mean, std = _load_normalization(config, dataset)
    model = build_model_from_config(config, device)
    checkpoint = load_checkpoint(model, config.model_path, device)

    runtime = None
    if args.measure_runtime:
        if device.type != "cuda":
            raise ValueError("--measure-runtime requires a CUDA device")
        runtime_dataset = EvUAV(config.root, mode=args.runtime_split, max_events_num=None)
        runtime_clips = [runtime_dataset[i] for i in range(len(runtime_dataset))]
        fixture = next((c for c in runtime_clips if c["name"] == "train_014"), runtime_clips[0])
        batch = prepare_clip(
            fixture, device, mean, std,
            tau_scale=float(config.tau_scale), n_blocks=int(config.n_blocks),
        )
        runtime = _runtime(model, batch, int(config.warmup), int(config.repeats))

    evaluator = evalute(config, threshold=float(config.threshold))
    total_events = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for sample, clip in enumerate(clips):
            batch = prepare_clip(
                clip, device, mean, std,
                tau_scale=float(config.tau_scale), n_blocks=int(config.n_blocks),
            )
            probability = _forward(model, batch).detach().cpu().numpy()
            evaluator.matches[str(sample)] = {
                "seg_pred": probability,
                "seg_gt": np.asarray(clip["seg"], dtype=np.int64),
            }
            if "ts" in clip and "idx" in clip:
                evaluator.roc_update(
                    clip["ts"], probability, clip["idx"], clip["seg"],
                    np.asarray(clip["locs"])[:, 1:4],
                )
            total_events += int(batch["n"])
    elapsed = time.perf_counter() - start
    metrics = evaluator.event_metrics()
    pd, fa = evaluator.cal_roc()
    metrics.update({"Pd": 100.0 * pd, "Fa_10k": 10000.0 * fa})
    result = {
        "method": "MoVeR",
        "split": config.split,
        "root": str(config.root),
        "model_path": str(config.model_path),
        "threshold": float(config.threshold),
        "n_clips": len(clips),
        "total_events": total_events,
        "metrics": metrics,
        "loop_wall_time_s": elapsed,
        "runtime": runtime,
        "pd_fa_protocol": {
            "pd_detT_ms": float(config.pd_detT),
            "correct_thresh": float(config.correct_thresh),
            "false_alarm_unit": "8-connected components per frame normalized by 346*260",
        },
        "protocol": {
            "batch_size": 1,
            "one_archive_per_forward": True,
            "prediction_mapping": "event order preserved",
            "runtime_scope": "forward only; no CPU preparation, file I/O, or H2D transfer",
            "deterministic": bool(args.deterministic or config.deterministic),
        },
        "checkpoint_payload_keys": sorted(checkpoint),
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
