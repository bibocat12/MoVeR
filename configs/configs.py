from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import yaml


_ALLOWED = {
    "STRUCTURE": {
        "model_name", "c_stem", "c_graph", "k",
    },
    "DATA": {
        "root", "whole_t", "res", "normalization_path", "max_events_num",
        "event_feature_channels", "tau_scale", "n_blocks",
    },
    "TRAIN": {
        "epochs", "batch_size", "train_workers", "optim", "optimizer_mode", "lr",
        "weight_decay", "stem_lr", "local_lr", "graph_lr", "verification_lr",
        "scheduler", "eta_min", "velocity_loss", "lovasz_w", "lambda_v", "pre_weight",
        "deterministic", "seed", "model_save_root",
    },
    "TEST": {
        "eval", "roc", "threshold", "pd_detT", "correct_thresh",
        "model_path", "save", "output_root", "deterministic", "warmup", "repeats", "split",
    },
}

_DEFAULTS: dict[str, Any] = {
    "model_name": "mover",
    "c_stem": 64,
    "c_graph": 16,
    "k": 8,
    "root": "data/EV-UAV",
    "whole_t": 8000,
    "res": [346, 260],
    "normalization_path": "configs/train_normalization.json",
    "max_events_num": None,
    "event_feature_channels": 4,
    "tau_scale": 64.0,
    "n_blocks": 32,
    "epochs": 120,
    "batch_size": 1,
    "train_workers": 0,
    "optim": "AdamW",
    "optimizer_mode": "single",
    "lr": 2e-4,
    "weight_decay": 1e-2,
    "stem_lr": 5e-4,
    "local_lr": 5e-4,
    "graph_lr": 1e-3,
    "verification_lr": 1e-3,
    "scheduler": "cosine",
    "eta_min": 1e-5,
    "velocity_loss": "smooth_l1",
    "lovasz_w": 0.15,
    "lambda_v": 0.02,
    "pre_weight": 0.10,
    "deterministic": False,
    "seed": 37,
    "model_save_root": "checkpoints/mover",
    "eval": True,
    "roc": True,
    "threshold": 0.90,
    "pd_detT": 50.0,
    "correct_thresh": 0.0001,
    "model_path": "",
    "save": True,
    "output_root": "results",
    "deterministic": False,
    "warmup": 10,
    "repeats": 50,
    "split": "test",
}


def _flatten_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    unknown_sections = set(raw) - set(_ALLOWED)
    if unknown_sections:
        raise ValueError(f"unknown config section(s): {sorted(unknown_sections)}")

    flat = dict(_DEFAULTS)
    for section, values in raw.items():
        if values is None:
            continue
        if not isinstance(values, Mapping):
            raise ValueError(f"config section {section} must be a mapping")
        unknown = set(values) - _ALLOWED[section]
        if unknown:
            raise ValueError(
                f"unknown config key(s) in {section}: {sorted(unknown)}"
            )
        flat.update(values)

    if int(flat["batch_size"]) != 1:
        raise ValueError("MoVeR currently requires batch_size=1 for clip isolation")
    if len(flat["res"]) != 2:
        raise ValueError("DATA.res must be [width, height]")
    if int(flat["c_stem"]) <= 0 or int(flat["c_graph"]) <= 0:
        raise ValueError("channel widths must be positive")
    if int(flat["k"]) <= 0:
        raise ValueError("k must be positive")
    return flat


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> SimpleNamespace:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    flat = _flatten_config(raw)
    if overrides:
        flat.update(overrides)
    cfg = SimpleNamespace(**flat)
    cfg.config_path = str(path)
    return cfg


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MoVeR event-stream segmentation")
    parser.add_argument("--config", default="configs/mover_evuav.yaml")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def parse_config_args(argv: list[str] | None = None) -> SimpleNamespace:
    args = get_parser().parse_args(argv)
    overrides = {} if args.seed is None else {"seed": args.seed}
    return load_config(args.config, overrides=overrides)
