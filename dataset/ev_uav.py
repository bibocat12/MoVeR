from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_KEYS = {"evs_norm", "ev_loc", "ev"}
CACHED_KEYS = {"locs", "feats", "seg"}


def convert_cached_arrays(arrays: dict[str, Any], name: str) -> dict[str, Any]:
    missing = CACHED_KEYS - set(arrays)
    if missing:
        raise ValueError(f"missing flat MoVeR cache arrays: {sorted(missing)}")
    locs = np.array(arrays["locs"], dtype=np.int64, copy=True)
    feats = np.array(arrays["feats"], dtype=np.float32, copy=True)
    seg = np.array(arrays["seg"], dtype=np.int64, copy=True)
    if locs.ndim != 2 or locs.shape[1] != 4:
        raise ValueError("cached locs must have shape (N, 4) = (clip, x, y, t)")
    if feats.ndim != 2 or feats.shape[1] < 4:
        raise ValueError("cached feats must have at least four channels")
    if seg.ndim != 1:
        raise ValueError("cached seg must be one-dimensional")
    n = min(len(locs), len(feats), len(seg))
    if n == 0:
        raise ValueError("cached MoVeR clip is empty")
    idx = np.array(arrays.get("idx", np.zeros(n, dtype=np.int64)), dtype=np.int64, copy=True)[:n]
    return {
        "name": str(name),
        "locs": locs[:n],
        "ts": locs[:n, 3].astype(np.float64, copy=True),
        "seg": seg[:n],
        "feats": feats[:n, :4],
        "idx": idx,
    }


def convert_official_arrays(arrays: dict[str, Any], name: str) -> dict[str, Any]:
    """Convert one official EV-UAV archive into MoVeR's clip representation.

    The official release stores event locations in ``ev_loc`` as ``(x, y, t)``;
    labels and target IDs are taken from the structured ``ev`` record. The
    normalized four event channels are copied from ``evs_norm[:, :4]``.
    """
    missing = REQUIRED_KEYS - set(arrays)
    if missing:
        raise ValueError(f"missing official EV-UAV arrays: {sorted(missing)}")

    ev_loc = np.asarray(arrays["ev_loc"])
    evs_norm = np.asarray(arrays["evs_norm"])
    ev = np.asarray(arrays["ev"])
    if ev_loc.ndim != 2 or ev_loc.shape[1] != 3:
        raise ValueError("ev_loc must have shape (N, 3) = (x, y, t)")
    if evs_norm.ndim != 2 or evs_norm.shape[1] < 4:
        raise ValueError("evs_norm must have at least four feature channels")
    if ev.ndim != 1 or not ev.dtype.names:
        raise ValueError("ev must be a structured event array")
    for field in ("label", "name"):
        if field not in ev.dtype.names:
            raise ValueError(f"official ev array lacks field {field!r}")
    n = min(len(ev_loc), len(evs_norm), len(ev))
    if n == 0:
        raise ValueError("official EV-UAV clip is empty")

    ev_loc = ev_loc[:n]
    evs_norm = evs_norm[:n]
    ev = ev[:n]
    locs = np.zeros((n, 4), dtype=np.int64)
    locs[:, 1:] = ev_loc.astype(np.int64, copy=False)
    return {
        "name": str(name),
        "locs": locs,
        "ts": np.array(ev["t"], dtype=np.float64, copy=True),
        "seg": np.array(ev["label"], dtype=np.int64, copy=True),
        "feats": np.array(evs_norm[:, :4], dtype=np.float32, copy=True),
        "idx": np.array(ev["name"], dtype=np.int64, copy=True),
    }


class EvUAV:
    """EV-UAV-style dataset: one archive is one isolated forward sample."""

    def __init__(
        self,
        root: str | Path,
        mode: str = "train",
        max_events_num: int | None = None,
        sample_seed: int = 0,
    ):
        base = Path(root)
        self.root = base / mode
        self.mode = mode
        self.max_events_num = max_events_num
        self.sample_seed = int(sample_seed)
        self.file_list = sorted(self.root.glob("*.npz"))
        # The legacy paper cache stores train_*.npz/val_*.npz/test_*.npz
        # directly under one root rather than in split directories.
        if not self.file_list:
            self.root = base
            self.file_list = sorted(base.glob(f"{mode}_*.npz"))
        if not self.file_list:
            raise FileNotFoundError(
                f"no EV-UAV archives found under {base / mode} or {base / (mode + '_*.npz')}"
            )

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.file_list[index]
        with np.load(path, allow_pickle=True) as arrays:
            raw = {key: arrays[key] for key in arrays.files}
        if CACHED_KEYS.issubset(raw):
            clip = convert_cached_arrays(raw, path.stem)
        else:
            clip = convert_official_arrays(raw, path.stem)
        if self.mode == "train" and self.max_events_num is not None:
            n = len(clip["locs"])
            if n > self.max_events_num:
                # Derive the capped subset from the run seed and archive name,
                # so indexing order and process restarts cannot silently alter
                # the training population.
                import hashlib
                token = f"{self.sample_seed}:{self.mode}:{path.name}".encode()
                seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
                rng = np.random.default_rng(seed)
                keep = rng.choice(n, self.max_events_num, replace=False)
                keep.sort()
                for key in ("locs", "seg", "feats", "idx"):
                    clip[key] = clip[key][keep]
        return clip

    @staticmethod
    def collate_one(batch: list[dict[str, Any]]) -> dict[str, Any]:
        if len(batch) != 1:
            raise ValueError("MoVeR uses one EV-UAV clip per forward; set batch_size=1")
        return batch[0]
