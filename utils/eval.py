from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class EventMetrics:
    threshold: float = 0.90

    def compute(self, predictions: list[np.ndarray], labels: list[np.ndarray]) -> dict[str, float | int]:
        prob = np.concatenate([np.asarray(x).reshape(-1) for x in predictions])
        truth = np.concatenate([np.asarray(x).reshape(-1).astype(bool) for x in labels])
        pred = prob >= self.threshold
        tp = int(np.count_nonzero(pred & truth))
        fp = int(np.count_nonzero(pred & ~truth))
        fn = int(np.count_nonzero(~pred & truth))
        tn = int(np.count_nonzero(~pred & ~truth))
        return {
            "IoU": 100.0 * tp / max(1, tp + fp + fn),
            "SegAcc": 100.0 * tp / max(1, tp + fn),
            "Precision": 100.0 * tp / max(1, tp + fp),
            "Accuracy": 100.0 * (tp + tn) / max(1, len(truth)),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "total_events": int(len(truth)),
        }


class evalute:
    """EV-UAV-compatible event evaluator with a deterministic threshold contract."""

    def __init__(self, cfg: Any = None, threshold: float | None = None):
        self.matches: dict[str, dict[str, Any]] = {}
        if threshold is None:
            threshold = float(getattr(cfg, "threshold", 0.90)) if cfg is not None else 0.90
        self.threshold = float(threshold)
        self.pd_detT = float(getattr(cfg, "pd_detT", 50.0)) if cfg is not None else 50.0
        self.correct_thresh = float(getattr(cfg, "correct_thresh", 0.0001)) if cfg is not None else 0.0001
        self._roc_records: list[dict[str, Any]] = []

    def roc_update(self, ts, preds, idx, label, ev_locs, thresh: float | None = None):
        """Accumulate the EV-UAV object/frame Pd and false-alarm protocol."""
        ts = np.asarray(ts).reshape(-1)
        preds = np.asarray(preds).reshape(-1)
        idx = np.asarray(idx).reshape(-1)
        label = np.asarray(label).reshape(-1)
        ev_locs = np.asarray(ev_locs)
        if not (len(ts) == len(preds) == len(idx) == len(label) == len(ev_locs)):
            raise ValueError("roc_update inputs must have the same event count")
        if ev_locs.ndim != 2 or ev_locs.shape[1] < 3:
            raise ValueError("ev_locs must contain x, y, and t columns")
        threshold = self.threshold if thresh is None else float(thresh)
        self._roc_records.append({
            "ts": ts,
            "preds": preds,
            "idx": idx,
            "label": label,
            "ev_locs": ev_locs,
            "threshold": threshold,
        })

    @staticmethod
    def _component_count(points: np.ndarray) -> int:
        """Count 8-connected false-alarm pixels without adding an OpenCV dependency."""
        if len(points) == 0:
            return 0
        pending = {(int(x), int(y)) for x, y in points[:, :2]}
        components = 0
        while pending:
            components += 1
            stack = [pending.pop()]
            while stack:
                x, y = stack.pop()
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        neighbour = (x + dx, y + dy)
                        if neighbour in pending:
                            pending.remove(neighbour)
                            stack.append(neighbour)
        return components

    def event_metrics(self) -> dict[str, float | int]:
        predictions = [v["seg_pred"] for v in self.matches.values()]
        labels = [v["seg_gt"] for v in self.matches.values()]
        if not predictions:
            raise RuntimeError("no predictions registered")
        return EventMetrics(self.threshold).compute(predictions, labels)

    def evaluate_semantic_segmantation_miou(self, thresh: float | None = None):
        if thresh is not None:
            old = self.threshold
            self.threshold = float(thresh)
            value = self.event_metrics()["IoU"]
            self.threshold = old
        else:
            value = self.event_metrics()["IoU"]
        return torch.tensor(float(value))

    def evaluate_semantic_segmantation_accuracy(self, thresh: float | None = None):
        if thresh is not None:
            old = self.threshold
            self.threshold = float(thresh)
            value = self.event_metrics()["SegAcc"]
            self.threshold = old
        else:
            value = self.event_metrics()["SegAcc"]
        return torch.tensor(float(value))

    def cal_roc(self):
        if not self._roc_records:
            return 0.0, 0.0
        correct = 0
        objects = 0
        false_components = 0
        frames = 0
        for record in self._roc_records:
            ts = record["ts"]
            preds = record["preds"]
            idx = record["idx"]
            labels = record["label"].astype(bool)
            ev_locs = record["ev_locs"]
            threshold = float(record["threshold"])
            pred = preds >= threshold
            duration = max(0.0, float(ts.max() - ts.min())) if len(ts) else 0.0
            n_frames = int(duration / self.pd_detT)
            frames += n_frames
            for frame in range(n_frames + 1):
                mask = (ts > frame * self.pd_detT) & (ts < (frame + 1) * self.pd_detT)
                if not mask.any():
                    continue
                for object_id in np.unique(idx[mask]):
                    if object_id == 0:
                        continue
                    objects += 1
                    object_mask = mask & (idx == object_id)
                    if (
                        (pred[object_mask] == labels[object_mask]).sum()
                        / max(1, int(labels[object_mask].sum()))
                        >= self.correct_thresh
                    ):
                        correct += 1
                false_points = ev_locs[mask & (~labels) & pred]
                false_components += self._component_count(false_points[:, :2])
        pd = correct / max(1, objects)
        fa = false_components / max(1, frames * 346 * 260)
        return pd, fa
