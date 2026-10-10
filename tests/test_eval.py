import numpy as np
import pytest
import torch

from utils.eval import evalute


def test_evuav_style_evaluator_reports_event_metrics():
    evaluator = evalute(threshold=0.90)
    evaluator.matches["0"] = {
        "seg_pred": torch.tensor([0.99, 0.95, 0.10, 0.05]),
        "seg_gt": torch.tensor([1.0, 0.0, 1.0, 0.0]),
    }

    assert evaluator.evaluate_semantic_segmantation_miou().item() == pytest.approx(33.333333, rel=1e-5)
    assert evaluator.evaluate_semantic_segmantation_accuracy().item() == 50.0


def test_evaluator_accepts_numpy_predictions():
    evaluator = evalute(threshold=0.90)
    evaluator.matches["0"] = {
        "seg_pred": np.array([0.99, 0.01]),
        "seg_gt": np.array([1, 0]),
    }
    metrics = evaluator.event_metrics()
    assert metrics["IoU"] == 100.0
    assert metrics["SegAcc"] == 100.0


def test_roc_matches_evuav_object_and_connected_false_alarm_units():
    evaluator = evalute(threshold=0.90)
    ts = np.array([1.0, 2.0, 3.0, 4.0])
    probability = np.array([0.99, 0.99, 0.99, 0.01])
    target_ids = np.array([5, 5, 0, 0])
    labels = np.array([1, 1, 0, 0])
    # x/y points (the false-positive point is one isolated component).
    ev_locs = np.array([[10, 10, 1], [11, 10, 2], [100, 100, 3], [200, 200, 4]])
    evaluator.roc_update(ts, probability, target_ids, labels, ev_locs)

    pd, fa = evaluator.cal_roc()
    assert pd == pytest.approx(1.0)
    assert fa > 0.0
