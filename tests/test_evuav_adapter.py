from pathlib import Path

import numpy as np

from dataset.ev_uav import EvUAV, convert_official_arrays


def _sample_arrays():
    evs_norm = np.array(
        [
            [0.1, 0.2, 0.0, 1.0, 0.0, 0.0],
            [0.3, 0.4, 0.5, 0.0, 1.0, 7.0],
            [0.5, 0.6, 1.0, 1.0, 1.0, 7.0],
        ],
        dtype=np.float32,
    )
    ev_loc = np.array([[10, 20, 0], [30, 40, 500], [50, 60, 1000]], dtype=np.int64)
    ev = np.array(
        [(10, 20, 0.0, 1, 0, 0), (30, 40, 500.0, 0, 1, 7), (50, 60, 1000.0, 1, 1, 7)],
        dtype=[("x", "i2"), ("y", "i2"), ("t", "f8"), ("p", "i1"), ("label", "i1"), ("name", "i1")],
    )
    return {"evs_norm": evs_norm, "ev_loc": ev_loc, "ev": ev}


def test_official_evuav_arrays_are_mapped_to_mover_clip():
    clip = convert_official_arrays(_sample_arrays(), name="val_000")

    assert clip["name"] == "val_000"
    assert clip["locs"].tolist() == [[0, 10, 20, 0], [0, 30, 40, 500], [0, 50, 60, 1000]]
    np.testing.assert_array_equal(clip["ts"], [0.0, 500.0, 1000.0])
    np.testing.assert_array_equal(clip["seg"], [0, 1, 1])
    np.testing.assert_array_equal(clip["feats"], _sample_arrays()["evs_norm"][:, :4])
    np.testing.assert_array_equal(clip["idx"], [0, 7, 7])


def test_official_label_and_name_fields_are_the_authoritative_metadata():
    arrays = _sample_arrays()
    arrays["evs_norm"][:, 4:] = 0
    clip = convert_official_arrays(arrays, name="val_000")

    np.testing.assert_array_equal(clip["seg"], arrays["ev"]["label"])
    np.testing.assert_array_equal(clip["idx"], arrays["ev"]["name"])


def test_official_archive_layout_is_loaded(tmp_path: Path):
    (tmp_path / "train").mkdir()
    np.savez(tmp_path / "train" / "train_000.npz", **_sample_arrays())

    clip = EvUAV(tmp_path, mode="train")[0]

    assert clip["name"] == "train_000"
    assert clip["locs"].shape == (3, 4)
    assert clip["seg"].tolist() == [0, 1, 1]


def test_flat_cached_layout_is_loaded_without_reencoding(tmp_path: Path):
    for split in ("train", "val", "test"):
        (tmp_path / split).mkdir()
    locs = np.arange(12).reshape(3, 4)
    np.savez(
        tmp_path / "train" / "train_000.npz",
        locs=locs,
        feats=np.ones((3, 4), dtype=np.float32),
        seg=np.array([0, 1, 0], dtype=np.int64),
        idx=np.array([0, 0, 0], dtype=np.int64),
    )

    clip = EvUAV(tmp_path, mode="train")[0]

    assert clip["name"] == "train_000"
    np.testing.assert_array_equal(clip["locs"], locs)
    assert clip["seg"].tolist() == [0, 1, 0]


def test_train_cap_is_reproducible_for_flat_cache(tmp_path: Path):
    for split in ("train", "val", "test"):
        (tmp_path / split).mkdir()
    n = 20
    np.savez(
        tmp_path / "train" / "train_000.npz",
        locs=np.arange(n * 4).reshape(n, 4),
        feats=np.ones((n, 4), dtype=np.float32),
        seg=np.zeros(n, dtype=np.int64),
        idx=np.zeros(n, dtype=np.int64),
    )

    first = EvUAV(tmp_path, mode="train", max_events_num=5)[0]
    second = EvUAV(tmp_path, mode="train", max_events_num=5)[0]

    np.testing.assert_array_equal(first["locs"], second["locs"])
