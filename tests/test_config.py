from pathlib import Path

import pytest

from configs.configs import load_config


def test_ev_uav_style_config_is_flattened(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        """
STRUCTURE:
  model_name: mover
  c_stem: 64
DATA:
  root: /data/EV-UAV
TRAIN:
  batch_size: 1
TEST:
  eval: true
""",
        encoding="utf-8",
    )

    cfg = load_config(cfg_file)

    assert cfg.model_name == "mover"
    assert cfg.c_stem == 64
    assert cfg.root == "/data/EV-UAV"
    assert cfg.batch_size == 1
    assert cfg.eval is True


def test_unknown_section_key_fails_closed(tmp_path: Path):
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text("DATA:\n  typo_not_supported: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown config key"):
        load_config(cfg_file)
