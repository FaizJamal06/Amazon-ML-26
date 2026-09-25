"""Tests for ber.config. Run: python -m pytest code/business_entity_resolution/tests  (or run this file directly)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber.config import load_config  # noqa: E402


def test_defaults_paths_and_hash():
    """Base config loads, paths are absolute under the repo root, and the hash is stable."""
    cfg = load_config()
    assert cfg.seed == 42 and cfg.n_folds == 5 and cfg.subworld_frac == 0.1
    assert cfg.path("train.source1").is_absolute()
    assert str(cfg.path("train.source1")).startswith(str(cfg.root))
    assert cfg.artifact("candidates", "train", subworld=True).name == "candidates_train_sw.parquet"
    assert load_config().hash == cfg.hash


def test_overrides_change_hash_and_are_frozen():
    """--set and override yaml win over base values, change the hash, and the result is read-only."""
    cfg = load_config(sets=["decide.threshold=0.7", "max_cands=20"])
    assert cfg.get("decide.threshold") == 0.7 and cfg.max_cands == 20
    assert cfg.hash != load_config().hash
    try:
        cfg.data["seed"] = 1
        raise AssertionError("config must be read-only")
    except TypeError:
        pass


if __name__ == "__main__":
    test_defaults_paths_and_hash()
    test_overrides_change_hash_and_are_frozen()
    print("config tests passed")
