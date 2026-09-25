"""Tests for ber.pipeline wiring (no data needed). Run directly or with pytest."""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber.config import load_config  # noqa: E402
from ber.pipeline import CHAIN, OWNER_STAGES, R1_STAGES, peak_memory_mb, resolve  # noqa: E402


def test_every_stage_resolves_and_placeholders_name_the_owner():
    """All CLI stages resolve; missing owner functions raise NotImplementedError mentioning the owner.

    Only placeholders are called: an implemented stage would process whatever is in the real cache (this test once
    started a full-train normalize after ingest had written cache/source*_train.parquet).
    """
    cfg = load_config()
    for stage in [*OWNER_STAGES, *R1_STAGES]:
        fn = resolve(stage)
        assert callable(fn), stage
    for stage, (module, func, owner) in OWNER_STAGES.items():
        if getattr(importlib.import_module(module), func, None) is not None:
            continue  # implemented: never run it here
        try:
            resolve(stage)(cfg, "train", False)
        except NotImplementedError as e:
            assert owner in str(e) and func in str(e)
        else:
            raise AssertionError(f"placeholder for {stage} did not raise")


def test_chains_only_use_known_stages():
    """Every stage in every 'all' chain exists; the fallback chains never need featurize/train/predict."""
    known = {*OWNER_STAGES, *R1_STAGES}
    for (split, scorer), stages in CHAIN.items():
        assert set(stages) <= known, (split, scorer)
        if scorer == "fallback":
            assert not {"featurize", "train", "predict"} & set(stages)


def test_peak_memory_is_positive():
    """The memory probe works on this platform."""
    assert peak_memory_mb() > 1


if __name__ == "__main__":
    test_every_stage_resolves_and_placeholders_name_the_owner()
    test_chains_only_use_known_stages()
    test_peak_memory_is_positive()
    print("pipeline tests passed")
