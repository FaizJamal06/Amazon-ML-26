"""Tests for ber.pipeline wiring (no data needed). Run directly or with pytest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber.config import load_config  # noqa: E402
from ber.pipeline import OWNER_STAGES, R1_STAGES, peak_memory_mb, resolve  # noqa: E402


def test_every_stage_resolves_and_placeholders_name_the_owner():
    """All CLI stages resolve; missing owner functions raise NotImplementedError mentioning the owner."""
    cfg = load_config()
    for stage in [*OWNER_STAGES, *R1_STAGES]:
        fn = resolve(stage)
        assert callable(fn), stage
    module, func, owner = OWNER_STAGES["normalize"]
    try:
        resolve("normalize")(cfg, "train", False)
    except NotImplementedError as e:
        assert owner in str(e) and func in str(e)
    else:  # owner has implemented it: fine, nothing to check
        pass


def test_peak_memory_is_positive():
    """The memory probe works on this platform."""
    assert peak_memory_mb() > 1


if __name__ == "__main__":
    test_every_stage_resolves_and_placeholders_name_the_owner()
    test_peak_memory_is_positive()
    print("pipeline tests passed")
