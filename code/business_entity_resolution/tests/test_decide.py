"""Tests for ber.decide. Run directly or with pytest."""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber.decide.assign import assign_one_to_one  # noqa: E402
from ber.decide.select import best_threshold, select_threshold, threshold_curve, threshold_grid  # noqa: E402


def test_assign_keeps_best_s1_with_tiebreaks():
    """Highest p wins; equal p -> higher block_score; equal both -> smaller s1_id."""
    s = pl.DataFrame({"s1_id": ["S1-a", "S1-b", "S1-a", "S1-b", "S1-b", "S1-a"],
                      "cand_id": ["S2-1", "S2-1", "S2-2", "S2-2", "S3-3", "S3-3"],
                      "p": [0.9, 0.8, 0.7, 0.7, 0.5, 0.5],
                      "block_score": [0.1, 0.9, 0.2, 0.6, 0.4, 0.4]})
    got = dict(assign_one_to_one(s).select("cand_id", "s1_id").iter_rows())
    assert got == {"S2-1": "S1-a", "S2-2": "S1-b", "S3-3": "S1-a"}


def test_assign_joins_block_score_from_candidates():
    """block_score is looked up from candidates when scores lack it."""
    s = pl.DataFrame({"s1_id": ["S1-a", "S1-b"], "cand_id": ["S2-1", "S2-1"], "p": [0.5, 0.5]})
    c = pl.DataFrame({"s1_id": ["S1-a", "S1-b"], "cand_id": ["S2-1", "S2-1"], "block_score": [0.1, 0.2]})
    assert assign_one_to_one(s, c)["s1_id"].to_list() == ["S1-b"]


def test_threshold_and_curve():
    """Threshold + margin filter; curve picks the t that drops the false merge but keeps the true match."""
    scores = pl.DataFrame({"s1_id": ["S1-a", "S1-b"], "cand_id": ["S2-1", "S2-2"], "p": [0.9, 0.6],
                           "block_score": [1.0, 1.0]})
    assert select_threshold(scores, 0.5, margin=0.2)["cand_id"].to_list() == ["S2-1"]
    gt = pl.DataFrame({"s1_id": ["S1-a"], "match_id": ["S2-1"]})
    s1 = pl.DataFrame({"s1_id": ["S1-a", "S1-b"], "country": ["US", "France"]})
    curve = threshold_curve(scores, gt, s1, threshold_grid(0.3, 0.95, 0.05))
    assert curve.height == 14 and curve["t"][0] == 0.3 and curve["t"][-1] == 0.95
    assert curve.filter(pl.col("t") == 0.5)["macro_f05"][0] == 0.5      # S1-b false merge
    assert curve.filter(pl.col("t") == 0.7)["macro_f05"][0] == 1.0
    assert best_threshold(curve) == 0.9 and "f05_France" in curve.columns


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} decide tests passed")
