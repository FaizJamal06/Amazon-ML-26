"""Tests for ber.eval.split. Run: python -m pytest code/business_entity_resolution/tests  (or run this file directly)."""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber.eval.split import make_folds, sample_subworld  # noqa: E402


def toy():
    """1000 S1s over two countries with 0-6 matches each, plus candidates with one decoy per S1."""
    s1 = pl.DataFrame({"s1_id": [f"S1-{i}" for i in range(1000)], "country": ["US", "India"] * 500})
    gt = pl.DataFrame([(f"S1-{i}", f"S2-{i}-{j}") for i in range(1000) for j in range(i % 7)],
                      schema=["s1_id", "match_id"], orient="row")
    cands = pl.concat([
        gt.select("s1_id", cand_id="match_id", rank_in_cand=pl.lit(1)),
        s1.select("s1_id", cand_id=pl.format("S3-d{}", "s1_id"), rank_in_cand=pl.lit(1)),
        s1.select("s1_id", cand_id=pl.format("S3-d{}", "s1_id").shift(1).fill_null("S3-x"), rank_in_cand=pl.lit(2)),
    ])
    return s1, gt, cands


def test_folds_balanced_deterministic_stratified():
    """Every S1 gets exactly one fold; strata are balanced; same seed -> same folds regardless of input order."""
    s1, gt, _ = toy()
    f = make_folds(s1, gt, 5, 42)
    assert f.height == 1000 and set(f["fold"].unique()) == {0, 1, 2, 3, 4}
    sizes = f.group_by("country", pl.col("n_matches").clip(upper_bound=5), "fold").len()
    spread = sizes.group_by("country", "n_matches").agg((pl.col("len").max() - pl.col("len").min()).alias("d"))
    assert spread["d"].max() <= 1
    assert f.sort("s1_id").equals(make_folds(s1.reverse(), gt, 5, 42).sort("s1_id"))
    assert not f.sort("s1_id").equals(make_folds(s1, gt, 5, 7).sort("s1_id"))


def test_subworld_closed():
    """Sampled S1s ~10% per country; all their matches and owned decoys are in; candidates stay inside the world."""
    s1, gt, cands = toy()
    world, cw = sample_subworld(s1, gt, cands, 0.1, 42)
    sampled = set(world.filter(pl.col("role") == "s1")["entity_id"])
    assert len(sampled) == 100
    want_matches = set(gt.filter(pl.col("s1_id").is_in(list(sampled)))["match_id"])
    assert set(world.filter(pl.col("role") == "match")["entity_id"]) == want_matches
    assert set(world.filter(pl.col("role") == "decoy")["entity_id"]) == {f"S3-d{s}" for s in sampled}
    assert set(cw["s1_id"]) <= sampled and set(cw["cand_id"]) <= set(world["entity_id"])
    assert "S3-x" not in set(world["entity_id"])


if __name__ == "__main__":
    test_folds_balanced_deterministic_stratified()
    test_subworld_closed()
    print("split tests passed")
