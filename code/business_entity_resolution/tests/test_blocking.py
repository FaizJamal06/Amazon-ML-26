"""Tests for ber.blocking.merge (pass merging, ranking, caps, config keys). Run directly or with pytest."""
import contextlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

PKG = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PKG / "src"), str(PKG / "scripts")]

import make_stub_data as stub  # noqa: E402
from ber.blocking.merge import (  # noqa: E402
    PASS_ADDR_KEY, PASS_HOUSE_NUM, PASS_NAME_TOKEN, _cap_candidates, _merge_passes, rank_scores)
from ber.pipeline import main  # noqa: E402

REC_COLS = ["entity_id", "name_core", "name_skeleton", "addr_street", "house_num"]


def _pass(rows: list[tuple[str, str, float]]) -> pl.DataFrame:
    """One pass's output: (cand_id, s1_id, score) rows."""
    return pl.DataFrame(rows, schema=["cand_id", "s1_id", "score"], orient="row")


def _merge(passes: list[tuple[pl.DataFrame, int]]) -> pl.DataFrame:
    """Raw hits of several passes (one country) -> merged pairs, normalized by each pass's max, like build_candidates."""
    raw = pl.concat([p.with_columns(bit=pl.lit(bit, pl.Int8)) for p, bit in passes])
    return _merge_passes(raw, dict(raw.group_by("bit").agg(pl.col("score").max()).iter_rows()))


def _cands(rows: list[tuple[str, str, float, float]]) -> pl.DataFrame:
    """Merged candidates with an explicit rank_score: (s1_id, cand_id, block_score, rank_score) rows."""
    return pl.DataFrame(rows, schema=["s1_id", "cand_id", "block_score", "rank_score"], orient="row") \
        .with_columns(block_mask=pl.lit(1, pl.Int32))


def test_merge_normalizes_each_pass():
    """Each pass's score is scaled to 0-1 by its own max; block_score sums over passes, block_mask ORs the bits."""
    passes = [(_pass([("c1", "A", 3.0), ("c9", "Z", 10.0)]), PASS_NAME_TOKEN),
              (_pass([("c1", "A", 2.0), ("c9", "Z", 4.0)]), PASS_ADDR_KEY),
              (_pass([("c1", "B", 9.0)]), PASS_HOUSE_NUM)]
    a = _merge(passes).filter(pl.col("s1_id") == "A")
    assert a["block_mask"][0] == 0b011 and abs(a["block_score"][0] - (0.3 + 0.5)) < 1e-9


def test_exact_name_same_street_outranks_shared_house_number():
    """S1 'A' (exact name + same street, no house number) outranks S1 'B' that only shares the house number, even
    though B's blocking score is higher; with k_per_query=1 only A survives. Shuffled input -> same output."""
    rec = pl.DataFrame([("c1", "anand food", "anand fud", "mg road", "12"),
                        ("A", "anand food", "anand fud", "mg road", ""),
                        ("B", "vijay textiles", "vijay textils", "station road", "12")], schema=REC_COLS, orient="row")
    passes = [(_pass([("c1", "A", 3.0), ("c9", "A", 10.0)]), PASS_NAME_TOKEN), (_pass([("c1", "B", 9.0)]), PASS_HOUSE_NUM)]
    merged = _merge(passes).filter(pl.col("cand_id") == "c1")
    merged = merged.with_columns(rank_scores(merged, rec))
    score = dict(merged.select("s1_id", "rank_score").rows())
    assert abs(score["A"] - 0.90) < 1e-6 and score["B"] < 0.5, score
    assert merged.filter(pl.col("s1_id") == "B")["block_score"][0] > merged.filter(pl.col("s1_id") == "A")["block_score"][0]
    outs = [_cap_candidates(merged.sample(fraction=1.0, shuffle=True, seed=s), k_per_query=1, max_cands=10)
            .select("s1_id", "cand_id", "rank_in_cand").rows() for s in range(3)]
    assert outs == [[("A", "c1", 1)]] * 3


def test_rank_in_cand_is_computed_before_capping():
    """c1's best S1 'X' drops c1 under the per-S1 cap (X has a better candidate); c1 keeps rank 2 with 'Y'
    (not renumbered to 1). The min_score floor (on block_score) does not renumber either."""
    cand = _cands([("X", "c1", 0.9, 0.8), ("X", "c0", 0.9, 0.9), ("Y", "c1", 0.9, 0.7), ("W", "c1", 0.9, 0.1)])
    out = _cap_candidates(cand, k_per_query=3, max_cands=1)
    assert out.filter(pl.col("cand_id") == "c1").select("s1_id", "rank_in_cand").rows() == [("Y", 2), ("W", 3)]
    cand = _cands([("X", "c1", 0.1, 0.9), ("Y", "c1", 0.9, 0.5)])
    out = _cap_candidates(cand, k_per_query=3, max_cands=10, min_score=0.5)
    assert out.select("s1_id", "rank_in_cand").rows() == [("Y", 2)]


def test_ties_are_deterministic():
    """Equal rank_score and block_score: s1_id asc wins, whatever the input row order."""
    rows = [(s, "c1", 0.5, 0.7) for s in ("S1-9", "S1-3", "S1-5")] + [("S1-1", "c2", 0.5, 0.7)]
    outs = []
    for seed in range(5):
        shuffled = [rows[i] for i in np.random.default_rng(seed).permutation(len(rows))]
        out = _cap_candidates(_cands(shuffled), k_per_query=2, max_cands=10)
        outs.append(out.sort("cand_id", "rank_in_cand").select("cand_id", "s1_id", "rank_in_cand").rows())
    assert all(o == outs[0] for o in outs)
    assert outs[0][:2] == [("c1", "S1-3", 1), ("c1", "S1-5", 2)]


def test_config_k_per_query_is_honored():
    """normalize + block on a stub world: --set blocking.k_per_query=1 gives fewer pairs than the default (2),
    and every kept rank_in_cand is <= k. Streaming in tiny chunks (7 query records) gives the identical output."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        cache = Path(d)
        rec, _, _ = stub.build_split("train", ["US", "India"], 30, np.random.default_rng(42))
        for src in (1, 2, 3):
            rec.filter(pl.col("source") == src).select(
                "entity_id", business_name="name_raw", business_address="addr_raw", country="country",
                source=pl.col("source").cast(pl.String)).write_parquet(cache / f"source{src}_train.parquet")
        sets = ["--set", f"paths.cache_dir={cache}"]
        with contextlib.redirect_stdout(io.StringIO()):
            main(["normalize", "--split", "train", *sets])
            main(["block", "--split", "train", *sets])
            k2 = pl.read_parquet(cache / "candidates_train.parquet")
            main(["block", "--split", "train", *sets, "--set", "blocking.k_per_query=1"])
            k1 = pl.read_parquet(cache / "candidates_train.parquet")
            main(["block", "--split", "train", *sets, "--set", "blocking.chunk_rows=7"])
            k2_small_chunks = pl.read_parquet(cache / "candidates_train.parquet")
    assert k2["rank_in_cand"].max() == 2 and k1["rank_in_cand"].max() == 1
    assert k1.height < k2.height
    assert k2_small_chunks.equals(k2)


if __name__ == "__main__":
    test_merge_normalizes_each_pass()
    test_exact_name_same_street_outranks_shared_house_number()
    test_rank_in_cand_is_computed_before_capping()
    test_ties_are_deterministic()
    test_config_k_per_query_is_honored()
    print("blocking tests passed")
