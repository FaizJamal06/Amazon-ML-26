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
from ber.blocking.merge import PASS_ADDR_KEY, PASS_HOUSE_NUM, PASS_NAME_TOKEN, _cap_candidates, _merge_passes  # noqa: E402
from ber.pipeline import main  # noqa: E402


def _pass(rows: list[tuple[str, str, float]]) -> pl.LazyFrame:
    """One pass's output: (cand_id, s1_id, score) rows."""
    return pl.DataFrame(rows, schema=["cand_id", "s1_id", "score"], orient="row").lazy()


def _block(passes, **cap) -> pl.DataFrame:
    """Merge passes and cap, like build_candidates does for one country."""
    return _cap_candidates(_merge_passes(passes).collect(), **cap)


def test_name_and_address_outrank_rare_house_number():
    """S1 'A' hit by name + address beats S1 'B' hit only by a rare house number, although B's raw house-number
    score (9.0) is larger than any raw score of A (max-across-passes used to rank B first)."""
    passes = [(_pass([("c1", "A", 3.0), ("c9", "Z", 10.0)]), PASS_NAME_TOKEN),
              (_pass([("c1", "A", 2.0), ("c9", "Z", 4.0)]), PASS_ADDR_KEY),
              (_pass([("c1", "B", 9.0)]), PASS_HOUSE_NUM)]
    merged = _merge_passes(passes).collect()
    a = merged.filter(pl.col("s1_id") == "A")
    assert a["block_mask"][0] == 0b011 and abs(a["block_score"][0] - (0.3 + 0.5)) < 1e-9   # normalized per pass
    out = _block(passes, k_per_query=1, max_cands=10)
    assert out.filter(pl.col("cand_id") == "c1")["s1_id"].to_list() == ["A"]
    assert _block(passes, k_per_query=2, max_cands=10).filter(pl.col("s1_id") == "B")["rank_in_cand"].to_list() == [2]


def test_rank_in_cand_is_computed_before_capping():
    """c1's best S1 'X' drops c1 under the per-S1 cap (X has a stronger candidate); c1 keeps rank 2 with 'Y'
    (not renumbered to 1), and min_score does not renumber either."""
    passes = [(_pass([("c1", "X", 5.0), ("c0", "X", 10.0), ("c1", "Y", 4.0), ("c1", "W", 1.0)]), PASS_NAME_TOKEN)]
    out = _block(passes, k_per_query=3, max_cands=1)
    assert out.filter(pl.col("cand_id") == "c1").select("s1_id", "rank_in_cand").rows() == [("Y", 2), ("W", 3)]
    # X: 2 passes but a low sum (0.1 + 0.2) ranks above Y (1 pass, 0.9); the min_score floor removes X, Y stays rank 2
    passes = [(_pass([("c1", "X", 1.0), ("c1", "Y", 9.0), ("c0", "Z", 10.0)]), PASS_NAME_TOKEN),
              (_pass([("c1", "X", 1.0), ("c0", "Z", 5.0)]), PASS_ADDR_KEY)]
    out = _block(passes, k_per_query=3, max_cands=10, min_score=0.5)
    assert out.filter(pl.col("cand_id") == "c1").select("s1_id", "rank_in_cand").rows() == [("Y", 2)]


def test_ties_are_deterministic():
    """Equal n_passes and block_score: s1_id asc wins, whatever the input row order."""
    rows = [("c1", s, 2.0) for s in ("S1-9", "S1-3", "S1-5")] + [("c2", "S1-1", 2.0)]
    outs = []
    for seed in range(5):
        shuffled = [rows[i] for i in np.random.default_rng(seed).permutation(len(rows))]
        out = _block([(_pass(shuffled), PASS_NAME_TOKEN)], k_per_query=2, max_cands=10)
        outs.append(out.sort("cand_id", "rank_in_cand").select("cand_id", "s1_id", "rank_in_cand").rows())
    assert all(o == outs[0] for o in outs)
    assert outs[0][:2] == [("c1", "S1-3", 1), ("c1", "S1-5", 2)]


def test_config_k_per_query_is_honored():
    """normalize + block on a stub world: --set blocking.k_per_query=1 gives fewer pairs than the default (2),
    and every kept rank_in_cand is <= k."""
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
    assert k2["rank_in_cand"].max() == 2 and k1["rank_in_cand"].max() == 1
    assert k1.height < k2.height


if __name__ == "__main__":
    test_name_and_address_outrank_rare_house_number()
    test_rank_in_cand_is_computed_before_capping()
    test_ties_are_deterministic()
    test_config_k_per_query_is_honored()
    print("blocking tests passed")
