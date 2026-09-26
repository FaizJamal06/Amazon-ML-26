"""Tests for ber.features.graph — graph/cluster-consistency features.

Hand-made scenario: one S1 with 2 true duplicates + 1 decoy.
The 2 true duplicates agree with each other (same name, same house_num).
The decoy has a shifted house number — name similarity is the same, but house-number agreement differs.

Expected key properties:
- graph_same_hnum: true duplicates share house_num with siblings; decoy does NOT.
- graph_top1_agree: true match that agrees with the top-1 sibling gets 1; decoy gets 0.
- graph_cross_src_agree: true S2 match agrees with S3 sibling; decoy does NOT.
- All features fill 0.0 for S1s with only one candidate.

Run directly with: PYTHONPATH=src python tests/test_graph.py
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber.features.graph import STRONG_NAME_THR, graph_features  # noqa: E402


def _make_scenario():
    """Build a 1-S1 world with 2 true duplicates (S2, S3) and 1 decoy (S2).

    S1:           entity_id=S1-1, name_core="sunrise bakery", house_num="42"
    True1 (S2-1): name_core="sunrise bakery",     house_num="42"  ← true match
    True2 (S3-1): name_core="sunrise bakery ltd",  house_num="42"  ← true match (legal variant)
    Decoy (S2-2): name_core="sunrise bakery",     house_num="69"  ← decoy (shifted house number)

    Observations:
    - True1 and True2 share house_num=42 with each other → graph_same_hnum ≥ 1.
    - Decoy has house_num=69 → no sibling shares 69 → graph_same_hnum = 0.
    - Decoy name == True1 name, so name_tset = 1.0 for all pairs (name alone can't distinguish).
    - graph_top1_agree: checks name AND house_num agreement with top-1 sibling (True1).
      True2 agrees (name~0.9, hnum equal). Decoy disagrees (hnum differs).
    - graph_cross_src_agree: True1 (S2) agrees with True2 (S3) across sources. Decoy (S2) does NOT.
    """
    rec = pl.DataFrame({
        "entity_id":  ["S1-1", "S2-1",           "S3-1",                "S2-2"],
        "source":     [1,       2,                 3,                     2],
        "name_core":  ["sunrise bakery",
                       "sunrise bakery",
                       "sunrise bakery ltd",
                       "sunrise bakery"],
        "addr_street":["main st", "main st", "main st", "main st"],
        "house_num":  ["42",      "42",       "42",      "69"],
        "legal_form": ["",        "",        "ltd",      ""],
        "name_script":["Latin",   "Latin",   "Latin",   "Latin"],
    })
    cand = pl.DataFrame({
        "s1_id":       ["S1-1", "S1-1", "S1-1"],
        "cand_id":     ["S2-1", "S3-1", "S2-2"],
        "rank_score":  [0.95,   0.90,   0.88],
        "block_mask":  [1,      1,      1],
        "block_score": [1.0,    1.0,    1.0],
        "tfidf_name":  [None,   None,   None],
        "tfidf_full":  [None,   None,   None],
        "rank_in_cand":[1,      1,      1],
        "country":     ["US",   "US",   "US"],
        "source":      [2,      3,      2],
    })
    return rec, cand


def test_same_hnum_decoy_gets_zero():
    """Decoy (house_num=69) has no sibling sharing its house number → graph_same_hnum = 0."""
    rec, cand = _make_scenario()
    feats = graph_features(cand, rec)
    assert feats.height == 3, f"Expected 3 rows, got {feats.height}"

    true1 = feats.filter(pl.col("cand_id") == "S2-1").row(0, named=True)
    true2 = feats.filter(pl.col("cand_id") == "S3-1").row(0, named=True)
    decoy = feats.filter(pl.col("cand_id") == "S2-2").row(0, named=True)

    # True duplicates share house_num=42 with at least one sibling
    assert true1["graph_same_hnum"] >= 1, f"true1 graph_same_hnum={true1['graph_same_hnum']}"
    assert true2["graph_same_hnum"] >= 1, f"true2 graph_same_hnum={true2['graph_same_hnum']}"
    # Decoy's house_num=69 is not shared by any sibling
    assert decoy["graph_same_hnum"] == 0, \
        f"decoy graph_same_hnum={decoy['graph_same_hnum']} (expected 0)"

    print(f"  true1: same_hnum={true1['graph_same_hnum']}")
    print(f"  true2: same_hnum={true2['graph_same_hnum']}")
    print(f"  decoy: same_hnum={decoy['graph_same_hnum']}")


def test_top1_agree_decoy_gets_zero():
    """The decoy (house_num=69) does NOT agree with the top-1 sibling (S2-1, house_num=42)."""
    rec, cand = _make_scenario()
    feats = graph_features(cand, rec)

    decoy = feats.filter(pl.col("cand_id") == "S2-2").row(0, named=True)
    # Decoy has different house_num from top-1 sibling → graph_top1_agree = 0
    assert decoy["graph_top1_agree"] == 0, \
        f"decoy graph_top1_agree={decoy['graph_top1_agree']} (expected 0)"

    true2 = feats.filter(pl.col("cand_id") == "S3-1").row(0, named=True)
    print(f"  true2 top1_agree={true2['graph_top1_agree']}, decoy top1_agree={decoy['graph_top1_agree']}")


def test_cross_src_agree_decoy_gets_zero():
    """Decoy (S2, house_num=69) does NOT agree with any S3 sibling (S3-1 has house_num=42)."""
    rec, cand = _make_scenario()
    feats = graph_features(cand, rec)

    decoy = feats.filter(pl.col("cand_id") == "S2-2").row(0, named=True)
    # Decoy house_num=69 ≠ S3-1 house_num=42 → cross_src_agree = 0
    assert decoy["graph_cross_src_agree"] == 0, \
        f"decoy graph_cross_src_agree={decoy['graph_cross_src_agree']} (expected 0)"

    true1 = feats.filter(pl.col("cand_id") == "S2-1").row(0, named=True)
    print(f"  true1 cross_src={true1['graph_cross_src_agree']}, decoy cross_src={decoy['graph_cross_src_agree']}")


def test_cluster_size():
    """Cluster size counts candidates whose name_core is similar to the S1 name_core."""
    rec, cand = _make_scenario()
    feats = graph_features(cand, rec)
    sizes = feats["graph_cluster_size"].to_list()
    # All three candidates belong to the same S1, so cluster_size is uniform
    assert all(s == sizes[0] for s in sizes), f"Cluster sizes differ within one S1: {sizes}"
    # At least 2 of the 3 candidates are similar to the S1 name ("sunrise bakery")
    assert sizes[0] >= 2, f"Expected cluster_size >= 2, got {sizes[0]}"
    print(f"  cluster_size={sizes[0]}")


def test_no_siblings_fills_zero():
    """An S1 with only one candidate gets 0.0 for all sibling-agreement features."""
    rec = pl.DataFrame({
        "entity_id":  ["S1-99", "S2-99"],
        "source":     [1,        2],
        "name_core":  ["lone shop", "lone shop"],
        "addr_street":["elm st",    "elm st"],
        "house_num":  ["10",        "10"],
        "legal_form": ["",         ""],
        "name_script":["Latin",    "Latin"],
    })
    cand = pl.DataFrame({
        "s1_id":       ["S1-99"],
        "cand_id":     ["S2-99"],
        "rank_score":  [0.9],
        "block_mask":  [1],
        "block_score": [1.0],
        "tfidf_name":  [None],
        "tfidf_full":  [None],
        "rank_in_cand":[1],
        "country":     ["US"],
        "source":      [2],
    })
    feats = graph_features(cand, rec)
    assert feats.height == 1
    row = feats.row(0, named=True)
    for col in ["graph_name_max", "graph_name_mean", "graph_addr_max", "graph_addr_mean",
                "graph_same_hnum", "graph_top1_agree", "graph_cross_src_agree"]:
        assert row[col] == 0.0, f"{col}={row[col]} (expected 0.0 for singleton S1)"
    print("  singleton fills zeros: OK")


def test_decoy_lower_hnum_score_than_true():
    """Composite: true duplicates outscore decoy on the house-number agreement axis."""
    rec, cand = _make_scenario()
    feats = graph_features(cand, rec)
    true1 = feats.filter(pl.col("cand_id") == "S2-1").row(0, named=True)
    true2 = feats.filter(pl.col("cand_id") == "S3-1").row(0, named=True)
    decoy = feats.filter(pl.col("cand_id") == "S2-2").row(0, named=True)

    # Combined: at least one of {same_hnum, top1_agree, cross_src_agree} must be higher for true matches
    score = lambda r: r["graph_same_hnum"] + r["graph_top1_agree"] + r["graph_cross_src_agree"]  # noqa
    assert score(true1) > score(decoy), (
        f"true1 combined={score(true1)} not > decoy combined={score(decoy)}")
    assert score(true2) > score(decoy), (
        f"true2 combined={score(true2)} not > decoy combined={score(decoy)}")
    print(f"  true1 score={score(true1)}, true2 score={score(true2)}, decoy score={score(decoy)}")


if __name__ == "__main__":
    test_same_hnum_decoy_gets_zero()
    print("test_same_hnum_decoy_gets_zero: PASS")
    test_top1_agree_decoy_gets_zero()
    print("test_top1_agree_decoy_gets_zero: PASS")
    test_cross_src_agree_decoy_gets_zero()
    print("test_cross_src_agree_decoy_gets_zero: PASS")
    test_cluster_size()
    print("test_cluster_size: PASS")
    test_no_siblings_fills_zero()
    print("test_no_siblings_fills_zero: PASS")
    test_decoy_lower_hnum_score_than_true()
    print("test_decoy_lower_hnum_score_than_true: PASS")
    print("\nAll graph feature tests passed.")
