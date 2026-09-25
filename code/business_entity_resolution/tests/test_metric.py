"""Tests for ber.eval.metric. Run: python -m pytest code/business_entity_resolution/tests  (or run this file directly)."""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber.eval.metric import gt_long, macro_f05, macro_f05_by, per_entity_f05  # noqa: E402


def pairs(rows):
    """Long (s1_id, match_id) frame from a list of tuples."""
    return pl.DataFrame(rows, schema=["s1_id", "match_id"], orient="row")


def score(pred, gt, s1_ids):
    """Macro F0.5 over the given S1 ids."""
    return macro_f05(pairs(pred), pairs(gt), pl.DataFrame({"s1_id": s1_ids}))


def test_readme_example():
    """README: pred {47,193,812} vs GT {47,812} -> P=2/3, R=1 -> 0.714."""
    got = score([("S1-00001", "S2-00047"), ("S1-00001", "S2-00193"), ("S1-00001", "S3-00812")],
                [("S1-00001", "S2-00047"), ("S1-00001", "S3-00812")], ["S1-00001"])
    assert round(got, 3) == 0.714, got


def test_empty_gt_empty_pred_is_one():
    """Correct singleton."""
    assert score([], [], ["S1-1"]) == 1.0


def test_empty_gt_nonempty_pred_is_zero():
    """False merge on a singleton."""
    assert score([("S1-1", "S2-1")], [], ["S1-1"]) == 0.0


def test_nonempty_gt_empty_pred_is_zero():
    """Missed entity entirely."""
    assert score([], [("S1-1", "S2-1")], ["S1-1"]) == 0.0


def test_nonempty_both_perfect_and_disjoint():
    """Exact set -> 1; no overlap -> 0; partial recall with full precision -> 0.625."""
    assert score([("S1-1", "S2-1")], [("S1-1", "S2-1")], ["S1-1"]) == 1.0
    assert score([("S1-1", "S2-9")], [("S1-1", "S2-1")], ["S1-1"]) == 0.0
    gt = [("S1-1", f"S2-{i}") for i in range(4)]
    assert round(score([("S1-1", "S2-0")], gt, ["S1-1"]), 4) == 0.625


def test_macro_average_and_duplicates():
    """Macro = mean over entities incl. singletons; duplicate predictions are not double-counted."""
    pred = [("S1-1", "S2-1"), ("S1-1", "S2-1")]
    gt = [("S1-1", "S2-1")]
    assert score(pred, gt, ["S1-1", "S1-2"]) == 1.0                    # S1-2: empty/empty -> 1
    assert score(pred + [("S1-2", "S3-5")], gt, ["S1-1", "S1-2"]) == 0.5  # S1-2 becomes a false merge
    assert score([("S1-X", "S2-1")], [], ["S1-1"]) == 1.0                # preds outside the universe are ignored


def test_cand_id_column_and_grouping():
    """``cand_id`` is accepted and per-group scores are returned."""
    pred = pl.DataFrame({"s1_id": ["S1-1"], "cand_id": ["S2-1"]})
    s1 = pl.DataFrame({"s1_id": ["S1-1", "S1-2"], "country": ["US", "France"]})
    by = macro_f05_by(pred, pairs([("S1-1", "S2-1")]), s1, "country")
    assert dict(by.select("country", "macro_f05").iter_rows()) == {"France": 1.0, "US": 1.0}
    assert per_entity_f05(pred, pairs([]), s1).filter(pl.col("s1_id") == "S1-1")["f05"][0] == 0.0


def test_random_against_naive():
    """Vectorised metric equals a plain-Python set implementation on random data."""
    import random
    rnd = random.Random(0)
    s1_ids = [f"S1-{i}" for i in range(300)]
    ids = [f"S2-{i}" for i in range(40)]
    gt = {s: set(rnd.sample(ids, rnd.choice([0, 0, 1, 2, 4]))) for s in s1_ids}
    pred = {s: set(rnd.sample(ids, rnd.choice([0, 1, 2, 3]))) | (set(list(gt[s])[:1]) if rnd.random() < .7 else set())
            for s in s1_ids}

    def naive(p, g):
        if not p and not g:
            return 1.0
        tp = len(p & g)
        if not tp:
            return 0.0
        pr, rc = tp / len(p), tp / len(g)
        return 1.25 * pr * rc / (0.25 * pr + rc)

    expected = sum(naive(pred[s], gt[s]) for s in s1_ids) / len(s1_ids)
    got = score([(s, m) for s in s1_ids for m in pred[s]], [(s, m) for s in s1_ids for m in gt[s]], s1_ids)
    assert abs(got - expected) < 1e-12, (got, expected)


def test_gt_long():
    """Comma lists explode to rows; empty cells produce none; whitespace is stripped."""
    wide = pl.DataFrame({"source1_entity_id": ["S1-1", "S1-2", "S1-3"],
                         "matched_entity_ids": ["S2-1,S3-2", "", " S2-7 "]})
    assert sorted(gt_long(wide).rows()) == [("S1-1", "S2-1"), ("S1-1", "S3-2"), ("S1-3", "S2-7")]


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} metric tests passed")
