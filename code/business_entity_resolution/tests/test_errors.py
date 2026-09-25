"""Tests for ber.eval.errors. Run directly or with pytest."""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber.eval.errors import classify, error_report  # noqa: E402


def world():
    """S1-a owns S2-1, S2-2, S3-3; S1-b is a singleton. Pred: S2-1 (TP), S2-2 -> S1-b (FP); S3-3 not blocked."""
    gt = pl.DataFrame({"s1_id": ["S1-a"] * 3, "match_id": ["S2-1", "S2-2", "S3-3"]})
    cands = pl.DataFrame({"s1_id": ["S1-a", "S1-a", "S1-b", "S1-b"], "cand_id": ["S2-1", "S2-2", "S2-2", "S2-9"]})
    scores = cands.with_columns(p=pl.Series([0.9, 0.3, 0.6, 0.1]))
    matches = pl.DataFrame({"s1_id": ["S1-a", "S1-b"], "cand_id": ["S2-1", "S2-2"], "p": [0.9, 0.6]})
    ids = ["S1-a", "S1-b", "S2-1", "S2-2", "S3-3", "S2-9"]
    records = pl.DataFrame({"entity_id": ids, "source": [1, 1, 2, 2, 3, 2], "country": ["US"] * 6,
                            "name_raw": ["A|co", "B"] + ["x"] * 4, "addr_raw": ["1 Main St"] * 6,
                            "name_script": ["Latin"] * 6, "house_num": ["1", "2", "1", "", "7", "1"]})
    return matches, scores, cands, records, gt


def test_classify_kinds_and_owners():
    """TP / FP / FN_model / FN_blocking, with the GT owner and the S1 the candidate went to instead."""
    matches, scores, cands, _, gt = world()
    got = {(r["s1_id"], r["cand_id"]): r for r in classify(matches, scores, cands, gt).iter_rows(named=True)}
    assert got[("S1-a", "S2-1")]["kind"] == "TP"
    assert got[("S1-b", "S2-2")]["kind"] == "FP" and got[("S1-b", "S2-2")]["gt_owner"] == "S1-a"
    assert got[("S1-a", "S2-2")]["kind"] == "FN_model" and got[("S1-a", "S2-2")]["assigned_to"] == "S1-b"
    assert got[("S1-a", "S3-3")]["kind"] == "FN_blocking" and got[("S1-a", "S3-3")]["p"] is None


def test_report_renders_markdown():
    """The report has all sections, escapes pipes, and lists the FP with its p."""
    text = error_report(*world(), title="t", n=10)
    for head in ["Summary by country", "Summary by candidate name_script", "match-count bucket",
                 "Worst false positives", "blocking misses", "model misses"]:
        assert head in text, head
    assert r"A\|co" in text and "| S1-b | S2-2 | US | Latin | 0.6 |" in text


if __name__ == "__main__":
    test_classify_kinds_and_owners()
    test_report_renders_markdown()
    print("errors tests passed")
