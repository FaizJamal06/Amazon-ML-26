"""Tests for ber.eval.blocking_report on a hand-computed toy world. Run directly or with pytest."""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber.eval.blocking_report import blocking_report  # noqa: E402


def test_toy_numbers():
    """2 S1 x 4 S2/S3 (space 8), 3 candidate pairs, 1 of 2 GT pairs found."""
    records = pl.DataFrame({"entity_id": ["S1-a", "S1-b", "S2-1", "S3-2", "S2-9", "S3-7"],
                            "source": [1, 1, 2, 3, 2, 3], "country": ["US"] * 6,
                            "name_script": ["Latin", "Latin", "Latin", "Devanagari", "Latin", "Latin"]})
    gt = pl.DataFrame({"s1_id": ["S1-a", "S1-a"], "match_id": ["S2-1", "S3-2"]})
    cands = pl.DataFrame({"s1_id": ["S1-a", "S1-a", "S1-b"], "cand_id": ["S2-1", "S2-9", "S3-2"],
                          "country": ["US"] * 3, "block_mask": [1, 2, 3]})
    rep = blocking_report(cands, gt, records)
    s = rep["summary"].filter(pl.col("country") == "ALL").row(0, named=True)
    assert (s["n_s1"], s["pairs"], s["gt_pairs"]) == (2, 3, 2)
    assert s["pair_recall"] == 0.5 and s["full_gt_covered"] == 0.0 and s["reduction_ratio"] == 1 - 3 / 8
    assert dict(rep["by_script"].select("name_script", "pair_recall").iter_rows()) == {"Devanagari": 0.0, "Latin": 1.0}
    bp = {r["bit"]: r for r in rep["by_pass"].iter_rows(named=True)}
    assert (bp[0]["pairs"], bp[0]["recall"], bp[0]["unique_recall"]) == (2, 0.5, 0.5)
    assert (bp[1]["pairs"], bp[1]["recall"]) == (2, 0.0)


if __name__ == "__main__":
    test_toy_numbers()
    print("blocking report tests passed")
