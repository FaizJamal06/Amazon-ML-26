"""Tests for ber.features. Run directly or with pytest."""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber.features.rarity import token_idf  # noqa: E402
from ber.features.structured import house_num_class, token_diff  # noqa: E402

HN_CASES = [("6800", "6821", "small_shift"), ("635", "646", "small_shift"), ("12", "120", "digit_dropped"),
            ("12/3", "12", "same_base_diff_suffix"), ("45-a", "45", "same_base_diff_suffix"),
            ("6821", "6812", "transposed"), ("007", "7", "equal"), ("", "12", "one_missing"),
            ("", "", "both_missing"), ("12", "900", "different")]


def test_house_num_class_exact():
    """Every spec case maps to exactly the expected class, in both argument orders."""
    df = pl.DataFrame([c[:2] for c in HN_CASES], schema=["a", "b"], orient="row")
    for a, b in (("a", "b"), ("b", "a")):
        rel, _, _ = house_num_class(a, b)
        assert df.select(rel=rel)["rel"].to_list() == [c[2] for c in HN_CASES], (a, b)


def test_house_num_diffs():
    """abs_diff / rel_diff on the bases; null when a side is missing."""
    df = pl.DataFrame({"a": ["6800", "007", "", "12/3"], "b": ["6821", "7", "12", "12"]})
    _, diff, rel_diff = house_num_class("a", "b")
    out = df.select(d=diff, r=rel_diff)
    assert out["d"].to_list() == [21, 0, None, 0]
    assert abs(out["r"][0] - 21 / 6821) < 1e-6 and out["r"][2] is None


def test_token_diff_one_extra_token():
    """Decoy 'one extra word' is flagged; two-sided or disjoint differences are not; IDF sums the extra tokens."""
    df = pl.DataFrame({"country": ["IN"] * 5,
                       "a": ["vm business india", "anand food", "west royal tankers", "a b", ""],
                       "b": ["vm business india overseas", "anand food", "west royal", "a c", "x"]})
    idf = token_idf(df.lazy().select("country", name_core=pl.concat_str("a", pl.lit(" "), "b")), "name_core")
    out = token_diff(df, "a", "b", "n", idf)
    assert out["n_only_a"].to_list() == [0, 0, 1, 1, 0]
    assert out["n_only_b"].to_list() == [1, 0, 0, 1, 1]
    assert out["n_one_extra"].to_list() == [1, 0, 1, 0, 0]   # empty side shares nothing -> not "one extra"
    overseas = idf.filter(pl.col("token") == "overseas")["idf"][0]
    assert abs(out["n_extra_idf_b"][0] - overseas) < 1e-6 and out["n_extra_idf_a"][0] == 0


if __name__ == "__main__":
    test_house_num_class_exact()
    test_house_num_diffs()
    test_token_diff_one_extra_token()
    print("feature tests passed")
