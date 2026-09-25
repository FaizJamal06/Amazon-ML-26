"""Tests for ber.fallback. Run directly or with pytest."""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber.fallback import house_num_relation, train_oof  # noqa: E402


def test_house_num_relation():
    """equal (after leading zeros) / one_missing / both_missing / different + abs diff."""
    df = pl.DataFrame({"a": ["0174", "12", "", "", "6800"], "b": ["174", "", "", "5", "6821"]})
    rel, diff = house_num_relation("a", "b")
    out = df.select(rel=rel, diff=diff)
    assert out["rel"].to_list() == ["equal", "one_missing", "both_missing", "one_missing", "different"]
    assert out["diff"].to_list() == [0, None, None, None, 21]


def test_train_oof_calibrated_and_separating():
    """OOF p is in [0, 1], every row gets a prediction, and it separates an easy synthetic signal."""
    rng = np.random.default_rng(0)
    n_s1, per = 400, 5
    s1 = np.repeat([f"S1-{i}" for i in range(n_s1)], per)
    y = rng.integers(0, 2, n_s1 * per)
    cols = ["block_score", "rank_in_cand", "hn_absdiff_log", "name_tset", "name_ratio", "street_ratio",
            "legal_conflict", "hn_rel_code"]
    feats = pl.DataFrame({c: rng.random(n_s1 * per) for c in cols}).with_columns(
        name_tset=pl.Series(y * 0.6 + rng.random(n_s1 * per) * 0.4),
        hn_rel_code=pl.Series(np.where(y == 1, 0, rng.integers(0, 4, n_s1 * per)).astype(float)))
    folds = np.repeat(np.arange(n_s1) % 5, per)
    p, model, cal = train_oof(feats, y, folds, pl.Series(s1), seed=42, max_train_s1=200)
    assert p.shape == y.shape and p.min() >= 0 and p.max() <= 1
    assert p[y == 1].mean() - p[y == 0].mean() > 0.5
    assert 0 <= cal.predict(model.predict_proba(feats.select(cols).to_numpy())[:, 1]).min()


if __name__ == "__main__":
    test_house_num_relation()
    test_train_oof_calibrated_and_separating()
    print("fallback tests passed")
