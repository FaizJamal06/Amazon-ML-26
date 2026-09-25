"""Leakage test for ber.features + end-to-end stub run (featurize -> train -> decide -> submit, official validator).
Run directly or with pytest."""
import contextlib
import io
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

PKG = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PKG / "src"), str(PKG / "scripts")]

import make_stub_data as stub  # noqa: E402
from ber.config import load_config  # noqa: E402
from ber.pipeline import main  # noqa: E402


def _stub_world(cache: Path, n_s1: int = 20) -> None:
    """Write stub records/candidates/gt (train + test) into ``cache`` and test_source{1,2,3}.tsv into ``cache/test``."""
    rng = np.random.default_rng(42)
    for split, countries in [("train", ["US", "India"]), ("test", ["US", "India", "France"])]:
        rec, cand, gt = stub.build_split(split, countries, n_s1, rng)
        rec.write_parquet(cache / f"records_{split}.parquet")
        cand.write_parquet(cache / f"candidates_{split}.parquet")
        if split == "train":
            gt.write_parquet(cache / f"gt_{split}.parquet")
    (cache / "test").mkdir()
    rec = pl.read_parquet(cache / "records_test.parquet")
    for src in (1, 2, 3):
        rec.filter(pl.col("source") == src).select(
            "entity_id", business_name="name_raw", business_address="addr_raw", country="country") \
            .write_csv(cache / "test" / f"test_source{src}.tsv", separator="\t")


def _run(cache: Path, *args: str) -> str:
    """Run one pipeline stage on the tmp cache and return its stdout."""
    sets = [f"paths.cache_dir={cache}", f"paths.output_dir={cache / 'output'}", f"paths.reports_dir={cache / 'reports'}",
            f"paths.test.source1={cache / 'test' / 'test_source1.tsv'}", "lgbm.params.min_data_in_leaf=5"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main([*args, *[x for s in sets for x in ("--set", s)]])
    return buf.getvalue()


def test_features_never_read_model_scores():
    """Features are identical with or without scores artifacts / a ``p`` column around, and the feature code never
    names a scores artifact: no model p can leak into features across folds."""
    for f in (PKG / "src" / "ber" / "features").glob("*.py"):
        assert not re.search(r"""["']scores""", f.read_text(encoding="utf-8")), f
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        cache = Path(d)
        _stub_world(cache)
        _run(cache, "featurize", "--split", "train")
        clean = pl.read_parquet(cache / "features_train.parquet")
        cand = pl.read_parquet(cache / "candidates_train.parquet")
        rand = cand.select("s1_id", "cand_id", p=pl.Series(np.random.default_rng(0).random(cand.height)))
        for name in ("scores", "scores_fallback"):
            rand.write_parquet(cache / f"{name}_train.parquet")
        cand.with_columns(p=rand["p"]).write_parquet(cache / "candidates_train.parquet")
        _run(cache, "featurize", "--split", "train")
        poisoned = pl.read_parquet(cache / "features_train.parquet")
    assert "p" not in clean.columns and clean.equals(poisoned)


def test_end_to_end_stub_validator_pass():
    """folds -> featurize -> train -> decide (train) -> featurize -> predict -> decide -> submit on a stub world;
    the official validator (with --check-ids) must PASS."""
    if not load_config().path("validator").exists():
        print("skip: official validator not found (student_resource/ not unpacked)")
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        cache = Path(d)
        _stub_world(cache)
        for stage, split in [("folds", "train"), ("featurize", "train"), ("train", "train"), ("decide", "train"),
                             ("featurize", "test"), ("predict", "test"), ("decide", "test")]:
            out = _run(cache, stage, "--split", split)
        scores = pl.read_parquet(cache / "scores_test.parquet")
        assert scores.height == pl.read_parquet(cache / "candidates_test.parquet").height
        assert scores["p"].is_between(0, 1).all()
        out = _run(cache, "submit", "--split", "test")
    assert "validator: PASS" in out, out


if __name__ == "__main__":
    test_features_never_read_model_scores()
    test_end_to_end_stub_validator_pass()
    print("lgbm tests passed")
