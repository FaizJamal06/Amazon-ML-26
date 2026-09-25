"""Tests for the parallel / vectorized parts of ber.normalize. Run directly or with pytest."""
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
from ber.normalize import remove_area_tokens, remove_area_tokens_expr  # noqa: E402
from ber.pipeline import main  # noqa: E402


def test_area_token_removal_vectorized_matches_row_version():
    """The polars expression gives exactly the row function's output, incl. fallbacks and countries without areas."""
    area = {"US": {"austin", "texas"}, "India": set()}
    df = pl.DataFrame({"country": ["US", "US", "US", "US", "India", "France"],
                       "addr_street": ["main  street austin", "austin texas", "", "oak   lane", "mg  road", "rue x"]})
    expected = [remove_area_tokens(s, area[c]) if area.get(c) else s for c, s in df.iter_rows()]
    assert df.select(remove_area_tokens_expr(area))["addr_street"].to_list() == expected
    assert expected == ["main street", "austin texas", "", "oak lane", "mg  road", "rue x"]


def test_records_identical_for_1_and_2_workers():
    """build_records gives the same records (sorted by entity_id) single-process and with a 2-worker pool."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        cache = Path(d)
        rec, _, _ = stub.build_split("test", ["US", "India", "France"], 30, np.random.default_rng(42))
        for src in (1, 2, 3):
            rec.filter(pl.col("source") == src).select(
                "entity_id", business_name="name_raw", business_address="addr_raw", country="country",
                source=pl.col("source").cast(pl.String)).write_parquet(cache / f"source{src}_test.parquet")
        out = []
        for workers in (1, 2):
            with contextlib.redirect_stdout(io.StringIO()):
                main(["normalize", "--split", "test", "--set", f"paths.cache_dir={cache}",
                      "--set", f"normalize.workers={workers}"])
            out.append(pl.read_parquet(cache / "records_test.parquet").sort("entity_id"))
    assert out[0].height == rec.height and out[0].equals(out[1])


if __name__ == "__main__":
    test_area_token_removal_vectorized_matches_row_version()
    test_records_identical_for_1_and_2_workers()
    print("normalize parallel tests passed")
