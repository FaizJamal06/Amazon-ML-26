"""Tests for ber.io.ingest on tiny TSVs with quotes, empty fields, literal NA/null values and CRLF endings.
Run directly or with pytest."""
import sys
import tempfile
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber.config import load_config  # noqa: E402
from ber.io import ingest  # noqa: E402

HEAD = "entity_id\tbusiness_name\tbusiness_address\tcountry\r\n"
FILES = {
    "source1": HEAD + 'S1-1\t"Joe\'s ""Best"" Pizza"\t12 Main St\tUS\r\n'
                      "S1-2\tNA\t\tIndia\r\n"
                      "S1-3\tnull\tNULL\tFrance\r\n",
    "source2": HEAD + "S2-1\tJoe's Pizza\tN/A\tUS\r\n"
                      "S2-2\tnan\tNone\tFrance\r\n",
    "source3": HEAD + 'S3-1\t""\t"12 Main St, ""Suite"" 4"\tUS\r\n',
    "ground_truth": "source1_entity_id\tmatched_entity_ids\r\nS1-1\tS2-1,S3-1\r\nS1-2\t\r\nS1-3\tS2-2\r\n",
}
EXPECTED = {"source1": 3, "source2": 2, "source3": 1, "ground_truth": 3, "gt_pairs": 3}


def _cfg(tmp: Path):
    """Config whose train paths point at the tiny TSVs in ``tmp`` and whose cache is ``tmp/cache``."""
    for name, text in FILES.items():
        (tmp / f"{name}.tsv").write_bytes(text.encode("utf-8"))            # bytes: keep the CRLF endings
    sets = [f"paths.train.{n}={(tmp / f'{n}.tsv').as_posix()}" for n in FILES]
    cfg = load_config(sets=sets + [f"paths.cache_dir={(tmp / 'cache').as_posix()}"])
    cfg.cache_dir.mkdir()
    return cfg


def test_ingest_keeps_raw_strings_and_writes_gt():
    """Quotes are unescaped, "" / NA / null / N/A / nan / None stay literal strings, no CR survives, gt is long."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(Path(tmp))
        ingest(cfg, "train", expected=EXPECTED)
        s1 = pl.read_parquet(cfg.artifact("source1", "train"))
        assert s1.columns == ["entity_id", "business_name", "business_address", "country", "source"]
        assert s1.schema["source"] == pl.Int8 and s1["source"].to_list() == [1, 1, 1]
        assert s1["business_name"].to_list() == ['Joe\'s "Best" Pizza', "NA", "null"]
        assert s1["business_address"].to_list() == ["12 Main St", "", "NULL"]
        assert s1["country"].to_list() == ["US", "India", "France"]
        s2 = pl.read_parquet(cfg.artifact("source2", "train"))
        assert s2.select("business_name", "business_address").rows() == [("Joe's Pizza", "N/A"), ("nan", "None")]
        s3 = pl.read_parquet(cfg.artifact("source3", "train"))
        assert s3.row(0) == ("S3-1", "", '12 Main St, "Suite" 4', "US", 3)
        gt = pl.read_parquet(cfg.artifact("gt", "train"))
        assert sorted(gt.rows()) == [("S1-1", "S2-1"), ("S1-1", "S3-1"), ("S1-3", "S2-2")]


def test_ingest_fails_loudly_on_row_count_mismatch():
    """A row count that differs from the expected one raises before anything is trusted downstream."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(Path(tmp))
        try:
            ingest(cfg, "train", expected={**EXPECTED, "source2": 5})
        except ValueError as e:
            assert "source2" in str(e)
        else:
            raise AssertionError("expected ValueError for a wrong row count")


if __name__ == "__main__":
    test_ingest_keeps_raw_strings_and_writes_gt()
    test_ingest_fails_loudly_on_row_count_mismatch()
    print("ingest tests passed")
