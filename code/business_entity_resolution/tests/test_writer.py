"""Tests for the submission writer in ber.io, incl. a real run of the official validator. Run directly or with pytest."""
import sys
import tempfile
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber.config import load_config  # noqa: E402
from ber.io import SubmissionError, write_submission  # noqa: E402

S1 = pl.Series(["S1-3", "S1-1", "S1-2", "S1-4"])            # S1-4 plays an unseen-country entity with no candidates
VALID = pl.Series(["S2-1", "S2-2", "S3-1", "S3-9"])
CANDS = pl.DataFrame({"s1_id": ["S1-1", "S1-1", "S1-1", "S1-2"], "cand_id": ["S3-1", "S2-1", "S2-2", "S3-9"]})
MATCHES = pl.DataFrame({"s1_id": ["S1-1", "S1-1"], "match_id": ["S3-1", "S2-1"]})


def _run(matches, cands, tmp):
    """Write into ``tmp`` against a fake test dir (test_source1 with ``S1``, test_source2/3 with ``VALID``)."""
    test_dir = Path(tmp) / "test"
    test_dir.mkdir(exist_ok=True)
    (test_dir / "test_source1.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n" + "".join(f"{s}\tx\ty\tZZ\n" for s in S1),
        encoding="utf-8")
    for src, ids in (("2", VALID.filter(VALID.str.starts_with("S2-"))), ("3", VALID.filter(VALID.str.starts_with("S3-")))):
        (test_dir / f"test_source{src}.tsv").write_text(
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n" + "".join(f"{i}\tx\ty\tZZ\n" for i in ids),
            encoding="utf-8")
    return write_submission(matches, cands, S1, VALID, Path(tmp) / "output",
                            load_config().path("validator"), test_dir)


def test_writes_exact_format_and_passes_validator():
    """Header, row order, "," without spaces, empty lists, no BOM, LF only; validator PASS."""
    with tempfile.TemporaryDirectory() as tmp:
        out = _run(MATCHES, CANDS, tmp)
        raw = Path(out["matching_results"]).read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw
        assert raw.decode() == ("source1_entity_id\tmatched_entity_ids\nS1-3\t\nS1-1\tS2-1,S3-1\nS1-2\t\nS1-4\t\n")
        cand = Path(out["candidate_pairs"]).read_text(encoding="utf-8").splitlines()
        assert cand[0] == "source1_entity_id\tcandidate_entity_ids" and cand[2] == "S1-1\tS2-1,S2-2,S3-1"
        assert len(out["matching_sha256"]) == 64


def test_rejects_rule_violations():
    """Matches outside candidates, unknown ids, S1 ids in lists, and duplicates all raise before writing."""
    bad_cases = [
        (pl.DataFrame({"s1_id": ["S1-2"], "cand_id": ["S2-1"]}), CANDS),                      # not a candidate
        (MATCHES, CANDS.vstack(pl.DataFrame({"s1_id": ["S1-2"], "cand_id": ["S2-77"]}))),     # unknown id
        (MATCHES, CANDS.vstack(pl.DataFrame({"s1_id": ["S1-2"], "cand_id": ["S1-1"]}))),      # S1 id in list
        (MATCHES.vstack(MATCHES.head(1)), CANDS),                                             # duplicate pair
    ]
    for m, c in bad_cases:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                _run(m, c, tmp)
            except SubmissionError:
                continue
            raise AssertionError(f"expected SubmissionError for\n{m}\n{c}")


if __name__ == "__main__":
    test_writes_exact_format_and_passes_validator()
    test_rejects_rule_violations()
    print("writer tests passed")
