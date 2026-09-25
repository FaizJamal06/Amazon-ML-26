"""TSV -> parquet cache; TSV writers for the submission files. The ONLY module that reads TSVs.

Contract:
In: student_resource/dataset/{train,test}/*.tsv read with sep='\\t', dtype=str, keep_default_na=False.
Out: parquet caches under cache/; output/matching_results.tsv and output/candidate_pairs.tsv
(tab-separated, no quoting, no index, no BOM, comma-joined ids without spaces).

Owner: ingest — Chris (R4); submission writer — Faiz (R1). Keep the two sections separate to avoid merge conflicts.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------------------------------------------------
# Ingest (Chris) — TSV -> parquet cache goes here.
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# Submission writer (Faiz)
# ---------------------------------------------------------------------------------------------------------------------
MATCH_HEADER = ("source1_entity_id", "matched_entity_ids")
CAND_HEADER = ("source1_entity_id", "candidate_entity_ids")


class SubmissionError(AssertionError):
    """Raised when submission pairs break a format rule or the official validator does not PASS."""


def _pairs(df: pl.DataFrame) -> pl.DataFrame:
    """``(s1_id, cand_id)`` string frame from pairs that may use ``match_id`` for the second column."""
    if "cand_id" not in df.columns and "match_id" in df.columns:
        df = df.rename({"match_id": "cand_id"})
    return df.select(pl.col("s1_id", "cand_id").cast(pl.String))


def _require(ok: bool, msg: str, bad: pl.DataFrame | None = None) -> None:
    """Raise ``SubmissionError`` with up to 5 offending rows when ``ok`` is false."""
    if not ok:
        raise SubmissionError(msg if bad is None else f"{msg} ({bad.height} rows), e.g.\n{bad.head(5)}")


def check_submission(matches: pl.DataFrame, candidates: pl.DataFrame, test_s1_ids: pl.Series,
                     valid_ids: pl.Series) -> None:
    """Assert every rule the scorer enforces, before anything is written.

    Rules: no duplicate pairs; S2-/S3- ids only, without commas/whitespace; S1 ids and S2/S3 ids exist in test;
    matches are a subset of candidates.
    """
    s1 = pl.DataFrame({"s1_id": test_s1_ids.cast(pl.String)})
    valid = pl.DataFrame({"cand_id": valid_ids.cast(pl.String)})
    for name, df in (("matches", matches), ("candidates", candidates)):
        _require(not df.is_duplicated().any(), f"{name}: duplicate (s1_id, cand_id) pairs", df.filter(df.is_duplicated()))
        bad = df.filter(~pl.col("cand_id").str.contains(r"^S[23]-[^\s,]+$"))
        _require(bad.is_empty(), f"{name}: ids must be S2-/S3- without commas or whitespace", bad)
        bad = df.join(s1, on="s1_id", how="anti")
        _require(bad.is_empty(), f"{name}: S1 ids not in test", bad)
        bad = df.join(valid, on="cand_id", how="anti")
        _require(bad.is_empty(), f"{name}: S2/S3 ids not in test", bad)
    bad = matches.join(candidates, on=["s1_id", "cand_id"], how="anti")
    _require(bad.is_empty(), "matches must be a subset of candidates", bad)


def id_lists(pairs: pl.DataFrame, test_s1_ids: pl.Series) -> pl.DataFrame:
    """One row per test S1 in the given order: ``(s1_id, ids)`` with sorted ids joined by "," ("" when none)."""
    lists = pairs.group_by("s1_id").agg(ids=pl.col("cand_id").sort().str.join(","))
    return (pl.DataFrame({"s1_id": test_s1_ids.cast(pl.String)})
            .join(lists, on="s1_id", how="left", maintain_order="left")
            .with_columns(pl.col("ids").fill_null("")))


def _write_tsv(df: pl.DataFrame, path: Path, header: tuple[str, str]) -> str:
    """Write a two-column TSV: UTF-8 without BOM, "\\n" line endings, no quoting. Returns its sha256."""
    df.rename(dict(zip(df.columns, header))).write_csv(path, separator="\t", quote_style="never",
                                                      line_terminator="\n", include_bom=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_validator(validator: Path, matching: Path, candidate: Path, test_dir: Path) -> str:
    """Run the official ``validate_submission.py``; raise ``SubmissionError`` unless it exits 0 and prints PASS."""
    res = subprocess.run([sys.executable, str(validator), "--matching", str(matching), "--candidate", str(candidate),
                          "--test-dir", str(test_dir)], capture_output=True, text=True, encoding="utf-8",
                         env={**os.environ, "PYTHONIOENCODING": "utf-8"})  # validator prints non-ASCII
    if res.returncode != 0 or "PASS" not in res.stdout:
        raise SubmissionError(f"official validator did not PASS (exit {res.returncode}):\n{res.stdout}\n{res.stderr}")
    return res.stdout


def write_submission(matches: pl.DataFrame, candidates: pl.DataFrame, test_s1_ids: pl.Series, valid_ids: pl.Series,
                     out_dir: Path, validator: Path, test_dir: Path) -> dict[str, str]:
    """Check, write ``matching_results.tsv`` + ``candidate_pairs.tsv`` and run the official validator.

    ``matches`` / ``candidates``: long pairs ``(s1_id, cand_id)`` (``match_id`` accepted). ``test_s1_ids``: every S1 id of
    test_source1 in file order — all countries, incl. France and any unseen label, get a row (empty when none).
    ``valid_ids``: all test S2/S3 ids. Returns paths and sha256 hashes for notes/SUBMISSIONS.md.
    """
    matches, candidates = _pairs(matches), _pairs(candidates)
    check_submission(matches, candidates, test_s1_ids, valid_ids)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m_path, c_path = out_dir / "matching_results.tsv", out_dir / "candidate_pairs.tsv"
    m_sha = _write_tsv(id_lists(matches, test_s1_ids), m_path, MATCH_HEADER)
    c_sha = _write_tsv(id_lists(candidates, test_s1_ids), c_path, CAND_HEADER)
    run_validator(Path(validator), m_path, c_path, Path(test_dir))
    return {"matching_results": str(m_path), "matching_sha256": m_sha,
            "candidate_pairs": str(c_path), "candidate_sha256": c_sha}
