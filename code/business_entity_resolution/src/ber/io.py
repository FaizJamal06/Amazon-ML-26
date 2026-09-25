"""TSV -> parquet cache; TSV writers for the submission files. The ONLY module that reads TSVs.

Contract:
In: student_resource/dataset/{train,test}/*.tsv read with sep='\\t', dtype=str, keep_default_na=False.
Out: parquet caches under cache/; output/matching_results.tsv and output/candidate_pairs.tsv
(tab-separated, no quoting, no index, no BOM, comma-joined ids without spaces).

Owner: ingest — Faiz (R1, covering Chris); submission writer — Faiz (R1). Keep the two sections separate to avoid merge conflicts.
"""
from __future__ import annotations

import hashlib
import os
import random
import subprocess
import sys
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------------------------------------------------
# Ingest (Faiz, covering Chris) — TSV -> parquet cache.
# ---------------------------------------------------------------------------------------------------------------------
# Row counts from notes/DATA_CONTEXT.md; ingest fails loudly on any difference (truncated copy, parser drift).
EXPECTED_ROWS = {
    "train": {"source1": 2_206_821, "source2": 5_034_616, "source3": 5_285_603, "ground_truth": 2_206_821,
              "gt_pairs": 7_638_365},
    "test": {"source1": 1_732_544, "source2": 4_887_273, "source3": 5_082_316},
}
PANDAS_SAMPLE = 2000


def read_tsv(path: Path) -> pl.DataFrame:
    """Read an official TSV like the mandated pandas call: all columns Utf8, empty strings kept, CSV quoting."""
    return pl.read_csv(path, separator="\t", infer_schema=False, empty_string_is_null=False, quote_char='"')


def _pandas_rows(path: Path, positions: list[int]) -> tuple[list[tuple], int, list[str]]:
    """Rows at ``positions`` (0-based), total row count and header from the mandated pandas read, in chunks."""
    import pandas as pd
    keep, rows, n, cols = set(positions), {}, 0, None
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=1_000_000):
        hit = chunk[chunk.index.isin(keep)]
        rows.update(zip(hit.index, hit.itertuples(index=False, name=None)))
        n, cols = n + len(chunk), list(chunk.columns)
    return [rows.get(i) for i in positions], n, cols


def _check_table(path: Path, df: pl.DataFrame, n_expected: int | None, seed: int) -> str:
    """Raise ValueError unless ``df`` has the expected rows, no nulls, no "\\r" in its last column (country / id list)
    and ``PANDAS_SAMPLE`` random rows identical to the mandated pandas read. Returns a one-line report."""
    pos = sorted(random.Random(seed).sample(range(df.height), min(PANDAS_SAMPLE, df.height)))
    pd_rows, pd_n, pd_cols = _pandas_rows(path, pos)
    checks = {
        f"rows == {n_expected}": n_expected is None or df.height == n_expected,
        f"pandas rows ({pd_n}) == polars rows": pd_n == df.height,
        "header == pandas header": pd_cols == df.columns,
        f"{len(pos)} sampled rows == pandas": pd_rows == df.select(pl.all().gather(pos)).rows(),
        "0 nulls": sum(df.null_count().row(0)) == 0,
        f'no "\\r" in {df.columns[-1]}': not df[df.columns[-1]].str.contains("\r", literal=True).any(),
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise ValueError(f"ingest check failed for {path}: {bad} (got {df.height:,} rows)")
    return f"{path.name}: {df.height:,} rows x {df.width} cols; {len(pos)} rows == pandas; 0 nulls; no CR - OK"


def ingest(cfg, split: str, subworld: bool = False, expected: dict[str, int] | None = None) -> None:
    """Raw TSVs -> ``source{1,2,3}_{split}`` parquet (raw columns as Utf8 + ``source`` Int8, zstd) and, for train,
    ``gt_train`` (long ``s1_id, match_id`` via ``gt_long``). Every file passes ``_check_table`` before it is written;
    ``expected`` replaces ``EXPECTED_ROWS[split]`` (tests). Prints a short report."""
    from ber.eval.metric import gt_long
    if subworld:
        raise SystemExit("ingest reads the raw TSVs: run it without --subworld")
    expected = EXPECTED_ROWS[split] if expected is None else expected
    report, s1_ids = [], None
    for s in (1, 2, 3):
        path = cfg.path(f"{split}.source{s}")
        df = read_tsv(path)
        report.append(_check_table(path, df, expected.get(f"source{s}"), cfg.seed + s))
        if s == 1:
            s1_ids = set(df["entity_id"])
        df.with_columns(source=pl.lit(s, pl.Int8)).write_parquet(cfg.artifact(f"source{s}", split), compression="zstd")
        del df
    if split == "train":
        path = cfg.path("train.ground_truth")
        raw = read_tsv(path)
        report.append(_check_table(path, raw, expected.get("ground_truth"), cfg.seed))
        gt = gt_long(raw)
        n_pairs, n_s1 = gt.height, gt["s1_id"].n_unique()
        if expected.get("gt_pairs") not in (None, n_pairs) or set(raw["source1_entity_id"]) != s1_ids:
            raise ValueError(f"gt_train: {n_pairs:,} pairs (expected {expected.get('gt_pairs')}) "
                             "or GT S1 ids != source1 ids")
        gt.write_parquet(cfg.artifact("gt", split), compression="zstd")
        report.append(f"gt_train: {n_pairs:,} pairs over {n_s1:,} matched S1s "
                      f"({raw.height:,} GT rows = all S1s, {raw.height - n_s1:,} singletons) - OK")
    print("\n".join(report))


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
    """Run the official ``validate_submission.py`` with ``--check-ids``; raise ``SubmissionError`` unless it exits 0,
    prints PASS and actually ran the id-existence check (it silently skips it when test_source2/3.tsv are missing).
    ``--check-ids`` loads all test S2/S3 ids (a few GB of RAM on the full test set)."""
    res = subprocess.run([sys.executable, str(validator), "--matching", str(matching), "--candidate", str(candidate),
                          "--test-dir", str(test_dir), "--check-ids"], capture_output=True, text=True, encoding="utf-8",
                         env={**os.environ, "PYTHONIOENCODING": "utf-8"})  # validator prints non-ASCII
    if res.returncode != 0 or "PASS" not in res.stdout or "valid S2/S3 match IDs" not in res.stdout:
        raise SubmissionError(f"official validator did not PASS with --check-ids (exit {res.returncode}):\n{res.stdout}\n{res.stderr}")
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
