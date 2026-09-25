"""Streaming union of blocking passes with a provenance bitmask; re-rank; per-record cut; per-S1 safety cap.

Contract:
In: outputs of keys / tfidf_knn / embed_knn.
Out: candidates_{split}.parquet with s1_id, cand_id, country, block_mask, tfidf_name, tfidf_full, knn_rank,
     rank_in_cand, block_score, rank_score.
Every retrieved pair (union of passes, before any cap) gets a cheap re-rank similarity ``rank_score``; each S2/S3
record keeps its top ``blocking.k_per_query`` S1s by rank_score, then each S1 keeps at most ``max_cands``.

Owner: Dhanishkaa (R2 Normalize / Blocking)
"""
from __future__ import annotations

import time
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz

from ber.blocking.keys import (
    build_address_index,
    build_house_index,
    build_name_index,
    compute_df_counts,
    compute_token_idf,
    query_address_index,
    query_house_index,
    query_name_index,
)
from ber import rules
from ber.config import Config
from ber.features.pairwise import cpdist
from ber.normalize import _normalize_base, name_skeleton

# Pass bit assignments (bitmask)
PASS_NAME_TOKEN = 0    # bit 0 (1 << 0 = 1)
PASS_ADDR_KEY = 1      # bit 1 (1 << 1 = 2)
PASS_HOUSE_NUM = 2     # bit 2 (1 << 2 = 4)

PASS_NAMES: dict[int, str] = {
    PASS_NAME_TOKEN: "name_token",
    PASS_ADDR_KEY: "addr_key",
    PASS_HOUSE_NUM: "house_num",
}

# rank_score = weighted cheap similarities (0-1 scale; a house-number conflict can push it below 0)
RANK_WEIGHTS = {"name_tsr": 0.45, "skel_ratio": 0.15, "street_tsr": 0.30, "hn_equal": 0.10, "hn_conflict": -0.20}
RANK_FIELDS = ("name_core", "name_skeleton", "addr_street", "house_num")
RANK_CHUNK = 5_000_000  # pairs per re-rank batch (bounds the joined strings in memory)


def rank_scores(pairs: pl.DataFrame, records: pl.DataFrame) -> pl.Series:
    """Cheap re-rank similarity for every (s1_id, cand_id) pair, row-aligned with ``pairs``.

    rank_score = 0.45 * token_set_ratio(name_core) + 0.15 * ratio(name_skeleton) + 0.30 * token_set_ratio(addr_street)
                 + 0.10 * [house_num equal] - 0.20 * [house_num conflict]  (similarities in 0-1).
    The skeleton term is 0 when either side has no skeleton; a missing house number on either side is neither equal
    nor a conflict. Vectorized: pairs are joined to the record fields in chunks and scored with rapidfuzz cpdist.
    """
    def side(key: str, prefix: str) -> pl.DataFrame:
        """Record fields used for re-ranking, keyed by ``key`` and prefixed."""
        return records.select(pl.col("entity_id").alias(key), *[pl.col(c).alias(prefix + c) for c in RANK_FIELDS])

    a_side, b_side = side("s1_id", "a_"), side("cand_id", "b_")
    out = []
    for lo in range(0, pairs.height, RANK_CHUNK):  # chunks of pairs, not rows
        df = (pairs.slice(lo, RANK_CHUNK).select("s1_id", "cand_id")
              .join(a_side, on="s1_id", how="left", maintain_order="left")
              .join(b_side, on="cand_id", how="left", maintain_order="left")
              .with_columns(pl.col("^[ab]_.*$").fill_null("").str.strip_chars()))
        sim = lambda f, scorer: cpdist(df[f"a_{f}"].to_list(), df[f"b_{f}"].to_list(), scorer)  # noqa: E731
        both = lambda f: ((df[f"a_{f}"] != "") & (df[f"b_{f}"] != "")).to_numpy()  # noqa: E731
        hn_eq = both("house_num") & (df["a_house_num"] == df["b_house_num"]).to_numpy()
        hn_conflict = both("house_num") & ~hn_eq
        out.append(RANK_WEIGHTS["name_tsr"] * sim("name_core", fuzz.token_set_ratio)
                   + RANK_WEIGHTS["skel_ratio"] * sim("name_skeleton", fuzz.ratio) * both("name_skeleton")
                   + RANK_WEIGHTS["street_tsr"] * sim("addr_street", fuzz.token_set_ratio)
                   + RANK_WEIGHTS["hn_equal"] * hn_eq + RANK_WEIGHTS["hn_conflict"] * hn_conflict)
    return pl.Series("rank_score", np.concatenate(out) if out else np.empty(0), dtype=pl.Float64)


RANK_BY, RANK_DESC = ["rank_score", "block_score", "s1_id", "cand_id"], [True, True, False, False]
RECORD_COLS = ["entity_id", "source", "country", "name_skeleton", "name_script", "name_core",
               "house_num", "addr_street", "addr_norm"]
EMPTY_CANDIDATES = {"s1_id": pl.String, "cand_id": pl.String, "block_mask": pl.Int32, "block_score": pl.Float64,
                    "rank_score": pl.Float64, "country": pl.String, "rank_in_cand": pl.Int32}


def _merge_passes(raw: pl.DataFrame, pass_max: dict[int, float]) -> pl.DataFrame:
    """Merge raw pass hits (cand_id, s1_id, score, bit) into (s1_id, cand_id, block_mask, block_score).

    Each pass scores on its own scale, so its score is first normalized to 0-1 by ``pass_max[bit]`` (that pass's
    max score over the whole country, so every chunk is scaled the same way). block_mask ORs the pass bits;
    block_score = sum of the normalized pass scores (0 to n_passes).
    """
    bits, maxes = list(pass_max), [float(pass_max[b]) for b in pass_max]
    return (raw.with_columns(score=(pl.col("score").cast(pl.Float64)
                                    / pl.col("bit").replace_strict(bits, maxes, return_dtype=pl.Float64)).fill_nan(0.0),
                             bit_val=pl.col("bit").replace_strict(list(PASS_NAMES), [1 << b for b in PASS_NAMES],
                                                               return_dtype=pl.Int32))
            .group_by("s1_id", "cand_id").agg(
                block_mask=pl.col("bit_val").bitwise_or(),
                block_score=pl.col("score").sum().round(9),   # round: group_by sum order varies -> last-bit noise
            ))


def _rank_per_cand(candidates: pl.DataFrame, k_per_query: int, min_score: float | None = None) -> pl.DataFrame:
    """Rank the S1s of every cand_id (RANK_BY) -> rank_in_cand, then keep rank_in_cand <= k_per_query.

    rank_in_cand is computed on the full retrieval list of the cand_id, BEFORE min_score / k_per_query remove
    anything. Needs every pair of a cand_id in ``candidates`` (true for a chunk of query records).
    """
    ranked = (candidates.sort(RANK_BY, descending=RANK_DESC)
              .with_columns(rank_in_cand=(pl.int_range(pl.len()).over("cand_id") + 1).cast(pl.Int32)))
    if min_score:
        ranked = ranked.filter(pl.col("block_score") >= min_score)
    return ranked.filter(pl.col("rank_in_cand") <= k_per_query)


def _cap_per_s1(candidates: pl.DataFrame, max_cands: int) -> pl.DataFrame:
    """Per-S1 safety cap: keep each S1's best ``max_cands`` candidates (same RANK_BY order)."""
    return candidates.sort(RANK_BY, descending=RANK_DESC).filter(pl.int_range(pl.len()).over("s1_id") < max_cands)


def _cap_candidates(candidates: pl.DataFrame, k_per_query: int = 2,
                    max_cands: int = 50, min_score: float | None = None) -> pl.DataFrame:
    """Rank, then cap: per S2/S3 keep the top k S1s, then per S1 keep the top max_cands candidates.

    Ordering everywhere: rank_score desc, block_score desc, s1_id asc, cand_id asc (deterministic ties).
    rank_in_cand (1 = best S1 for this cand_id) is computed on the full retrieval list, BEFORE min_score /
    k_per_query / max_cands remove anything. build_candidates applies the same two steps chunk by chunk.

    Args:
        candidates: DataFrame with (s1_id, cand_id, block_mask, block_score, rank_score).
        k_per_query: per S2/S3 record, keep at most this many S1 candidates.
        max_cands: per S1, cap total candidates at this number (a safety net).
        min_score: drop pairs with block_score below this (None or 0 = off).

    Returns:
        Capped DataFrame with an added rank_in_cand column.
    """
    return _cap_per_s1(_rank_per_cand(candidates, k_per_query, min_score), max_cands)


def _block_country(recs: pl.DataFrame, country: str, tmp: Path, chunk_rows: int, k_per_query: int,
                   min_score: float | None) -> list[Path]:
    """Stream one country: S1 indexes built once, S2/S3 queried in chunks of ``chunk_rows`` records.

    Phase 1: every pass runs on each chunk; raw hits go to disk and the per-pass max score is tracked.
    Phase 2: each chunk is normalized with the country-wide pass maxima, merged, re-ranked (rank_score) and cut to
    the top ``k_per_query`` S1s per cand_id; the (small) result goes to disk. Returns the result files.
    Memory is bounded by the chunk (+ the country's S1 indexes), not by the country's pair count.
    """
    s1_recs, query_recs = recs.filter(pl.col("source") == 1), recs.filter(pl.col("source") > 1)
    print(f"\n  -- {country}: {s1_recs.height:,} S1, {query_recs.height:,} S2/S3 --")
    if s1_recs.height == 0 or query_recs.height == 0:
        return []
    name_idf, name_df = compute_token_idf(recs, "name_core", country), compute_df_counts(recs, "name_core", country)
    skel_idf = compute_token_idf(recs, "name_skeleton", country)
    skel_df = compute_df_counts(recs, "name_skeleton", country)
    addr_idf, addr_df = compute_token_idf(recs, "addr_norm", country), compute_df_counts(recs, "addr_norm", country)
    house_idf, house_df = compute_token_idf(recs, "house_num", country), compute_df_counts(recs, "house_num", country)
    t = time.time()
    legal_skel = frozenset(t for form in rules.legal_forms(country) for t in name_skeleton(_normalize_base(form)).split())
    name_index = build_name_index(s1_recs, name_idf, name_df, skel_idf, skel_df, legal_skel)
    addr_index = build_address_index(s1_recs, addr_idf, addr_df)
    house_index = build_house_index(s1_recs, house_idf, house_df)
    print(f"    indexes built in {time.time() - t:.1f}s")

    raw_files, pass_max, n_raw = [], {}, 0
    for i, lo in enumerate(range(0, query_recs.height, chunk_rows)):  # chunks of records, not rows
        part = query_recs.slice(lo, chunk_rows)
        raw = pl.concat([
            query_name_index(name_index, part, name_idf, name_df, skel_idf, skel_df, legal_skel)
            .with_columns(bit=PASS_NAME_TOKEN),
            query_address_index(addr_index, part, addr_idf, addr_df).with_columns(bit=PASS_ADDR_KEY),
            query_house_index(house_index, part, house_df).with_columns(bit=PASS_HOUSE_NUM),
        ]).with_columns(pl.col("bit").cast(pl.Int8))
        for bit, mx in raw.group_by("bit").agg(pl.col("score").max()).iter_rows():
            pass_max[bit] = max(pass_max.get(bit, 0.0), mx)
        n_raw += raw.height
        raw_files.append(tmp / f"{country}_raw_{i:05d}.parquet")
        raw.write_parquet(raw_files[-1])
    del name_index, addr_index, house_index
    print(f"    phase 1: {n_raw:,} raw pass hits in {len(raw_files)} chunks ({time.time() - t:.1f}s)")

    rank_recs = recs.select("entity_id", *RANK_FIELDS)
    kept_files, n_merged, n_kept = [], 0, 0
    for i, path in enumerate(raw_files):
        merged = _merge_passes(pl.read_parquet(path), pass_max)
        merged = merged.with_columns(rank_scores(merged, rank_recs), country=pl.lit(country))
        kept = _rank_per_cand(merged, k_per_query, min_score)
        n_merged, n_kept = n_merged + merged.height, n_kept + kept.height
        kept_files.append(tmp / f"{country}_kept_{i:05d}.parquet")
        kept.write_parquet(kept_files[-1])
        path.unlink()
    print(f"    phase 2: {n_merged:,} unique pairs re-ranked, {n_kept:,} kept (k={k_per_query}) "
          f"({time.time() - t:.1f}s)")
    return kept_files


def build_candidates(cfg: Config, split: str, subworld: bool = False,
                     k_per_query: int | None = None, max_cands: int | None = None) -> None:
    """Build candidates_{split}.parquet from records_{split}.parquet, streaming per country and per chunk.

    Per country: S1 indexes once, S2/S3 in chunks of ``blocking.chunk_rows`` (default 50k): passes -> union ->
    rank_score -> top ``blocking.k_per_query`` per cand_id -> disk. Then the per-S1 ``max_cands`` safety cap on the
    concatenated (small) result. Only the columns blocking needs are read, one country at a time.
    """
    t0 = time.time()
    path = cfg.artifact("records", split, subworld)
    countries = sorted(pl.scan_parquet(path).select(pl.col("country").unique()).collect()["country"].to_list())
    print(f"[block] split={split}, countries: {countries}")
    max_cands = cfg.max_cands if max_cands is None else max_cands
    k_per_query = int(cfg.get("blocking.k_per_query", 2)) if k_per_query is None else k_per_query
    min_score = cfg.get("blocking.min_score", None)   # None or 0 = off
    chunk_rows = int(cfg.get("blocking.chunk_rows", 50_000))
    print(f"  k_per_query={k_per_query}, max_cands={max_cands}, min_score={min_score}, chunk_rows={chunk_rows}")

    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cfg.cache_dir, prefix="tmp_block_", ignore_cleanup_errors=True) as tmp:
        kept_files = []
        for country in countries:  # countries, not rows
            tc = time.time()
            recs = pl.scan_parquet(path).filter(pl.col("country") == country).select(RECORD_COLS).collect()
            kept_files += _block_country(recs, country, Path(tmp), chunk_rows, k_per_query, min_score)
            del recs
            print(f"    {country} done in {time.time() - tc:.1f}s")
        candidates = (pl.concat([pl.read_parquet(f) for f in kept_files]) if kept_files
                      else pl.DataFrame(schema=EMPTY_CANDIDATES))

    candidates = _cap_per_s1(candidates, max_cands)
    candidates = candidates.with_columns(
        tfidf_name=pl.lit(None, dtype=pl.Float64),   # contract columns, not produced by these passes
        tfidf_full=pl.lit(None, dtype=pl.Float64),
        knn_rank=pl.lit(None, dtype=pl.Int32),
    ).select(
        "s1_id", "cand_id", "country", "block_mask",
        "tfidf_name", "tfidf_full", "knn_rank", "rank_in_cand", "block_score", "rank_score",
    ).sort(["s1_id", "rank_score", "cand_id"], descending=[False, True, False])

    out_path = cfg.artifact("candidates", split, subworld)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.write_parquet(out_path)

    per_s1 = candidates.group_by("s1_id").len()["len"]
    print(f"\n[block] wrote {candidates.height:,} candidate pairs to {out_path}")
    print(f"  S1s with candidates: {candidates['s1_id'].n_unique():,}")
    if candidates.height:
        print(f"  candidates per S1: mean={per_s1.mean():.1f}, median={per_s1.median():.0f}, "
              f"p95={per_s1.quantile(0.95):.0f}, max={per_s1.max()}")
    print(f"  runtime: {time.time() - t0:.1f}s")
