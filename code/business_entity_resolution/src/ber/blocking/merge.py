"""Union of blocking passes with a provenance bitmask; caps candidates per S1 at max_cands.

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
    address_key_pass,
    compute_df_counts,
    compute_token_idf,
    house_num_pass,
    name_token_pass,
)
from ber.config import Config
from ber.features.pairwise import cpdist

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


def _merge_passes(pass_results: list[tuple[pl.LazyFrame, int]]) -> pl.LazyFrame:
    """Merge the blocking passes of one country into (s1_id, cand_id, block_mask, block_score).

    Each pass scores on its own scale, so its score is first normalized to 0-1 within the country
    (score / that pass's max score). block_mask ORs the pass bits; block_score = sum of the normalized
    pass scores (0 to n_passes).

    Args:
        pass_results: list of (lazy_pairs, bit_position) where lazy_pairs has (cand_id, s1_id, score) for one country.

    Returns:
        LazyFrame with (s1_id, cand_id, block_mask, block_score).
    """
    if not pass_results:
        return pl.DataFrame(schema={
            "s1_id": pl.String, "cand_id": pl.String,
            "block_mask": pl.Int32, "block_score": pl.Float64,
        }).lazy()
    merged = pl.concat([
        lazy_pairs.select("s1_id", "cand_id",
                          score=(pl.col("score").cast(pl.Float64) / pl.col("score").max()).fill_nan(0.0),
                          bit_val=pl.lit(1 << bit, dtype=pl.Int32))
        for lazy_pairs, bit in pass_results
    ])
    return merged.group_by("s1_id", "cand_id").agg(
        block_mask=pl.col("bit_val").bitwise_or(),
        block_score=pl.col("score").sum(),
    )


def _cap_candidates(candidates: pl.DataFrame, k_per_query: int = 2,
                    max_cands: int = 50, min_score: float | None = None) -> pl.DataFrame:
    """Rank, then cap: per S2/S3 keep the top k S1s, then per S1 keep the top max_cands candidates.

    Ordering everywhere: rank_score desc, block_score desc, s1_id asc, cand_id asc (deterministic ties).
    rank_in_cand (1 = best S1 for this cand_id) is computed on the full retrieval list, BEFORE min_score /
    k_per_query / max_cands remove anything.

    Args:
        candidates: DataFrame with (s1_id, cand_id, block_mask, block_score, rank_score).
        k_per_query: per S2/S3 record, keep at most this many S1 candidates.
        max_cands: per S1, cap total candidates at this number (a safety net).
        min_score: drop pairs with block_score below this (None or 0 = off).

    Returns:
        Capped DataFrame with an added rank_in_cand column.
    """
    ranked = (
        candidates
        .sort(["rank_score", "block_score", "s1_id", "cand_id"], descending=[True, True, False, False])
        .with_columns(rank_in_cand=(pl.int_range(pl.len()).over("cand_id") + 1).cast(pl.Int32))
    )
    if min_score:
        ranked = ranked.filter(pl.col("block_score") >= min_score)
    return (
        ranked
        .filter(pl.col("rank_in_cand") <= k_per_query)
        .filter(pl.int_range(pl.len()).over("s1_id") < max_cands)   # same ordering, frame is still sorted
    )


def build_candidates(cfg: Config, split: str, subworld: bool = False,
                     k_per_query: int | None = None, max_cands: int | None = None) -> None:
    """Build candidates_{split}.parquet from records_{split}.parquet.

    Runs blocking passes (name tokens + address keys + house numbers) per country,
    merges, caps, and writes the §4 candidates artifact.
    """
    t0 = time.time()
    records = pl.read_parquet(cfg.artifact("records", split, subworld))
    print(f"[block] loaded {records.height:,} records, split={split}")

    countries = sorted(records["country"].unique().to_list())
    print(f"  countries: {countries}")

    all_candidates = []

    for country in countries:
        tc = time.time()
        country_recs = records.filter(pl.col("country") == country)
        s1_recs = country_recs.filter(pl.col("source") == 1)
        query_recs = country_recs.filter(pl.col("source") > 1)

        print(f"\n  -- {country}: {s1_recs.height:,} S1, {query_recs.height:,} S2/S3 --")

        if s1_recs.height == 0 or query_recs.height == 0:
            continue

        # Compute IDF for name tokens
        print(f"    computing name token IDF...")
        name_idf = compute_token_idf(country_recs, "name_tokens", country)
        name_df = compute_df_counts(country_recs, "name_tokens", country)

        # Compute IDF for skeleton tokens
        print(f"    computing skeleton token IDF...")
        skel_idf = compute_token_idf(country_recs, "name_skeleton", country)
        skel_df = compute_df_counts(country_recs, "name_skeleton", country)

        # Compute IDF for address tokens on addr_norm (so all street & area tokens have accurate IDF)
        print(f"    computing addr token IDF...")
        addr_idf = compute_token_idf(country_recs, "addr_norm", country)
        addr_df = compute_df_counts(country_recs, "addr_norm", country)

        # Compute IDF for house numbers
        print(f"    computing house num IDF...")
        house_idf = compute_token_idf(country_recs, "house_num", country)
        house_df = compute_df_counts(country_recs, "house_num", country)

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            pass_results = []

            # Pass A: name token blocking
            print(f"    Pass A: name token blocking...")
            name_pairs, name_cnt = name_token_pass(s1_recs, query_recs, name_idf, name_df, skel_idf, skel_df, country, out_dir,
                                         freq_cap=2000, k_tokens=3)
            print(f"      {name_cnt:,} pairs from name tokens")
            if name_cnt > 0:
                pass_results.append((name_pairs, PASS_NAME_TOKEN))

            # Pass B: address key blocking (house_num + rare street token)
            print(f"    Pass B: address key blocking...")
            addr_pairs, addr_cnt = address_key_pass(s1_recs, query_recs, addr_idf, addr_df, country, out_dir,
                                          freq_cap=2000, k_tokens=2)
            print(f"      {addr_cnt:,} pairs from address keys")
            if addr_cnt > 0:
                pass_results.append((addr_pairs, PASS_ADDR_KEY))

            # Pass C: house number blocking
            print(f"    Pass C: house number blocking...")
            house_pairs, house_cnt = house_num_pass(s1_recs, query_recs, house_idf, house_df, country, out_dir,
                                         freq_cap=50)
            print(f"      {house_cnt:,} pairs from house numbers")
            if house_cnt > 0:
                pass_results.append((house_pairs, PASS_HOUSE_NUM))

            # Merge passes lazily then collect
            merged_lazy = _merge_passes(pass_results)
            merged_lazy = merged_lazy.with_columns(country=pl.lit(country))
            merged = merged_lazy.collect()
            print(f"    merged: {merged.height:,} unique pairs, "
                  f"{merged['s1_id'].n_unique():,} S1s with candidates")

        all_candidates.append(merged)
        print(f"    {country} done in {time.time() - tc:.1f}s")

    if not all_candidates:
        candidates = pl.DataFrame(schema={
            "s1_id": pl.String, "cand_id": pl.String, "country": pl.String,
            "block_mask": pl.Int32, "block_score": pl.Float64,
        })
    else:
        candidates = pl.concat(all_candidates)

    print(f"\n  total merged: {candidates.height:,} pairs")

    # Re-rank every retrieved pair (union of passes, before any cap) with cheap similarities
    t_rank = time.time()
    candidates = candidates.with_columns(rank_scores(candidates, records))
    print(f"  rank_score for {candidates.height:,} pairs in {time.time() - t_rank:.1f}s")

    # Cap candidates: keep top k_per_query per query, cap at max_cands per S1
    if max_cands is None:
        max_cands = cfg.max_cands
    if k_per_query is None:
        k_per_query = int(cfg.get("blocking.k_per_query", 2))
    min_score = cfg.get("blocking.min_score", None)   # None or 0 = off
    print(f"  capping: k_per_query={k_per_query}, max_cands={max_cands}, min_score={min_score}")
    candidates = _cap_candidates(candidates, k_per_query=k_per_query, max_cands=max_cands, min_score=min_score)
    print(f"  after capping: {candidates.height:,} pairs, "
          f"{candidates['s1_id'].n_unique():,} S1s")

    # Add contract columns (tfidf_name, tfidf_full, knn_rank filled with null for now)
    candidates = candidates.with_columns(
        tfidf_name=pl.lit(None, dtype=pl.Float64),
        tfidf_full=pl.lit(None, dtype=pl.Float64),
        knn_rank=pl.lit(None, dtype=pl.Int32),
    )

    # Reorder columns to match §4 contract
    candidates = candidates.select(
        "s1_id", "cand_id", "country", "block_mask",
        "tfidf_name", "tfidf_full", "knn_rank", "rank_in_cand", "block_score", "rank_score",
    ).sort(["s1_id", "rank_score", "cand_id"], descending=[False, True, False])

    # Write output
    out_path = cfg.artifact("candidates", split, subworld)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.write_parquet(out_path)

    # Report summary stats
    per_s1 = candidates.group_by("s1_id").len()["len"]
    runtime = time.time() - t0
    print(f"\n[block] wrote {candidates.height:,} candidate pairs to {out_path}")
    print(f"  S1s with candidates: {candidates['s1_id'].n_unique():,}")
    print(f"  candidates per S1: mean={per_s1.mean():.1f}, median={per_s1.median():.0f}, "
          f"p95={per_s1.quantile(0.95):.0f}, max={per_s1.max()}")
    print(f"  runtime: {runtime:.1f}s")
