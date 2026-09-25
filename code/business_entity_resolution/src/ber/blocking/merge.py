"""Union of blocking passes with a provenance bitmask; caps candidates per S1 at max_cands.

Contract:
In: outputs of keys / tfidf_knn / embed_knn.
Out: candidates_{split}.parquet with s1_id, cand_id, country, block_mask, tfidf_name, tfidf_full, knn_rank,
     rank_in_cand, block_score.

Owner: Dhanishkaa (R2 Normalize / Blocking)
"""
from __future__ import annotations

import time

import polars as pl

from ber.blocking.keys import (
    address_key_pass,
    compute_df_counts,
    compute_token_idf,
    name_token_pass,
)
from ber.config import Config

# Pass bit assignments (bitmask)
PASS_NAME_TOKEN = 0    # bit 0
PASS_ADDR_KEY = 1      # bit 1

PASS_NAMES: dict[int, str] = {
    PASS_NAME_TOKEN: "name_token",
    PASS_ADDR_KEY: "addr_key",
}


def _merge_passes(pass_results: list[tuple[pl.DataFrame, int]]) -> pl.DataFrame:
    """Merge multiple blocking pass results into a single DataFrame with a bitmask.

    Args:
        pass_results: list of (pairs_df, bit_position) where pairs_df has (cand_id, s1_id, score).

    Returns:
        DataFrame with (s1_id, cand_id, block_mask, block_score).
    """
    all_pairs = []
    for pairs, bit in pass_results:
        if pairs.height == 0:
            continue
        tagged = pairs.with_columns(
            bit_val=pl.lit(1 << bit, dtype=pl.Int32),
        )
        all_pairs.append(tagged)

    if not all_pairs:
        return pl.DataFrame(schema={
            "s1_id": pl.String, "cand_id": pl.String,
            "block_mask": pl.Int32, "block_score": pl.Float64,
        })

    merged = pl.concat(all_pairs)

    # Group by (s1_id, cand_id): OR the bitmask, take the max score
    result = (
        merged
        .group_by("s1_id", "cand_id")
        .agg(
            block_mask=pl.col("bit_val").sum(),  # sum works as OR when bits are disjoint
            block_score=pl.col("score").max(),
        )
    )

    return result


def _cap_candidates(candidates: pl.DataFrame, k_per_query: int = 10,
                    max_cands: int = 50) -> pl.DataFrame:
    """Cap candidates: per S2/S3 keep top k S1s, then per S1 cap at max_cands by block_score.

    Args:
        candidates: DataFrame with (s1_id, cand_id, block_mask, block_score).
        k_per_query: per S2/S3 record, keep at most this many S1 candidates.
        max_cands: per S1, cap total candidates at this number.

    Returns:
        Capped DataFrame with added rank_in_cand column.
    """
    # Per cand_id (S2/S3): keep top k S1s by block_score
    capped = (
        candidates
        .with_columns(
            rank_for_cand=pl.col("block_score")
            .rank("ordinal", descending=True)
            .over("cand_id")
            .cast(pl.Int32)
        )
        .filter(pl.col("rank_for_cand") <= k_per_query)
        .drop("rank_for_cand")
    )

    # Per S1: cap at max_cands by block_score
    capped = (
        capped
        .with_columns(
            rank_for_s1=pl.col("block_score")
            .rank("ordinal", descending=True)
            .over("s1_id")
            .cast(pl.Int32)
        )
        .filter(pl.col("rank_for_s1") <= max_cands)
        .drop("rank_for_s1")
    )

    # Add rank_in_cand: rank of this S1 among all S1s retrieved for this cand_id (1 = best)
    capped = capped.with_columns(
        rank_in_cand=pl.col("block_score")
        .rank("ordinal", descending=True)
        .over("cand_id")
        .cast(pl.Int32)
    )

    return capped


def build_candidates(cfg: Config, split: str, subworld: bool = False) -> None:
    """Build candidates_{split}.parquet from records_{split}.parquet.

    Runs blocking passes (name tokens + address keys) per country, merges,
    caps, and writes the §4 candidates artifact.
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

        print(f"\n  ── {country}: {s1_recs.height:,} S1, {query_recs.height:,} S2/S3 ──")

        if s1_recs.height == 0 or query_recs.height == 0:
            continue

        # Compute IDF for name tokens
        print(f"    computing name token IDF...")
        name_idf = compute_token_idf(country_recs, "name_tokens", country)
        name_df = compute_df_counts(country_recs, "name_tokens", country)

        # Compute IDF for address tokens
        print(f"    computing addr token IDF...")
        addr_idf = compute_token_idf(country_recs, "addr_street", country)
        addr_df = compute_df_counts(country_recs, "addr_street", country)

        pass_results = []

        # Pass A: name token blocking
        print(f"    Pass A: name token blocking...")
        name_pairs = name_token_pass(s1_recs, query_recs, name_idf, name_df, country,
                                     freq_cap=2000, k_tokens=3)
        print(f"      {name_pairs.height:,} pairs from name tokens")
        pass_results.append((name_pairs, PASS_NAME_TOKEN))

        # Pass B: address key blocking
        print(f"    Pass B: address key blocking...")
        addr_pairs = address_key_pass(s1_recs, query_recs, addr_idf, addr_df, country,
                                      freq_cap=2000, k_tokens=2)
        print(f"      {addr_pairs.height:,} pairs from address keys")
        pass_results.append((addr_pairs, PASS_ADDR_KEY))

        # Merge passes
        merged = _merge_passes(pass_results)
        merged = merged.with_columns(country=pl.lit(country))
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

    # Cap candidates
    k_per_query = 10
    max_cands = cfg.max_cands
    print(f"  capping: k_per_query={k_per_query}, max_cands={max_cands}")
    candidates = _cap_candidates(candidates, k_per_query=k_per_query, max_cands=max_cands)
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
        "tfidf_name", "tfidf_full", "knn_rank", "rank_in_cand", "block_score",
    ).sort("s1_id", "block_score", descending=[False, True])

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
