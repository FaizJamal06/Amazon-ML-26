"""Ranks, gaps and one-to-one competition features, per S1 and per candidate.
Computed from blocking/string similarities only, never from model p (leakage).

Contract:
In: candidate pairs + pairwise similarities. Out: feature columns keyed by (s1_id, cand_id).

Owner: Nitish (R3 Features / LightGBM)
"""
from __future__ import annotations

import polars as pl

MASK_BITS = 8  # ponytail: blocking passes 0-7 expanded to flags; raise if R2 adds more passes


def context_features(cand: pl.DataFrame, score: str = "block_score") -> pl.DataFrame:
    """Competition features from the candidates table (all S1s of the split at once, so the one-to-one view is
    complete). Returns ``s1_id, cand_id`` + features; reads only blocking columns.

    - ``{score}`` itself, ``rank_in_cand``, ``tfidf_name``, ``tfidf_full``, one flag per ``block_mask`` bit;
    - ``gap_other_s1``: this pair's score minus the best score of the candidate with any *other* S1 (null if none);
    - ``n_s1_for_cand``: how many S1s retrieved this candidate;
    - ``cand_rank_in_s1`` / ``n_cands_s1`` / ``gap_to_s1_best``: position of the candidate in its S1's list.
    """
    s = pl.col(score)
    rank_c = s.rank("ordinal", descending=True).over("cand_id")
    best_c = s.max().over("cand_id")
    second_c = s.sort(descending=True).slice(1, 1).first().over("cand_id")
    return cand.select(
        "s1_id", "cand_id", score, "rank_in_cand", "tfidf_name", "tfidf_full",
        *[((pl.col("block_mask") // (1 << i)) % 2).cast(pl.Int8).alias(f"pass_{i}") for i in range(MASK_BITS)],
        gap_other_s1=s - pl.when(rank_c == 1).then(second_c).otherwise(best_c),
        n_s1_for_cand=pl.len().over("cand_id").cast(pl.Int16),
        cand_rank_in_s1=s.rank("ordinal", descending=True).over("s1_id").cast(pl.Int16),
        n_cands_s1=pl.len().over("s1_id").cast(pl.Int16),
        gap_to_s1_best=s.max().over("s1_id") - s,
    )
