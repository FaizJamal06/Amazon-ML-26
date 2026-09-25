"""One-to-one: each S2/S3 id is assigned to its single best S1 (or none).

Contract:
In: ``scores_{split}.parquet`` (``s1_id, cand_id, p``); ``block_score`` is taken from the scores frame if present,
otherwise joined from ``candidates_{split}.parquet``.
Out: the same pairs filtered to one S1 per ``cand_id`` (ties: highest ``block_score``, then smallest ``s1_id``).

Owner: Faiz (R1 Lead / Eval / Decision)
"""
from __future__ import annotations

import polars as pl


def assign_one_to_one(scores: pl.DataFrame, candidates: pl.DataFrame | None = None) -> pl.DataFrame:
    """Keep, for every ``cand_id``, only the pair with the highest ``p``.

    GT shows every S2/S3 record belongs to at most one S1, so any other pair of that record is a guaranteed false
    positive. Tie-break: higher ``block_score``, then lexicographically smaller ``s1_id`` (deterministic).
    """
    df = scores
    if "block_score" not in df.columns:
        if candidates is None:
            raise ValueError("scores lack block_score; pass candidates to join it")
        df = df.join(candidates.select("s1_id", "cand_id", "block_score"), on=["s1_id", "cand_id"], how="left")
    return (df.sort(["cand_id", "p", "block_score", "s1_id"], descending=[False, True, True, False], nulls_last=True)
            .unique("cand_id", keep="first", maintain_order=True))
