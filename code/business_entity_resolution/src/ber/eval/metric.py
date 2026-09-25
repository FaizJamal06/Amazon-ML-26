"""Exact macro F0.5 per S1 entity, mirroring the official rule (empty/empty = 1, empty vs non-empty = 0).

Contract:
In: predictions and ground truth as long polars frames ``(s1_id, match_id)`` (a ``cand_id`` column is accepted in place
of ``match_id``) + the S1 universe as a frame with ``s1_id`` (and optional group columns such as ``country``).
Out: macro F0.5 overall / per group; per-entity table ``s1_id, n_pred, n_gt, tp, f05``.
``gt_long`` turns the raw ground truth (``source1_entity_id, matched_entity_ids``) into the long form
stored as ``gt_train.parquet``.

Owner: Faiz (R1 Lead / Eval / Decision)
"""
from __future__ import annotations

from typing import Sequence

import polars as pl

BETA2 = 0.25  # beta = 0.5


def gt_long(gt_wide: pl.DataFrame) -> pl.DataFrame:
    """Explode raw GT ``(source1_entity_id, matched_entity_ids)`` to long ``(s1_id, match_id)``.

    Ids are comma-separated; an empty cell means no match and produces no row. Whitespace around ids is stripped.
    """
    return (
        gt_wide.select(s1_id=pl.col("source1_entity_id"), match_id=pl.col("matched_entity_ids").str.split(","))
        .explode("match_id")
        .with_columns(pl.col("match_id").str.strip_chars())
        .filter(pl.col("match_id").is_not_null() & (pl.col("match_id") != ""))
    )


def _pairs(df: pl.DataFrame) -> pl.DataFrame:
    """Normalise a long prediction/GT frame to unique ``(s1_id, match_id)`` pairs."""
    if "match_id" not in df.columns and "cand_id" in df.columns:
        df = df.rename({"cand_id": "match_id"})
    return df.select(pl.col("s1_id", "match_id").cast(pl.String)).unique()


def per_entity_f05(pred: pl.DataFrame, gt: pl.DataFrame, s1: pl.DataFrame) -> pl.DataFrame:
    """F0.5 for every S1 in ``s1`` (extra columns of ``s1`` are kept for grouping).

    Rules: GT empty & pred empty -> 1; exactly one of them empty -> 0; otherwise F0.5 of set precision/recall
    (0 when there is no true positive). Predictions for S1 ids outside ``s1`` are ignored.
    """
    pred, gt = _pairs(pred), _pairs(gt)
    counts = lambda df, name: df.group_by("s1_id").agg(pl.len().alias(name))  # noqa: E731
    tp = counts(pred.join(gt, on=["s1_id", "match_id"]), "tp")
    out = (
        s1.with_columns(pl.col("s1_id").cast(pl.String)).unique("s1_id")
        .join(counts(pred, "n_pred"), on="s1_id", how="left")
        .join(counts(gt, "n_gt"), on="s1_id", how="left")
        .join(tp, on="s1_id", how="left")
        .with_columns(pl.col("n_pred", "n_gt", "tp").fill_null(0).cast(pl.Int64))
    )
    p = pl.col("tp") / pl.col("n_pred")
    r = pl.col("tp") / pl.col("n_gt")
    return out.with_columns(
        f05=pl.when((pl.col("n_gt") == 0) & (pl.col("n_pred") == 0)).then(1.0)
        .when((pl.col("n_gt") == 0) | (pl.col("n_pred") == 0) | (pl.col("tp") == 0)).then(0.0)
        .otherwise((1 + BETA2) * p * r / (BETA2 * p + r))
    )


def macro_f05(pred: pl.DataFrame, gt: pl.DataFrame, s1: pl.DataFrame) -> float:
    """Official score: mean of per-entity F0.5 over all S1 ids in ``s1`` (singletons included)."""
    return float(per_entity_f05(pred, gt, s1)["f05"].mean())


def macro_f05_by(pred: pl.DataFrame, gt: pl.DataFrame, s1: pl.DataFrame, by: str | Sequence[str]) -> pl.DataFrame:
    """Macro F0.5 per group (e.g. ``by='country'``); ``s1`` must carry the group column(s)."""
    return (
        per_entity_f05(pred, gt, s1)
        .group_by(by)
        .agg(n_s1=pl.len(), macro_f05=pl.col("f05").mean(), singleton_share=(pl.col("n_gt") == 0).mean())
        .sort(by)
    )
