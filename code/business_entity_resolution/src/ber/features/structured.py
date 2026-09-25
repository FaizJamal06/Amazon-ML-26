"""Decoy-killers (build first): house-number relation, token-set difference, street similarity
without number/city/region, legal form, script, lengths, missingness.

Contract:
In: candidate pairs joined with normalized records. Out: feature columns keyed by (s1_id, cand_id).

Owner: Nitish (R3 Features / LightGBM)
"""
from __future__ import annotations

import polars as pl


def house_num_relation(a: str | pl.Expr, b: str | pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """House-number relation of two ``house_num`` columns and the absolute numeric difference.

    Relation: ``equal`` / ``one_missing`` / ``both_missing`` / ``different`` (after stripping leading zeros).
    The difference uses the first digit run of each side; null when either side has no digits.
    """
    a, b = (pl.col(x) if isinstance(x, str) else x for x in (a, b))
    a, b = a.fill_null("").str.strip_chars_start("0"), b.fill_null("").str.strip_chars_start("0")
    rel = (pl.when((a == "") & (b == "")).then(pl.lit("both_missing"))
           .when((a == "") | (b == "")).then(pl.lit("one_missing"))
           .when(a == b).then(pl.lit("equal")).otherwise(pl.lit("different")))
    num = lambda x: x.str.extract(r"(\d+)").cast(pl.Int64, strict=False)  # noqa: E731
    return rel, (num(a) - num(b)).abs()


def legal_conflict(a: str, b: str) -> pl.Expr:
    """1.0 when both sides carry a legal form and they differ, else 0.0."""
    return ((pl.col(a) != "") & (pl.col(b) != "") & (pl.col(a) != pl.col(b))).cast(pl.Float32)
