"""Full-world 5-fold assignment of S1 entities (seed 42, stratified by country and match count) and the
closed sub-world sampler (S1 sample + their GT matches + their decoys).

Contract:
In: S1 frame ``(s1_id, country)`` (from ``records_train.parquet``, source == 1), ``gt_train.parquet`` (long
``s1_id, match_id``), ``candidates_train.parquet`` (needs ``s1_id, cand_id, rank_in_cand``) and the records frame
``(entity_id, source, country)`` for the density report.
Out: ``folds_train.parquet`` (``s1_id, fold 0-4, country, n_matches``); ``subworld_train.parquet``
(``entity_id, source, role``) + ``candidates_train_sw.parquet`` (candidate pairs restricted to the world).

Owner: Faiz (R1 Lead / Eval / Decision)
"""
from __future__ import annotations

import numpy as np
import polars as pl

MATCH_CAP = 5  # strata use min(n_matches, 5)


def _random_key(n: int, seed: int) -> pl.Series:
    """A seeded random permutation of 0..n-1, used to shuffle within strata reproducibly."""
    return pl.Series("_key", np.random.default_rng(seed).permutation(n))


def match_counts(s1: pl.DataFrame, gt: pl.DataFrame) -> pl.DataFrame:
    """``(s1_id, country, n_matches)`` for every S1, sorted by id (0 matches for singletons)."""
    n = gt.group_by("s1_id").agg(n_matches=pl.len())
    return (s1.select("s1_id", "country").unique("s1_id").sort("s1_id")
            .join(n, on="s1_id", how="left").with_columns(pl.col("n_matches").fill_null(0).cast(pl.Int32)))


def make_folds(s1: pl.DataFrame, gt: pl.DataFrame, n_folds: int = 5, seed: int = 42) -> pl.DataFrame:
    """Assign every S1 to a fold, stratified by ``country x min(n_matches, 5)``.

    Within each stratum the S1s are shuffled with ``seed`` and dealt round-robin, so fold sizes differ by at most one
    per stratum. Deterministic for a given input set (input order does not matter).
    """
    df = match_counts(s1, gt)
    return (
        df.with_columns(_random_key(df.height, seed))
        .with_columns(fold=((pl.col("_key").rank("ordinal").over("country", pl.col("n_matches").clip(upper_bound=MATCH_CAP)) - 1)
                            % n_folds).cast(pl.Int8))
        .select("s1_id", "fold", "country", "n_matches")
    )


def _source(col: str) -> pl.Expr:
    """Source number (1/2/3) from an id prefix like ``S2-123``."""
    return pl.col(col).str.slice(1, 1).cast(pl.Int8)


def sample_subworld(s1: pl.DataFrame, gt: pl.DataFrame, candidates: pl.DataFrame,
                    frac: float = 0.1, seed: int = 42) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Closed sub-world: sampled S1s + all their GT matches + unmatched records whose top-1 S1 is sampled.

    - S1s: ``round(frac * n)`` per country (at least 1), seeded.
    - matches: every GT match of a sampled S1 (whether or not blocking found it).
    - decoys: S2/S3 ids that match nothing in GT and whose ``rank_in_cand == 1`` S1 is a sampled S1 ("decoy owner").
    Returns ``(world, candidates_sw)`` where ``world`` = ``(entity_id, source, role)`` and ``candidates_sw`` keeps only
    pairs whose S1 is sampled and whose candidate is in the world.
    """
    base = s1.select("s1_id", "country").unique("s1_id").sort("s1_id")
    sampled = (
        base.with_columns(_random_key(base.height, seed))
        .filter(pl.col("_key").rank("ordinal").over("country")
                <= (pl.len().over("country") * frac).round().clip(lower_bound=1))
        .select("s1_id")
    )
    matches = gt.join(sampled, on="s1_id").select(entity_id="match_id")
    decoys = (
        candidates.filter(pl.col("rank_in_cand") == 1)
        .join(sampled, on="s1_id")
        .join(gt.select(cand_id="match_id"), on="cand_id", how="anti")
        .select(entity_id="cand_id")
    )
    world = pl.concat([
        sampled.select(entity_id="s1_id", role=pl.lit("s1")),
        matches.with_columns(role=pl.lit("match")),
        decoys.with_columns(role=pl.lit("decoy")),
    ]).unique("entity_id", keep="first").with_columns(source=_source("entity_id")).select("entity_id", "source", "role")
    cands_sw = (candidates.join(sampled, on="s1_id")
                .join(world.filter(pl.col("role") != "s1").select(cand_id="entity_id"), on="cand_id", how="semi"))
    return world, cands_sw


def density_report(world: pl.DataFrame, records: pl.DataFrame) -> pl.DataFrame:
    """S2+S3 records per S1, per country: the sub-world vs the full record pool (should be close).

    ``records`` needs ``entity_id, source, country``; world ids get their country from it.
    """
    def ratio(df: pl.DataFrame, name: str) -> pl.DataFrame:
        return df.group_by("country").agg(
            ((pl.col("source") > 1).sum() / (pl.col("source") == 1).sum()).alias(name))
    w = world.join(records.select("entity_id", "country"), on="entity_id", how="left")
    return (ratio(records, "full_s23_per_s1").join(ratio(w, "world_s23_per_s1"), on="country", how="left")
            .join(w.group_by("country").agg(world_s1=(pl.col("role") == "s1").sum(),
                                             world_matches=(pl.col("role") == "match").sum(),
                                             world_decoys=(pl.col("role") == "decoy").sum()), on="country", how="left")
            .sort("country"))
