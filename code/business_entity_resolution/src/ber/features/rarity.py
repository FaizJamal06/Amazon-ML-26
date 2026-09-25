"""Token / name frequency (IDF) per country computed on the record pool (no external data).

Contract:
In: records_{split}.parquet (all sources, same split). Out: rarity lookups + feature columns keyed by (s1_id, cand_id).
Country is only a grouping key here (IDF is computed within each country label, open set) — never a feature.

Owner: Nitish (R3 Features / LightGBM)
"""
from __future__ import annotations

import polars as pl


def tokens(col: str) -> pl.Expr:
    """Whitespace tokens of a string column as a de-duplicated list (``[]`` for empty/null)."""
    return pl.col(col).fill_null("").str.extract_all(r"\S+").list.unique()


def token_idf(pool: pl.LazyFrame, col: str) -> pl.DataFrame:
    """``country, token, idf`` with idf = log(N_country / docfreq) of the tokens of ``col`` over all pool records."""
    docs = pool.select("country", token=tokens(col))
    n = docs.group_by("country").agg(n=pl.len())
    return (docs.explode("token").drop_nulls("token").group_by("country", "token").agg(df=pl.len())
            .join(n, on="country").select("country", "token", idf=(pl.col("n") / pl.col("df")).log().cast(pl.Float32))
            .collect())


def name_freq(pool: pl.LazyFrame) -> pl.DataFrame:
    """``country, name_core, name_freq``: number of pool records (all sources) with this exact non-empty name_core."""
    return (pool.filter(pl.col("name_core").fill_null("") != "").group_by("country", "name_core")
            .agg(name_freq=pl.len().cast(pl.Int32)).collect())


def idf_stats(lists: pl.DataFrame, col: str, idf: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """Row-aligned ``{prefix}_sum/_min/_mean`` of the IDF of the tokens in list column ``col`` (needs ``country``).

    Empty lists give sum 0 and null min/mean. Unknown tokens count as idf 0.
    """
    ex = (lists.select("country", pl.col(col).alias("token")).with_row_index("_r").explode("token")
          .join(idf, on=["country", "token"], how="left", maintain_order="left"))
    return ex.group_by("_r", maintain_order=True).agg(
        **{f"{prefix}_sum": pl.col("idf").fill_null(0).sum(), f"{prefix}_min": pl.col("idf").min(),
           f"{prefix}_mean": pl.col("idf").mean()}).drop("_r")


def rarity_features(df: pl.DataFrame, name_idf: pl.DataFrame, freq: pl.DataFrame) -> pl.DataFrame:
    """Add shared-name-token IDF (min/mean), exact name_core frequency per side and the two interactions.

    Needs ``country, a_name_core, b_name_core`` plus ``name_tset``, ``name_tsort`` (pairwise) and ``hn_conflict``
    (structured) — call after those two.
    """
    shared = df.select("country", shared=tokens("a_name_core").list.set_intersection(tokens("b_name_core")))
    stats = idf_stats(shared, "shared", name_idf, "shared_idf").drop("shared_idf_sum")
    side = lambda s: freq.rename({"name_core": f"{s}_name_core", "name_freq": f"{s}_name_freq"})  # noqa: E731
    return (pl.concat([df, stats], how="horizontal")
            .join(side("a"), on=["country", "a_name_core"], how="left", maintain_order="left")
            .join(side("b"), on=["country", "b_name_core"], how="left", maintain_order="left")
            .with_columns(
                a_name_freq_log=pl.col("a_name_freq").log1p().cast(pl.Float32),
                b_name_freq_log=pl.col("b_name_freq").log1p().cast(pl.Float32),
                name_sim_x_rarity=(pl.col("name_tset") * pl.col("shared_idf_mean")).cast(pl.Float32),
                name_match_hn_conflict=((pl.col("name_tsort") >= 0.9) & (pl.col("hn_conflict") == 1)).cast(pl.Int8))
            .drop("a_name_freq", "b_name_freq"))
