"""Pairwise feature engineering for (S1, candidate) pairs — the ``featurize`` stage.

Contract:
In: candidates_{split}.parquet + records_{split}.parquet (+ gt_train for the label). IDF / name frequencies use the
full-world records of the split (all sources), so sub-world rows get the same values as in the full world.
Out: features_{split}.parquet with s1_id, cand_id, <feature columns (Float32)>, label (Int8, train only).
Never reads any scores artifact (no model p -> no leakage across folds); tests/test_features.py checks this.

Owner: Nitish (R3 Features / LightGBM)
"""
from __future__ import annotations

import polars as pl
import pyarrow.parquet as pq

from ber.config import Config
from ber.features.context import context_features
from ber.features.pairwise import pairwise_features
from ber.features.rarity import name_freq, rarity_features, token_idf
from ber.features.structured import structured_features

KEYS = ["s1_id", "cand_id"]
SIDE_COLS = ["source", "name_core", "legal_form", "addr_street", "house_num", "name_script"]
OPTIONAL_SIDE_COLS = ["name_skeleton"]  # R2 records have it; used when present


def _side(rec: pl.DataFrame, prefix: str, key: str) -> pl.DataFrame:
    """The record columns features need, renamed ``{prefix}_{col}`` and keyed by ``key``."""
    cols = SIDE_COLS + [c for c in OPTIONAL_SIDE_COLS if c in rec.columns]
    return rec.select(pl.col("entity_id").alias(key), *[pl.col(c).alias(f"{prefix}_{c}") for c in cols])


def pair_features(ctx: pl.DataFrame, a_side: pl.DataFrame, b_side: pl.DataFrame, name_idf: pl.DataFrame,
                  street_idf: pl.DataFrame, freq: pl.DataFrame) -> pl.DataFrame:
    """Features for a chunk of pairs: ``ctx`` (country + context features) joined to both records.

    Returns ``s1_id, cand_id`` + every feature column cast to Float32 (country and raw strings dropped).
    """
    df = (ctx.join(a_side, on="s1_id", how="left", maintain_order="left")
          .join(b_side, on="cand_id", how="left", maintain_order="left"))
    df = rarity_features(pairwise_features(structured_features(df, name_idf, street_idf)), name_idf, freq)
    raw = {"country", *a_side.columns, *b_side.columns} - set(KEYS)
    return df.select(*KEYS, *[pl.col(c).cast(pl.Float32) for c in df.columns if c not in raw and c not in KEYS])


def build_features(cfg: Config, split: str, subworld: bool) -> None:
    """Stage ``featurize``: write ``features_{split}`` in chunks of ``features.chunk_pairs`` pairs (streamed)."""
    cand = pl.read_parquet(cfg.artifact("candidates", split, subworld)).sort(KEYS)
    rec = pl.read_parquet(cfg.artifact("records", split, subworld))
    full = cfg.artifact("records", split)
    pool = pl.scan_parquet(full if full.exists() else cfg.artifact("records", split, subworld)) \
        .select("country", "name_core", "addr_street")
    name_idf, street_idf, freq = token_idf(pool, "name_core"), token_idf(pool, "addr_street"), name_freq(pool)
    ctx = pl.concat([cand.select("country"), context_features(cand)], how="horizontal")
    a_side, b_side = _side(rec, "a", "s1_id"), _side(rec, "b", "cand_id")
    gt = None
    if split == "train":
        gt = pl.read_parquet(cfg.artifact("gt", split, subworld)).select(
            "s1_id", cand_id="match_id", label=pl.lit(1, pl.Int8))
    path, chunk, writer, n_pos = cfg.artifact("features", split, subworld), int(cfg.get("features.chunk_pairs",
                                                                                         5_000_000)), None, 0
    try:
        for lo in range(0, ctx.height, chunk):  # chunks of pairs, not rows
            out = pair_features(ctx.slice(lo, chunk), a_side, b_side, name_idf, street_idf, freq)
            if gt is not None:
                out = out.join(gt, on=KEYS, how="left", maintain_order="left").with_columns(pl.col("label").fill_null(0))
                n_pos += int(out["label"].sum())
            tbl = out.to_arrow()
            writer = writer or pq.ParquetWriter(path, tbl.schema)
            writer.write_table(tbl)
    finally:
        if writer:
            writer.close()
    n_feat = len(out.columns) - len(KEYS) - (gt is not None)
    print(f"featurize: {ctx.height:,} pairs, {n_feat} features"
          + (f", positives {n_pos:,} ({n_pos / max(ctx.height, 1):.3f})" if gt is not None else "") + f" -> {path}")
