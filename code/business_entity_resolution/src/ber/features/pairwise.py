"""Name/address string similarities (token set/sort, Jaro-Winkler, ...) for (S1, candidate) pairs.

Contract:
In: candidate pairs joined with normalized records (``a_*`` = S1 side, ``b_*`` = candidate side).
Out: feature columns keyed by (s1_id, cand_id).

Owner: Nitish (R3 Features / LightGBM)
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

CHUNK = 2_000_000  # pairs per rapidfuzz batch (bounds the Python string lists)
SCORERS = {"ratio": (fuzz.ratio, 100.0), "tset": (fuzz.token_set_ratio, 100.0),
           "tsort": (fuzz.token_sort_ratio, 100.0), "partial": (fuzz.partial_ratio, 100.0),
           "jw": (JaroWinkler.normalized_similarity, 1.0)}
SCRIPTS = ["Latin", "Devanagari", "Bengali", "Gurmukhi", "Gujarati", "Odia", "Tamil", "Telugu", "Kannada",
           "Malayalam"]  # R2's detect_script labels; anything else -> code len(SCRIPTS) ("other")


def cpdist(a: list[str], b: list[str], scorer, scale: float = 100.0) -> np.ndarray:
    """Element-wise rapidfuzz scores in [0, 1] for two equal-length string lists (multithreaded)."""
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32) / scale


def string_sims(df: pl.DataFrame, a: str, b: str, prefix: str, scorers=tuple(SCORERS)) -> pl.DataFrame:
    """Columns ``{prefix}_{scorer}`` in [0, 1] comparing string columns ``a`` and ``b``, row by row, in chunks."""
    out = {f"{prefix}_{k}": np.empty(df.height, np.float32) for k in scorers}
    for lo in range(0, df.height, CHUNK):  # chunks of pairs, not rows
        part = df.slice(lo, CHUNK)
        xa, xb = part[a].fill_null("").to_list(), part[b].fill_null("").to_list()
        for k in scorers:
            out[f"{prefix}_{k}"][lo:lo + part.height] = cpdist(xa, xb, *SCORERS[k])
    return pl.DataFrame(out)


def _len_ratio(a: str, b: str) -> pl.Expr:
    """min/max character length of two string columns; null when both are empty."""
    la, lb = pl.col(a).str.len_chars(), pl.col(b).str.len_chars()
    hi = pl.max_horizontal(la, lb)
    return pl.when(hi > 0).then(pl.min_horizontal(la, lb) / hi).cast(pl.Float32)


def _script_code(col: str) -> pl.Expr:
    """Integer code of a ``name_script`` column (fixed vocabulary, unknown scripts share one code)."""
    return pl.col(col).replace_strict(SCRIPTS, list(range(len(SCRIPTS))), default=len(SCRIPTS), return_dtype=pl.Int8)


def pairwise_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add string-similarity, script, length and source features to a joined pair frame.

    Needs ``a_/b_`` columns ``name_core, addr_street, name_script`` and ``b_source``; uses ``name_skeleton`` if both
    sides have it.
    """
    parts = [df, string_sims(df, "a_name_core", "b_name_core", "name"),
             string_sims(df, "a_addr_street", "b_addr_street", "street")]
    if {"a_name_skeleton", "b_name_skeleton"} <= set(df.columns):
        parts.append(string_sims(df, "a_name_skeleton", "b_name_skeleton", "skel", scorers=("ratio",)))
    return pl.concat(parts, how="horizontal").with_columns(
        s1_script=_script_code("a_name_script"), cand_script=_script_code("b_name_script"),
        same_script=(pl.col("a_name_script") == pl.col("b_name_script")).cast(pl.Int8),
        name_len_ratio=_len_ratio("a_name_core", "b_name_core"),
        street_len_ratio=_len_ratio("a_addr_street", "b_addr_street"),
        is_s3=(pl.col("b_source").cast(pl.Int8) == 3).cast(pl.Int8),
    )
