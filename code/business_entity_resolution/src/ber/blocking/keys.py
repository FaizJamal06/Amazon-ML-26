"""Exact / sorted-key blocking passes: name-token IDF join, address-key join, house-number join.

Each pass has an index builder (S1 records of one country, called once) and a query function (a chunk of that
country's S2/S3 records), so merge.py can stream the queries with bounded memory.

Contract:
In: records_{split}.parquet. Out: (cand_id, s1_id, score) pairs per pass for merge.py.

Owner: Dhanishkaa (R2 Normalize / Blocking)
"""
from __future__ import annotations

import heapq
import math

import polars as pl

# ═══════════════════════════════════════════════════════════════════════════════
# §1  TOKEN IDF COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════


def compute_token_idf(records: pl.DataFrame, column: str, country: str) -> dict[str, float]:
    """Compute IDF for each token in *column* within a country.

    IDF = log(N / df_t) where df_t = number of documents containing token t.

    Args:
        records: DataFrame with 'entity_id', 'country', and *column* (list[str] or str).
        column: Column name containing tokens (list[str]) or text (str to split).
        country: Country to filter on.

    Returns:
        dict mapping token → IDF score.
    """
    subset = records.filter(pl.col("country") == country)
    n_docs = subset.height

    if n_docs == 0:
        return {}

    # Explode tokens
    if subset.schema[column] == pl.List(pl.String):
        exploded = subset.select("entity_id", column).explode(column).rename({column: "token"})
    else:
        exploded = (
            subset.select("entity_id", column)
            .with_columns(pl.col(column).str.split(" ").alias("_tokens"))
            .explode("_tokens")
            .rename({"_tokens": "token"})
            .select("entity_id", "token")
        )

    # Document frequency
    df_counts = (
        exploded
        .filter(pl.col("token").is_not_null() & (pl.col("token") != ""))
        .unique(["entity_id", "token"])
        .group_by("token")
        .agg(df=pl.len())
    )

    idf: dict[str, float] = {}
    for row in df_counts.iter_rows():
        token, df_val = row[0], row[1]
        idf[token] = math.log(n_docs / max(df_val, 1))

    return idf


# ═══════════════════════════════════════════════════════════════════════════════
# §2  PASSES — index built once per country, queried per chunk of S2/S3 records
# ═══════════════════════════════════════════════════════════════════════════════


def _get_rarest_tokens(tokens: list[str], idf: dict[str, float], k: int = 3,
                       freq_cap: int = 2000, country_df: dict[str, int] | None = None) -> list[str]:
    """Return the k rarest tokens from the list (by IDF), skipping tokens above freq_cap.

    Args:
        tokens: list of normalized tokens.
        idf: token → IDF mapping.
        k: number of rarest tokens to return.
        freq_cap: skip tokens with document frequency > freq_cap.
        country_df: token → document frequency count (for cap check).
    """
    scored = []
    for t in tokens:
        if not t or len(t) < 2:
            continue
        if country_df and country_df.get(t, 0) > freq_cap:
            continue
        scored.append((t, idf.get(t, 10.0)))  # unseen tokens get high IDF

    scored.sort(key=lambda x: -x[1])
    return [t for t, _ in scored[:k]]


Index = dict  # token or (house_num, token) -> list of (s1_id, idf score)
PAIR_SCHEMA = {"cand_id": pl.String, "s1_id": pl.String, "score": pl.Float64}


def _pairs_frame(pairs: list[tuple[str, str, float]]) -> pl.DataFrame:
    """(cand_id, s1_id, score) rows -> DataFrame with a fixed schema (also when empty)."""
    return pl.DataFrame(pairs, schema=PAIR_SCHEMA, orient="row")


def _top_scores(index: Index, keys: list, top_k: int) -> list[tuple[str, float]]:
    """Sum the index scores of every S1 hit by ``keys`` and return the ``top_k`` best (s1_id, score).

    Ties keep index insertion order (heapq.nlargest is stable), so results are deterministic.
    """
    s1_scores: dict[str, float] = {}
    for key in keys:
        for s1_id, score in index.get(key, ()):
            s1_scores[s1_id] = s1_scores.get(s1_id, 0.0) + score
    return [(s, s1_scores[s]) for s in heapq.nlargest(min(top_k, len(s1_scores)), s1_scores, key=s1_scores.get)]


def _name_keys(row: dict, idf: dict[str, float], country_df: dict[str, int], skel_idf: dict[str, float],
               skel_df: dict[str, int], freq_cap: int, k_tokens: int, always_skeleton: bool) -> list[str]:
    """Rarest name tokens of a record (+ rarest skeleton tokens: always for S1, only non-Latin for queries)."""
    rare = _get_rarest_tokens(row["name_tokens"] or [], idf, k_tokens, freq_cap, country_df)
    if row["name_skeleton"] and (always_skeleton or row["name_script"] != "Latin"):
        skel_rare = _get_rarest_tokens(row["name_skeleton"].split(), skel_idf, k_tokens, freq_cap, skel_df)
        rare = list(dict.fromkeys(rare + skel_rare))  # ordered dedup: set order varies per process
    return rare


def build_name_index(s1_records: pl.DataFrame, idf: dict[str, float], country_df: dict[str, int],
                     skel_idf: dict[str, float], skel_df: dict[str, int], freq_cap: int = 2000,
                     k_tokens: int = 3) -> Index:
    """Pass A index, built once per country: rare name token (or skeleton token) -> [(s1_id, idf)].

    Skeleton tokens are always indexed for S1: the skeleton is the shared representation between Latin S1 names
    and transliterated non-Latin S2/S3 names.
    """
    index: Index = {}
    for row in s1_records.select("entity_id", "name_tokens", "name_skeleton", "name_script").iter_rows(named=True):
        for token in _name_keys(row, idf, country_df, skel_idf, skel_df, freq_cap, k_tokens, always_skeleton=True):
            index.setdefault(token, []).append((row["entity_id"], idf.get(token, skel_idf.get(token, 10.0))))
    return index


def query_name_index(index: Index, query_records: pl.DataFrame, idf: dict[str, float], country_df: dict[str, int],
                     skel_idf: dict[str, float], skel_df: dict[str, int], freq_cap: int = 2000, k_tokens: int = 3,
                     top_k_per_query: int = 50) -> pl.DataFrame:
    """Pass A on a chunk of S2/S3 records: each record retrieves its top S1s by the summed IDF of shared rare tokens.

    Direction S2/S3 -> S1, within one country. Returns (cand_id, s1_id, score).
    """
    pairs = []
    for row in query_records.select("entity_id", "name_tokens", "name_skeleton", "name_script").iter_rows(named=True):
        keys = _name_keys(row, idf, country_df, skel_idf, skel_df, freq_cap, k_tokens, always_skeleton=False)
        pairs += [(row["entity_id"], s1, sc) for s1, sc in _top_scores(index, keys, top_k_per_query)]
    return _pairs_frame(pairs)


def _street_keys(row: dict, idf: dict[str, float], country_df: dict[str, int], freq_cap: int,
                 k_tokens: int) -> list[tuple[str, str]]:
    """(house_num, rare street token) keys of a record; [] when it has no house number."""
    house = row["house_num"] or ""
    if not house:
        return []
    street_tokens = (row["addr_street"] or "").split()
    if not street_tokens:
        street_tokens = [t for t in (row["addr_norm"] or "").split() if t != house and t.lstrip("0") != house]
    return [(house, t) for t in _get_rarest_tokens(street_tokens, idf, k_tokens, freq_cap, country_df)]


def build_address_index(s1_records: pl.DataFrame, idf: dict[str, float], country_df: dict[str, int],
                        freq_cap: int = 2000, k_tokens: int = 2) -> Index:
    """Pass B index, built once per country: (house_num, rare street token) -> [(s1_id, idf)]."""
    index: Index = {}
    for row in s1_records.select("entity_id", "house_num", "addr_street", "addr_norm").iter_rows(named=True):
        for key in _street_keys(row, idf, country_df, freq_cap, k_tokens):
            index.setdefault(key, []).append((row["entity_id"], idf.get(key[1], 10.0)))
    return index


def query_address_index(index: Index, query_records: pl.DataFrame, idf: dict[str, float],
                        country_df: dict[str, int], freq_cap: int = 2000, k_tokens: int = 2,
                        top_k_per_query: int = 50) -> pl.DataFrame:
    """Pass B on a chunk of S2/S3 records: S1s sharing the house number + a rare street token. (cand_id, s1_id, score)."""
    pairs = []
    for row in query_records.select("entity_id", "house_num", "addr_street", "addr_norm").iter_rows(named=True):
        keys = _street_keys(row, idf, country_df, freq_cap, k_tokens)
        pairs += [(row["entity_id"], s1, sc) for s1, sc in _top_scores(index, keys, top_k_per_query)]
    return _pairs_frame(pairs)


def build_house_index(s1_records: pl.DataFrame, idf: dict[str, float], country_df: dict[str, int],
                      freq_cap: int = 50) -> Index:
    """Pass C index, built once per country: distinctive house number (doc freq <= freq_cap) -> [(s1_id, idf)].

    The cap prevents a Cartesian explosion on common numbers like '1', '2', '10'.
    """
    index: Index = {}
    for row in s1_records.select("entity_id", "house_num").iter_rows(named=True):
        house = row["house_num"] or ""
        if house and country_df.get(house, 0) <= freq_cap:
            index.setdefault(house, []).append((row["entity_id"], idf.get(house, 5.0)))
    return index


def query_house_index(index: Index, query_records: pl.DataFrame, country_df: dict[str, int],
                      freq_cap: int = 50) -> pl.DataFrame:
    """Pass C on a chunk of S2/S3 records: every S1 with the same distinctive house number. (cand_id, s1_id, score)."""
    pairs = []
    for row in query_records.select("entity_id", "house_num").iter_rows(named=True):
        house = row["house_num"] or ""
        if house and country_df.get(house, 0) <= freq_cap:
            pairs += [(row["entity_id"], s1, sc) for s1, sc in index.get(house, ())]
    return _pairs_frame(pairs)


# ═══════════════════════════════════════════════════════════════════════════════
# §5  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def compute_df_counts(records: pl.DataFrame, column: str, country: str) -> dict[str, int]:
    """Compute document frequency (count of unique entities containing each token) within a country.

    Args:
        records: DataFrame with 'entity_id', 'country', and *column*.
        column: Column name containing tokens (list[str]) or text (str).
        country: Country to filter on.

    Returns:
        dict mapping token → count.
    """
    subset = records.filter(pl.col("country") == country)

    if subset.height == 0:
        return {}

    if subset.schema[column] == pl.List(pl.String):
        exploded = subset.select("entity_id", column).explode(column).rename({column: "token"})
    else:
        exploded = (
            subset.select("entity_id", column)
            .with_columns(pl.col(column).str.split(" ").alias("_tokens"))
            .explode("_tokens")
            .rename({"_tokens": "token"})
            .select("entity_id", "token")
        )

    df_counts = (
        exploded
        .filter(pl.col("token").is_not_null() & (pl.col("token") != ""))
        .unique(["entity_id", "token"])
        .group_by("token")
        .agg(df=pl.len())
    )

    return {row[0]: row[1] for row in df_counts.iter_rows()}
