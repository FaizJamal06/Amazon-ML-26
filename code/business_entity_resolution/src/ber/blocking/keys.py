"""Exact / sorted-key blocking passes: name-token IDF join and address-key join.

Contract:
In: records_{split}.parquet. Out: (s1_id, cand_id, block_score, pass_bit) pairs for merge.py.

Owner: Dhanishkaa (R2 Normalize / Blocking)
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path
from collections import Counter
import heapq

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
# §2  PASS A — NAME TOKEN BLOCKING
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


def name_token_pass(s1_records: pl.DataFrame, query_records: pl.DataFrame,
                    idf: dict[str, float], country_df: dict[str, int],
                    skel_idf: dict[str, float], skel_df: dict[str, int],
                    country: str, out_dir: Path, freq_cap: int = 2000, k_tokens: int = 3, top_k_per_query: int = 50) -> tuple[pl.LazyFrame, int]:
    """Pass A: name-token blocking. For each S2/S3 record, find S1s sharing rare name tokens.

    Direction: S2/S3 → S1 (each S2/S3 retrieves its top-k S1 candidates).
    Always within the same country.

    Args:
        s1_records: S1 records for this country.
        query_records: S2/S3 records for this country.
        idf: token → IDF scores.
        country_df: token → document frequency.
        skel_idf: skeleton token → IDF scores.
        skel_df: skeleton token → document frequency.
        country: country label.
        freq_cap: skip tokens with DF > this.
        k_tokens: number of rarest tokens per record.
        top_k_per_query: cap S1 candidates per query inside the pass to save memory.

    Returns:
        DataFrame with (cand_id, s1_id, score) — score = sum of IDF of shared tokens.
    """
    # Build S1 inverted index: token → list of (s1_id, idf_score)
    s1_index: dict[str, list[tuple[str, float]]] = {}

    for row in s1_records.select("entity_id", "name_tokens", "name_skeleton", "name_script").iter_rows(named=True):
        tokens = row["name_tokens"] or []
        rare = _get_rarest_tokens(tokens, idf, k_tokens, freq_cap, country_df)

        # Always add skeleton tokens — skeleton is the shared representation
        # between Latin S1 names and transliterated non-Latin S2/S3 names
        if row["name_skeleton"]:
            skel_tokens = row["name_skeleton"].split()
            skel_rare = _get_rarest_tokens(skel_tokens, skel_idf, k_tokens, freq_cap, skel_df)
            rare = list(set(rare + skel_rare))

        for token in rare:
            if token not in s1_index:
                s1_index[token] = []
            s1_index[token].append((row["entity_id"], idf.get(token, skel_idf.get(token, 10.0))))

    chunk_files = []
    total_pairs = 0
    chunk_size = 50_000
    n_chunks = max(1, math.ceil(query_records.height / chunk_size))

    for i in range(n_chunks):
        pairs: list[tuple[str, str, float]] = []
        chunk = query_records.slice(i * chunk_size, chunk_size)

        for row in chunk.select("entity_id", "name_tokens", "name_skeleton", "name_script").iter_rows(named=True):
            tokens = row["name_tokens"] or []
            rare = _get_rarest_tokens(tokens, idf, k_tokens, freq_cap, country_df)

            # For non-Latin, also query with skeleton tokens
            if row["name_script"] != "Latin" and row["name_skeleton"]:
                skel_tokens = row["name_skeleton"].split()
                skel_rare = _get_rarest_tokens(skel_tokens, skel_idf, k_tokens, freq_cap, skel_df)
                rare = list(set(rare + skel_rare))

            # Accumulate scores per S1
            s1_scores: dict[str, float] = {}
            for token in rare:
                if token in s1_index:
                    for s1_id, token_idf in s1_index[token]:
                        s1_scores[s1_id] = s1_scores.get(s1_id, 0.0) + token_idf

            if s1_scores:
                top_k = min(top_k_per_query, len(s1_scores))
                for s1_id in heapq.nlargest(top_k, s1_scores, key=s1_scores.get):
                    pairs.append((row["entity_id"], s1_id, s1_scores[s1_id]))

        if pairs:
            chunk_df = pl.DataFrame(pairs, schema=["cand_id", "s1_id", "score"], orient="row")
            total_pairs += chunk_df.height
            chunk_path = out_dir / f"name_pass_{i}.parquet"
            chunk_df.write_parquet(chunk_path)
            chunk_files.append(chunk_path)

    if not chunk_files:
        return pl.DataFrame(schema={"cand_id": pl.String, "s1_id": pl.String, "score": pl.Float64}).lazy(), 0

    return pl.scan_parquet([str(p) for p in chunk_files]), total_pairs


# ═══════════════════════════════════════════════════════════════════════════════
# §3  PASS B — ADDRESS KEY BLOCKING
# ═══════════════════════════════════════════════════════════════════════════════


def address_key_pass(s1_records: pl.DataFrame, query_records: pl.DataFrame,
                     idf: dict[str, float], country_df: dict[str, int],
                     country: str, out_dir: Path, freq_cap: int = 2000, k_tokens: int = 2, top_k_per_query: int = 50) -> tuple[pl.LazyFrame, int]:
    """Pass B: address blocking. Key = house_num + each of the rarest addr_street tokens.

    Direction: S2/S3 → S1, within country.

    Args:
        s1_records: S1 records for this country.
        query_records: S2/S3 records for this country.
        idf: addr token → IDF scores.
        country_df: addr token → document frequency.
        country: country label.
        freq_cap: skip tokens with DF > this.
        k_tokens: number of rarest street tokens per record.
        top_k_per_query: cap S1 candidates per query inside the pass to save memory.

    Returns:
        DataFrame with (cand_id, s1_id, score).
    """
    # Build S1 inverted index: (house_num, street_token) → list of (s1_id, idf)
    s1_index: dict[tuple[str, str], list[tuple[str, float]]] = {}

    for row in s1_records.select("entity_id", "house_num", "addr_street", "addr_norm").iter_rows(named=True):
        house = row["house_num"] or ""
        if not house:
            continue
        street = row["addr_street"] or ""
        street_tokens = street.split()
        if not street_tokens:
            norm = row.get("addr_norm") or ""
            street_tokens = [t for t in norm.split() if t != house and t.lstrip("0") != house]
        rare = _get_rarest_tokens(street_tokens, idf, k_tokens, freq_cap, country_df)

        for token in rare:
            key = (house, token)
            if key not in s1_index:
                s1_index[key] = []
            s1_index[key].append((row["entity_id"], idf.get(token, 10.0)))

    # Query S2/S3 records
    chunk_files = []
    total_pairs = 0
    chunk_size = 50_000
    n_chunks = max(1, math.ceil(query_records.height / chunk_size))

    for i in range(n_chunks):
        pairs: list[tuple[str, str, float]] = []
        chunk = query_records.slice(i * chunk_size, chunk_size)

        for row in chunk.select("entity_id", "house_num", "addr_street", "addr_norm").iter_rows(named=True):
            house = row["house_num"] or ""
            if not house:
                continue
            street = row["addr_street"] or ""
            street_tokens = street.split()
            if not street_tokens:
                norm = row.get("addr_norm") or ""
                street_tokens = [t for t in norm.split() if t != house and t.lstrip("0") != house]
            rare = _get_rarest_tokens(street_tokens, idf, k_tokens, freq_cap, country_df)

            s1_scores: dict[str, float] = {}
            for token in rare:
                key = (house, token)
                if key in s1_index:
                    for s1_id, token_idf in s1_index[key]:
                        s1_scores[s1_id] = s1_scores.get(s1_id, 0.0) + token_idf

            if s1_scores:
                top_k = min(top_k_per_query, len(s1_scores))
                for s1_id in heapq.nlargest(top_k, s1_scores, key=s1_scores.get):
                    pairs.append((row["entity_id"], s1_id, s1_scores[s1_id]))

        if pairs:
            chunk_df = pl.DataFrame(pairs, schema=["cand_id", "s1_id", "score"], orient="row")
            total_pairs += chunk_df.height
            chunk_path = out_dir / f"addr_pass_{i}.parquet"
            chunk_df.write_parquet(chunk_path)
            chunk_files.append(chunk_path)

    if not chunk_files:
        return pl.DataFrame(schema={"cand_id": pl.String, "s1_id": pl.String, "score": pl.Float64}).lazy(), 0

    return pl.scan_parquet([str(p) for p in chunk_files]), total_pairs


# ═══════════════════════════════════════════════════════════════════════════════
# §4  PASS C — HOUSE NUMBER BLOCKING
# ═══════════════════════════════════════════════════════════════════════════════


def house_num_pass(s1_records: pl.DataFrame, query_records: pl.DataFrame,
                   idf: dict[str, float], country_df: dict[str, int],
                   country: str, out_dir: Path, freq_cap: int = 50) -> tuple[pl.LazyFrame, int]:
    """Pass C: house-number-only blocking for distinctive house numbers.

    Direction: S2/S3 → S1, within country.
    Only indexes/queries house numbers with document frequency <= freq_cap
    to prevent Cartesian explosion on common numbers like '1', '2', '10'.

    Args:
        s1_records: S1 records for this country.
        query_records: S2/S3 records for this country.
        idf: house_num → IDF scores.
        country_df: house_num → document frequency.
        country: country label.
        freq_cap: skip house numbers appearing in > freq_cap records.

    Returns:
        DataFrame with (cand_id, s1_id, score).
    """
    s1_index: dict[str, list[tuple[str, float]]] = {}

    for row in s1_records.select("entity_id", "house_num").iter_rows(named=True):
        house = row["house_num"] or ""
        if not house or country_df.get(house, 0) > freq_cap:
            continue
        if house not in s1_index:
            s1_index[house] = []
        s1_index[house].append((row["entity_id"], idf.get(house, 5.0)))

    chunk_files = []
    total_pairs = 0
    chunk_size = 200_000
    n_chunks = max(1, math.ceil(query_records.height / chunk_size))

    for i in range(n_chunks):
        pairs: list[tuple[str, str, float]] = []
        chunk = query_records.slice(i * chunk_size, chunk_size)

        for row in chunk.select("entity_id", "house_num").iter_rows(named=True):
            house = row["house_num"] or ""
            if not house or country_df.get(house, 0) > freq_cap or house not in s1_index:
                continue
            for s1_id, score in s1_index[house]:
                pairs.append((row["entity_id"], s1_id, score))

        if pairs:
            chunk_df = pl.DataFrame(pairs, schema=["cand_id", "s1_id", "score"], orient="row")
            total_pairs += chunk_df.height
            chunk_path = out_dir / f"house_pass_{i}.parquet"
            chunk_df.write_parquet(chunk_path)
            chunk_files.append(chunk_path)

    if not chunk_files:
        return pl.DataFrame(schema={"cand_id": pl.String, "s1_id": pl.String, "score": pl.Float64}).lazy(), 0

    return pl.scan_parquet([str(p) for p in chunk_files]), total_pairs


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
