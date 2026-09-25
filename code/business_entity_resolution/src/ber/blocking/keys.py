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


def compute_token_idf(records: pl.DataFrame, column: str, country: str,
                      df_counts: dict[str, int] | None = None) -> dict[str, float]:
    """Compute IDF for each token in *column* within a country.

    IDF = log(N / df_t) where df_t = number of documents containing token t and N = records of the country.

    Args:
        records: DataFrame with 'country' and *column* (list[str] or str).
        column: Column name containing tokens (list[str]) or text (str to split).
        country: Country to filter on.
        df_counts: the country's compute_df_counts result, if already computed (avoids a second pass).

    Returns:
        dict mapping token → IDF score.
    """
    n_docs = records.filter(pl.col("country") == country).height
    if n_docs == 0:
        return {}
    df_counts = compute_df_counts(records, column, country) if df_counts is None else df_counts
    return {token: math.log(n_docs / max(df_val, 1)) for token, df_val in df_counts.items()}


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


def _pair_keys(tokens: list[str], idf: dict[str, float], country_df: dict[str, int], freq_cap: int,
               k: int = 3) -> list[tuple[tuple[str, str], float]]:
    """Scale-proof keys: sorted (a, b) pairs among the k rarest tokens (ignoring the cap) where at least one of the
    two is above freq_cap, with weight idf(a) + idf(b). Two frequent words ('shree', 'precision') are still a rare
    pair, so these keys keep working when a bigger world pushes single tokens over the absolute cap."""
    top = _get_rarest_tokens(tokens, idf, k, freq_cap, None)  # country_df=None -> no cap
    capped = {t for t in top if country_df.get(t, 0) > freq_cap}
    return [(tuple(sorted((a, b))), idf.get(a, 10.0) + idf.get(b, 10.0))
            for i, a in enumerate(top) for b in top[i + 1:] if a in capped or b in capped]


def _name_keys(row: dict, idf: dict[str, float], country_df: dict[str, int], skel_idf: dict[str, float],
               skel_df: dict[str, int], freq_cap: int, k_tokens: int, always_skeleton: bool,
               legal_skel: frozenset[str], pair_keys: bool = False) -> list[tuple]:
    """(key, weight) of a record: rarest name_core tokens (legal form removed) + rarest skeleton tokens that are not
    the skeleton of a legal-form word (always for S1, only non-Latin for queries), so 'limited' / 'private' / 'llc'
    and their transliterations ('limiteda' -> 'lmtd') never take the slots; + _pair_keys when ``pair_keys``."""
    core = (row["name_core"] or "").split()
    rare = _get_rarest_tokens(core, idf, k_tokens, freq_cap, country_df)
    pairs = _pair_keys(core, idf, country_df, freq_cap) if pair_keys else []
    if row["name_skeleton"] and (always_skeleton or row["name_script"] != "Latin"):
        skel_tokens = [t for t in row["name_skeleton"].split() if t not in legal_skel]
        skel_rare = _get_rarest_tokens(skel_tokens, skel_idf, k_tokens, freq_cap, skel_df)
        rare = list(dict.fromkeys(rare + skel_rare))  # ordered dedup: set order varies per process
        if pair_keys:
            pairs += _pair_keys(skel_tokens, skel_idf, skel_df, freq_cap)
    return [(t, idf.get(t, skel_idf.get(t, 10.0))) for t in rare] + list(dict(pairs).items())


def build_name_index(s1_records: pl.DataFrame, idf: dict[str, float], country_df: dict[str, int],
                     skel_idf: dict[str, float], skel_df: dict[str, int], legal_skel: frozenset[str] = frozenset(),
                     freq_cap: int = 2000, k_tokens: int = 3, pair_keys: bool = False) -> Index:
    """Pass A index, built once per country: rare name_core token (or skeleton token) -> [(s1_id, idf)].

    Skeleton tokens are always indexed for S1: the skeleton is the shared representation between Latin S1 names
    and transliterated non-Latin S2/S3 names. With ``pair_keys``, (a, b) pair keys are indexed too; a pair shared by
    more than freq_cap // 4 S1s is dropped (S1 postings vs all-source document frequency, ~1:4.7).
    """
    index: Index = {}
    for row in s1_records.select("entity_id", "name_core", "name_skeleton", "name_script").iter_rows(named=True):
        for key, weight in _name_keys(row, idf, country_df, skel_idf, skel_df, freq_cap, k_tokens,
                                      always_skeleton=True, legal_skel=legal_skel, pair_keys=pair_keys):
            index.setdefault(key, []).append((row["entity_id"], weight))
    for key in [k for k, v in index.items() if isinstance(k, tuple) and len(v) > freq_cap // 4]:
        del index[key]
    return index


def query_name_index(index: Index, query_records: pl.DataFrame, idf: dict[str, float], country_df: dict[str, int],
                     skel_idf: dict[str, float], skel_df: dict[str, int], legal_skel: frozenset[str] = frozenset(),
                     freq_cap: int = 2000, k_tokens: int = 3, top_k_per_query: int = 50,
                     pair_keys: bool = False) -> pl.DataFrame:
    """Pass A on a chunk of S2/S3 records: each record retrieves its top S1s by the summed IDF of shared rare tokens.

    Direction S2/S3 -> S1, within one country. Returns (cand_id, s1_id, score).
    """
    pairs = []
    for row in query_records.select("entity_id", "name_core", "name_skeleton", "name_script").iter_rows(named=True):
        keys = [k for k, _ in _name_keys(row, idf, country_df, skel_idf, skel_df, freq_cap, k_tokens,
                                          always_skeleton=False, legal_skel=legal_skel, pair_keys=pair_keys)]
        pairs += [(row["entity_id"], s1, sc) for s1, sc in _top_scores(index, keys, top_k_per_query)]
    return _pairs_frame(pairs)


def _street_keys(row: dict, idf: dict[str, float], country_df: dict[str, int], freq_cap: int,
                 k_tokens: int) -> list[tuple[str, str]]:
    """(number, rare street token) keys for EVERY number in the address (addr_nums, not only house_num), so a
    spurious extra number ('818 F-25', 'No. 337 1/598') does not hide the real one; [] when there is no number."""
    nums = list(dict.fromkeys(n for n in (row["addr_nums"] or []) if n))
    if not nums:
        return []
    street_tokens = (row["addr_street"] or "").split()
    if not street_tokens:
        num_set = set(nums)
        street_tokens = [t for t in (row["addr_norm"] or "").split() if t not in num_set and t.lstrip("0") not in num_set]
    rare = _get_rarest_tokens(street_tokens, idf, k_tokens, freq_cap, country_df)
    return [(n, t) for n in nums for t in rare]


def build_address_index(s1_records: pl.DataFrame, idf: dict[str, float], country_df: dict[str, int],
                        freq_cap: int = 2000, k_tokens: int = 2) -> Index:
    """Pass B index, built once per country: (any address number, rare street token) -> [(s1_id, idf)]."""
    index: Index = {}
    for row in s1_records.select("entity_id", "addr_nums", "addr_street", "addr_norm").iter_rows(named=True):
        for key in _street_keys(row, idf, country_df, freq_cap, k_tokens):
            index.setdefault(key, []).append((row["entity_id"], idf.get(key[1], 10.0)))
    return index


def query_address_index(index: Index, query_records: pl.DataFrame, idf: dict[str, float],
                        country_df: dict[str, int], freq_cap: int = 2000, k_tokens: int = 2,
                        top_k_per_query: int = 50) -> pl.DataFrame:
    """Pass B on a chunk of S2/S3 records: S1s sharing any address number + a rare street token.
    (cand_id, s1_id, score)."""
    pairs = []
    for row in query_records.select("entity_id", "addr_nums", "addr_street", "addr_norm").iter_rows(named=True):
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
    """Compute document frequency (count of records containing each token) within a country.

    Tokens are de-duplicated per record (list.unique) before counting, so no (entity_id, token) table is built:
    the transient memory is one token column instead of id + token pairs.

    Args:
        records: DataFrame with 'country' and *column*.
        column: Column name containing tokens (list[str]) or text (str, split on spaces).
        country: Country to filter on.

    Returns:
        dict mapping token → count.
    """
    col = pl.col(column)
    tokens = col if records.schema[column] == pl.List(pl.String) else col.str.split(" ")
    counts = (records.lazy().filter(pl.col("country") == country)
              .select(tokens.list.unique().alias("token")).explode("token")
              .filter(pl.col("token").is_not_null() & (pl.col("token") != ""))
              .group_by("token").agg(df=pl.len()).collect())
    return dict(zip(counts["token"].to_list(), counts["df"].to_list()))
