"""Streaming union of blocking passes with a provenance bitmask; re-rank; per-record cut; per-S1 safety cap.

Contract:
In: outputs of keys / tfidf_knn / embed_knn.
Out: candidates_{split}.parquet with s1_id, cand_id, country, block_mask, tfidf_name, tfidf_full, knn_rank,
     rank_in_cand, block_score, rank_score.
Every retrieved pair (union of passes, before any cap) gets a cheap re-rank similarity ``rank_score``; each S2/S3
record keeps its top ``blocking.k_per_query`` S1s by rank_score, then each S1 keeps at most ``max_cands``.

Owner: Dhanishkaa (R2 Normalize / Blocking)
"""
from __future__ import annotations

import time
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz

from ber.blocking.fuzzy import FuzzyIndex, build_fuzzy_index, query_fuzzy
from ber.blocking.keys import (
    PAIR_SCHEMA,
    build_address_index,
    build_combo_index,
    build_house_index,
    build_name_index,
    compute_df_counts,
    compute_token_idf,
    query_address_index,
    query_combo_index,
    query_house_index,
    query_name_index,
)
from ber import rules
from ber.config import Config
from ber.features.pairwise import cpdist
from ber.normalize import _normalize_base, name_skeleton

# Pass bit assignments (bitmask)
PASS_NAME_TOKEN = 0    # bit 0 (1 << 0 = 1)
PASS_ADDR_KEY = 1      # bit 1 (1 << 1 = 2)
PASS_HOUSE_NUM = 2     # bit 2 (1 << 2 = 4)
PASS_NAME_STREET = 3   # bit 3 (1 << 3 = 8): exact name + (name token, street token) keys, scale-proof
PASS_FUZZY = 4         # bit 4 (1 << 4 = 16): char 3-gram TF-IDF cosine on names (fuzzy.py), weak records only

PASS_NAMES: dict[int, str] = {
    PASS_NAME_TOKEN: "name_token",
    PASS_ADDR_KEY: "addr_key",
    PASS_HOUSE_NUM: "house_num",
    PASS_NAME_STREET: "name_street",
    PASS_FUZZY: "fuzzy",
}

# rank_score = weighted cheap similarities (0-1 scale; a house-number conflict can push it below 0)
RANK_WEIGHTS = {"name_tsr": 0.45, "skel_ratio": 0.15, "street_tsr": 0.30, "hn_equal": 0.10, "hn_conflict": -0.20}
RANK_FIELDS = ("name_core", "name_skeleton", "addr_street", "house_num")
RANK_CHUNK = 5_000_000  # pairs per re-rank batch (bounds the joined strings in memory)


def rank_scores(pairs: pl.DataFrame, records: pl.DataFrame) -> pl.Series:
    """Cheap re-rank similarity for every (s1_id, cand_id) pair, row-aligned with ``pairs``.

    rank_score = 0.45 * token_set_ratio(name_core) + 0.15 * ratio(name_skeleton) + 0.30 * token_set_ratio(addr_street)
                 + 0.10 * [house_num equal] - 0.20 * [house_num conflict]  (similarities in 0-1).
    The skeleton term is 0 when either side has no skeleton; a missing house number on either side is neither equal
    nor a conflict. Vectorized: pairs are joined to the record fields in chunks and scored with rapidfuzz cpdist.
    """
    def side(key: str, prefix: str) -> pl.DataFrame:
        """Record fields used for re-ranking, keyed by ``key`` and prefixed."""
        return records.select(pl.col("entity_id").alias(key), *[pl.col(c).alias(prefix + c) for c in RANK_FIELDS])

    a_side, b_side = side("s1_id", "a_"), side("cand_id", "b_")
    out = []
    for lo in range(0, pairs.height, RANK_CHUNK):  # chunks of pairs, not rows
        df = (pairs.slice(lo, RANK_CHUNK).select("s1_id", "cand_id")
              .join(a_side, on="s1_id", how="left", maintain_order="left")
              .join(b_side, on="cand_id", how="left", maintain_order="left")
              .with_columns(pl.col("^[ab]_.*$").fill_null("").str.strip_chars()))
        sim = lambda f, scorer: cpdist(df[f"a_{f}"].to_list(), df[f"b_{f}"].to_list(), scorer)  # noqa: E731
        both = lambda f: ((df[f"a_{f}"] != "") & (df[f"b_{f}"] != "")).to_numpy()  # noqa: E731
        hn_eq = both("house_num") & (df["a_house_num"] == df["b_house_num"]).to_numpy()
        hn_conflict = both("house_num") & ~hn_eq
        out.append(RANK_WEIGHTS["name_tsr"] * sim("name_core", fuzz.token_set_ratio)
                   + RANK_WEIGHTS["skel_ratio"] * sim("name_skeleton", fuzz.ratio) * both("name_skeleton")
                   + RANK_WEIGHTS["street_tsr"] * sim("addr_street", fuzz.token_set_ratio)
                   + RANK_WEIGHTS["hn_equal"] * hn_eq + RANK_WEIGHTS["hn_conflict"] * hn_conflict)
    return pl.Series("rank_score", np.concatenate(out) if out else np.empty(0), dtype=pl.Float64)


RANK_BY, RANK_DESC = ["rank_score", "block_score", "s1_id", "cand_id"], [True, True, False, False]
RECORD_COLS = ["entity_id", "source", "country", "name_skeleton", "name_script", "name_core",
               "house_num", "addr_nums", "addr_street", "addr_norm"]
EMPTY_CANDIDATES = {"s1_id": pl.String, "cand_id": pl.String, "block_mask": pl.Int32, "block_score": pl.Float64,
                    "rank_score": pl.Float64, "country": pl.String, "rank_in_cand": pl.Int32}


def _merge_passes(raw: pl.DataFrame, pass_max: dict[int, float]) -> pl.DataFrame:
    """Merge raw pass hits (cand_id, s1_id, score, bit) into (s1_id, cand_id, block_mask, block_score).

    Each pass scores on its own scale, so its score is first normalized to 0-1 by ``pass_max[bit]`` (that pass's
    max score over the whole country, so every chunk is scaled the same way). block_mask ORs the pass bits;
    block_score = sum of the normalized pass scores (0 to n_passes).
    """
    bits, maxes = list(pass_max), [float(pass_max[b]) for b in pass_max]
    return (raw.with_columns(score=(pl.col("score").cast(pl.Float64)
                                    / pl.col("bit").replace_strict(bits, maxes, return_dtype=pl.Float64)).fill_nan(0.0),
                             bit_val=pl.col("bit").replace_strict(list(PASS_NAMES), [1 << b for b in PASS_NAMES],
                                                               return_dtype=pl.Int32))
            .group_by("s1_id", "cand_id").agg(
                block_mask=pl.col("bit_val").bitwise_or(),
                block_score=pl.col("score").sum().round(9),   # round: group_by sum order varies -> last-bit noise
            ))


def _rank_per_cand(candidates: pl.DataFrame, k_per_query: int, min_score: float | None = None) -> pl.DataFrame:
    """Rank the S1s of every cand_id (RANK_BY) -> rank_in_cand, then keep rank_in_cand <= k_per_query.

    rank_in_cand is computed on the full retrieval list of the cand_id, BEFORE min_score / k_per_query remove
    anything. Needs every pair of a cand_id in ``candidates`` (true for a chunk of query records).
    """
    ranked = (candidates.sort(RANK_BY, descending=RANK_DESC)
              .with_columns(rank_in_cand=(pl.int_range(pl.len()).over("cand_id") + 1).cast(pl.Int32)))
    if min_score:
        ranked = ranked.filter(pl.col("block_score") >= min_score)
    return ranked.filter(pl.col("rank_in_cand") <= k_per_query)


def _cap_per_s1(candidates: pl.DataFrame, max_cands: int) -> pl.DataFrame:
    """Per-S1 safety cap: keep each S1's best ``max_cands`` candidates (same RANK_BY order)."""
    return candidates.sort(RANK_BY, descending=RANK_DESC).filter(pl.int_range(pl.len()).over("s1_id") < max_cands)


def _cap_candidates(candidates: pl.DataFrame, k_per_query: int = 2,
                    max_cands: int = 50, min_score: float | None = None) -> pl.DataFrame:
    """Rank, then cap: per S2/S3 keep the top k S1s, then per S1 keep the top max_cands candidates.

    Ordering everywhere: rank_score desc, block_score desc, s1_id asc, cand_id asc (deterministic ties).
    rank_in_cand (1 = best S1 for this cand_id) is computed on the full retrieval list, BEFORE min_score /
    k_per_query / max_cands remove anything. build_candidates applies the same two steps chunk by chunk.

    Args:
        candidates: DataFrame with (s1_id, cand_id, block_mask, block_score, rank_score).
        k_per_query: per S2/S3 record, keep at most this many S1 candidates.
        max_cands: per S1, cap total candidates at this number (a safety net).
        min_score: drop pairs with block_score below this (None or 0 = off).

    Returns:
        Capped DataFrame with an added rank_in_cand column.
    """
    return _cap_per_s1(_rank_per_cand(candidates, k_per_query, min_score), max_cands)


PASS_LIMITS = {"name_freq_cap": 2000, "addr_freq_cap": 2000, "house_freq_cap": 50, "top_k_per_query": 50,
               "name_pair_keys": 0, "addr_key_cap": 0, "name_street_cap": 0,
               "fuzzy": 0, "fuzzy_min_score": 0.8, "fuzzy_max_df": 0.002, "fuzzy_top_k": 20}
# name_pair_keys: 1 = also key on pairs of frequent name tokens (keys._pair_keys).
# addr_key_cap: 0 = street tokens above addr_freq_cap are skipped; N > 0 = no token cap, but (number, token) keys shared
# by more than N S1s are dropped (keys._street_keys).
# name_street_cap: 0 = pass D off; N > 0 = pass D on (exact-name + (name token, street token) keys, keys shared by
# more than N S1s dropped; keys._combo_keys).
# fuzzy: 1 = pass E on (fuzzy.py) for query records whose best rank_score from the other passes is < fuzzy_min_score.


def pass_limits(cfg: Config) -> dict:
    """PASS_LIMITS overridden by the ``blocking.*`` config keys of the same name (bools -> 0/1)."""
    return {k: type(v)(cfg.get(f"blocking.{k}", v)) for k, v in PASS_LIMITS.items()}


@dataclass
class CountryIndex:
    """What every pass needs for one country: pool document frequencies / IDF and the S1 indexes (built once)."""
    country: str
    lim: dict
    dfs: dict
    idfs: dict
    legal_skel: frozenset
    name: dict
    addr: dict
    house: dict
    combo: dict | None
    fuzzy: FuzzyIndex | None


def build_country_index(recs: pl.DataFrame, country: str, limits: dict | None = None) -> CountryIndex:
    """IDF over all records of ``country`` (all sources) and the S1 indexes of every enabled pass."""
    lim = {**PASS_LIMITS, **(limits or {})}
    s1_recs = recs.filter(pl.col("source") == 1)
    dfs = {c: compute_df_counts(recs, c, country) for c in ("name_core", "name_skeleton", "addr_norm", "house_num")}
    idfs = {c: compute_token_idf(recs, c, country, df) for c, df in dfs.items()}  # idf from df: one pass per column
    legal_skel = frozenset(tok for form in rules.legal_forms(country) for tok in name_skeleton(_normalize_base(form)).split())
    return CountryIndex(
        country, lim, dfs, idfs, legal_skel,
        name=build_name_index(s1_recs, idfs["name_core"], dfs["name_core"], idfs["name_skeleton"], dfs["name_skeleton"],
                              legal_skel, freq_cap=lim["name_freq_cap"], pair_keys=bool(lim["name_pair_keys"])),
        addr=build_address_index(s1_recs, idfs["addr_norm"], dfs["addr_norm"], freq_cap=lim["addr_freq_cap"],
                                 key_cap=lim["addr_key_cap"]),
        house=build_house_index(s1_recs, idfs["house_num"], dfs["house_num"], freq_cap=lim["house_freq_cap"]),
        combo=(build_combo_index(s1_recs, idfs["name_core"], idfs["name_skeleton"], idfs["addr_norm"], legal_skel,
                                 key_cap=lim["name_street_cap"]) if lim["name_street_cap"] else None),
        fuzzy=build_fuzzy_index(s1_recs, lim["fuzzy_max_df"]) if lim["fuzzy"] else None,
    )


def query_passes(ix: CountryIndex, part: pl.DataFrame) -> pl.DataFrame:
    """Key passes A-D for a chunk of S2/S3 records -> raw hits (cand_id, s1_id, score, bit). Pass E runs later."""
    lim, dfs, idfs = ix.lim, ix.dfs, ix.idfs
    frames = [
        query_name_index(ix.name, part, idfs["name_core"], dfs["name_core"], idfs["name_skeleton"],
                         dfs["name_skeleton"], ix.legal_skel, freq_cap=lim["name_freq_cap"],
                         top_k_per_query=lim["top_k_per_query"], pair_keys=bool(lim["name_pair_keys"]))
        .with_columns(bit=PASS_NAME_TOKEN),
        query_address_index(ix.addr, part, idfs["addr_norm"], dfs["addr_norm"], freq_cap=lim["addr_freq_cap"],
                            top_k_per_query=lim["top_k_per_query"], key_cap=lim["addr_key_cap"])
        .with_columns(bit=PASS_ADDR_KEY),
        query_house_index(ix.house, part, dfs["house_num"], freq_cap=lim["house_freq_cap"]).with_columns(bit=PASS_HOUSE_NUM),
    ]
    if ix.combo is not None:
        frames.append(query_combo_index(ix.combo, part, idfs["name_core"], idfs["name_skeleton"], idfs["addr_norm"],
                                        ix.legal_skel, top_k_per_query=lim["top_k_per_query"])
                      .with_columns(bit=PASS_NAME_STREET))
    return pl.concat(frames).with_columns(pl.col("bit").cast(pl.Int8))


def rerank_chunk(ix: CountryIndex, raw: pl.DataFrame, part: pl.DataFrame, pass_max: dict[int, float],
                 rank_recs: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Merge a chunk's raw hits (normalized by the country-wide pass maxima), add rank_score, and, when pass E is on,
    add fuzzy hits for the records whose best rank_score is below fuzzy_min_score (or that got nothing).

    Returns (merged pairs with block_mask / block_score / rank_score / country, the fuzzy raw hits).
    """
    merged = _merge_passes(raw, pass_max)
    merged = merged.with_columns(rank_scores(merged, rank_recs))
    fz = pl.DataFrame(schema={**PAIR_SCHEMA, "bit": pl.Int8})
    if ix.fuzzy is not None:
        best = merged.group_by("cand_id").agg(best=pl.col("rank_score").max())
        weak = (part.join(best, left_on="entity_id", right_on="cand_id", how="left")
                .filter(pl.col("best").is_null() | (pl.col("best") < ix.lim["fuzzy_min_score"])))
        fz = query_fuzzy(ix.fuzzy, weak, top_k=ix.lim["fuzzy_top_k"]).with_columns(bit=pl.lit(PASS_FUZZY, pl.Int8))
        if fz.height:
            known = merged.select("s1_id", "cand_id", "rank_score")
            merged = (_merge_passes(pl.concat([raw, fz]), {**pass_max, PASS_FUZZY: 1.0})   # cosine is already 0-1
                      .join(known, on=["s1_id", "cand_id"], how="left"))
            new = merged.filter(pl.col("rank_score").is_null()).drop("rank_score")
            merged = pl.concat([merged.filter(pl.col("rank_score").is_not_null()),
                                new.with_columns(rank_scores(new, rank_recs))])
    return merged.with_columns(country=pl.lit(ix.country)), fz


def _block_country(recs: pl.DataFrame, country: str, tmp: Path, chunk_rows: int, k_per_query: int,
                   min_score: float | None, limits: dict | None = None) -> list[Path]:
    """Stream one country: S1 indexes built once, S2/S3 queried in chunks of ``chunk_rows`` records.

    Phase 1: passes A-D run on each chunk; raw hits go to disk and the per-pass max score is tracked.
    Phase 2: each chunk is normalized with the country-wide pass maxima, merged, re-ranked (rank_score; pass E for
    weak records when on) and cut to the top ``k_per_query`` S1s per cand_id; the (small) result goes to disk.
    Returns the result files. Memory is bounded by the chunk (+ the country's S1 indexes), not by the pair count.
    """
    s1_n, query_recs = recs.filter(pl.col("source") == 1).height, recs.filter(pl.col("source") > 1)
    print(f"\n  -- {country}: {s1_n:,} S1, {query_recs.height:,} S2/S3 --", flush=True)
    if s1_n == 0 or query_recs.height == 0:
        return []
    t = time.time()
    ix = build_country_index(recs, country, limits)
    print(f"    indexes built in {time.time() - t:.1f}s", flush=True)

    raw_files, pass_max, n_raw = [], {}, 0
    for i, lo in enumerate(range(0, query_recs.height, chunk_rows)):  # chunks of records, not rows
        raw = query_passes(ix, query_recs.slice(lo, chunk_rows))
        for bit, mx in raw.group_by("bit").agg(pl.col("score").max()).iter_rows():
            pass_max[bit] = max(pass_max.get(bit, 0.0), mx)
        n_raw += raw.height
        raw_files.append(tmp / f"{country}_raw_{i:05d}.parquet")
        raw.write_parquet(raw_files[-1])
    ix.name = ix.addr = ix.house = ix.combo = None   # free the key indexes; pass E (if on) is still needed
    print(f"    phase 1: {n_raw:,} raw pass hits in {len(raw_files)} chunks ({time.time() - t:.1f}s)", flush=True)

    rank_recs = recs.select("entity_id", *RANK_FIELDS)
    kept_files, n_merged, n_kept, n_fuzzy = [], 0, 0, 0
    for i, path in enumerate(raw_files):
        merged, fz = rerank_chunk(ix, pl.read_parquet(path), query_recs.slice(i * chunk_rows, chunk_rows), pass_max,
                                  rank_recs)
        kept = _rank_per_cand(merged, k_per_query, min_score)
        n_merged, n_kept, n_fuzzy = n_merged + merged.height, n_kept + kept.height, n_fuzzy + fz.height
        kept_files.append(tmp / f"{country}_kept_{i:05d}.parquet")
        kept.write_parquet(kept_files[-1])
        path.unlink()
    print(f"    phase 2: {n_merged:,} unique pairs re-ranked ({n_fuzzy:,} fuzzy hits), {n_kept:,} kept "
          f"(k={k_per_query}) ({time.time() - t:.1f}s)", flush=True)
    return kept_files


def build_candidates(cfg: Config, split: str, subworld: bool = False,
                     k_per_query: int | None = None, max_cands: int | None = None) -> None:
    """Build candidates_{split}.parquet from records_{split}.parquet, streaming per country and per chunk.

    Per country: S1 indexes once, S2/S3 in chunks of ``blocking.chunk_rows`` (default 50k): passes -> union ->
    rank_score -> top ``blocking.k_per_query`` per cand_id -> disk. Then the per-S1 ``max_cands`` safety cap on the
    concatenated (small) result. Only the columns blocking needs are read, one country at a time.
    """
    t0 = time.time()
    path = cfg.artifact("records", split, subworld)
    countries = sorted(pl.scan_parquet(path).select(pl.col("country").unique()).collect()["country"].to_list())
    print(f"[block] split={split}, countries: {countries}")
    max_cands = cfg.max_cands if max_cands is None else max_cands
    k_per_query = int(cfg.get("blocking.k_per_query", 2)) if k_per_query is None else k_per_query
    min_score = cfg.get("blocking.min_score", None)   # None or 0 = off
    chunk_rows = int(cfg.get("blocking.chunk_rows", 50_000))
    limits = pass_limits(cfg)
    print(f"  k_per_query={k_per_query}, max_cands={max_cands}, min_score={min_score}, chunk_rows={chunk_rows}, "
          f"limits={limits}")

    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cfg.cache_dir, prefix="tmp_block_", ignore_cleanup_errors=True) as tmp:
        kept_files = []
        for country in countries:  # countries, not rows
            tc = time.time()
            cols = RECORD_COLS + (["name_raw"] if limits["fuzzy"] else [])   # pass E cleans handles from name_raw
            recs = pl.scan_parquet(path).filter(pl.col("country") == country).select(cols).collect()
            kept_files += _block_country(recs, country, Path(tmp), chunk_rows, k_per_query, min_score, limits)
            del recs
            print(f"    {country} done in {time.time() - tc:.1f}s")
        candidates = (pl.concat([pl.read_parquet(f) for f in kept_files]) if kept_files
                      else pl.DataFrame(schema=EMPTY_CANDIDATES))

    candidates = _cap_per_s1(candidates, max_cands)
    candidates = candidates.with_columns(
        tfidf_name=pl.lit(None, dtype=pl.Float64),   # contract columns, not produced by these passes
        tfidf_full=pl.lit(None, dtype=pl.Float64),
        knn_rank=pl.lit(None, dtype=pl.Int32),
    ).select(
        "s1_id", "cand_id", "country", "block_mask",
        "tfidf_name", "tfidf_full", "knn_rank", "rank_in_cand", "block_score", "rank_score",
    ).sort(["s1_id", "rank_score", "cand_id"], descending=[False, True, False])

    out_path = cfg.artifact("candidates", split, subworld)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.write_parquet(out_path)

    per_s1 = candidates.group_by("s1_id").len()["len"]
    print(f"\n[block] wrote {candidates.height:,} candidate pairs to {out_path}")
    print(f"  S1s with candidates: {candidates['s1_id'].n_unique():,}")
    if candidates.height:
        print(f"  candidates per S1: mean={per_s1.mean():.1f}, median={per_s1.median():.0f}, "
              f"p95={per_s1.quantile(0.95):.0f}, max={per_s1.max()}")
    print(f"  runtime: {time.time() - t0:.1f}s")
