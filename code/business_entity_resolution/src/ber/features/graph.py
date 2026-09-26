"""Cluster-consistency features: S2<->S3 agreement / similarity to other top candidates of the same S1.

Idea: the true matches of one S1 (which are S2/S3 duplicates of the same business) look like each other.
A decoy (shifted house number or one extra word) agrees with fewer of the genuine cluster members.
All features are computed from string similarities of record fields only — never from model scores or labels.

Contract:
In:  candidates_{split}.parquet  (s1_id, cand_id, rank_score, source)
     records_{split}.parquet     (entity_id, name_core, addr_street, house_num, source)
Out: DataFrame keyed by (s1_id, cand_id) with graph feature columns (Float32).
     Returned by ``graph_features()``, then horizontally concatenated in ``build_features``.

Rules:
- NO labels and NO model scores — only record fields + blocking columns (rank_score, rank_in_cand).
- Vectorized: candidates self-joined on s1_id (top-5 cap), then rapidfuzz.process.cpdist(workers=-1),
  chunked by S1 groups so memory stays bounded.
- Every function has a docstring.

Owner: Nitish (R3 Features / LightGBM)
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

# How many top (by rank_score) sibling candidates to compare against.
TOP_K_SIBLINGS = 5
# Minimum name_core token_set_ratio (0-1) to count as strong agreement.
STRONG_NAME_THR = 0.90
# Rapidfuzz batch size (keeps Python string lists bounded).
_CHUNK = 500_000


def _cpdist(a: list[str], b: list[str], scorer, scale: float = 100.0) -> np.ndarray:
    """Element-wise rapidfuzz scores in [0, 1] for two equal-length string lists (multi-threaded)."""
    if not a:
        return np.empty(0, np.float32)
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32) / scale


def _top_k_siblings(cand: pl.DataFrame, k: int) -> pl.DataFrame:
    """Return the top-k sibling rows per s1_id (ranked by rank_score, descending).

    Each row in the returned frame is one sibling candidate: s1_id, cand_id, name_core,
    addr_street, house_num, source, rank_score. Used to self-join against the full candidate set.
    """
    return (
        cand
        .with_columns(
            _sib_rank=pl.col("rank_score").rank("ordinal", descending=True).over("s1_id")
        )
        .filter(pl.col("_sib_rank") <= k)
        .drop("_sib_rank")
    )


def _cross_join_siblings(cand: pl.DataFrame, siblings: pl.DataFrame) -> pl.DataFrame:
    """Self-join candidates with their siblings on s1_id, excluding self-pairs (cand_id == sib_cand_id).

    Returns a frame with columns:
        s1_id, cand_id, name_core, addr_street, house_num, source,
        sib_cand_id, sib_name_core, sib_addr_street, sib_house_num, sib_source, sib_rank_score
    """
    left = cand.select(
        "s1_id", "cand_id",
        name_core=pl.col("name_core"),
        addr_street=pl.col("addr_street"),
        house_num=pl.col("house_num"),
        source=pl.col("source"),
    )
    right = siblings.select(
        "s1_id",
        sib_cand_id=pl.col("cand_id"),
        sib_name_core=pl.col("name_core"),
        sib_addr_street=pl.col("addr_street"),
        sib_house_num=pl.col("house_num"),
        sib_source=pl.col("source"),
        sib_rank_score=pl.col("rank_score"),
    )
    return (
        left.join(right, on="s1_id", how="inner")
        .filter(pl.col("cand_id") != pl.col("sib_cand_id"))
    )


def _compute_pairwise_sims(pairs: pl.DataFrame) -> pl.DataFrame:
    """Add name_tset and addr_tset columns (token_set_ratio, 0-1) to a cross-joined sibling frame.

    Processed in chunks of ``_CHUNK`` rows to keep peak memory bounded.
    """
    n = pairs.height
    name_tset = np.empty(n, np.float32)
    addr_tset = np.empty(n, np.float32)
    for lo in range(0, n, _CHUNK):
        part = pairs.slice(lo, _CHUNK)
        na = part["name_core"].fill_null("").to_list()
        nb = part["sib_name_core"].fill_null("").to_list()
        aa = part["addr_street"].fill_null("").to_list()
        ab = part["sib_addr_street"].fill_null("").to_list()
        name_tset[lo: lo + part.height] = _cpdist(na, nb, fuzz.token_set_ratio)
        addr_tset[lo: lo + part.height] = _cpdist(aa, ab, fuzz.token_set_ratio)
    return pairs.with_columns(
        name_tset=pl.Series(name_tset),
        addr_tset=pl.Series(addr_tset),
    )


def graph_features(cand: pl.DataFrame, rec: pl.DataFrame) -> pl.DataFrame:
    """Compute cluster-consistency graph features for every (s1_id, cand_id) pair.

    For each pair we compare the candidate against the S1's top-5 other candidates (by rank_score):

    - ``graph_name_max``         : max name_core token_set_ratio to those siblings [0, 1].
    - ``graph_name_mean``        : mean name_core token_set_ratio to those siblings [0, 1]; 0 if no siblings.
    - ``graph_addr_max``         : max addr_street token_set_ratio to those siblings [0, 1].
    - ``graph_addr_mean``        : mean addr_street token_set_ratio to those siblings [0, 1].
    - ``graph_same_hnum``        : number of siblings that share the same non-empty house_num as this candidate.
    - ``graph_top1_agree``       : 1 iff the candidate strongly agrees (name ≥ STRONG_NAME_THR and same
                                   house_num) with the S1's single top-ranked sibling (by rank_score).
                                   0 when no top-1 or when either house_num is empty.
    - ``graph_cross_src_agree``  : number of candidates from the OTHER source (S2 vs S3) that strongly
                                   agree with this candidate (name ≥ STRONG_NAME_THR and same house_num).
    - ``graph_cluster_size``     : number of candidates of this S1 whose name_core token_set_ratio to the
                                   S1 record is ≥ STRONG_NAME_THR.

    Parameters
    ----------
    cand:
        Candidates table — must contain s1_id, cand_id, rank_score, source (int).
    rec:
        Records table — entity_id, source (int), name_core, addr_street, house_num.
        Must include both S1 records (source=1) and S2/S3 records.

    Returns
    -------
    DataFrame with s1_id, cand_id and the graph feature columns (all Float32), sorted by (s1_id, cand_id).
    """
    # ── 1. Join candidate record fields ───────────────────────────────────────────────────────────
    # Fallback: use block_score for ranking when rank_score is absent (e.g. in stub data).
    _score_col = "rank_score" if "rank_score" in cand.columns else "block_score"
    # Fallback: if candidates already carry source (S2/S3 source id), use it; otherwise get from rec.
    _has_cand_source = "source" in cand.columns

    s2s3_rec = rec.filter(pl.col("source") != 1).select(
        entity_id=pl.col("entity_id"),
        name_core=pl.col("name_core").fill_null(""),
        addr_street=pl.col("addr_street").fill_null(""),
        house_num=pl.col("house_num").fill_null(""),
        rec_source=pl.col("source").cast(pl.Int8),
    )
    s1_rec = rec.filter(pl.col("source") == 1).select(
        s1_id=pl.col("entity_id"),
        s1_name_core=pl.col("name_core").fill_null(""),
    )

    _cand_sel = ["s1_id", "cand_id", pl.col(_score_col).alias("rank_score")]
    if _has_cand_source:
        _cand_sel.append(pl.col("source").cast(pl.Int8).alias("cand_src"))
    cand_aug = (
        cand.select(*_cand_sel)
        .join(s2s3_rec, left_on="cand_id", right_on="entity_id", how="left")
        .with_columns(
            name_core=pl.col("name_core").fill_null(""),
            addr_street=pl.col("addr_street").fill_null(""),
            house_num=pl.col("house_num").fill_null(""),
            source=(
                pl.col("cand_src") if _has_cand_source
                else pl.col("rec_source")
            ).fill_null(2).cast(pl.Int8),
        )
        .drop(["rec_source"] + (["cand_src"] if _has_cand_source else []))
    )

    # ── 2. Build sibling table (top-K per S1 by rank_score) ───────────────────────────────────────
    siblings = _top_k_siblings(cand_aug, TOP_K_SIBLINGS)

    # ── 3. Cross-join: each candidate × siblings of the same S1 ──────────────────────────────────
    pairs = _cross_join_siblings(cand_aug, siblings)

    # ── 4. Compute string similarities ────────────────────────────────────────────────────────────
    if pairs.height > 0:
        pairs = _compute_pairwise_sims(pairs)
    else:
        pairs = pairs.with_columns(
            name_tset=pl.lit(0.0).cast(pl.Float32),
            addr_tset=pl.lit(0.0).cast(pl.Float32),
        )

    # ── 5. Identify top-1 sibling per S1 ─────────────────────────────────────────────────────────
    top1_sib = (
        siblings
        .sort("sib_rank_score" if "sib_rank_score" in siblings.columns else "rank_score", descending=True)
        .group_by("s1_id")
        .first()
        .select("s1_id", top1_sib_id=pl.col("cand_id"))
    )

    # Re-read rank_score from siblings frame — it was renamed in _cross_join_siblings
    # Actually siblings still has rank_score; use that directly here
    top1_sib = (
        siblings
        .sort("rank_score", descending=True)
        .group_by("s1_id")
        .first()
        .select("s1_id", top1_sib_id=pl.col("cand_id"))
    )

    pairs_with_top1 = pairs.join(top1_sib, on="s1_id", how="left")

    # ── 6. Boolean helpers ────────────────────────────────────────────────────────────────────────
    hn_match = (
        (pl.col("house_num") != "") &
        (pl.col("sib_house_num") != "") &
        (pl.col("house_num") == pl.col("sib_house_num"))
    )
    strong_agree = (pl.col("name_tset") >= STRONG_NAME_THR) & hn_match
    top1_agree = strong_agree & (pl.col("sib_cand_id") == pl.col("top1_sib_id"))
    cross_src = pl.col("sib_source") != pl.col("source")

    # ── 7. Aggregate over siblings ────────────────────────────────────────────────────────────────
    agg = (
        pairs_with_top1
        .group_by("s1_id", "cand_id")
        .agg(
            graph_name_max=pl.col("name_tset").max(),
            graph_name_mean=pl.col("name_tset").mean(),
            graph_addr_max=pl.col("addr_tset").max(),
            graph_addr_mean=pl.col("addr_tset").mean(),
            graph_same_hnum=hn_match.cast(pl.Int32).sum(),
            graph_top1_agree=top1_agree.cast(pl.Int32).max(),
            graph_cross_src_agree=(cross_src & strong_agree).cast(pl.Int32).sum(),
        )
    )

    # ── 8. Cluster size: candidates whose name_core is close to the S1's name_core ───────────────
    cand_vs_s1 = cand_aug.join(s1_rec, on="s1_id", how="left").with_columns(
        s1_name_core=pl.col("s1_name_core").fill_null("")
    )
    n = cand_vs_s1.height
    name_vs_s1_arr = np.empty(n, np.float32)
    for lo in range(0, n, _CHUNK):
        part = cand_vs_s1.slice(lo, _CHUNK)
        na = part["name_core"].to_list()
        nb = part["s1_name_core"].to_list()
        name_vs_s1_arr[lo: lo + part.height] = _cpdist(na, nb, fuzz.token_set_ratio)

    cluster_size = (
        cand_vs_s1
        .with_columns(_name_vs_s1=pl.Series(name_vs_s1_arr))
        .group_by("s1_id")
        .agg(graph_cluster_size=(pl.col("_name_vs_s1") >= STRONG_NAME_THR).cast(pl.Int32).sum())
    )

    # ── 9. Final join — fill missing rows (singleton S1s with no siblings) with 0 ─────────────────
    cand_base = cand_aug.select("s1_id", "cand_id")
    result = (
        cand_base
        .join(agg, on=["s1_id", "cand_id"], how="left")
        .join(cluster_size, on="s1_id", how="left")
        .with_columns(
            graph_name_max=pl.col("graph_name_max").fill_null(0.0).cast(pl.Float32),
            graph_name_mean=pl.col("graph_name_mean").fill_null(0.0).cast(pl.Float32),
            graph_addr_max=pl.col("graph_addr_max").fill_null(0.0).cast(pl.Float32),
            graph_addr_mean=pl.col("graph_addr_mean").fill_null(0.0).cast(pl.Float32),
            graph_same_hnum=pl.col("graph_same_hnum").fill_null(0).cast(pl.Float32),
            graph_top1_agree=pl.col("graph_top1_agree").fill_null(0).cast(pl.Float32),
            graph_cross_src_agree=pl.col("graph_cross_src_agree").fill_null(0).cast(pl.Float32),
            graph_cluster_size=pl.col("graph_cluster_size").fill_null(0).cast(pl.Float32),
        )
        .sort(["s1_id", "cand_id"])
    )
    return result
