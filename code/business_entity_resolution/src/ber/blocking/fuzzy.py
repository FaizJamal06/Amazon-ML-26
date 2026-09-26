"""Pass E (fuzzy): char 3-gram TF-IDF cosine between names, per country, for records the key passes left weak.

The key passes need an exact shared token; typos, OCR digits ('bu1kers'), transliteration variants and names written
as one word ('reedpizza' vs 'reed pizza') share none. Here every name is a char_wb 3-gram TF-IDF vector over
``name_core + no-space name_core + non-legal name_skeleton`` (handles/domains cleaned first, normalize.strip_handle), and
a query retrieves its top S1s by cosine. N-grams in more than ``max_df`` of the country's S1 names are dropped so the
sparse products stay small. Only run for query records whose best rank_score from the other passes is weak (merge.py).

Contract:
In: records of one country (S1 for the index, a chunk of S2/S3 for queries).
Out: (cand_id, s1_id, score) with score = cosine in [0, 1], like the other passes.

Owner: Dhanishkaa (R2 Normalize / Blocking); pass E by Faiz.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from ber.blocking.keys import PAIR_SCHEMA
from ber.normalize import is_handle_expr, strip_handle_expr

TEXT_COLS = ["entity_id", "name_raw", "name_core", "name_skeleton"]


def fuzzy_text(records: pl.DataFrame) -> pl.Series:
    """Text vectorized for pass E: cleaned name (handles/domains -> words) + the same without spaces + skeleton."""
    core = pl.when(is_handle_expr("name_raw")).then(strip_handle_expr("name_raw")).otherwise(pl.col("name_core"))
    return records.select(pl.concat_str(core.fill_null(""), core.fill_null("").str.replace_all(" ", ""),
                                        pl.col("name_skeleton").fill_null(""), separator=" ")).to_series()


@dataclass
class FuzzyIndex:
    """TF-IDF vectorizer fitted on one country's S1 names + the transposed S1 matrix (vocabulary x S1)."""
    vectorizer: TfidfVectorizer
    s1_t: sp.csr_matrix
    s1_ids: np.ndarray


def build_fuzzy_index(s1_records: pl.DataFrame, max_df: float = 0.002) -> FuzzyIndex:
    """Fit char_wb 3-gram TF-IDF (l2-normalized, float32) on the S1 names; n-grams in > max_df of S1s are dropped.

    The cap never goes below 50 documents, so a small country (or the stub world) keeps a usable vocabulary.
    """
    max_df = min(1.0, max(max_df, 50 / max(s1_records.height, 1)))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), max_df=max_df, dtype=np.float32, sublinear_tf=True)
    m = vec.fit_transform(fuzzy_text(s1_records).to_list())
    return FuzzyIndex(vec, m.T.tocsr(), s1_records["entity_id"].to_numpy())


def row_top_k(m: sp.csr_matrix, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(row, col, value) of the k largest stored values of every row of a CSR matrix, vectorized (no Python row loop).

    Ties are broken by column index, so the result is deterministic.
    """
    counts = np.diff(m.indptr)
    rows = np.repeat(np.arange(m.shape[0]), counts)
    order = np.lexsort((m.indices, -m.data, rows))              # by row, then value desc, then column
    rank = np.arange(order.size) - np.repeat(m.indptr[:-1], counts)
    keep = order[rank < k]
    return rows[keep], m.indices[keep], m.data[keep]


def query_fuzzy(ix: FuzzyIndex, query_records: pl.DataFrame, top_k: int = 20, min_cos: float = 0.3,
                block_rows: int = 1000) -> pl.DataFrame:
    """Pass E on S2/S3 records: top_k S1s by cosine (>= min_cos) per record, in blocks of ``block_rows`` queries.
    Returns (cand_id, s1_id, score)."""
    if query_records.height == 0:
        return pl.DataFrame(schema=PAIR_SCHEMA)
    q = ix.vectorizer.transform(fuzzy_text(query_records).to_list())
    ids = query_records["entity_id"].to_numpy()
    out = []
    for lo in range(0, q.shape[0], block_rows):  # blocks of queries, not rows
        sim = (q[lo:lo + block_rows] @ ix.s1_t).tocsr()
        sim.data[sim.data < min_cos] = 0
        sim.eliminate_zeros()
        r, c, v = row_top_k(sim, top_k)
        out.append(pl.DataFrame({"cand_id": ids[lo + r], "s1_id": ix.s1_ids[c], "score": v.astype(np.float64)}))
    return pl.concat(out) if out else pl.DataFrame(schema=PAIR_SCHEMA)
