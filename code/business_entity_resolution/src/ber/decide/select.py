"""Per-S1 set selection after one-to-one assignment.

v1 ``threshold``: global threshold on calibrated p, effective threshold = ``t + decision_margin``.
v2 ``ef05_exact`` / ``ef05_approx``: per S1, keep the top-k (by p) that maximises expected F0.5 (CLAUDE.md §5.1).
Exact: TP_k ~ Poisson-binomial(top-k p), R_k ~ Poisson-binomial(rest of the kept p), plus ``miss_mass`` m:
E[F|k] = sum_t sum_r P(TP_k=t) P(R_k=r) * 1.25 t / (0.25 (t + r + m) + k);  E[F|0] = prod(1 - p) * exp(-m).
Approx: E[F|k] ~= 1.25 * sum_{i<=k} p_i / (0.25 * (sum_i p_i + m) + k). k grows from 0 only while each step gains
more than ``decision_margin``. Use v2 only after it beats the threshold on real full-world OOF.

Contract:
In: assigned pairs ``(s1_id, cand_id, p)`` from ``decide.assign``.
Out: long selected pairs ``(s1_id, cand_id)`` -> ``io.write_submission``; ``threshold_curve`` returns the OOF
macro F0.5 sweep (overall + per country) used to pick ``decide.threshold``.

Owner: Faiz (R1 Lead / Eval / Decision)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import polars as pl

from ber.decide.assign import assign_one_to_one
from ber.eval.metric import per_entity_f05


def select_threshold(assigned: pl.DataFrame, t: float, margin: float = 0.0) -> pl.DataFrame:
    """v1: keep assigned pairs with ``p >= t + margin``; returns ``(s1_id, cand_id, p)``."""
    return assigned.filter(pl.col("p") >= t + margin).select("s1_id", "cand_id", "p")


def pad_topk(assigned: pl.DataFrame, topk: int) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Keep the top ``topk`` pairs per S1 by p and lay their p out as an (N, K) matrix sorted descending.

    Returns ``(long, P, n)``: ``long`` = kept pairs with ``_row`` (S1 index) and ``_col`` (rank - 1); ``P`` padded with 0;
    ``n`` = number of real candidates per row. Ties in p are broken by ``cand_id`` (deterministic).
    """
    long = (assigned.select("s1_id", "cand_id", "p").sort(["s1_id", "p", "cand_id"], descending=[False, True, False])
            .with_columns(_col=pl.int_range(pl.len()).over("s1_id"))
            .filter(pl.col("_col") < topk)
            .with_columns(_row=pl.col("s1_id").rank("dense").cast(pl.Int64) - 1))
    n_rows = int(long["_row"].max() + 1) if long.height else 0
    P = np.zeros((n_rows, topk))
    rows, cols = long["_row"].to_numpy(), long["_col"].to_numpy()
    P[rows, cols] = long["p"].to_numpy()
    n = np.bincount(rows, minlength=n_rows)
    return long, P, n


def _poisson_binomial_prefix(P: np.ndarray) -> np.ndarray:
    """``pre[k, i, t]`` = P(t successes among the first k columns of row i), k = 0..K (DP over columns)."""
    N, K = P.shape
    pre = np.zeros((K + 1, N, K + 1))
    pre[0, :, 0] = 1.0
    for k in range(1, K + 1):  # K <= topk columns, vectorised over all rows
        p = P[:, k - 1:k]
        pre[k] = pre[k - 1] * (1 - p)
        pre[k, :, 1:] += pre[k - 1, :, :-1] * p
    return pre


def _poisson_binomial_suffix(P: np.ndarray) -> np.ndarray:
    """``suf[k, i, r]`` = P(r successes among columns k..K-1 of row i), k = 0..K."""
    N, K = P.shape
    suf = np.zeros((K + 1, N, K + 1))
    suf[K, :, 0] = 1.0
    for k in range(K - 1, -1, -1):
        p = P[:, k:k + 1]
        suf[k] = suf[k + 1] * (1 - p)
        suf[k, :, 1:] += suf[k + 1, :, :-1] * p
    return suf


def expected_f05(P: np.ndarray, n: np.ndarray, method: str = "exact", miss_mass: float = 0.0) -> np.ndarray:
    """Expected F0.5 of selecting the top-k, for k = 0..K: an (N, K+1) matrix (-inf where k > n).

    ``P`` rows must be sorted descending (``pad_topk``). ``method`` is ``exact`` (Poisson-binomial) or ``approx``
    (ratio of expectations).
    """
    N, K = P.shape
    E = np.full((N, K + 1), -np.inf)
    E[:, 0] = np.prod(1 - P, axis=1) * np.exp(-miss_mass)
    if method == "approx":
        k = np.arange(1, K + 1)
        E[:, 1:] = 1.25 * np.cumsum(P, axis=1) / (0.25 * (P.sum(axis=1, keepdims=True) + miss_mass) + k)
    elif method == "exact":
        pre, suf = _poisson_binomial_prefix(P), _poisson_binomial_suffix(P)
        t = np.arange(K + 1)[:, None]
        r = np.arange(K + 1)[None, :]
        for k in range(1, K + 1):  # K <= topk
            W = 1.25 * t / (0.25 * (t + r + miss_mass) + k)
            E[:, k] = np.einsum("nt,nr,tr->n", pre[k], suf[k], W)
    else:
        raise ValueError(f"unknown method {method!r}")
    E[np.arange(K + 1)[None, :] > n[:, None]] = -np.inf
    return E


def choose_k(E: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Per row, start at k = 0 and step k-1 -> k only while E[k] - E[k-1] > ``margin``; returns the stopping k."""
    with np.errstate(invalid="ignore"):
        ok = np.diff(E, axis=1) > margin          # (N, K); -inf steps and nan are never ok
    return np.where(ok.all(axis=1), E.shape[1] - 1, ok.argmin(axis=1))


def select_expected_f05(assigned: pl.DataFrame, method: str = "exact", topk: int = 15, miss_mass: float = 0.0,
                        margin: float = 0.0, chunk: int = 50_000) -> pl.DataFrame:
    """v2 selection: per S1 keep the top-k maximising expected F0.5 (``method`` exact | approx).

    Input: assigned pairs ``(s1_id, cand_id, p)`` with calibrated p. S1s are processed in chunks of ``chunk`` rows to
    bound the (K+1, chunk, K+1) PMF arrays (~100 MB each at chunk=50k, K=15). Returns ``(s1_id, cand_id, p)``.
    """
    long, P, n = pad_topk(assigned, topk)
    k = np.zeros(len(n), dtype=np.int64)
    for lo in range(0, len(n), chunk):  # chunks of S1s, not rows
        k[lo:lo + chunk] = choose_k(expected_f05(P[lo:lo + chunk], n[lo:lo + chunk], method, miss_mass), margin)
    chosen = pl.DataFrame({"_row": np.arange(len(k), dtype=np.int64), "_k": k})
    return long.join(chosen, on="_row").filter(pl.col("_col") < pl.col("_k")).select("s1_id", "cand_id", "p")


def select(assigned: pl.DataFrame, method: str, threshold: float = 0.5, topk: int = 15, miss_mass: float = 0.0,
           margin: float = 0.0) -> pl.DataFrame:
    """Dispatch on ``decide.method``: ``threshold`` | ``ef05_exact`` | ``ef05_approx``."""
    if method == "threshold":
        return select_threshold(assigned, threshold, margin)
    if method in ("ef05_exact", "ef05_approx"):
        return select_expected_f05(assigned, method.split("_")[1], topk, miss_mass, margin)
    raise ValueError(f"unknown decide.method {method!r}")


def threshold_grid(start: float, stop: float, step: float) -> list[float]:
    """Inclusive float grid, rounded to avoid 0.30000000000000004-style keys."""
    return [round(float(x), 4) for x in np.arange(start, stop + step / 2, step)]


def threshold_curve(scores: pl.DataFrame, gt: pl.DataFrame, s1: pl.DataFrame, grid: Sequence[float],
                    candidates: pl.DataFrame | None = None, margin: float = 0.0) -> pl.DataFrame:
    """Macro F0.5 for each threshold in ``grid`` after one-to-one assignment (use OOF p on train).

    ``s1`` = the evaluated S1 universe with a ``country`` column (singletons included). Returns one row per threshold
    with ``macro_f05`` overall, one ``f05_<country>`` column per country, and ``pred_pairs``.
    """
    assigned = assign_one_to_one(scores, candidates)
    rows = []
    for t in grid:
        sel = select_threshold(assigned, t, margin)
        ent = per_entity_f05(sel, gt, s1)
        row = {"t": t, "macro_f05": ent["f05"].mean(), "pred_pairs": sel.height}
        row.update({f"f05_{c}": v for c, v in ent.group_by("country").agg(pl.col("f05").mean()).iter_rows()})
        rows.append(row)
    return pl.DataFrame(rows).select("t", "macro_f05", pl.selectors.starts_with("f05_"), "pred_pairs")


def best_threshold(curve: pl.DataFrame) -> float:
    """Threshold with the highest overall macro F0.5 (ties -> the higher, more conservative t)."""
    return float(curve.sort(["macro_f05", "t"], descending=True)["t"][0])
