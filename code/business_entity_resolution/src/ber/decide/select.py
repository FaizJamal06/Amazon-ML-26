"""Per-S1 set selection after one-to-one assignment.

v1 (live): global threshold on calibrated p, effective threshold = ``t + decision_margin``.
v2 (stub): expected-F0.5 optimal top-k per S1 (CLAUDE.md §5.1) — to be implemented next.

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


def select_expected_f05(assigned: pl.DataFrame, shrink: float = 1.0, miss_mass: float = 0.0,
                        margin: float = 0.0) -> pl.DataFrame:
    """v2 (not implemented yet): per S1, choose k maximising expected F0.5.

    For each S1 sort candidates by calibrated p (after ``shrink``); for k = 1..K
    E[F0.5](k) ~= 1.25 * sum_{i<=k} p_i / (0.25 * E|G| + k), with E|G| = sum_i p_i + ``miss_mass``;
    k = 0 scores prod_i (1 - p_i) (probability the entity is a singleton). Pick the argmax, requiring the winner to
    beat k = 0 by ``margin``.
    """
    raise NotImplementedError("expected-F0.5 selection (CLAUDE.md §5.1) is planned for Day 2")


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
