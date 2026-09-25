"""Name/address string similarities (token set/sort, Jaro-Winkler, ...) for (S1, candidate) pairs.

Contract:
In: candidate pairs joined with normalized records (``a_*`` = S1 side, ``b_*`` = candidate side).
Out: feature columns keyed by (s1_id, cand_id).

Owner: Nitish (R3 Features / LightGBM)
"""
from __future__ import annotations

import numpy as np
from rapidfuzz import process


def cpdist(a: list[str], b: list[str], scorer, scale: float = 100.0) -> np.ndarray:
    """Element-wise rapidfuzz scores in [0, 1] for two equal-length string lists (multithreaded)."""
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32) / scale
