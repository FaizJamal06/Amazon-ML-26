"""Tests for expected-F0.5 selection (ber.decide.select v2). Run directly or with pytest."""
import itertools
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber.decide.select import choose_k, expected_f05, pad_topk, select_expected_f05  # noqa: E402


def brute(p: np.ndarray, m: float) -> np.ndarray:
    """E[F|k] for k = 0..n by enumerating all 2^n truth outcomes (p sorted descending)."""
    n = len(p)
    E = np.zeros(n + 1)
    for x in itertools.product([0, 1], repeat=n):
        x = np.array(x)
        w = np.prod(np.where(x == 1, p, 1 - p))
        for k in range(1, n + 1):
            t, r = x[:k].sum(), x[k:].sum()
            E[k] += w * (1.25 * t / (0.25 * (t + r + m) + k) if t else 0.0)
    E[0] = np.prod(1 - p) * np.exp(-m)
    return E


def brute_metric(p: np.ndarray) -> np.ndarray:
    """Same with m = 0 but scored with the official per-entity F0.5 definition (P/R form, empty/empty = 1)."""
    n = len(p)
    E = np.zeros(n + 1)
    for x in itertools.product([0, 1], repeat=n):
        x = np.array(x)
        w = np.prod(np.where(x == 1, p, 1 - p))
        E[0] += w * (1.0 if x.sum() == 0 else 0.0)
        for k in range(1, n + 1):
            t, g = x[:k].sum(), x.sum()
            E[k] += w * (0.0 if t == 0 else 1.25 * (t / k) * (t / g) / (0.25 * (t / k) + t / g))
    return E


def test_exact_matches_enumeration():
    """Exact E[F|k] equals brute-force enumeration for random small cases (n <= 8), with and without miss mass."""
    rng = np.random.default_rng(0)
    K = 8
    for _ in range(40):
        n = int(rng.integers(1, K + 1))
        m = float(rng.choice([0.0, 0.3, 1.5]))
        p = np.sort(rng.random(n))[::-1]
        P = np.zeros((1, K))
        P[0, :n] = p
        got = expected_f05(P, np.array([n]), "exact", m)[0, : n + 1]
        assert np.allclose(got, brute(p, m), atol=1e-12), (p, m)
        assert np.all(np.isneginf(expected_f05(P, np.array([n]), "exact", m)[0, n + 1:]))
        if m == 0:
            assert np.allclose(got, brute_metric(p), atol=1e-12)


def test_tiny_p_selects_nothing_and_certain_p_selects_all():
    """All p tiny -> k = 0 (predict singleton); all p ~ 1 -> take all; for both methods."""
    for method in ("exact", "approx"):
        P = np.array([[0.02, 0.01, 0.01, 0.0], [0.999, 0.998, 0.997, 0.0]])
        k = choose_k(expected_f05(P, np.array([3, 3]), method, 0.0), 0.0)
        assert k.tolist() == [0, 3], (method, k)


def test_margin_is_monotone():
    """A larger decision_margin never selects more candidates."""
    rng = np.random.default_rng(1)
    P = -np.sort(-rng.random((500, 10)), axis=1)
    n = rng.integers(0, 11, 500)
    P[np.arange(10)[None, :] >= n[:, None]] = 0
    for method in ("exact", "approx"):
        E = expected_f05(P, n, method, 0.2)
        ks = [choose_k(E, mg) for mg in (0.0, 0.005, 0.02, 0.1, 0.5)]
        assert all((a >= b).all() for a, b in zip(ks, ks[1:])), method
        assert (ks[0] <= n).all()


def test_select_end_to_end_and_topk():
    """Long-frame selection keeps the chosen top-k per S1, respects topk, and handles chunking."""
    assigned = pl.DataFrame({
        "s1_id": ["S1-a"] * 3 + ["S1-b"] * 2 + ["S1-c"] * 20,
        "cand_id": [f"S2-{i}" for i in range(25)],
        "p": [0.99, 0.97, 0.02, 0.01, 0.02] + [0.99] * 20,
    })
    got = select_expected_f05(assigned, "exact", topk=15, chunk=2)
    per = dict(got.group_by("s1_id").len().iter_rows())
    assert per == {"S1-a": 2, "S1-c": 15}                     # S1-b -> empty (singleton), S1-c capped at topk
    assert set(got.filter(pl.col("s1_id") == "S1-a")["cand_id"]) == {"S2-0", "S2-1"}
    long, P, n = pad_topk(assigned, 15)
    assert P.shape == (3, 15) and n.tolist() == [3, 2, 15] and (np.diff(P, axis=1) <= 0).all()


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} select v2 tests passed")
