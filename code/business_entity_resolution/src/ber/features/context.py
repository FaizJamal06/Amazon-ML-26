"""Ranks, gaps and one-to-one competition features, per S1 and per candidate.
Computed from blocking/string similarities only, never from model p (leakage).

Contract:
In: candidate pairs + pairwise similarities. Out: feature columns keyed by (s1_id, cand_id).

Owner: Nitish (R3 Features / LightGBM)
"""
