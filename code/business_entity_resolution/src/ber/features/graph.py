"""Cluster-consistency features: S2<->S3 agreement / similarity to other top candidates of the same S1.

Contract:
In: candidate pairs + pairwise similarities. Out: feature columns keyed by (s1_id, cand_id).

Owner: Nitish (R3 Features / LightGBM)
"""
