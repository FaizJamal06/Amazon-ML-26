"""Candidate generation: multi-pass blocking, always within country, chunked.

Contract:
In: records_{split}.parquet. Out: candidates_{split}.parquet (= candidate_pairs.tsv).

Owner: Dhanishkaa (R2 Normalize / Blocking)
"""
from ber.blocking.merge import PASS_NAMES, build_candidates

__all__ = ["build_candidates", "PASS_NAMES"]
