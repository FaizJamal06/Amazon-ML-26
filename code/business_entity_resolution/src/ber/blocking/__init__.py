"""Candidate generation: multi-pass blocking, always within country, chunked.

Contract:
In: records_{split}.parquet. Out: candidates_{split}.parquet (= candidate_pairs.tsv).

Owner: Dhanishkaa (R2 Normalize / Blocking)
"""
