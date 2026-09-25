"""Char n-gram TF-IDF top-k per country via chunked sparse matmul (S1 chunks of ~50k).

Contract:
In: records_{split}.parquet. Out: (s1_id, cand_id, tfidf_name, tfidf_full, knn_rank) pairs for merge.py.

Owner: Dhanishkaa (R2 Normalize / Blocking)
"""
