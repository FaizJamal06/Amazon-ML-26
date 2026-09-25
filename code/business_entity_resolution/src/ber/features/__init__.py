"""Pairwise feature engineering for (S1, candidate) pairs.

Contract:
In: candidates_{split}.parquet + records_{split}.parquet.
Out: features_{split}.parquet with s1_id, cand_id, <feature columns>, label (train only).

Owner: Nitish (R3 Features / LightGBM)
"""
