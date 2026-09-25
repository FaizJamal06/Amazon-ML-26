"""Token / name frequency (IDF) per country computed on the record pool (no external data).

Contract:
In: records_{split}.parquet. Out: rarity lookups + feature columns keyed by (s1_id, cand_id).

Owner: Nitish (R3 Features / LightGBM)
"""
