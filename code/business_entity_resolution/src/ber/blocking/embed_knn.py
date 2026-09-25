"""Optional pass: multilingual-e5 embeddings + FAISS top-k per country (helps native-script names).

Contract:
In: records_{split}.parquet. Out: (s1_id, cand_id, pass_bit, score) pairs for merge.py.
Model license + HF revision must be recorded in docs/MODELS.md.

Owner: Chris (R4 Deep models / AWS), for R2
"""
