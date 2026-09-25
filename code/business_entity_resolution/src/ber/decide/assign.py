"""One-to-one: each S2/S3 id is assigned to its single best S1 (or none).

Contract:
In: scores_{split}.parquet (s1_id, cand_id, p). Out: the same pairs filtered to one S1 per cand_id.

Owner: Faiz (R1 Lead / Eval / Decision)
"""
