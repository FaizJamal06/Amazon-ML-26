"""Full-world 5-fold assignment of S1 entities (seed 42, stratified by country and match count) and the
closed sub-world sampler (S1 sample + their GT matches + their decoys).

Contract:
In: train records + ground truth + cached candidates.
Out: folds_train.parquet (s1_id, fold 0-4, country, n_matches); sub-world id lists.

Owner: Faiz (R1 Lead / Eval / Decision)
"""
