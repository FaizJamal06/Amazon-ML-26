"""LightGBM training (5-fold OOF), prediction and isotonic calibration on held-out folds.

Contract:
In: features_{split}.parquet + folds_train.parquet.
Out: scores_{split}.parquet with s1_id, cand_id, p (calibrated); OOF on train.

Owner: Nitish (R3 Features / LightGBM)
"""
