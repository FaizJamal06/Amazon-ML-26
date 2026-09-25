"""SageMaker-trainable pair classifier (cross-encoder) fine-tuned with hard negatives from
our own blocking; run on the top-K per S1 after LightGBM.

Contract:
In: candidate pairs + normalized records + folds_train.parquet.
Out: an out-of-fold score column keyed by (s1_id, cand_id), used as a LightGBM feature.
Model must be MIT/Apache-2.0, <= 8B params, recorded in docs/MODELS.md.

Owner: Chris (R4 Deep models / AWS)
"""
