# Methodology notes

Each owner writes one paragraph per module as it is built. R1 assembles the final `Documentation_template.md` from this on Day 3.

## R1 Lead / Eval / Decision — Faiz

## R2 Normalize / Blocking — Dhanishkaa

## R3 Features / LightGBM — Nitish

**Features (63 total across five groups).** The primary decoy-killers live in `features/structured.py`: an 8-way house-number relation (`equal`, `one_missing`, `both_missing`, `same_base_diff_suffix`, `transposed`, `digit_dropped`, `small_shift`, `different`) plus log/relative numeric diff and a conflict flag, computed in a single polars expression pass with no Python loops. Token-set difference on `name_core` and `addr_street` counts tokens present on one side only, IDF-weights them (per-country IDF from the record pool), and flags the "exactly one extra word" decoy pattern. `features/pairwise.py` adds five rapidfuzz string similarities (ratio, token_set, token_sort, partial, Jaro-Winkler) on `name_core` and `addr_street`, script codes, length ratios, and source indicator. `features/context.py` derives blocking-column features (pass flags, rank-within-S1, competition gaps) from the candidates table before any model score exists. `features/rarity.py` computes per-country token IDF and exact name frequency over the full-world record pool; features include shared-token IDF statistics and the interactions `name_sim × name_rarity` and `name_match & hn_conflict`. `features/graph.py` (Task A, Day 2) adds eight cluster-consistency features: for each pair (s1, cand), the max/mean `name_core` and `addr_street` token_set_ratio of cand to the S1's top-5 other candidates (by rank_score), the count of siblings sharing the same house_num, a flag for strong agreement with the top-ranked sibling, a cross-source agreement count (S2 vs S3), and the cluster size (candidates whose name is similar to the S1 record). All graph features are computed via `rapidfuzz.process.cpdist(workers=-1)` in batches of 500k rows. Country is never used as a feature; IDF lookups use the country string as a grouping key only, so France and any other unseen country are handled transparently.

**Model.** LightGBM binary classification (`lgbm.py`): 5-fold OOF, each fold trained on up to 300k S1s with early stopping on a held-out 10% S1-grouped inner split. Raw OOF scores are calibrated by cross-fitted isotonic regression (fold k fitted on the OOF of the other folds) to avoid inflated probabilities. The final model is trained on a sample of all folds at the mean best-iteration count. Feature importance (gain) is logged per fold and for the final model to `reports/lgbm_importance_train.csv`. A `transfer_check.py` script (Task B) measures the US→India and India→US transfer gap and runs a drop-one feature-group ablation to identify features that hurt cross-country generalisation (critical for unseen France). Findings are written to `notes/TRANSFER_CHECK.md` without changing any pipeline defaults.

## R4 Deep models / AWS — Chris
