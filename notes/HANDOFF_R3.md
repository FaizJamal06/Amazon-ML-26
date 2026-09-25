# HANDOFF — R3 Features / LightGBM (Faiz → Nitish)

Faiz covered R3 on the night of Fri 25 Sep. The branch is `r3/features-lgbm`. Everything below is **stub-tested only**,
because real `records_train` / `candidates_train` had not landed yet.

## What's built

| module | what it does |
|---|---|
| `features/structured.py` | **Decoy-killers.** `house_num_class`: 8-way relation checked in this order: both_missing, one_missing, equal, same_base_diff_suffix, transposed, digit_dropped, small_shift (≤50), different. The base is the leading digits without leading zeros; the suffix is the rest, alphanumerics only. Also `hn_abs_diff_log`, `hn_rel_diff` and `hn_conflict` (small_shift or different). Token-set difference on `name_core` and on `addr_street`: counts per side, IDF-weighted sums, and a one-extra-token flag. `legal_rel` takes equal / one_missing / both_missing / conflict. `street_missing`. The fallback's original `house_num_relation` and `legal_conflict` also live here now; `fallback.py` imports them. |
| `features/pairwise.py` | ratio, token_set, token_sort, partial and Jaro-Winkler on `name_core` and on `addr_street`. `skel_ratio` on `name_skeleton` when both sides have it. Script codes per side plus `same_script`, length ratios, and `is_s3`. `cpdist` is shared with the fallback. |
| `features/context.py` | Blocking columns only: `block_score`, `pass_0..7` (bits of `block_mask`), `rank_in_cand`, `tfidf_*`. Also `gap_other_s1` (this pair's score minus the candidate's best score with any *other* S1), `n_s1_for_cand`, `cand_rank_in_s1`, `n_cands_s1` and `gap_to_s1_best`. |
| `features/rarity.py` | Per-country token IDF and exact `name_core` frequency, computed on the **full-world** records of the split (all sources). Features: `shared_idf_min/mean`, `a/b_name_freq_log`, `name_sim_x_rarity` (= name_tset × shared_idf_mean) and `name_match_hn_conflict` (= name_tsort ≥ 0.9 and hn_conflict). |
| `features/__init__.py` | The `featurize` stage. Joins pairs to records once, then streams `features_{split}` in chunks of `features.chunk_pairs` (default 5M) pairs. All features are Float32; train rows also get `label`. Country is used only as a join key for IDF, never as a feature. |
| `model/lgbm.py` | The `train` stage (`train_oof`): 5 folds from `folds_train`, grouped by `s1_id`. Each fold trains on up to `lgbm.max_train_s1` (300k) S1s from the other folds, with early stopping on a 10% S1-grouped inner split. OOF p comes from **cross-fitted** isotonic: fold k is mapped by an isotonic fit on the other folds' OOF. The final model is trained on a sample of all folds for the mean best-iteration count. The calibrator is isotonic on all OOF raw scores. Output goes to `cache/models/lgbm[_sw].joblib` and `reports/lgbm_importance_train[_sw].csv`. The `predict` stage is chunked. |
| `pipeline.py` | `featurize` / `train` / `predict` resolve through the existing `OWNER_STAGES` names, so dispatch is unchanged. `compare-decide` now reports LightGBM (`scores`) and the fallback (`scores_fallback`) side by side on the same folds and prints which one to use. |

55 features in total. Categorical codes: `hn_rel`, `legal_rel`, `s1_script`, `cand_script`. Params are in `lgbm.PARAMS`;
override any of them with `--set lgbm.params.<name>=<value>`.

## How to run

From `code/business_entity_resolution/src`:

```bash
# stub (tiny): regenerate, then the chain; min_data_in_leaf=5 because the stub has only 444 pairs
python ../scripts/make_stub_data.py
S="--set paths.cache_dir=cache/stub --set lgbm.params.min_data_in_leaf=5"
python -m ber.pipeline folds --split train $S
python -m ber.pipeline featurize --split train $S
python -m ber.pipeline train --split train $S
python -m ber.pipeline fallback-train --split train $S
python -m ber.pipeline compare-decide --split train $S
```

Real data: see "Real-data commands" below.

## What's tested

Every test file runs with `python tests/test_*.py`; pytest is not installed in the venv.

- `tests/test_features.py`: every house-number case from the brief, in both argument orders; the abs and rel diffs; the token-set difference, including the "exactly one extra token" flag and the IDF weights.
- `tests/test_lgbm.py`:
  - **Leakage:** features are byte-identical when `scores` / `scores_fallback` artifacts and a `p` column are present, and no `features/*.py` names a scores artifact.
  - **End to end:** builds a stub world in a temp dir and runs folds → featurize → train → decide → featurize → predict → decide → submit. The official validator, run with `--check-ids`, must PASS.
- `tests/test_fallback.py`: unchanged and passing. The fallback's `pair_features` output is identical to `main` on the stub.

## Known gaps

- **No real-data run yet.** Runtime and RAM on ~2.2M S1 × up to 50 candidates are untested.
  - `featurize` streams its output, but the per-chunk rapidfuzz work plus the list explodes in `idf_stats` are the likely hot spots.
  - Try `--set features.chunk_pairs=2000000` if RAM is tight.
- Context features use `block_score` only.
  - Competition gaps on a string composite (e.g. name_tset + street_tset) are probably stronger.
  - Adding one is a single `context_features(cand, score=...)` call once that score exists on the candidates table.
- R2's `candidates` v1 writes `tfidf_name`/`tfidf_full` as all-null. LightGBM ignores all-NaN columns, so this is harmless.
- `pass_0..7` assumes at most 8 blocking passes; raise `MASK_BITS` if R2 adds more.
- `house_num_class` looks only at the first 10 base digits for the transposed and digit_dropped checks.
- `hn_conflict` counts only small_shift and different. transposed and digit_dropped are treated as noise, because the EDA saw them in true pairs.
- No per-country or per-script importance, and no cross-country transfer run yet.
- The final model uses the fold models' mean best iteration, with no scaling for the extra 25% of data.
- On the stub, the fallback beats LightGBM: 0.9864 vs 0.9842 OOF. That is meaningless with 444 easy pairs. The real comparison is `compare-decide` on real data.

## Your next tasks (Nitish)

1. **Run on real data** as soon as `candidates_train` lands (commands below). Post the `scores_train` handoff, then log OOF F0.5 overall / US / India / Latin / native in `notes/EXPERIMENTS.md`.
2. **Beat the fallback's OOF score** in `compare-decide`. We use LightGBM only if it wins.
   - Read `reports/errors_train_scores.md`, produced by `python -m ber.pipeline errors --split train`.
   - Look at the 50 worst FP / FN.
   - The likely wins are the string-composite competition gaps and better handling of native-script names.
3. **Graph / cluster features** (`features/graph.py`, CLAUDE.md §5.3):
   - max and mean similarity of each candidate to the other top candidates of the same S1;
   - agreement between S2 and S3 candidates.
   - Compute them from string sims only, never from p.
4. **US→India transfer check** (CLAUDE.md §5.5a): train on US folds, score India, and the reverse. Drop features that do not transfer; France is unseen.

## Real-data commands

Run these when `records_train`, `candidates_train`, `records_test` and `candidates_test` are in `cache/`. From `code/business_entity_resolution/src`:

```bash
# train world
python -m ber.pipeline folds --split train              # if folds_train.parquet is not there yet
python -m ber.pipeline featurize --split train
python -m ber.pipeline train --split train             # -> scores_train (OOF), cache/models/lgbm.joblib
python -m ber.pipeline fallback-train --split train    # same folds, for the comparison
python -m ber.pipeline compare-decide --split train    # LightGBM vs fallback: overall / country / script
python -m ber.pipeline decide --split train            # prints the OOF threshold curve; set decide.threshold from it
python -m ber.pipeline errors --split train

# test (only if LightGBM won compare-decide; otherwise use --scorer fallback / fallback-predict)
python -m ber.pipeline featurize --split test
python -m ber.pipeline predict --split test            # -> scores_test
python -m ber.pipeline decide --split test --set decide.threshold=<best OOF t>
python -m ber.pipeline submit --split test             # writes output/*.tsv; official validator must PASS
```

Fast loop first: add `--subworld` to the train-side commands, after Faiz's `subworld` stage has written `*_train_sw`.
Sub-world features still use the full-world IDF.
