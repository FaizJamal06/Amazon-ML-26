# business_entity_resolution

Pipeline for the Amazon ML Challenge 2026 Business Entity Resolution task. Design, rules and data contracts: `CLAUDE.md` at the repo root.

## Setup

Python 3.12. From the repo root (`ml26/`):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt
```

## Data

The data is **not** in git. Unzip the official bundle at the repo root so these paths exist:

```
student_resource/dataset/train/train_source{1,2,3}.tsv
student_resource/dataset/train/train_ground_truth.tsv
student_resource/dataset/test/test_source{1,2,3}.tsv
student_resource/utils/validate_submission.py
```

Paths are set in `configs/base.yaml`, relative to the repo root. Never modify files under `student_resource/`.

## Running

```bash
cd code/business_entity_resolution/src
python -m ber.pipeline <stage> --split {train,test} [--subworld] [--config override.yaml] [--set key=value ...]
```

| stage | split | owner | what it does → output (in `cache/` unless noted) | status |
|---|---|---|---|---|
| `ingest` | both | Chris | raw TSVs → `source{1,2,3}_{split}` parquet (+ `gt_train`) | placeholder on `main` |
| `normalize` | both | Dhanishkaa | transliteration, name/address normalisation, house numbers → `records_{split}` | PR pending |
| `block` | both | Dhanishkaa | multi-pass blocking within country → `candidates_{split}` | PR pending |
| `folds` | train | Faiz | 5 S1-level folds, stratified by country × match count → `folds_train` | ready |
| `subworld` | train | Faiz | closed ~10% sub-world (S1s + their matches + their decoys) → `*_train_sw` | ready |
| `blocking-report` | train | Faiz | recall ceiling, candidates/S1 and reduction ratio, as markdown on stdout | ready |
| `featurize` | both | Nitish | pair features (house-number class, token-set diff, string sims, context, rarity) → `features_{split}` | ready |
| `train` | train | Nitish | LightGBM 5-fold OOF + isotonic → `scores_train`, `cache/models/lgbm.joblib` | ready |
| `predict` | both | Nitish | saved LightGBM + calibrator → `scores_{split}` | ready |
| `fallback-train` | train | Faiz | rule-feature fallback scorer, OOF on the same folds → `scores_fallback_train` | ready |
| `fallback-predict` | both | Faiz | saved fallback model → `scores_fallback_{split}` | ready |
| `decide` | both | Faiz | one-to-one assignment + `decide.method` selection → `matches_{split}` (OOF F0.5 on train) | ready |
| `compare-decide` | train | Faiz | OOF macro F0.5 of LightGBM vs fallback × threshold / ef05, per country and per script | ready |
| `errors` | train | Faiz | worst FP / FN with raw strings → `reports/errors_*.md` | ready |
| `submit` | test | Faiz | `output/*.tsv` + official validator (`--check-ids`), which must PASS | ready |
| `all` | both | – | the whole chain for the split (`--scorer lgbm` default, or `--scorer fallback`) | – |

**End-to-end order** (from `code/business_entity_resolution/src`):

```bash
# train world
python -m ber.pipeline ingest --split train
python -m ber.pipeline normalize --split train
python -m ber.pipeline block --split train
python -m ber.pipeline folds --split train
python -m ber.pipeline blocking-report --split train
python -m ber.pipeline featurize --split train
python -m ber.pipeline train --split train
python -m ber.pipeline fallback-train --split train
python -m ber.pipeline compare-decide --split train     # use LightGBM only if it beats the fallback
python -m ber.pipeline decide --split train             # pick decide.threshold from the OOF curve
python -m ber.pipeline errors --split train
# test
python -m ber.pipeline ingest --split test
python -m ber.pipeline normalize --split test
python -m ber.pipeline block --split test
python -m ber.pipeline featurize --split test
python -m ber.pipeline predict --split test             # or fallback-predict + --scorer fallback on decide
python -m ber.pipeline decide --split test --set decide.threshold=<t>
python -m ber.pipeline submit --split test
```

Fast loop: run `subworld` once, then add `--subworld` to the train-side stages.

`--subworld` runs a stage on the closed sub-world (`subworld_frac` of the S1s, CLAUDE.md §6). Every stage logs
the config hash, git commit, runtime and peak memory.

**Stub data** (no real data needed): `python code/business_entity_resolution/scripts/make_stub_data.py` from the
repo root, then add `--set paths.cache_dir=cache/stub` to any stage (for `train` also `--set lgbm.params.min_data_in_leaf=5`:
the stub has only ~450 pairs).

**Tests:** `python -m pytest code/business_entity_resolution/tests`, or run any `tests/test_*.py` file directly.

## Validate before every upload

From `student_resource/`:

```bash
python utils/validate_submission.py \
    --matching ../output/matching_results.tsv \
    --candidate ../output/candidate_pairs.tsv \
    --test-dir dataset/test
```

It must print `PASS`. Add `--check-ids` to also check that every id exists in test. It needs a few GB of RAM.
