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

| stage | split | owner | output (in `cache/`) | status |
|---|---|---|---|---|
| `ingest` | both | Chris | raw parquet, `gt_train.parquet` | planned |
| `normalize` | both | Dhanishkaa | `records_{split}.parquet` | planned |
| `block` | both | Dhanishkaa | `candidates_{split}.parquet` | planned |
| `folds` | train | Faiz | `folds_train.parquet` | ready |
| `subworld` | train | Faiz | `subworld_train.parquet` + `*_train_sw.parquet` | ready |
| `blocking-report` | train | Faiz | markdown report on stdout | ready |
| `featurize` | both | Nitish | `features_{split}.parquet` | planned |
| `train` / `predict` | train / test | Nitish | `scores_{split}.parquet` | planned |
| `decide` | both | Faiz | `matches_{split}.parquet` (+ OOF threshold curve on train) | ready |
| `submit` | test | Faiz | `output/*.tsv`, official validator must PASS | ready |
| `all` | both | – | the whole chain for the split | – |

`--subworld` runs a stage on the closed sub-world (`subworld_frac` of the S1s, CLAUDE.md §6). Every stage logs
the config hash, git commit, runtime and peak memory.

**Stub data** (no real data needed): `python code/business_entity_resolution/scripts/make_stub_data.py` from the
repo root, then add `--set paths.cache_dir=cache/stub` to any stage.

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
