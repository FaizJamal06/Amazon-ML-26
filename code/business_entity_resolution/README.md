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

## Running (planned — stages are stubs today)

```bash
cd code/business_entity_resolution/src
python -m ber.pipeline ingest      # TSV -> parquet cache            (planned)
python -m ber.pipeline block       # candidates_{split}.parquet      (planned)
python -m ber.pipeline featurize   # features_{split}.parquet        (planned)
python -m ber.pipeline train       # 5-fold OOF LightGBM + calibration (planned)
python -m ber.pipeline predict     # scores_test.parquet             (planned)
python -m ber.pipeline submit      # output/*.tsv + validator        (planned)
python -m ber.pipeline all         # everything above                (planned)
```

Every stage will accept `--subworld 0.1` for the fast loop.

## Validate before every upload

From `student_resource/`:

```bash
python utils/validate_submission.py \
    --matching ../output/matching_results.tsv \
    --candidate ../output/candidate_pairs.tsv \
    --test-dir dataset/test
```

It must print `PASS`. Add `--check-ids` to also check that every id exists in test. It needs a few GB of RAM.
