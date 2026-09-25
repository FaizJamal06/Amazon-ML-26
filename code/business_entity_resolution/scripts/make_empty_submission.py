"""Format-probe submission: every test S1 id with an EMPTY match list (and an empty candidate list).

Usage (from the repo root):  python code/business_entity_resolution/scripts/make_empty_submission.py

Writes output/matching_results.tsv + output/candidate_pairs.tsv through ber.io.write_submission, which runs the
official validator and raises unless it prints PASS. Expected public LB ~= share of singletons (train: 5.6%).
"""
import sys
from pathlib import Path

import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber.config import load_config  # noqa: E402
from ber.io import write_submission  # noqa: E402


def main() -> None:
    """Read test S1 ids with the mandated read, write the all-empty submission and print the hashes."""
    cfg = load_config()
    s1 = pd.read_csv(cfg.path("test.source1"), sep="\t", dtype=str, keep_default_na=False)
    ids = pl.Series("s1_id", s1["entity_id"].tolist(), dtype=pl.String)
    empty = pl.DataFrame(schema={"s1_id": pl.String, "cand_id": pl.String})
    out = write_submission(empty, empty, ids, pl.Series([], dtype=pl.String), cfg.output_dir,
                           cfg.path("validator"), cfg.path("test.source1").parent)
    print(f"test S1 rows: {len(ids):,}")
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
