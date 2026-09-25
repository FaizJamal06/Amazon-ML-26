import sys
import time
from pathlib import Path
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber.config import load_config
from ber.blocking.merge import build_candidates, PASS_NAMES
from ber.eval.blocking_report import blocking_report

# redirect stdout to hide noise
import os
import contextlib

cfg = load_config(sets=["paths.cache_dir=cache/stub"])
records = pl.read_parquet(cfg.artifact("records", "train"))
gt = pl.read_parquet(cfg.artifact("gt", "train"))

print("| k_per_query | pairs | cand_mean | cand_p95 | pair_recall | full_gt_covered | reduction_ratio |")
print("|---|---|---|---|---|---|---|")

for k in [1, 2, 3, 5]:
    cfg = load_config(sets=["paths.cache_dir=cache/stub"])
    
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        build_candidates(cfg, "train", k_per_query=k)
        candidates = pl.read_parquet(cfg.artifact("candidates", "train"))
        report = blocking_report(candidates, gt, records, PASS_NAMES)
    
    all_row = None
    for row in report["summary"].iter_rows(named=True):
        if row["country"] == "ALL":
            all_row = row
            break
            
    if all_row:
        print(f"| {k} | {all_row['pairs']} | {all_row['cand_mean']:.2f} | {all_row['cand_p95']:.2f} | {all_row['pair_recall']:.4f} | {all_row['full_gt_covered']:.4f} | {all_row['reduction_ratio']:.4f} |")
