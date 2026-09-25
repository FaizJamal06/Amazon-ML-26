"""Quick test: normalize the stub records and run blocking on them.

Usage: python scripts/test_normalize_block.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl

from ber.config import load_config
from ber.normalize import (
    _normalize_base,
    _show_examples,
    compute_area_tokens,
    detect_script,
    name_skeleton,
    normalize_address,
    normalize_name,
    remove_area_tokens,
    transliterate,
)

cfg = load_config(sets=["paths.cache_dir=cache/stub"])

# Read the stub records (already has name_raw, addr_raw)
stub_records = pl.read_parquet(cfg.artifact("records", "train"))

print("=" * 80)
print("TEST: Normalization on stub data")
print("=" * 80)

# Re-normalize from raw columns
t0 = time.time()
result_rows = []
for row in stub_records.iter_rows(named=True):
    name_fields = normalize_name(row["name_raw"], row["country"])
    addr_fields = normalize_address(row["addr_raw"], row["country"])
    result_rows.append({
        "entity_id": row["entity_id"],
        "source": row["source"],
        "country": row["country"],
        "name_raw": row["name_raw"],
        "addr_raw": row["addr_raw"],
        **name_fields,
        **addr_fields,
    })

records = pl.DataFrame(result_rows, schema={
    "entity_id": pl.String,
    "source": pl.Int8,
    "country": pl.String,
    "name_raw": pl.String,
    "addr_raw": pl.String,
    "name_script": pl.String,
    "name_norm": pl.String,
    "name_core": pl.String,
    "legal_form": pl.String,
    "name_tokens": pl.List(pl.String),
    "name_skeleton": pl.String,
    "addr_norm": pl.String,
    "addr_street": pl.String,
    "house_num": pl.String,
    "addr_nums": pl.List(pl.String),
})

# Compute area tokens
area_tokens = compute_area_tokens(records, threshold_frac=0.15)
for country, tokens in area_tokens.items():
    print(f"  {country}: {len(tokens)} area tokens: {sorted(tokens)[:20]}")

# Remove area tokens from addr_street
addr_street_cleaned = []
for row in records.select("addr_street", "country").iter_rows():
    area_set = area_tokens.get(row[1], set())
    addr_street_cleaned.append(remove_area_tokens(row[0], area_set))
records = records.with_columns(pl.Series("addr_street", addr_street_cleaned))

norm_time = time.time() - t0
print(f"\nNormalization: {records.height} records in {norm_time:.2f}s")

# Show examples
_show_examples(records, area_tokens)

# Save normalized records for blocking test
out_path = cfg.artifact("records", "train")
records.write_parquet(out_path)
print(f"\nWrote {records.height} records to {out_path}")

# ── Now test blocking ─────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("TEST: Blocking on stub data")
print("=" * 80)

from ber.blocking.merge import build_candidates

t0 = time.time()
build_candidates(cfg, "train")
block_time = time.time() - t0

# Read the candidates back and check
candidates = pl.read_parquet(cfg.artifact("candidates", "train"))
gt = pl.read_parquet(cfg.artifact("gt", "train"))

print(f"\nBlocking: {candidates.height} candidate pairs in {block_time:.2f}s")
print(f"GT pairs: {gt.height}")

# Check recall
gt_pairs = set(zip(gt["s1_id"].to_list(), gt["match_id"].to_list()))
cand_pairs = set(zip(candidates["s1_id"].to_list(), candidates["cand_id"].to_list()))
found = gt_pairs & cand_pairs
recall = len(found) / max(len(gt_pairs), 1)
print(f"Pair recall: {recall:.4f} ({len(found)}/{len(gt_pairs)})")

# Missed pairs
missed = gt_pairs - cand_pairs
if missed:
    print(f"\nMissed {len(missed)} GT pairs:")
    for s1, cand in list(missed)[:10]:
        s1_rec = records.filter(pl.col("entity_id") == s1).head(1)
        c_rec = records.filter(pl.col("entity_id") == cand).head(1)
        if s1_rec.height > 0 and c_rec.height > 0:
            print(f"  S1={s1}: {s1_rec['name_raw'][0][:40]} / {s1_rec['addr_raw'][0][:40]}")
            print(f"  C ={cand}: {c_rec['name_raw'][0][:40]} / {c_rec['addr_raw'][0][:40]}")

per_s1 = candidates.group_by("s1_id").len()["len"]
print(f"\nCandidates per S1: mean={per_s1.mean():.1f}, median={per_s1.median():.0f}, max={per_s1.max()}")

# Test blocking report
print("\n" + "=" * 80)
print("TEST: Blocking report")
print("=" * 80)
from ber.eval.blocking_report import blocking_report, to_markdown
from ber.blocking.merge import PASS_NAMES
report = blocking_report(candidates, gt, records, PASS_NAMES)
print(to_markdown(report))
