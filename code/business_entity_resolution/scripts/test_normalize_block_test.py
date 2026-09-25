"""Quick test: normalize + block the test stub (includes France).

Usage: python scripts/test_normalize_block_test.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl

from ber.config import load_config
from ber.normalize import (
    _show_examples,
    compute_area_tokens,
    normalize_address,
    normalize_name,
    remove_area_tokens,
)

cfg = load_config(sets=["paths.cache_dir=cache/stub"])

# Read the stub test records
stub_records = pl.read_parquet(cfg.artifact("records", "test"))

print("=" * 80)
print("TEST: Normalization on test stub data (includes France)")
print("=" * 80)

t0 = time.time()
result_rows = []
for row in stub_records.iter_rows(named=True):
    fields = normalize_name(row["name_raw"], row["country"])
    addr_fields = normalize_address(row["addr_raw"], row["country"])
    result_rows.append({
        "entity_id": row["entity_id"],
        "source": row["source"],
        "country": row["country"],
        "name_raw": row["name_raw"],
        "addr_raw": row["addr_raw"],
        **fields,
        **addr_fields,
    })

records = pl.DataFrame(result_rows, schema={
    "entity_id": pl.String, "source": pl.Int8, "country": pl.String,
    "name_raw": pl.String, "addr_raw": pl.String, "name_script": pl.String,
    "name_norm": pl.String, "name_core": pl.String, "legal_form": pl.String,
    "name_tokens": pl.List(pl.String), "name_skeleton": pl.String,
    "addr_norm": pl.String, "addr_street": pl.String, "house_num": pl.String,
    "addr_nums": pl.List(pl.String),
})

area_tokens = compute_area_tokens(records, threshold_frac=0.15)
for country, tokens in area_tokens.items():
    print(f"  {country}: {len(tokens)} area tokens: {sorted(tokens)[:15]}")

addr_street_cleaned = []
for row in records.select("addr_street", "country").iter_rows():
    area_set = area_tokens.get(row[1], set())
    addr_street_cleaned.append(remove_area_tokens(row[0], area_set))
records = records.with_columns(pl.Series("addr_street", addr_street_cleaned))

norm_time = time.time() - t0
print(f"\nNormalization: {records.height} records in {norm_time:.2f}s")
_show_examples(records, area_tokens)

records.write_parquet(cfg.artifact("records", "test"))

# Blocking
print("\n" + "=" * 80)
print("TEST: Blocking on test stub (including France)")
print("=" * 80)

from ber.blocking.merge import build_candidates
t0 = time.time()
build_candidates(cfg, "test")
block_time = time.time() - t0

candidates = pl.read_parquet(cfg.artifact("candidates", "test"))
gt = pl.read_parquet(cfg.artifact("gt", "train"))  # no GT for test, use train GT

print(f"\nBlocking test: {candidates.height} candidate pairs in {block_time:.2f}s")
per_s1 = candidates.group_by("s1_id").len()["len"]
print(f"Candidates per S1: mean={per_s1.mean():.1f}, median={per_s1.median():.0f}, max={per_s1.max()}")
print(f"Countries in candidates: {sorted(candidates['country'].unique().to_list())}")
print(f"France candidates: {candidates.filter(pl.col('country') == 'France').height}")
