"""Early real-data smoke test of normalize + block on a small closed world built from the train TSVs.

Usage (from the repo root):  python code/business_entity_resolution/scripts/smoke_real_blocking.py [--n-s1 20000] [--rebuild]

World: --n-s1 S1s sampled per country (seed 42) + all their GT matches + unmatched S2/S3 records (matched to no S1 in
the full GT) sampled per country so that (S2+S3)/S1 equals the full train ratio of that country. Written as
source{1,2,3}_train + gt_train into cache/smoke/ (git-ignored); every stage runs with --set paths.cache_dir=cache/smoke.
Then: normalize once; block + blocking-report for each blocking.k_per_query in KS; one uncapped block run (to tell
"retrieved but capped out" from "no pass retrieved it"). Prints per-country / per-script recall, all-matches-found
share, candidates per S1, runtime + peak RAM, and 15 missed GT pairs. Full text -> reports/smoke_real_blocking.md.

Caveat: randomly sampled unmatched records are easier than the real decoys (near-copies of specific S1s), so recall
here is meaningful but candidate counts are optimistic.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import polars as pl

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))
from ber.blocking.merge import PASS_NAMES  # noqa: E402
from ber.config import load_config  # noqa: E402
from ber.eval.metric import gt_long, s1_groups  # noqa: E402

KS = [1, 2, 3, 5]
UNCAPPED = ["--set", "blocking.k_per_query=100000", "--set", "max_cands=100000"]
CAVEAT = ("Caveat: randomly sampled unmatched records are easier than the real decoys (near-copies of specific S1s), "
          "so recall here is meaningful but candidate counts are optimistic.")
TRAIN_ROWS = {"source1": 2_206_821, "source2": 5_034_616, "source3": 5_285_603, "ground_truth": 2_206_821}


def read_tsv(path: Path) -> pl.DataFrame:
    """Read an official TSV like the mandated pandas call: all columns Utf8, empty strings kept, CSV quoting."""
    return pl.read_csv(path, separator="\t", infer_schema=False, empty_string_is_null=False, quote_char='"')


def build_world(cfg, n_s1: int, seed: int) -> None:
    """Sample the closed world from the train TSVs and write source{1,2,3}_train + gt_train to the smoke cache."""
    raw = {k: read_tsv(cfg.path(f"train.{k}")) for k in TRAIN_ROWS}
    for k, n in TRAIN_ROWS.items():
        assert raw[k].height == n, f"{k}: {raw[k].height} rows, expected {n}"
    s1 = raw["source1"].with_columns(source=pl.lit(1, pl.Int8))
    s23 = pl.concat([raw[f"source{s}"].with_columns(source=pl.lit(s, pl.Int8)) for s in (2, 3)])
    assert not s1["country"].str.contains("\r").any(), "CRLF leaked into country"
    gt = gt_long(raw["ground_truth"])
    ratio = (s23.group_by("country").agg(n23=pl.len())
             .join(s1.group_by("country").agg(n1=pl.len()), on="country")
             .with_columns(ratio=pl.col("n23") / pl.col("n1")))
    picks = pl.concat([s1.filter(pl.col("country") == c).sample(n_s1, seed=seed) for c in sorted(ratio["country"])])
    gt_w = gt.join(picks.select(s1_id="entity_id"), on="s1_id", how="semi")
    matches = s23.join(gt_w.select(entity_id="match_id"), on="entity_id", how="semi")
    unmatched = s23.join(gt.select(entity_id="match_id"), on="entity_id", how="anti")
    fill = []
    for c, r in ratio.select("country", "ratio").iter_rows():  # countries, not rows
        need = round(r * picks.filter(pl.col("country") == c).height) - matches.filter(pl.col("country") == c).height
        fill.append(unmatched.filter(pl.col("country") == c).sample(max(need, 0), seed=seed))
    pool = pl.concat([matches, *fill])
    out = cfg.cache_dir
    out.mkdir(parents=True, exist_ok=True)
    picks.write_parquet(cfg.artifact("source1", "train"))
    for s in (2, 3):
        pool.filter(pl.col("source") == s).write_parquet(cfg.artifact(f"source{s}", "train"))
    gt_w.write_parquet(cfg.artifact("gt", "train"))
    print(f"world: {picks.height:,} S1, {matches.height:,} matched + {pool.height - matches.height:,} unmatched S2/S3 "
          f"(ratio {pool.height / picks.height:.2f}/S1; full train {ratio['n23'].sum() / ratio['n1'].sum():.2f}), "
          f"{gt_w.height:,} GT pairs -> {out}")


def stage(name: str, *extra: str) -> tuple[str, float, float]:
    """Run one pipeline stage on the smoke cache in a fresh process; return (stdout, seconds, peak MB)."""
    res = subprocess.run([sys.executable, "-m", "ber.pipeline", name, "--split", "train",
                          "--set", "paths.cache_dir=cache/smoke", *extra], cwd=PKG / "src", capture_output=True,
                         text=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if res.returncode:
        raise SystemExit(f"{name} failed:\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}")
    m = re.search(rf"\[{name}\] done in ([\d.]+)s, peak memory (\d+) MB", res.stdout)
    return res.stdout, float(m.group(1)), float(m.group(2))


def metrics(cand: pl.DataFrame, gt: pl.DataFrame, groups: pl.DataFrame) -> pl.DataFrame:
    """Per country, per name script (of the S1's matches) and ALL: pair recall, share of matched S1s with every
    match found, mean / p95 candidates per S1."""
    found = gt.join(cand.select("s1_id", match_id="cand_id"), on=["s1_id", "match_id"], how="semi")
    per_s1 = (groups.join(gt.group_by("s1_id").agg(n_gt=pl.len()), on="s1_id", how="left")
              .join(found.group_by("s1_id").agg(n_found=pl.len()), on="s1_id", how="left")
              .join(cand.group_by("s1_id").agg(n_cand=pl.len()), on="s1_id", how="left")
              .with_columns(pl.col("n_gt", "n_found", "n_cand").fill_null(0)))
    agg = dict(n_s1=pl.len(), pair_recall=pl.col("n_found").sum() / pl.col("n_gt").sum(),
               all_found=(pl.col("n_found") == pl.col("n_gt")).filter(pl.col("n_gt") > 0).mean(),
               cand_mean=pl.col("n_cand").mean(), cand_p95=pl.col("n_cand").quantile(0.95))
    return pl.concat([
        per_s1.group_by(group="country").agg(**agg).sort("group"),
        per_s1.group_by(group=pl.lit("script=") + pl.col("script")).agg(**agg).sort("group"),
        per_s1.select(group=pl.lit("ALL"), **agg)])


def missed_pairs(cand: pl.DataFrame, uncapped: pl.DataFrame, gt: pl.DataFrame, rec: pl.DataFrame,
                 n: int, seed: int) -> str:
    """``n`` random GT pairs missing from ``cand``: raw strings of both sides + what the passes did for the candidate."""
    miss = gt.join(cand.select("s1_id", match_id="cand_id"), on=["s1_id", "match_id"], how="anti")
    miss = miss.sample(min(n, miss.height), seed=seed)
    raw = rec.select("entity_id", "country", "name_raw", "addr_raw", "name_script", "house_num")
    passes = lambda m: "+".join(v for b, v in PASS_NAMES.items() if m >> b & 1) or "-"  # noqa: E731
    lines = []
    for s1_id, cid in miss.iter_rows():  # 15 pairs, not rows
        a, b = (raw.filter(pl.col("entity_id") == x).row(0, named=True) for x in (s1_id, cid))
        hit = uncapped.filter((pl.col("s1_id") == s1_id) & (pl.col("cand_id") == cid))
        others = uncapped.filter(pl.col("cand_id") == cid)
        why = (f"retrieved by {passes(hit['block_mask'][0])}, rank_in_cand {hit['rank_in_cand'][0]} -> capped out"
               if hit.height else
               f"not retrieved; this record got {others.height} other S1s via "
               f"{passes(int(others.select(pl.col('block_mask').bitwise_or()).item())) if others.height else 'no pass'}")
        lines.append(f"- [{a['country']}] S1 `{a['name_raw']}` | `{a['addr_raw']}` (hn {a['house_num'] or '-'})\n"
                     f"  cand ({b['name_script']}) `{b['name_raw']}` | `{b['addr_raw']}` (hn {b['house_num'] or '-'})\n"
                     f"  -> {why}")
    return "\n".join(lines)


def main() -> None:
    """Build the smoke world (once), run the sweep, print and save the report."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-s1", type=int, default=20_000, help="S1s sampled per country (default 20000)")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the world even if the smoke sources exist")
    args = ap.parse_args()
    cfg = load_config(sets=["paths.cache_dir=cache/smoke"])
    if args.rebuild or not cfg.artifact("source1", "train").exists():
        build_world(cfg, args.n_s1, cfg.seed)
    out = [f"# Smoke: real-data blocking, {args.n_s1:,} S1/country\n", CAVEAT, ""]
    _, t, mb = stage("normalize")
    runs = [("normalize", "-", t, mb)]
    rec, gt = pl.read_parquet(cfg.artifact("records", "train")), pl.read_parquet(cfg.artifact("gt", "train"))
    groups = s1_groups(rec, gt)
    _, t, mb = stage("block", *UNCAPPED)
    runs.append(("block", "uncapped", t, mb))
    uncapped = pl.read_parquet(cfg.artifact("candidates", "train"))
    tables, cands = [("uncapped", metrics(uncapped, gt, groups))], {}
    for k in KS:
        _, t, mb = stage("block", "--set", f"blocking.k_per_query={k}")
        runs.append(("block", f"k={k}", t, mb))
        cands[k] = pl.read_parquet(cfg.artifact("candidates", "train"))
        tables.append((f"k={k}", metrics(cands[k], gt, groups)))
        report, _, _ = stage("blocking-report")
        out += [f"## blocking-report, k_per_query={k}", report.split("\n", 1)[1].rsplit("\n[blocking-report]", 1)[0]]
    sweep = pl.concat([tb.select(pl.lit(k).alias("k"), pl.all()) for k, tb in tables])
    with pl.Config(tbl_rows=100, tbl_cols=-1, tbl_width_chars=200, float_precision=4, tbl_formatting="MARKDOWN",
                   tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
        text = [f"## Sweep (max_cands={cfg.max_cands})", str(sweep), "", "## Runtime / peak RAM",
                str(pl.DataFrame(runs, schema=["stage", "run", "seconds", "peak_MB"], orient="row")), "",
                "## 15 missed GT pairs at k=2 (default)", missed_pairs(cands[2], uncapped, gt, rec, 15, cfg.seed)]
    print("\n".join([CAVEAT, "", *text]))
    path = cfg.path("reports_dir") / "smoke_real_blocking.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out[:3] + text + [""] + out[3:]), encoding="utf-8")
    print(f"\nfull report -> {path}")


if __name__ == "__main__":
    main()
