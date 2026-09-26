"""Full-density blocking eval: the FULL train S1 index per country, queried by a sample of S2/S3 records.

Usage (from the repo root):
    python code/business_entity_resolution/scripts/eval_fulldensity_blocking.py [--n-queries 50000]
        [--set blocking.fuzzy=true ...] [--label name]

Why: sampled worlds (smoke 20k/200k) shrink the S1 index, so tokens look rarer and passes look better than on the
real data (20k world 0.947 vs full train 0.718). Here every query competes against ALL S1s of its country, with IDF /
document frequencies over ALL records of the country, exactly like `block` (same build_country_index /
query_passes / rerank_chunk code), so the numbers transfer to the full run.

Sample: --n-queries S2/S3 records of cache/records_train.parquet (seed 42); records with a GT owner and unmatched
ones in their population proportion. For queries with an owner: "retrieved at all" (the owner is in the union of the
passes) and recall@k (owner within rank_in_cand <= k) for k = 1, 2, 3, 5, per country / name script of the query /
pass. Unmatched queries report how many S1s they keep at k=2 (decoy load).
Approximations vs `block`: pass scores are normalized by maxima measured on the sample (not the whole country), and
the per-S1 max_cands safety cap is not applied. Report -> reports/fulldensity_blocking[_label].md.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import polars as pl

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))
from ber.blocking.merge import (PASS_FUZZY, PASS_NAME_STREET, PASS_NAMES, RANK_FIELDS, RECORD_COLS,  # noqa: E402
                                _rank_per_cand, build_country_index, pass_limits, query_passes, rerank_chunk)
from ber.config import load_config  # noqa: E402
from ber.pipeline import peak_memory_mb  # noqa: E402

KS = [1, 2, 3, 5]


def sample_queries(records: Path, gt: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    """``n`` S2/S3 records (entity_id, country, owner) with owners and unmatched ones in population proportion."""
    q = (pl.scan_parquet(records).filter(pl.col("source") > 1).select("entity_id", "country").collect()
         .join(gt.select(entity_id="match_id", owner="s1_id"), on="entity_id", how="left"))
    matched, unmatched = q.filter(pl.col("owner").is_not_null()), q.filter(pl.col("owner").is_null())
    n_m = round(n * matched.height / q.height)
    return pl.concat([matched.sample(n_m, seed=seed), unmatched.sample(n - n_m, seed=seed)])


def profile_country(ix, part: pl.DataFrame, rank_recs: pl.DataFrame, pass_max: dict, n: int, seed: int) -> dict:
    """ms/record of each key pass (A-D), of the re-rank (merge + rank_score) and of pass E, on ``n`` sampled
    query records (pass E: only the records whose best rank_score is weak run it; the cost is spread over all n)."""
    p = part.sample(min(n, part.height), seed=seed)
    ms = lambda t: round(1000 * t / max(p.height, 1), 3)  # noqa: E731
    out, raws = {"country": ix.country, "records": p.height}, []
    for bit, name in PASS_NAMES.items():  # passes, not rows
        if bit == PASS_FUZZY or (bit == PASS_NAME_STREET and ix.combo is None):
            continue
        t = time.time()
        raws.append(query_passes(ix, p, only={bit}))
        out[f"{name}_ms"] = ms(time.time() - t)
    raw = pl.concat(raws)
    t = time.time()
    rerank_chunk(dataclasses.replace(ix, fuzzy=None), raw, p, pass_max, rank_recs)
    out["rerank_ms"] = ms(time.time() - t)
    if ix.fuzzy is not None:
        t = time.time()
        _, fz = rerank_chunk(ix, raw, p, pass_max, rank_recs)
        out["fuzzy_ms"] = round(ms(time.time() - t) - out["rerank_ms"], 3)
        out["fuzzy_hits"] = fz.height
    out["total_ms"] = round(sum(v for k, v in out.items() if k.endswith("_ms")), 3)
    return out


def eval_country(records: Path, country: str, sample: pl.DataFrame, limits: dict,
                 profile_n: int = 0, seed: int = 42) -> tuple[pl.DataFrame, dict, dict | None]:
    """Build the full-country index, run the sampled queries; per-query outcome rows + timings (+ profile)."""
    t0 = time.time()
    cols = RECORD_COLS + (["name_raw"] if limits["fuzzy"] else [])
    recs = pl.scan_parquet(records).filter(pl.col("country") == country).select(cols).collect()
    t_load = time.time()
    ix = build_country_index(recs, country, limits)
    t_index = time.time()
    part = recs.join(sample.select("entity_id"), on="entity_id", how="semi")
    raw = query_passes(ix, part)
    pass_max = dict(raw.group_by("bit").agg(pl.col("score").max()).iter_rows())
    rank_recs = recs.select("entity_id", *RANK_FIELDS)
    t_prof = time.time()
    prof = profile_country(ix, part, rank_recs, pass_max, profile_n, seed) if profile_n else None
    t_prof = time.time() - t_prof
    merged, fz = rerank_chunk(ix, raw, part, pass_max, rank_recs)
    ranked = _rank_per_cand(merged, k_per_query=10**9)
    t_query = time.time() - t_prof   # profiling time excluded
    hits = pl.concat([raw.select("cand_id", "s1_id", "bit"), fz.select("cand_id", "s1_id", "bit")])
    owner_hits = (sample.filter(pl.col("country") == country)
                  .join(hits, left_on=["entity_id", "owner"], right_on=["cand_id", "s1_id"], how="left")
                  .group_by("entity_id").agg(pass_bits=pl.col("bit").drop_nulls().unique()))
    out = (sample.filter(pl.col("country") == country)
           .join(part.select("entity_id", "name_script"), on="entity_id", how="left")
           .join(ranked.select(entity_id="cand_id", owner="s1_id", rank="rank_in_cand"), on=["entity_id", "owner"],
                 how="left")
           .join(ranked.filter(pl.col("rank_in_cand") <= 2).group_by("cand_id").agg(kept_k2=pl.len())
                 .rename({"cand_id": "entity_id"}), on="entity_id", how="left")
           .join(owner_hits, on="entity_id", how="left")
           .with_columns(script=pl.when(pl.col("name_script").fill_null("Latin") == "Latin").then(pl.lit("latin"))
                         .otherwise(pl.lit("native")), kept_k2=pl.col("kept_k2").fill_null(0)))
    times = {"country": country, "s1": recs.filter(pl.col("source") == 1).height, "records": recs.height,
             "queries": part.height, "load_s": t_load - t0, "index_s": t_index - t_load, "query_s": t_query - t_index,
             "fuzzy_hits": fz.height}
    return out, times, prof


def summarize(rows: pl.DataFrame) -> pl.DataFrame:
    """Retrieved-at-all and recall@k (queries with an owner) + decoy load (unmatched) per country / script / ALL."""
    m = rows.filter(pl.col("owner").is_not_null())
    agg = dict(n_q=pl.len(), retrieved=pl.col("rank").is_not_null().mean(),
               **{f"recall@{k}": (pl.col("rank") <= k).fill_null(False).mean() for k in KS})
    decoy = rows.filter(pl.col("owner").is_null())
    parts = [m.group_by(group="country").agg(**agg).sort("group"),
             m.group_by(group=pl.lit("script=") + pl.col("script")).agg(**agg).sort("group"),
             m.select(group=pl.lit("ALL"), **agg)]
    table = pl.concat(parts)
    load = pl.concat([decoy.group_by(group="country").agg(unmatched_kept_k2=pl.col("kept_k2").mean()),
                      decoy.select(group=pl.lit("ALL"), unmatched_kept_k2=pl.col("kept_k2").mean())])
    return table.join(load, on="group", how="left")


def by_pass(rows: pl.DataFrame) -> pl.DataFrame:
    """Per pass: share of owner-queries whose owner that pass retrieved, and share retrieved by that pass only."""
    m = rows.filter(pl.col("owner").is_not_null()).with_columns(pl.col("pass_bits").fill_null([]))
    out = []
    for bit, name in PASS_NAMES.items():  # passes, not rows
        has = pl.col("pass_bits").list.contains(bit)
        out.append(m.select(pass_=pl.lit(name), retrieved=has.mean(),
                            only_this=(has & (pl.col("pass_bits").list.len() == 1)).mean()))
    return pl.concat(out)


def main() -> None:
    """Parse args, run every country, print + save the report."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-queries", type=int, default=50_000)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override (repeatable)")
    ap.add_argument("--label", default="", help="suffix of the report file")
    ap.add_argument("--profile", type=int, default=0, metavar="N",
                    help="also time each pass / the re-rank / pass E on N query records per country")
    args = ap.parse_args()
    cfg = load_config(sets=args.set)
    limits, records = pass_limits(cfg), cfg.artifact("records", "train")
    t0 = time.time()
    sample = sample_queries(records, pl.read_parquet(cfg.artifact("gt", "train")), args.n_queries, cfg.seed)
    rows, times, profs = [], [], []
    for country in sorted(sample["country"].unique()):  # countries, not rows
        r, t, prof = eval_country(records, country, sample, limits, args.profile, cfg.seed)
        rows.append(r)
        times.append(t)
        profs += [prof] if prof else []
        print(f"  {country}: {t}", flush=True)
        if prof:
            print(f"  profile: {prof}", flush=True)
    rows = pl.concat(rows)
    total = time.time() - t0
    with pl.Config(tbl_formatting="MARKDOWN", tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True,
                   float_precision=4, tbl_rows=50, tbl_cols=-1, tbl_width_chars=200):
        text = [f"# Full-density blocking eval {args.label}".rstrip(), "",
                f"config {cfg.hash}, overrides {args.set or 'none'}, limits {limits}", "",
                f"{args.n_queries:,} sampled S2/S3 queries (seed {cfg.seed}) against the full train S1 index per country.",
                "", "## Recall (queries with a GT owner) + decoy load (unmatched queries: S1s kept at k=2)",
                str(summarize(rows)), "", "## Per pass (owner retrieved by the pass / by that pass only)",
                str(by_pass(rows)), "", "## Runtime",
                str(pl.DataFrame(times).with_columns(pl.col("^.*_s$").round(1))), "",
                *(["## Profile (ms per query record)", str(pl.DataFrame(profs))] if profs else []),
                f"total {total:.0f} s, peak memory {peak_memory_mb():.0f} MB"]
    print("\n".join(text))
    path = cfg.path("reports_dir") / f"fulldensity_blocking{'_' + args.label if args.label else ''}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text), encoding="utf-8")
    print(f"report -> {path}")


if __name__ == "__main__":
    main()
