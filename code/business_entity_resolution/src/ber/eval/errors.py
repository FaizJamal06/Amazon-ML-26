"""Dumps the worst false positives / false negatives with raw strings for manual review.

Contract:
In: ``matches`` (decide output), the scores artifact decide used, ``candidates``, ``records``, ``gt`` (train).
Out: markdown report ``reports/errors_{split}[_sw]_{scores_name}.md`` with
- a summary: TP / FP / FN (blocking vs model misses) per country, per candidate name_script, per S1 match-count bucket;
- the worst false positives (highest p first);
- the worst false negatives, split into blocking misses (GT pair not in candidates) and model misses (lowest p first).
Each row shows S1 vs candidate raw name/address, country, name_script, p, house-number relation, the GT owner of the
candidate (FPs) and the S1 it was assigned to instead (FNs).

Owner: Faiz (R1 Lead / Eval / Decision)
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from ber.eval.metric import macro_f05
from ber.fallback import house_num_relation

BUCKETS = ["0", "1", "2", "3", "4+"]


def _pairs(df: pl.DataFrame) -> pl.DataFrame:
    """``(s1_id, cand_id)`` pairs (accepts ``match_id``)."""
    return (df.rename({"match_id": "cand_id"}) if "match_id" in df.columns else df).select("s1_id", "cand_id")


def classify(matches: pl.DataFrame, scores: pl.DataFrame, candidates: pl.DataFrame, gt: pl.DataFrame) -> pl.DataFrame:
    """Every predicted or GT pair with ``kind`` in {TP, FP, FN_blocking, FN_model}, its p, the GT owner of the
    candidate and the S1 the candidate was assigned to by decide (null if none)."""
    pred, truth = _pairs(matches).unique(), _pairs(gt).unique()
    allp = pl.concat([pred, truth]).unique()
    owner = truth.select("cand_id", gt_owner="s1_id").unique("cand_id")
    assigned = pred.select("cand_id", assigned_to="s1_id").unique("cand_id")
    return (
        allp.join(pred.with_columns(_pred=pl.lit(True)), on=["s1_id", "cand_id"], how="left")
        .join(truth.with_columns(_gt=pl.lit(True)), on=["s1_id", "cand_id"], how="left")
        .join(_pairs(candidates).with_columns(_cand=pl.lit(True)), on=["s1_id", "cand_id"], how="left")
        .join(scores.select("s1_id", "cand_id", "p"), on=["s1_id", "cand_id"], how="left")
        .join(owner, on="cand_id", how="left").join(assigned, on="cand_id", how="left")
        .with_columns(pl.col("_pred", "_gt", "_cand").fill_null(False))
        .with_columns(kind=pl.when(pl.col("_pred") & pl.col("_gt")).then(pl.lit("TP"))
                      .when(pl.col("_pred")).then(pl.lit("FP"))
                      .when(~pl.col("_cand")).then(pl.lit("FN_blocking")).otherwise(pl.lit("FN_model")))
        .drop("_pred", "_gt", "_cand")
    )


def enrich(pairs: pl.DataFrame, records: pl.DataFrame, gt: pl.DataFrame) -> pl.DataFrame:
    """Add raw S1/candidate strings, country, candidate name_script, house-number relation and S1 match bucket."""
    rec = records.select("entity_id", "name_raw", "addr_raw", "country", "name_script", "house_num")
    s1 = rec.select(s1_id="entity_id", s1_name="name_raw", s1_addr="addr_raw", country="country", _s1_hn="house_num")
    c = rec.select(cand_id="entity_id", cand_name="name_raw", cand_addr="addr_raw", name_script="name_script",
                   _c_hn="house_num")
    n_gt = _pairs(gt).unique().group_by("s1_id").agg(_n=pl.len())
    rel, diff = house_num_relation("_s1_hn", "_c_hn")
    return (pairs.join(s1, on="s1_id", how="left").join(c, on="cand_id", how="left")
            .join(n_gt, on="s1_id", how="left")
            .with_columns(hn_rel=rel, hn_diff=diff,
                          s1_bucket=pl.col("_n").fill_null(0).clip(upper_bound=4).cast(pl.String).replace("4", "4+"))
            .drop("_s1_hn", "_c_hn", "_n"))


def summary(enriched: pl.DataFrame, by: str) -> pl.DataFrame:
    """Counts of TP / FP / FN_blocking / FN_model grouped by ``by`` (e.g. country, name_script, s1_bucket)."""
    kinds = ["TP", "FP", "FN_blocking", "FN_model"]
    return (enriched.group_by(by).agg([(pl.col("kind") == k).sum().alias(k) for k in kinds])
            .with_columns(FN=pl.col("FN_blocking") + pl.col("FN_model")).sort(by))


def worst(enriched: pl.DataFrame, n: int = 50, seed: int = 42) -> dict[str, pl.DataFrame]:
    """Worst FPs (highest p), and n FNs split proportionally into blocking misses (random, seeded) and model misses
    (lowest p first)."""
    cols = ["s1_id", "cand_id", "country", "name_script", "p", "hn_rel", "hn_diff", "s1_name", "cand_name",
            "s1_addr", "cand_addr"]
    fp = enriched.filter(pl.col("kind") == "FP").sort("p", descending=True, nulls_last=True).head(n)
    fn_b = enriched.filter(pl.col("kind") == "FN_blocking")
    fn_m = enriched.filter(pl.col("kind") == "FN_model")
    n_b = round(n * fn_b.height / max(fn_b.height + fn_m.height, 1))
    return {
        "Worst false positives (highest p first)": fp.select(*cols, "gt_owner"),
        "False negatives — blocking misses (GT pair not in candidates)":
            fn_b.sample(min(n_b, fn_b.height), seed=seed).select(*cols, "assigned_to"),
        "False negatives — model misses (in candidates, not predicted; lowest p first)":
            fn_m.sort("p", nulls_last=True).head(n - min(n_b, fn_b.height)).select(*cols, "assigned_to"),
    }


def _md(df: pl.DataFrame) -> str:
    """Markdown table (floats to 3 decimals, pipes escaped, nothing truncated). Report tables are small (<= ~150 rows)."""
    cells = df.with_columns(pl.col(pl.Float32, pl.Float64).round(3)).cast(pl.String).fill_null("") \
        .with_columns(pl.all().str.replace_all("|", r"\|", literal=True))
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * df.width]
    lines += ["| " + " | ".join(row) + " |" for row in cells.iter_rows()]
    return "\n".join(lines)


def error_report(matches: pl.DataFrame, scores: pl.DataFrame, candidates: pl.DataFrame, records: pl.DataFrame,
                 gt: pl.DataFrame, title: str, n: int = 50, seed: int = 42) -> str:
    """Build the full markdown error report."""
    e = enrich(classify(matches, scores, candidates, gt), records, gt)
    s1 = records.filter(pl.col("source") == 1).select(s1_id="entity_id", country="country")
    counts = dict(e.group_by("kind").len().iter_rows())
    parts = [f"# {title}\n",
             f"macro F0.5 = {macro_f05(_pairs(matches), _pairs(gt), s1):.4f} over {s1.height:,} S1s; "
             + ", ".join(f"{k} = {counts.get(k, 0):,}" for k in ["TP", "FP", "FN_blocking", "FN_model"]) + "\n"]
    for by, label in [("country", "country"), ("name_script", "candidate name_script"),
                      ("s1_bucket", "S1 match-count bucket (GT matches of the S1)")]:
        parts.append(f"## Summary by {label}\n\n{_md(summary(e, by))}\n")
    for head, df in worst(e, n, seed).items():
        parts.append(f"## {head} — {df.height} rows\n\n{_md(df)}\n")
    return "\n".join(parts)


def write_report(text: str, path: Path) -> Path:
    """Write the report (UTF-8) and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
