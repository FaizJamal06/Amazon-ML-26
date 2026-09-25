"""Blocking quality: recall ceiling, reduction ratio, candidates per S1; by country, script and pass.

Contract:
In: ``candidates_{split}.parquet`` (``s1_id, cand_id, country, block_mask``), ``gt_train.parquet`` (long
``s1_id, match_id``) and ``records_{split}.parquet`` (``entity_id, source, country, name_script``).
Out: a dict of polars frames (``summary`` per country + overall, ``by_script`` of the matched record, ``by_pass`` per
``block_mask`` bit) and ``to_markdown()`` for EXPERIMENTS.md / PR descriptions.

Owner: Faiz (R1 Lead / Eval / Decision)
"""
from __future__ import annotations

import polars as pl

MAX_BITS = 16


def _found(candidates: pl.DataFrame, gt: pl.DataFrame) -> pl.DataFrame:
    """GT pairs with a ``found`` flag and the ``block_mask`` of the candidate pair (0 when not found)."""
    c = candidates.select("s1_id", pl.col("cand_id").alias("match_id"), "block_mask")
    return (gt.select("s1_id", "match_id").unique().join(c, on=["s1_id", "match_id"], how="left")
            .with_columns(found=pl.col("block_mask").is_not_null(), block_mask=pl.col("block_mask").fill_null(0)))


def summary(candidates: pl.DataFrame, gt: pl.DataFrame, records: pl.DataFrame) -> pl.DataFrame:
    """Per country and overall (``ALL``): pair recall, full-GT-set coverage, candidates/S1, reduction ratio.

    Candidates/S1 counts S1s with no candidates as 0. Reduction ratio = 1 - pairs / sum_c(|S1_c| x |S2+S3_c|).
    Full coverage is over S1s that have at least one GT match.
    """
    s1 = records.filter(pl.col("source") == 1).select(s1_id="entity_id", country="country")
    per_s1 = (s1.join(candidates.group_by("s1_id").agg(n_cand=pl.len()), on="s1_id", how="left")
              .with_columns(pl.col("n_cand").fill_null(0)))
    found = _found(candidates, gt).join(s1, on="s1_id", how="left")
    cover = found.group_by("s1_id", "country").agg(all_found=pl.col("found").all())
    pool = records.group_by("country").agg(space=(pl.col("source") == 1).sum() * (pl.col("source") > 1).sum())

    def grouped(per_s1, found, cover, pool):
        """The summary columns grouped by ``country``."""
        return (per_s1.group_by("country").agg(n_s1=pl.len(), pairs=pl.col("n_cand").sum(),
                                                cand_mean=pl.col("n_cand").mean(),
                                                cand_p50=pl.col("n_cand").quantile(0.5),
                                                cand_p95=pl.col("n_cand").quantile(0.95))
                .join(found.group_by("country").agg(gt_pairs=pl.len(), pair_recall=pl.col("found").mean()),
                      on="country", how="left")
                .join(cover.group_by("country").agg(full_gt_covered=pl.col("all_found").mean()), on="country", how="left")
                .join(pool.group_by("country").agg(pl.col("space").sum()), on="country", how="left")
                .with_columns(reduction_ratio=1 - pl.col("pairs") / pl.col("space")).drop("space"))

    all_ = lambda df: df.with_columns(country=pl.lit("ALL"))  # noqa: E731
    return pl.concat([grouped(per_s1, found, cover, pool).sort("country"),
                      grouped(all_(per_s1), all_(found), all_(cover), all_(pool))])


def by_script(candidates: pl.DataFrame, gt: pl.DataFrame, records: pl.DataFrame) -> pl.DataFrame:
    """Pair recall by ``name_script`` of the matched S2/S3 record (e.g. Latin vs Devanagari), per country."""
    rec = records.select(pl.col("entity_id").alias("match_id"), "country", "name_script")
    return (_found(candidates, gt).join(rec, on="match_id", how="left")
            .group_by("country", "name_script").agg(gt_pairs=pl.len(), pair_recall=pl.col("found").mean())
            .sort("country", "name_script"))


def by_pass(candidates: pl.DataFrame, gt: pl.DataFrame, pass_names: dict[int, str] | None = None) -> pl.DataFrame:
    """Per ``block_mask`` bit: pairs flagged, GT recall of that pass, and unique recall (found only by that pass)."""
    f = _found(candidates, gt).filter(pl.col("found"))
    n_gt = gt.select("s1_id", "match_id").unique().height
    present = candidates.select(pl.col("block_mask").bitwise_or()).item() or 0
    rows = []
    for b in range(MAX_BITS):
        bit = 1 << b
        if not present & bit:
            continue
        has = (pl.col("block_mask") & bit) != 0
        rows.append({
            "bit": b, "pass": (pass_names or {}).get(b, f"bit{b}"),
            "pairs": candidates.filter(has).height,
            "recall": f.filter(has).height / max(n_gt, 1),
            "unique_recall": f.filter(pl.col("block_mask") == bit).height / max(n_gt, 1),
        })
    return pl.DataFrame(rows, schema={"bit": pl.Int32, "pass": pl.String, "pairs": pl.Int64,
                                      "recall": pl.Float64, "unique_recall": pl.Float64})


def blocking_report(candidates: pl.DataFrame, gt: pl.DataFrame, records: pl.DataFrame,
                    pass_names: dict[int, str] | None = None) -> dict[str, pl.DataFrame]:
    """All three views: ``summary``, ``by_script``, ``by_pass``."""
    return {"summary": summary(candidates, gt, records), "by_script": by_script(candidates, gt, records),
            "by_pass": by_pass(candidates, gt, pass_names)}


def to_markdown(report: dict[str, pl.DataFrame]) -> str:
    """Render the report as markdown tables (4 decimals)."""
    parts = []
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_hide_column_data_types=True, tbl_hide_dataframe_shape=True,
                   float_precision=4, tbl_rows=100, tbl_cols=30, tbl_width_chars=400):
        for name, df in report.items():
            parts.append(f"### {name}\n\n{df}\n")
    return "\n".join(parts)
