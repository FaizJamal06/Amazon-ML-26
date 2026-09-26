"""US→India cross-country transfer check (CLAUDE.md §5.5a / task brief §4).

Trains LightGBM on one country's S1s and evaluates OOF-style on the other.
Reports the transfer gap and runs a drop-one ablation over feature GROUPS.

Usage (from code/business_entity_resolution/src):
    python ../scripts/transfer_check.py --split train [--set paths.cache_dir=cache/smoke] [OPTIONS]

The script reads the same features_{split}.parquet and folds_train.parquet that the main pipeline uses.
It never changes any model config defaults — results are only written to notes/TRANSFER_CHECK.md.

Owner: Nitish (R3 Features / LightGBM)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

# Allow running from src/ or from repo root
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent / "src"))

from ber.config import load_config  # noqa: E402
from ber.model.lgbm import KEYS, _params, _x, cross_fit_isotonic, fit  # noqa: E402

# ── Feature groups (for the drop-one ablation) ──────────────────────────────────────────────────
# Any feature not matching a prefix below falls into the 'other' group.
FEATURE_GROUPS: dict[str, list[str]] = {
    "pairwise": ["name_ratio", "name_tset", "name_tsort", "name_partial", "name_jw",
                 "street_ratio", "street_tset", "street_tsort", "street_partial", "street_jw",
                 "skel_ratio", "name_len_ratio", "street_len_ratio", "s1_script", "cand_script",
                 "same_script", "is_s3"],
    "structured": ["hn_rel", "hn_abs_diff_log", "hn_rel_diff", "hn_conflict",
                   "name_only_a", "name_only_b", "name_extra_idf_a", "name_extra_idf_b", "name_one_extra",
                   "street_only_a", "street_only_b", "street_extra_idf_a", "street_extra_idf_b", "street_one_extra",
                   "legal_rel", "street_missing"],
    "context": ["block_score", "rank_in_cand", "tfidf_name", "tfidf_full",
                "gap_other_s1", "n_s1_for_cand", "cand_rank_in_s1", "n_cands_s1", "gap_to_s1_best"],
    "rarity": ["shared_idf_min", "shared_idf_mean", "a_name_freq_log", "b_name_freq_log",
               "name_sim_x_rarity", "name_match_hn_conflict"],
    "graph": ["graph_name_max", "graph_name_mean", "graph_addr_max", "graph_addr_mean",
              "graph_same_hnum", "graph_top1_agree", "graph_cross_src_agree", "graph_cluster_size"],
}
# Also capture pass_* bits (context group) and any unrecognised features
_CONTEXT_PATTERNS = ["pass_"]


def _assign_group(feat: str) -> str:
    """Return the feature group name for a feature column (prefix matching)."""
    for group, members in FEATURE_GROUPS.items():
        if feat in members:
            return group
    if any(feat.startswith(p) for p in _CONTEXT_PATTERNS):
        return "context"
    return "other"


def _load_features(path: Path, s1_ids: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Load features parquet filtered to a set of S1 ids; return (df, feature_col_names)."""
    schema = pq.read_schema(path)
    feats = [c for c in schema.names if c not in (*KEYS, "label")]
    df = pl.scan_parquet(path).join(s1_ids.select("s1_id").lazy(), on="s1_id", how="semi").collect()
    return df, feats


def _oof_on_target(train_df: pl.DataFrame, test_df: pl.DataFrame, feats: list[str],
                   params: dict, seed: int) -> tuple[float, np.ndarray]:
    """Train on train_df (all pairs), predict on test_df, calibrate with cross-fold isotonic.

    Returns (macro-avg p, per-pair p array aligned to test_df order).
    This simulates the OOF protocol: train model on source country, evaluate on target country.
    A cross-fold isotonic calibrator is fitted on training OOF (5-fold within the training set)
    and applied to the test predictions — matching the pipeline's calibration path.
    """
    # Build 5 OOF folds within the training set for calibration
    rng = np.random.default_rng(seed)
    s1_arr = train_df["s1_id"].unique().sort().to_numpy()
    fold_idx = rng.integers(0, 5, len(s1_arr))
    s1_fold_map = dict(zip(s1_arr.tolist(), fold_idx.tolist()))
    train_df = train_df.with_columns(
        _fold=pl.col("s1_id").map_elements(lambda x: s1_fold_map.get(x, 0), return_dtype=pl.Int8)
    )
    # Cross-validated isotonic on training data
    raw_tr, y_tr, folds_tr = [], [], []
    boosters = []
    for k in range(5):
        tr_k = train_df.filter(pl.col("_fold") != k)
        va_k = train_df.filter(pl.col("_fold") == k)
        if tr_k.height == 0 or va_k.height == 0:
            continue
        b = fit(tr_k.drop("_fold"), feats, params, seed)
        raw_tr.append(b.predict(_x(va_k, feats)))
        y_tr.append(va_k["label"].to_numpy())
        folds_tr.append(np.full(va_k.height, k, dtype=np.int8))
        boosters.append(b)
    if not raw_tr:
        return float("nan"), np.array([])
    raw_arr = np.concatenate(raw_tr)
    y_arr = np.concatenate(y_tr)
    folds_arr = np.concatenate(folds_tr)
    _ = cross_fit_isotonic(raw_arr, y_arr, folds_arr)   # calibration fit on train
    # Final model on all train data
    rounds = max(1, int(np.mean([b.best_iteration or b.current_iteration() for b in boosters])))
    final = fit(train_df.drop("_fold"), feats, params, seed, rounds)
    # Isotonic calibrator on full training OOF
    from sklearn.isotonic import IsotonicRegression
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_arr, y_arr)
    # Predict on target country
    raw_test = final.predict(_x(test_df, feats))
    p_test = calibrator.predict(raw_test)
    return float(np.mean(p_test)), p_test


def _macro_f05(scores_df: pl.DataFrame, gt: pl.DataFrame, s1_universe: pl.DataFrame,
               threshold: float) -> float:
    """Simple threshold-based macro F0.5 on a scores frame (s1_id, cand_id, p)."""
    from ber.eval.metric import macro_f05
    from ber.decide.assign import assign_one_to_one
    from ber.decide.select import select
    # We need a candidates table to do one-to-one; scores_df must have s1_id, cand_id, p.
    # Skip one-to-one for transfer check (target S1s are disjoint from train S1s, so it is trivial).
    matched = scores_df.filter(pl.col("p") >= threshold).select("s1_id", "cand_id", "p")
    return macro_f05(matched, gt, s1_universe)


def _feature_groups_present(feats: list[str]) -> dict[str, list[str]]:
    """Map group_name -> [feature_cols] for features that are actually in the data."""
    result: dict[str, list[str]] = {}
    for f in feats:
        g = _assign_group(f)
        result.setdefault(g, []).append(f)
    return result


def run_transfer_check(cfg, split: str) -> None:
    """Main entry point: run transfer check and write notes/TRANSFER_CHECK.md."""
    path = cfg.artifact("features", split)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run featurize first")
    folds_path = cfg.artifact("folds", split)
    if not folds_path.exists():
        raise FileNotFoundError(f"{folds_path} missing — run folds stage first")

    folds = pl.read_parquet(folds_path)   # s1_id, fold, country, n_matches
    params = _params(cfg)
    seed = cfg.seed

    # Full feature schema
    schema = pq.read_schema(path)
    feats = [c for c in schema.names if c not in (*KEYS, "label")]

    # Country sets
    us_s1 = folds.filter(pl.col("country") == "US").select("s1_id")
    in_s1 = folds.filter(pl.col("country") == "India").select("s1_id")

    # Load full feature table (chunked read)
    print("transfer_check: loading features ...", flush=True)
    all_df = pl.read_parquet(path)
    us_df = all_df.join(us_s1, on="s1_id", how="semi")
    in_df = all_df.join(in_s1, on="s1_id", how="semi")
    print(f"  US pairs: {us_df.height:,}  India pairs: {in_df.height:,}", flush=True)

    rows = []

    def _run(train: pl.DataFrame, test: pl.DataFrame, label: str, feats_: list[str]) -> float:
        """Train on train, evaluate on test; return macro F0.5 at threshold 0.5."""
        t0 = time.time()
        _, p_test = _oof_on_target(train, test, feats_, params, seed)
        elapsed = time.time() - t0
        if len(p_test) == 0:
            return float("nan")
        # Threshold-based: p >= 0.5 counted as match
        pred = test.select("s1_id", "cand_id").with_columns(p=pl.Series(p_test))
        matched = pred.filter(pl.col("p") >= 0.5)
        gt_test = test.filter(pl.col("label") == 1).select("s1_id", "cand_id")
        # Per-S1 F0.5
        from ber.eval.metric import macro_f05 as _mf05
        # Build a simple s1_universe from the test set
        s1_u = test.select("s1_id").unique().with_columns(country=pl.lit("?"), n_matches=pl.lit(0, pl.Int32))
        f = _mf05(matched, gt_test.rename({"cand_id": "match_id"}), s1_u)
        print(f"  {label}: macro F0.5={f:.4f}  ({elapsed:.1f}s)", flush=True)
        return f

    # ── Baseline: in-country OOF ──────────────────────────────────────────────────────────────────
    print("\n=== Baseline: in-country OOF ===")
    # Use first-half folds as train, last fold as eval (rough proxy, not full 5-fold)
    us_train = us_df.filter(pl.col("s1_id").is_in(
        folds.filter((pl.col("country") == "US") & (pl.col("fold") < 4))["s1_id"]))
    us_eval = us_df.filter(pl.col("s1_id").is_in(
        folds.filter((pl.col("country") == "US") & (pl.col("fold") == 4))["s1_id"]))
    in_train = in_df.filter(pl.col("s1_id").is_in(
        folds.filter((pl.col("country") == "India") & (pl.col("fold") < 4))["s1_id"]))
    in_eval = in_df.filter(pl.col("s1_id").is_in(
        folds.filter((pl.col("country") == "India") & (pl.col("fold") == 4))["s1_id"]))

    f_us_incountry = _run(us_train, us_eval, "US in-country", feats)
    f_in_incountry = _run(in_train, in_eval, "India in-country", feats)
    rows.append({"scenario": "US→US (in-country fold-4)", "drop_group": "none", "macro_f05": f_us_incountry})
    rows.append({"scenario": "India→India (in-country fold-4)", "drop_group": "none", "macro_f05": f_in_incountry})

    # ── Transfer: US→India ────────────────────────────────────────────────────────────────────────
    print("\n=== Transfer: US→India ===")
    f_us2in = _run(us_df, in_df, "US→India", feats)
    rows.append({"scenario": "US→India", "drop_group": "none", "macro_f05": f_us2in})
    gap_us2in = f_in_incountry - f_us2in

    # ── Transfer: India→US ────────────────────────────────────────────────────────────────────────
    print("\n=== Transfer: India→US ===")
    f_in2us = _run(in_df, us_df, "India→US", feats)
    rows.append({"scenario": "India→US", "drop_group": "none", "macro_f05": f_in2us})
    gap_in2us = f_us_incountry - f_in2us

    # ── Drop-one group ablation ───────────────────────────────────────────────────────────────────
    print("\n=== Drop-one group ablation (US→India direction) ===")
    group_map = _feature_groups_present(feats)
    ablation_rows = []
    for group, group_feats in group_map.items():
        remaining = [f for f in feats if f not in group_feats]
        if not remaining:
            continue
        f_drop = _run(us_df, in_df, f"US→India drop [{group}]", remaining)
        # In-country check: make sure dropping this group doesn't hurt US in-country by >0.002
        f_drop_us = _run(us_train, us_eval, f"US in-country drop [{group}]", remaining)
        delta_transfer = f_drop - f_us2in     # positive = gap narrows (good)
        delta_incountry = f_drop_us - f_us_incountry   # should be > -0.002
        ablation_rows.append({
            "group": group, "n_feats_dropped": len(group_feats),
            "transfer_us2in_with_drop": f_drop,
            "delta_transfer": delta_transfer,
            "incountry_us_with_drop": f_drop_us,
            "delta_incountry_us": delta_incountry,
            "recommendation": "KEEP" if delta_incountry < -0.002 else
                              ("DROP" if delta_transfer > 0.002 else "NEUTRAL"),
        })
        rows.append({"scenario": f"US→India drop [{group}]", "drop_group": group, "macro_f05": f_drop})

    # ── Write markdown report ─────────────────────────────────────────────────────────────────────
    root = Path(__file__).resolve().parents[3]   # repo root
    out_path = root / "notes" / "TRANSFER_CHECK.md"

    lines = [
        "# Transfer Check — US ↔ India (France safety proxy)",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M IST')}",
        f"Split: {split}  |  Config hash: {cfg.hash}",
        "",
        "## Motivation",
        "France is unseen at train time. We use the US↔India transfer gap as a proxy for",
        "how well features generalise to an unseen country. Features that hurt transfer",
        "without helping in-country performance should be dropped or regularised.",
        "",
        "## Main results",
        "",
        "| scenario | macro F0.5 |",
        "|---|---|",
        f"| US in-country (fold-4 OOF) | {f_us_incountry:.4f} |",
        f"| India in-country (fold-4 OOF) | {f_in_incountry:.4f} |",
        f"| US→India transfer | {f_us2in:.4f} |",
        f"| India→US transfer | {f_in2us:.4f} |",
        "",
        f"**Transfer gap US→India**: {gap_us2in:+.4f}  (India in-country minus US→India)",
        f"**Transfer gap India→US**: {gap_in2us:+.4f}  (US in-country minus India→US)",
        "",
        "## Drop-one group ablation (US→India direction)",
        "",
        "Threshold for recommendation: drop/regularise group if its removal narrows the transfer gap",
        "by > 0.002 **and** does not hurt US in-country performance by more than 0.002.",
        "",
        "| group | n dropped | transfer (all) | transfer (drop) | Δ transfer | US OOF (drop) | Δ US OOF | rec |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for ar in ablation_rows:
        lines.append(
            f"| {ar['group']} | {ar['n_feats_dropped']} | {f_us2in:.4f} | {ar['transfer_us2in_with_drop']:.4f} "
            f"| {ar['delta_transfer']:+.4f} | {ar['incountry_us_with_drop']:.4f} "
            f"| {ar['delta_incountry_us']:+.4f} | {ar['recommendation']} |"
        )
    lines += [
        "",
        "## All raw numbers",
        "",
        "| scenario | group dropped | macro F0.5 |",
        "|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['scenario']} | {r['drop_group']} | {r['macro_f05']:.4f} |")
    lines += [
        "",
        "## Notes",
        "- One-to-one assignment is skipped in this check (cross-country S1 sets are disjoint by design).",
        "- Threshold is fixed at 0.5; the decision layer is not tuned here.",
        "- Do NOT change model defaults based on this report alone — present findings to Faiz first.",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nTransfer check written to {out_path}")
    print(f"Transfer gap US→India: {gap_us2in:+.4f}")
    print(f"Transfer gap India→US: {gap_in2us:+.4f}")
    print("\nDrop-one ablation:")
    for ar in ablation_rows:
        print(f"  drop [{ar['group']:12s}]: Δtransfer={ar['delta_transfer']:+.4f}  "
              f"Δincountry={ar['delta_incountry_us']:+.4f}  → {ar['recommendation']}")


def main() -> None:
    """CLI entry point for transfer_check.py."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["train"], default="train",
                    help="Only train split has ground truth for OOF evaluation.")
    ap.add_argument("--config", help="Override yaml merged on top of configs/base.yaml.")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="Dotted config override, repeatable (e.g. --set paths.cache_dir=cache/smoke).")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    run_transfer_check(cfg, args.split)


if __name__ == "__main__":
    main()
