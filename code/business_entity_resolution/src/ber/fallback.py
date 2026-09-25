"""Rule-feature fallback scorer — insurance for Submission 1 if LightGBM slips.

Needs only ``records`` + ``candidates`` (+ ``gt`` and ``folds`` for training). Self-contained features:
block_score, rank_in_cand, house-number relation + abs diff, rapidfuzz name/street similarities, legal-form conflict.
Model: a small HistGradientBoostingClassifier trained per fold (group = S1) -> OOF raw scores -> cross-fitted isotonic
calibration. The final model (all train) + isotonic calibrator are saved to ``cache/models/`` for the test split.

Contract:
In: ``records``, ``candidates`` (both splits); ``gt``, ``folds`` (train).
Out: ``scores_fallback`` artifact (``s1_id, cand_id, p``; OOF + calibrated on train) — never overwrites ``scores``.
Stages: ``fallback-train`` (train) and ``fallback-predict`` (test).

Owner: Faiz (R1 Lead / Eval / Decision)
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from ber.config import Config
from ber.features.pairwise import cpdist
from ber.features.structured import house_num_relation, legal_conflict

SCORES_NAME = "scores_fallback"
HN_RELATIONS = ["equal", "one_missing", "both_missing", "different"]
NUMERIC = ["block_score", "rank_in_cand", "hn_absdiff_log", "name_tset", "name_ratio", "street_ratio",
           "legal_conflict", "hn_rel_code"]
CHUNK = 2_000_000  # pairs per rapidfuzz batch (bounds the Python string lists)


def pair_features(candidates: pl.DataFrame, records: pl.DataFrame) -> pl.DataFrame:
    """Fallback features for every candidate pair: ``s1_id, cand_id, country`` + ``NUMERIC`` columns + ``hn_rel``."""
    side = lambda pre: records.select(  # noqa: E731
        pl.col("entity_id").alias(f"{pre}_id" if pre == "s1" else "cand_id"),
        pl.col("name_core").alias(f"{pre}_core"), pl.col("addr_street").alias(f"{pre}_street"),
        pl.col("house_num").alias(f"{pre}_hn"), pl.col("legal_form").alias(f"{pre}_legal"))
    df = (candidates.select("s1_id", "cand_id", "country", "block_score", "rank_in_cand")
          .join(side("s1"), on="s1_id", how="left").join(side("c"), on="cand_id", how="left")
          .with_columns(pl.col("^(s1|c)_(core|street|hn|legal)$").fill_null("")))
    rel, diff = house_num_relation("s1_hn", "c_hn")
    df = df.with_columns(
        hn_rel=rel,
        hn_absdiff_log=diff.cast(pl.Float64).log1p().fill_null(-1.0),
        legal_conflict=legal_conflict("s1_legal", "c_legal"),
    ).with_columns(hn_rel_code=pl.col("hn_rel").replace_strict(HN_RELATIONS, list(range(4)), return_dtype=pl.Int8))
    sims = {k: np.empty(df.height, np.float32) for k in ("name_tset", "name_ratio", "street_ratio")}
    for lo in range(0, df.height, CHUNK):  # chunks of pairs, not rows
        part = df.slice(lo, CHUNK)
        s1c, cc = part["s1_core"].to_list(), part["c_core"].to_list()
        sims["name_tset"][lo:lo + part.height] = cpdist(s1c, cc, fuzz.token_set_ratio)
        sims["name_ratio"][lo:lo + part.height] = cpdist(s1c, cc, fuzz.ratio)
        sims["street_ratio"][lo:lo + part.height] = cpdist(part["s1_street"].to_list(), part["c_street"].to_list(),
                                                            fuzz.ratio)
    return df.with_columns(**{k: pl.Series(v) for k, v in sims.items()}).select(
        "s1_id", "cand_id", "country", "hn_rel", *NUMERIC)


def _model(seed: int) -> HistGradientBoostingClassifier:
    """The fallback classifier: small, regularised gradient boosting; hn_rel_code is categorical."""
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0,
                                          categorical_features=[NUMERIC.index("hn_rel_code")], random_state=seed)


def _xy(feats: pl.DataFrame) -> np.ndarray:
    """Feature matrix in ``NUMERIC`` order."""
    return feats.select(NUMERIC).to_numpy().astype(np.float32)


def train_oof(feats: pl.DataFrame, labels: np.ndarray, folds: np.ndarray, s1_ids: pl.Series, seed: int,
              max_train_s1: int) -> tuple[np.ndarray, HistGradientBoostingClassifier, IsotonicRegression]:
    """Fold-wise OOF raw scores, cross-fitted isotonic calibration, and a final model + calibrator on all data.

    Each fold's model trains on at most ``max_train_s1`` S1s of the other folds (all their pairs) and predicts the
    whole held-out fold. Fold k's calibrated p uses an isotonic fit on the OOF raw scores of the other folds.
    Returns ``(p_oof_calibrated, final_model, final_calibrator)``.
    """
    X = _xy(feats)
    s1_fold = pl.DataFrame({"s1_id": s1_ids, "fold": folds}).unique("s1_id").sort("s1_id")
    s1_fold = s1_fold.with_columns(_r=pl.Series(np.random.default_rng(seed).random(s1_fold.height)))
    raw = np.zeros(len(labels))
    fold_ids = np.unique(folds)
    for k in fold_ids:  # 5 folds, not rows
        tr_s1 = s1_fold.filter(pl.col("fold") != k).sort("_r").head(max_train_s1)["s1_id"]
        tr = s1_ids.is_in(tr_s1).to_numpy()
        raw[folds == k] = _model(seed).fit(X[tr], labels[tr]).predict_proba(X[folds == k])[:, 1]
    p = np.zeros_like(raw)
    for k in fold_ids:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw[folds != k], labels[folds != k])
        p[folds == k] = iso.predict(raw[folds == k])
    final_s1 = s1_fold.sort("_r").head(max_train_s1)["s1_id"]
    sel = s1_ids.is_in(final_s1).to_numpy()
    final = _model(seed).fit(X[sel], labels[sel])
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, labels)
    return p, final, calibrator


def _model_path(cfg: Config, subworld: bool) -> Path:
    """Where the fitted fallback model + calibrator live (git-ignored cache/models/)."""
    return cfg.cache_dir / "models" / f"fallback{'_sw' if subworld else ''}.joblib"


def fallback_train(cfg: Config, split: str, subworld: bool) -> None:
    """Stage ``fallback-train``: OOF calibrated scores on train -> ``scores_fallback_train``; saves model."""
    if split != "train":
        raise SystemExit("fallback-train needs --split train")
    cand = pl.read_parquet(cfg.artifact("candidates", split, subworld))
    rec = pl.read_parquet(cfg.artifact("records", split, subworld))
    gt = pl.read_parquet(cfg.artifact("gt", split, subworld))
    folds = pl.read_parquet(cfg.artifact("folds", split)).select("s1_id", "fold")   # folds are full-world
    feats = (pair_features(cand, rec).join(folds, on="s1_id", how="left")
             .join(gt.select("s1_id", cand_id="match_id", label=pl.lit(1, pl.Int8)), on=["s1_id", "cand_id"], how="left")
             .with_columns(pl.col("label").fill_null(0)))
    if feats["fold"].null_count():
        raise ValueError("some candidate S1s have no fold — run the folds stage on the same train set")
    p, model, calibrator = train_oof(feats, feats["label"].to_numpy(), feats["fold"].to_numpy(), feats["s1_id"],
                                     cfg.seed, int(cfg.get("fallback.max_train_s1", 300_000)))
    path = _model_path(cfg, subworld)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "calibrator": calibrator, "features": NUMERIC, "config_hash": cfg.hash}, path)
    out = feats.select("s1_id", "cand_id", p=pl.Series(p))
    out.write_parquet(cfg.artifact(SCORES_NAME, split, subworld))
    y = feats["label"].to_numpy()
    print(f"fallback: {out.height:,} pairs, positives {y.mean():.3f}, OOF mean p {p.mean():.3f}, "
          f"Brier {np.mean((p - y) ** 2):.4f}; model -> {path}")


def fallback_predict(cfg: Config, split: str, subworld: bool) -> None:
    """Stage ``fallback-predict``: score ``candidates_{split}`` with the saved model -> ``scores_fallback_{split}``."""
    path = _model_path(cfg, subworld)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run fallback-train first")
    bundle = joblib.load(path)
    feats = pair_features(pl.read_parquet(cfg.artifact("candidates", split, subworld)),
                          pl.read_parquet(cfg.artifact("records", split, subworld)))
    p = bundle["calibrator"].predict(bundle["model"].predict_proba(_xy(feats))[:, 1])
    feats.select("s1_id", "cand_id", p=pl.Series(p)).write_parquet(cfg.artifact(SCORES_NAME, split, subworld))
    print(f"fallback-predict: {feats.height:,} pairs scored (model trained with config {bundle['config_hash']})")
