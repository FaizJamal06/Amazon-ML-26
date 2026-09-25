"""LightGBM training (5-fold OOF), prediction and isotonic calibration on held-out folds.

Contract:
In: features_{split}.parquet + folds_train.parquet (full-world folds, group = s1_id).
Out: scores_{split}.parquet with s1_id, cand_id, p (calibrated; OOF on train). Final model (sample of all folds,
rounds = mean best iteration of the fold models) + isotonic calibrator (fit on all OOF raw scores) ->
cache/models/lgbm[_sw].joblib; gain importance -> reports/lgbm_importance_train[_sw].csv.
Stages: ``train`` (train_oof, train split) and ``predict`` (any split, uses the saved model).
Tunables (optional config keys): ``lgbm.params.*`` (merged over PARAMS), ``lgbm.max_train_s1``, ``features.chunk_pairs``.

Owner: Nitish (R3 Features / LightGBM)
"""
from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from ber.config import Config

KEYS = ["s1_id", "cand_id"]
CATEGORICAL = ["hn_rel", "legal_rel", "s1_script", "cand_script"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, deterministic=True,
              force_row_wise=True)
MAX_ROUNDS, EARLY_STOP, INNER_VALID = 3000, 100, 0.1


def _params(cfg: Config) -> dict:
    """PARAMS + the global seed + any ``lgbm.params.*`` overrides from the config."""
    return {**PARAMS, "seed": cfg.seed, **dict(cfg.get("lgbm.params", {}))}


def _x(df: pl.DataFrame, feats: list[str]) -> np.ndarray:
    """Float32 feature matrix in ``feats`` order (nulls -> NaN, which LightGBM treats as missing)."""
    return df.select(pl.col(feats).cast(pl.Float32)).to_numpy()


def _dataset(df: pl.DataFrame, feats: list[str], reference: lgb.Dataset | None = None) -> lgb.Dataset:
    """LightGBM dataset with the categorical code columns declared."""
    return lgb.Dataset(_x(df, feats), df["label"].to_numpy(), feature_name=feats, reference=reference,
                       categorical_feature=[c for c in CATEGORICAL if c in feats])


def fit(train: pl.DataFrame, feats: list[str], params: dict, seed: int, rounds: int | None = None) -> lgb.Booster:
    """Train one booster. ``rounds=None``: early stopping on an S1-grouped inner split (INNER_VALID of the S1s)."""
    if rounds is not None:
        return lgb.train(params, _dataset(train, feats), rounds)
    val_s1 = train["s1_id"].unique().sort().sample(fraction=INNER_VALID, seed=seed)
    is_val = train["s1_id"].is_in(val_s1)
    dtr = _dataset(train.filter(~is_val), feats)
    return lgb.train(params, dtr, MAX_ROUNDS, valid_sets=[_dataset(train.filter(is_val), feats, dtr)],
                     callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])


def cross_fit_isotonic(raw: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Calibrated OOF p: fold k is mapped by an isotonic fit on the OOF raw scores of the other folds."""
    p = np.zeros_like(raw)
    for k in np.unique(folds):  # folds, not rows
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw[folds != k], y[folds != k])
        p[folds == k] = iso.predict(raw[folds == k])
    return p


def _model_path(cfg: Config, subworld: bool) -> Path:
    """Where the final booster + calibrator live (git-ignored cache/models/)."""
    return cfg.cache_dir / "models" / f"lgbm{'_sw' if subworld else ''}.joblib"


def train_oof(cfg: Config, split: str, subworld: bool) -> None:
    """Stage ``train``: 5-fold OOF raw scores -> cross-fitted isotonic p -> ``scores_train``; saves the final model,
    the calibrator and gain importances.

    Each fold model trains on at most ``lgbm.max_train_s1`` S1s of the other folds (all their pairs) and predicts
    every pair of the held-out fold.
    """
    if split != "train":
        raise SystemExit("train needs --split train")
    path = cfg.artifact("features", split, subworld)
    feats = [c for c in pq.read_schema(path).names if c not in (*KEYS, "label")]
    folds = pl.read_parquet(cfg.artifact("folds", split)).select("s1_id", "fold")   # folds are full-world
    ids = pl.scan_parquet(path).select("s1_id").unique().collect().join(folds, on="s1_id", how="left").sort("s1_id")
    if ids["fold"].null_count():
        raise ValueError("some feature S1s have no fold — run the folds stage on the same train set")
    ids = ids.with_columns(_r=pl.Series(np.random.default_rng(cfg.seed).random(ids.height)))
    params, max_s1, scan = _params(cfg), int(cfg.get("lgbm.max_train_s1", 300_000)), pl.scan_parquet(path)
    rows = lambda s1: scan.join(s1.select("s1_id").lazy(), on="s1_id", how="semi").collect()  # noqa: E731
    oof, boosters = [], []
    for k in sorted(ids["fold"].unique()):  # 5 folds, not rows
        booster = fit(rows(ids.filter(pl.col("fold") != k).sort("_r").head(max_s1)), feats, params, cfg.seed)
        held = rows(ids.filter(pl.col("fold") == k))
        oof.append(held.select(*KEYS, "label", raw=pl.Series(booster.predict(_x(held, feats))), fold=pl.lit(k)))
        boosters.append(booster)
        print(f"fold {k}: best iteration {booster.best_iteration}, {held.height:,} held-out pairs")
    oof = pl.concat(oof)
    raw, y, fold = oof["raw"].to_numpy(), oof["label"].to_numpy(), oof["fold"].to_numpy()
    p = cross_fit_isotonic(raw, y, fold)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)
    rounds = max(1, int(np.mean([b.best_iteration or b.current_iteration() for b in boosters])))
    final = fit(rows(ids.sort("_r").head(max_s1)), feats, params, cfg.seed, rounds)
    mpath = _model_path(cfg, subworld)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final.model_to_string(), "calibrator": calibrator, "features": feats, "rounds": rounds,
                 "config_hash": cfg.hash}, mpath)
    imp = pl.DataFrame({"feature": feats, "gain_final": final.feature_importance("gain"),
                        "gain_folds_mean": np.mean([b.feature_importance("gain") for b in boosters], axis=0)}) \
        .sort("gain_final", descending=True)
    ipath = cfg.path("reports_dir") / f"lgbm_importance_{split}{'_sw' if subworld else ''}.csv"
    ipath.parent.mkdir(parents=True, exist_ok=True)
    imp.write_csv(ipath)
    oof.select(*KEYS, p=pl.Series(p)).write_parquet(cfg.artifact("scores", split, subworld))
    auc = roc_auc_score(y, raw) if 0 < y.mean() < 1 else float("nan")
    print(f"lgbm: {oof.height:,} pairs, {len(feats)} features, positives {y.mean():.3f}, OOF AUC {auc:.4f}, "
          f"mean p {p.mean():.3f}, Brier {np.mean((p - y) ** 2):.4f}; final model {rounds} rounds -> {mpath}")
    print(f"top gain: {imp.head(10)['feature'].to_list()} (full table -> {ipath})")


def predict(cfg: Config, split: str, subworld: bool) -> None:
    """Stage ``predict``: score ``features_{split}`` in chunks with the saved model + calibrator -> ``scores_{split}``."""
    mpath = _model_path(cfg, subworld)
    if not mpath.exists():
        raise FileNotFoundError(f"{mpath} missing — run the train stage first")
    bundle = joblib.load(mpath)
    booster, feats = lgb.Booster(model_str=bundle["model"]), bundle["features"]
    out = []
    batches = pq.ParquetFile(cfg.artifact("features", split, subworld)).iter_batches(
        batch_size=int(cfg.get("features.chunk_pairs", 5_000_000)), columns=KEYS + feats)
    for batch in batches:  # chunks of pairs, not rows
        df = pl.from_arrow(batch)
        out.append(df.select(*KEYS, p=pl.Series(bundle["calibrator"].predict(booster.predict(_x(df, feats))))))
    scores = pl.concat(out)
    scores.write_parquet(cfg.artifact("scores", split, subworld))
    print(f"lgbm predict: {scores.height:,} pairs scored, mean p {scores['p'].mean():.3f} "
          f"(model trained with config {bundle['config_hash']})")
