"""CLI entry point: ``python -m ber.pipeline <stage> --split {train,test} [--subworld] [--config x.yaml] [--set k=v]``.

Contract:
In: config + cached artifacts of the previous stage (CLAUDE.md §4). Out: the artifacts of the requested stage.
Every owner stage function has the signature ``fn(cfg: Config, split: str, subworld: bool) -> None`` and reads/writes
``cfg.artifact(name, split, subworld)``. Stages whose owner has not pushed the function yet raise NotImplementedError.
Run on stub data with ``--set paths.cache_dir=cache/stub``.

Owner: Faiz (R1 Lead / Eval / Decision)
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import time
from typing import Callable

import polars as pl

from ber.config import Config, load_config

# stage -> (module, function, owner) for stages implemented in other owners' modules
OWNER_STAGES: dict[str, tuple[str, str, str]] = {
    "ingest": ("ber.io", "ingest", "Chris (R4)"),                          # raw TSV -> parquet (+ gt_train.parquet)
    "normalize": ("ber.normalize", "build_records", "Dhanishkaa (R2)"),    # -> records_{split}.parquet
    "block": ("ber.blocking.merge", "build_candidates", "Dhanishkaa (R2)"),  # -> candidates_{split}.parquet
    "featurize": ("ber.features", "build_features", "Nitish (R3)"),        # -> features_{split}.parquet
    "train": ("ber.model.lgbm", "train_oof", "Nitish (R3)"),               # -> scores_train.parquet (OOF)
    "predict": ("ber.model.lgbm", "predict", "Nitish (R3)"),               # -> scores_test.parquet
}
CHAIN = {"train": ["ingest", "normalize", "block", "folds", "featurize", "train", "decide"],
         "test": ["ingest", "normalize", "block", "featurize", "predict", "decide", "submit"]}


def peak_memory_mb() -> float:
    """Peak resident memory of this process in MB (Windows: PeakWorkingSetSize; Unix: ru_maxrss)."""
    if sys.platform == "win32":
        import ctypes
        import ctypes.wintypes as wt

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t)] + [(f"f{i}", ctypes.c_size_t) for i in range(6)]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wt.HANDLE
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / 2**20
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


def git_commit() -> str:
    """Short commit hash of the working tree (``+dirty`` if uncommitted changes), or ``unknown``."""
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout
        return h + ("+dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _read(cfg: Config, name: str, split: str, subworld: bool) -> pl.DataFrame:
    """Read a §4 artifact, with a clear error naming the stage that produces it."""
    path = cfg.artifact(name, split, subworld)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run the stage that produces '{name}' first")
    return pl.read_parquet(path)


def _s1_universe(cfg: Config, split: str, subworld: bool) -> pl.DataFrame:
    """All S1 ids of the (sub-)world with their country, from records."""
    rec = _read(cfg, "records", split, subworld)
    return rec.filter(pl.col("source") == 1).select(s1_id="entity_id", country="country")


# --------------------------------------------------------------------------------------------- R1 stages
def stage_folds(cfg: Config, split: str, subworld: bool) -> None:
    """folds_train.parquet from records (S1s) + gt_train; prints fold sizes per country."""
    from ber.eval.split import make_folds
    _train_only("folds", split)
    folds = make_folds(_s1_universe(cfg, split, False), _read(cfg, "gt", split, False), cfg.n_folds, cfg.seed)
    folds.write_parquet(cfg.artifact("folds", split))
    print(folds.group_by("country", "fold").agg(n_s1=pl.len(), mean_matches=pl.col("n_matches").mean())
          .sort("country", "fold"))


def stage_subworld(cfg: Config, split: str, subworld: bool) -> None:
    """Closed sub-world: writes subworld / records / gt / candidates ``*_train_sw.parquet`` + density report."""
    from ber.eval.split import density_report, sample_subworld
    _train_only("subworld", split)
    rec, gt, cand = (_read(cfg, n, split, False) for n in ("records", "gt", "candidates"))
    s1 = rec.filter(pl.col("source") == 1).select(s1_id="entity_id", country="country")
    world, cand_sw = sample_subworld(s1, gt, cand, cfg.subworld_frac, cfg.seed)
    world.write_parquet(cfg.artifact("subworld", split))
    rec.join(world.select("entity_id"), on="entity_id", how="semi").write_parquet(cfg.artifact("records", split, True))
    gt.join(world.filter(pl.col("role") == "s1").select(s1_id="entity_id"), on="s1_id", how="semi") \
        .write_parquet(cfg.artifact("gt", split, True))
    cand_sw.write_parquet(cfg.artifact("candidates", split, True))
    print(density_report(world, rec))


def stage_blocking_report(cfg: Config, split: str, subworld: bool) -> None:
    """Print the blocking report (recall ceiling, coverage, cands/S1, reduction ratio) as markdown."""
    from ber.eval.blocking_report import blocking_report, to_markdown
    _train_only("blocking-report", split)
    print(to_markdown(blocking_report(*(_read(cfg, n, split, subworld) for n in ("candidates", "gt", "records")))))


def _decide_params(cfg: Config) -> dict:
    """Selection parameters from the ``decide`` config block."""
    return dict(threshold=float(cfg.get("decide.threshold")), topk=int(cfg.get("decide.topk", 15)),
                miss_mass=float(cfg.get("decide.miss_mass", 0.0)), margin=cfg.decision_margin)


def stage_decide(cfg: Config, split: str, subworld: bool) -> None:
    """One-to-one + ``decide.method`` selection -> matches_{split}.parquet; on train also prints the OOF threshold curve
    and the OOF macro F0.5 of the configured method."""
    from ber.decide.assign import assign_one_to_one
    from ber.decide.select import best_threshold, select, threshold_curve, threshold_grid
    from ber.eval.metric import macro_f05
    scores_name, method = cfg.get("decide.scores_name", "scores"), cfg.get("decide.method", "threshold")
    scores, cand = _read(cfg, scores_name, split, subworld), _read(cfg, "candidates", split, subworld)
    print(f"decide: reading {scores_name}, method {method}")
    if split == "train":
        curve = threshold_curve(scores, _read(cfg, "gt", split, subworld), _s1_universe(cfg, split, subworld),
                                threshold_grid(*cfg.get("decide.grid")), cand, cfg.decision_margin)
        with pl.Config(tbl_rows=50, float_precision=4):
            print(curve)
        print(f"best OOF threshold: {best_threshold(curve)} (config decide.threshold = {cfg.get('decide.threshold')})")
    sel = select(assign_one_to_one(scores, cand), method, **_decide_params(cfg))
    sel.write_parquet(cfg.artifact("matches", split, subworld))
    if split == "train":
        f = macro_f05(sel, _read(cfg, "gt", split, subworld), _s1_universe(cfg, split, subworld))
        print(f"OOF macro F0.5 with {method}: {f:.4f}")
    print(f"{sel.height:,} matched pairs over {sel['s1_id'].n_unique():,} S1s")


def stage_compare_decide(cfg: Config, split: str, subworld: bool) -> None:
    """OOF macro F0.5 of threshold (best t) vs ef05_approx vs ef05_exact: overall / per country / per script + runtime.

    Note: the threshold row picks t on the same OOF it is scored on (slightly optimistic); ef05 has no tuned knob.
    """
    from ber.decide.assign import assign_one_to_one
    from ber.decide.select import best_threshold, select, threshold_curve, threshold_grid
    from ber.eval.metric import macro_f05_by, per_entity_f05, s1_groups
    _train_only("compare-decide", split)
    scores_name = cfg.get("decide.scores_name", "scores")
    rec, gt, cand = (_read(cfg, n, split, subworld) for n in ("records", "gt", "candidates"))
    s1 = s1_groups(rec, gt)
    assigned = assign_one_to_one(_read(cfg, scores_name, split, subworld), cand)
    params = _decide_params(cfg)
    t0 = time.time()
    params["threshold"] = best_threshold(threshold_curve(assigned, gt, s1, threshold_grid(*cfg.get("decide.grid")),
                                                         margin=cfg.decision_margin))
    rows = []
    for method in ("threshold", "ef05_approx", "ef05_exact"):
        t1 = time.time()
        sel = select(assigned, method, **params)
        secs = time.time() - t1 + (t1 - t0 if method == "threshold" else 0.0)
        row = {"method": method + (f" (t={params['threshold']})" if method == "threshold" else ""),
               "macro_f05": per_entity_f05(sel, gt, s1)["f05"].mean(), "pred_pairs": sel.height,
               "runtime_s": round(secs, 2)}
        for by in ("country", "script"):
            row.update({f"{by}={g}": v for g, v in macro_f05_by(sel, gt, s1, by).select(by, "macro_f05").iter_rows()})
        rows.append(row)
    with pl.Config(tbl_cols=-1, tbl_width_chars=250, float_precision=4):
        print(f"compare-decide on {scores_name} ({'sub-world' if subworld else 'full world'}), "
              f"topk={params['topk']}, miss_mass={params['miss_mass']}, margin={params['margin']}")
        print(pl.DataFrame(rows))


def stage_submit(cfg: Config, split: str, subworld: bool) -> None:
    """Write output/*.tsv from matches_test + candidates_test, run the official validator, print SUBMISSIONS line."""
    from ber.io import write_submission
    if split != "test" or subworld:
        raise SystemExit("submit runs on the full test split only")
    rec = _read(cfg, "records", split, False)
    s1_ids = rec.filter(pl.col("source") == 1)["entity_id"]
    out = write_submission(_read(cfg, "matches", split, False), _read(cfg, "candidates", split, False), s1_ids,
                           rec.filter(pl.col("source") > 1)["entity_id"], cfg.output_dir, cfg.path("validator"),
                           cfg.path("test.source1").parent)
    print("validator: PASS")
    print(f"| # | <date-time IST> | {git_commit()} | <change> | <OOF F0.5> | <LB> | {out['matching_sha256']} | "
          f"{out['candidate_sha256']} | <Drive/S3 path> |")


def _train_only(stage: str, split: str) -> None:
    """Guard for stages that need ground truth."""
    if split != "train":
        raise SystemExit(f"{stage} needs ground truth: use --split train")


def stage_errors(cfg: Config, split: str, subworld: bool) -> None:
    """Markdown error report (worst FP / FN, summaries) -> reports/errors_{split}[_sw]_{scores_name}.md."""
    from ber.eval.errors import error_report, write_report
    _train_only("errors", split)
    scores_name = cfg.get("decide.scores_name", "scores")
    arts = {n: _read(cfg, n, split, subworld) for n in ("matches", "candidates", "records", "gt")}
    title = f"Errors — {split}{' sub-world' if subworld else ''}, {scores_name}, config {cfg.hash}, commit {git_commit()}"
    text = error_report(arts["matches"], _read(cfg, scores_name, split, subworld), arts["candidates"],
                        arts["records"], arts["gt"], title, seed=cfg.seed)
    path = write_report(text, cfg.path("reports_dir") / f"errors_{split}{'_sw' if subworld else ''}_{scores_name}.md")
    print(text.split("\n## ")[0])
    print(f"report -> {path}")


def stage_fallback_train(cfg: Config, split: str, subworld: bool) -> None:
    """Fallback scorer: OOF calibrated ``scores_fallback`` on train + saved model (ber.fallback)."""
    from ber.fallback import fallback_train
    fallback_train(cfg, split, subworld)


def stage_fallback_predict(cfg: Config, split: str, subworld: bool) -> None:
    """Fallback scorer: ``scores_fallback`` for the given split from the saved model (ber.fallback)."""
    from ber.fallback import fallback_predict
    fallback_predict(cfg, split, subworld)


R1_STAGES: dict[str, Callable[[Config, str, bool], None]] = {
    "fallback-train": stage_fallback_train, "fallback-predict": stage_fallback_predict,
    "compare-decide": stage_compare_decide, "errors": stage_errors, "folds": stage_folds, "subworld": stage_subworld, "blocking-report": stage_blocking_report,
    "decide": stage_decide, "submit": stage_submit,
}


def resolve(stage: str) -> Callable[[Config, str, bool], None]:
    """Return the callable for ``stage``; owner stages raise NotImplementedError until their function exists."""
    if stage in R1_STAGES:
        return R1_STAGES[stage]
    module, func, owner = OWNER_STAGES[stage]
    fn = getattr(importlib.import_module(module), func, None)
    if fn is None:
        def missing(cfg: Config, split: str, subworld: bool) -> None:
            """Placeholder for a stage whose owner has not pushed it yet."""
            raise NotImplementedError(f"{module}.{func}(cfg, split, subworld) not implemented yet — owner {owner}")
        return missing
    return fn


def run(stage: str, cfg: Config, split: str, subworld: bool) -> None:
    """Run one stage and log config hash, commit, runtime and peak memory."""
    t0 = time.time()
    print(f"[{stage}] split={split} subworld={subworld} config={cfg.hash} commit={git_commit()}", flush=True)
    resolve(stage)(cfg, split, subworld)
    print(f"[{stage}] done in {time.time() - t0:.1f}s, peak memory {peak_memory_mb():.0f} MB", flush=True)


def main(argv: list[str] | None = None) -> None:
    """Parse the CLI and run the requested stage (``all`` runs the whole chain for the split)."""
    stages = [*OWNER_STAGES, *R1_STAGES]
    ap = argparse.ArgumentParser(prog="python -m ber.pipeline", description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=[*stages, "all"])
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--subworld", action="store_true", help="run on the closed sub-world (cfg.subworld_frac)")
    ap.add_argument("--config", help="override yaml merged on top of configs/base.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="dotted override, repeatable")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    for stage in CHAIN[args.split] if args.stage == "all" else [args.stage]:
        run(stage, cfg, args.split, args.subworld)


if __name__ == "__main__":
    main()
