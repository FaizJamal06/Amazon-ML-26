"""Paths, seeds and all tunables, loaded from configs/*.yaml.

Contract:
In: configs/base.yaml (+ optional override yaml + ``--set key=value`` CLI overrides).
Out: a frozen ``Config`` used by every stage. All ``paths.*`` values are resolved to absolute paths relative to the
repo root (``ml26/``; override with the ``BER_ROOT`` env var, e.g. inside the final submission zip).
``Config.hash`` is a short, machine-independent hash of the merged settings for artifact logging.

Owner: Faiz (R1 Lead / Eval / Decision)
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import yaml

PKG_DIR = Path(__file__).resolve().parents[2]            # code/business_entity_resolution
BASE_CONFIG = PKG_DIR / "configs" / "base.yaml"
_MISSING = object()


def repo_root() -> Path:
    """Return the repo root: ``$BER_ROOT`` if set, else the folder that contains ``code/``."""
    return Path(os.environ["BER_ROOT"]).resolve() if "BER_ROOT" in os.environ else PKG_DIR.parents[1]


def _deep_merge(base: dict, override: Mapping) -> dict:
    """Recursively merge ``override`` into a copy of ``base`` (override wins; nested dicts are merged)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, Mapping) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _apply_set(cfg: dict, assignment: str) -> None:
    """Apply one ``dotted.key=value`` override in place; the value is parsed as YAML (so 0.5, true, [1,2] work)."""
    key, sep, raw = assignment.partition("=")
    if not sep:
        raise ValueError(f"override must look like key=value, got {assignment!r}")
    *parents, leaf = key.strip().split(".")
    node = cfg
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = yaml.safe_load(raw)


def _resolve_paths(node: Any, root: Path) -> Any:
    """Turn every string under ``paths`` into an absolute path string relative to ``root``."""
    if isinstance(node, Mapping):
        return {k: _resolve_paths(v, root) for k, v in node.items()}
    return str((root / node).resolve()) if isinstance(node, str) else node


def _freeze(node: Any) -> Any:
    """Recursively convert dicts to read-only mappings and lists to tuples."""
    if isinstance(node, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in node.items()})
    if isinstance(node, list):
        return tuple(_freeze(v) for v in node)
    return node


@dataclass(frozen=True)
class Config:
    """Read-only merged configuration. Use attributes for the common knobs and ``get('a.b')`` for anything else."""

    data: Mapping[str, Any]
    root: Path
    hash: str

    def get(self, key: str, default: Any = _MISSING) -> Any:
        """Dotted lookup, e.g. ``cfg.get('decide.threshold')``; raises KeyError unless a default is given."""
        node: Any = self.data
        for part in key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                if default is _MISSING:
                    raise KeyError(key)
                return default
            node = node[part]
        return node

    def path(self, key: str) -> Path:
        """Absolute path for ``paths.<key>``, e.g. ``cfg.path('train.source1')``."""
        return Path(self.get(f"paths.{key}"))

    def artifact(self, name: str, split: str, subworld: bool = False) -> Path:
        """Parquet path of a §4 artifact in the cache, e.g. ``artifact('candidates', 'train')``.

        Sub-world runs use a ``_sw`` suffix so they never overwrite full-world artifacts.
        """
        return self.cache_dir / f"{name}_{split}{'_sw' if subworld else ''}.parquet"

    seed = property(lambda self: int(self.get("seed")), doc="Global random seed.")
    n_folds = property(lambda self: int(self.get("n_folds")), doc="Number of CV folds.")
    subworld_frac = property(lambda self: float(self.get("subworld_frac")), doc="S1 fraction of a sub-world.")
    max_cands = property(lambda self: int(self.get("max_cands")), doc="Candidate cap per S1.")
    decision_margin = property(lambda self: float(self.get("decision_margin")), doc="Extra decision conservatism.")
    cache_dir = property(lambda self: self.path("cache_dir"), doc="Absolute cache directory.")
    output_dir = property(lambda self: self.path("output_dir"), doc="Absolute submission output directory.")


def load_config(override: str | Path | None = None, sets: Sequence[str] = (), base: str | Path = BASE_CONFIG) -> Config:
    """Load base.yaml, deep-merge an optional override yaml, then apply ``key=value`` overrides.

    The hash is taken before path resolution, so the same settings give the same hash on every machine.
    """
    cfg = yaml.safe_load(Path(base).read_text(encoding="utf-8")) or {}
    if override:
        cfg = _deep_merge(cfg, yaml.safe_load(Path(override).read_text(encoding="utf-8")) or {})
    for s in sets:
        _apply_set(cfg, s)
    digest = hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]
    root = repo_root()
    cfg["paths"] = _resolve_paths(cfg.get("paths", {}), root)
    return Config(data=_freeze(cfg), root=root, hash=digest)
