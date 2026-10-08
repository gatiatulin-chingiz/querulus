"""Профили legacy (main_) vs shadow v2.0.0 (second_).

# CUTOVER: после shadow-периода перевести v2 в main_ и удалить legacy-константы фич.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from loguru import logger

import config

# CUTOVER: True пока 2.0.0 живёт только в second_
SHADOW_NEW_AS_SECOND: bool = True

# Плоский results/ (без querulus/2.0.0): pickle + metadata рядом.
DEFAULT_SHADOW_SUBDIR = ""
DEFAULT_SHADOW_STEM = "querulus_ansamble"
DEFAULT_SHADOW_META = "metadata.json"
DEFAULT_SHADOW_DQ = "dq_bounds.json"


def _base() -> Path:
    return Path(config.base_path)


def prod_models_dir() -> Path:
    models_path = Path(config.prod_models_path)
    if models_path.is_absolute():
        return models_path
    return (_base() / models_path).resolve()


def shadow_subdir() -> str:
    return (
        getattr(config, "shadow_models_subdir", None)
        or os.environ.get("SHADOW_MODELS_SUBDIR")
        or DEFAULT_SHADOW_SUBDIR
    ).strip().replace("\\", "/")


def shadow_models_dir() -> Path:
    """Каталог артефактов 2.0.0: плоский ``results/`` (опционально subdir из env)."""
    sub = shadow_subdir()
    if not sub or sub in {".", "./"}:
        return prod_models_dir()
    return (prod_models_dir() / Path(sub)).resolve()


def shadow_pickle_path(group_name: str) -> Path:
    """``{stem}.pickle`` в shadow_models_dir; fallback на legacy ``querulus/2.0.0/``."""
    stem = group_name.strip()
    primary = shadow_models_dir() / f"{stem}.pickle"
    if primary.is_file():
        return primary
    legacy = prod_models_dir() / "querulus" / "2.0.0" / f"{stem}.pickle"
    if legacy.is_file():
        return legacy
    return primary


def shadow_meta_path(group_name: str | None = None) -> Path:
    """metadata.json рядом с pickle (τ = best_threshold)."""
    name = getattr(config, "shadow_meta_filename", None) or DEFAULT_SHADOW_META
    primary = shadow_models_dir() / name
    if primary.is_file():
        return primary
    legacy = prod_models_dir() / "querulus" / "2.0.0" / name
    if legacy.is_file():
        return legacy
    if group_name:
        alt = shadow_models_dir() / f"querulus_meta_{group_name}.json"
        if alt.is_file():
            return alt
    return primary


def shadow_dq_bounds_path() -> Path:
    name = getattr(config, "shadow_dq_bounds_filename", None) or DEFAULT_SHADOW_DQ
    return shadow_models_dir() / name


def threshold_from_meta(meta_path: Path | None = None) -> Optional[float]:
    """best_threshold из metadata.json (τ-cal / prod из collect)."""
    path = meta_path or shadow_meta_path()
    if not path.is_file():
        logger.warning("Shadow meta не найден: {}", path)
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("best_threshold")
        if raw is None:
            return None
        return float(raw)
    except Exception:
        logger.exception("Не удалось прочитать best_threshold из {}", path)
        return None


def feature_names_from_model_result(model_result: dict[str, Any]) -> list[str]:
    """Имена фич из model_config pickle (порядок как в конфиге)."""
    cfg = model_result.get("model_config") or {}
    if not isinstance(cfg, dict):
        cfg = {}
    features = cfg.get("features") or []
    names: list[str] = []
    for item in features:
        if isinstance(item, dict) and item.get("name"):
            names.append(str(item["name"]))
        elif hasattr(item, "name"):
            names.append(str(item.name))
    return names
