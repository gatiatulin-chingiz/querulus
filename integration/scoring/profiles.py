"""Пути к артефактам main_ / second_ (pickle, калибраторы)."""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

try:
    from integration import config
except ImportError:
    import config

DEFAULT_SECOND_STEM = "UU_querulus_2026_10_09_16_51_20"
DEFAULT_CF_CALIBRATOR = "cf_calibrator.pickle"
DEFAULT_RG_CALIBRATOR = "rg_calibrator.pickle"


#reviewed
def _base() -> Path:
    return Path(config.base_path)


#reviewed
def prod_models_dir() -> Path:
    models_path = Path(config.prod_models_path)
    if models_path.is_absolute():
        return models_path
    return (_base() / models_path).resolve()


#reviewed
def second_subdir() -> str:
    return (
        getattr(config, "shadow_models_subdir", None)
        or os.environ.get("SHADOW_MODELS_SUBDIR")
        or ""
    ).strip().replace("\\", "/")


#reviewed
def second_models_dir() -> Path:
    sub = second_subdir()
    if not sub or sub in {".", "./"}:
        return prod_models_dir()
    return (prod_models_dir() / Path(sub)).resolve()


#reviewed
def second_pickle_path(group_name: str) -> Path:
    stem = group_name.strip()
    primary = second_models_dir() / f"{stem}.pickle"
    if primary.is_file():
        return primary
    legacy = prod_models_dir() / "querulus" / "2.0.0" / f"{stem}.pickle"
    if legacy.is_file():
        return legacy
    return primary


#new
def second_cf_calibrator_path() -> Path:
    name = getattr(config, "second_cf_calibrator_filename", None) or DEFAULT_CF_CALIBRATOR
    return second_models_dir() / name


#new
def second_rg_calibrator_path() -> Path:
    name = getattr(config, "second_rg_calibrator_filename", None) or DEFAULT_RG_CALIBRATOR
    return second_models_dir() / name


#new
def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


#new
def load_optional_pickle(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return load_pickle(path)


#reviewed
def feature_names_from_model_result(model_result: dict[str, Any]) -> list[str]:
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
