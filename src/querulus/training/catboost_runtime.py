"""CatBoost runtime helpers (без зависимости от pipeline → severity_training cycle)."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from querulus import PROJECT_ROOT
from querulus.training.config import TrainingConfig

try:
    from catboost import (
        CatBoostClassifier,
        CatBoostRegressor,
        EFeaturesSelectionAlgorithm,
        EShapCalcType,
        Pool,
    )
except ImportError:  # pragma: no cover
    CatBoostClassifier = None  # type: ignore[assignment, misc]
    CatBoostRegressor = None  # type: ignore[assignment, misc]
    EFeaturesSelectionAlgorithm = None  # type: ignore[assignment, misc]
    EShapCalcType = None  # type: ignore[assignment, misc]
    Pool = None  # type: ignore[assignment, misc]


def require_catboost():
    """Вернуть классы CatBoost; понятная ошибка, если пакет не установлен."""
    if (
        CatBoostClassifier is None
        or CatBoostRegressor is None
        or Pool is None
        or EFeaturesSelectionAlgorithm is None
        or EShapCalcType is None
    ):
        raise ImportError(
            "Для обучения нужен catboost. Установите зависимости окружения проекта."
        )
    return CatBoostClassifier, CatBoostRegressor, Pool, EFeaturesSelectionAlgorithm, EShapCalcType


def require_model_diagnostics(config: TrainingConfig):
    """Импортировать ModelDiagnostics из внешнего проекта."""
    candidates: list[Path] = []
    if config.modeldiagnostics_root is not None:
        candidates.append(Path(config.modeldiagnostics_root))
    candidates.extend([PROJECT_ROOT.parent])
    if len(PROJECT_ROOT.parents) > 2:
        candidates.append(PROJECT_ROOT.parents[2])
    for path in candidates:
        if path.exists():
            sys.path.insert(0, str(path))
    module = importlib.import_module("modeldiagnostics.src.modeldiagnostics")
    return module.ModelDiagnostics


def _looks_numeric(value: object) -> bool:
    """True, если значение можно привести к float (для cat→str)."""
    if value is None or value is pd.NA:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return not pd.isna(value)
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def stringify_categorical_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Привести категориальные признаки к строкам (CatBoost не принимает float в cat).

    Целочисленные/бинарные float (0.0/1.0) → ``\"0\"``/``\"1\"``, не ``\"1.0\"``.
    """
    result = df.copy()
    for column in columns:
        if column not in result.columns:
            continue
        series = result[column]
        numeric = pd.to_numeric(series, errors="coerce")
        # Если все non-null — целые (в т.ч. 0.0/1.0) — пишем без десятичной точки
        finite = numeric.dropna()
        if not finite.empty and bool((finite == finite.round()).all()):
            as_int = numeric.round().astype("Int64")
            result[column] = as_int.astype(str).replace({"<NA>": "nan", "None": "nan"})
            continue
        try:
            result[column] = series.map(
                lambda value: (
                    "nan"
                    if value is None or (isinstance(value, float) and pd.isna(value))
                    or value is pd.NA
                    else str(int(float(value)))
                    if _looks_numeric(value)
                    else str(value)
                )
            )
        except (ValueError, TypeError):
            result[column] = series.astype(str).replace({"<NA>": "nan", "None": "nan"})
    return result


def make_pool(
    features: pd.DataFrame,
    label: pd.Series | np.ndarray | None = None,
    *,
    cat_features: list[str],
    feature_names: list[str] | None = None,
    weight: pd.Series | np.ndarray | None = None,
):
    """Pool с гарантированным stringify cat-колонок (защита от float 1.0)."""
    _, _, pool_cls, _, _ = require_catboost()

    names = feature_names or list(features.columns)
    data = stringify_categorical_columns(features[names], cat_features)
    kwargs: dict[str, object] = {
        "data": data,
        "cat_features": cat_features,
        "feature_names": names,
    }
    if label is not None:
        kwargs["label"] = label
    if weight is not None:
        kwargs["weight"] = weight
    return pool_cls(**kwargs)
