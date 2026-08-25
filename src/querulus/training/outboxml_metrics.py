"""Метрики DSM в стиле collect (ModelDiagnostics / HPO), без правок outboxml.

OutBoxML ``BaseMetrics`` отдаёт урезанный набор (clf: f1/precision/recall/gini;
reg: mae/rmse/r2). Здесь считаем полный bundle как в ``train_loop`` / collect
и подставляем в ``result.metrics[*]['full']``.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from querulus.training.build_outboxml_configs import (
    ensure_predictable_model,
    unwrap_estimator,
)
from querulus.training.hpo import _fold_metrics_bundle
from querulus.training.pipeline import _model_metrics_table, format_metrics_table

TaskKind = Literal["classification", "regression"]


def _is_categorical_series(series: pd.Series) -> bool:
    return isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_categorical_dtype(
        series.dtype
    )


def _model_cat_feature_names(
    X: pd.DataFrame,
    estimator: Any,
    cat_features: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Имена cat-колонок: индексы модели + имена из DSM subset."""
    names: set[str] = set()
    if cat_features:
        names.update(name for name in cat_features if name in X.columns)
    getter = getattr(estimator, "_get_cat_feature_indices", None)
    if callable(getter):
        try:
            indices = list(getter())
        except Exception:
            indices = []
        columns = list(X.columns)
        for idx in indices:
            if isinstance(idx, (int, np.integer)) and 0 <= int(idx) < len(columns):
                names.add(columns[int(idx)])
    return list(names)


def _frame_for_catboost_predict(
    X: pd.DataFrame,
    estimator: Any,
    cat_features: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Подготовка кадра под CatBoost predict.

    - Колонки cat_features модели → string/int (не float): иначе
      ``cat_features must be integer or string``.
    - Остальные ``category`` → numeric: иначе
      ``dtype 'category' but is not in cat_features list``.
    """
    from querulus.training.pipeline import _stringify_categorical_columns

    out = X.copy()
    cat_names = _model_cat_feature_names(out, estimator, cat_features)
    cat_set = set(cat_names)

    for column in out.columns:
        if column in cat_set:
            continue
        series = out[column]
        if _is_categorical_series(series):
            out[column] = pd.to_numeric(series, errors="coerce")

    if cat_names:
        out = _stringify_categorical_columns(out, cat_names)
    return out


def _predict_scores(
    estimator: Any,
    X: pd.DataFrame,
    *,
    task_type: TaskKind,
    cat_features: list[str] | tuple[str, ...] | None = None,
) -> np.ndarray:
    frame = _frame_for_catboost_predict(X, estimator, cat_features)
    if task_type == "classification":
        if hasattr(estimator, "predict_proba"):
            return np.asarray(estimator.predict_proba(frame)[:, 1], dtype=float)
        return np.asarray(estimator.predict(frame), dtype=float)
    return np.asarray(estimator.predict(frame), dtype=float)


def collect_style_metrics_bundle(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    *,
    task_type: TaskKind,
) -> dict[str, float]:
    """Один сплит: те же ключи, что у collect (PR-AUC, ECE, MCC, … / MAE, …)."""
    return _fold_metrics_bundle(y_true, y_score, task_type=task_type)


def enrich_dsm_model_metrics(
    dsm: Any,
    model_name: str,
    *,
    task_type: TaskKind,
) -> pd.DataFrame:
    """Пересчитать full-метрики train/test для модели DSM и вернуть таблицу.

    Обновляет ``dsm.get_result()[model_name].metrics['train'|'test']['full']``.
    """
    result = dsm.get_result()[model_name]
    result.model = ensure_predictable_model(result.model)
    estimator = unwrap_estimator(result.model)
    subset = result.data_subset
    cat_features = list(subset.features_categorical or [])

    split_metrics: dict[str, dict[str, float]] = {}
    for split_name, x_split, y_split in (
        ("train", subset.X_train, subset.y_train),
        ("test", subset.X_test, subset.y_test),
    ):
        scores = _predict_scores(
            estimator,
            x_split,
            task_type=task_type,
            cat_features=cat_features,
        )
        bundle = collect_style_metrics_bundle(y_split, scores, task_type=task_type)
        split_metrics[split_name] = bundle
        if result.metrics is None:
            result.metrics = {"train": {}, "test": {}}
        bucket = result.metrics.setdefault(split_name, {})
        if not isinstance(bucket, dict):
            bucket = {}
            result.metrics[split_name] = bucket
        bucket["full"] = dict(bundle)

    table = _model_metrics_table(
        {
            "train": split_metrics.get("train", {}),
            "val": {},
            "test": split_metrics.get("test", {}),
        }
    )
    return table


def display_dsm_collect_metrics(
    dsm: Any,
    model_name: str,
    *,
    task_type: TaskKind,
    title: str | None = None,
) -> pd.DataFrame:
    """Enrich + человекочитаемая таблица (для ноутбука)."""
    from IPython.display import Markdown, display

    table = enrich_dsm_model_metrics(dsm, model_name, task_type=task_type)
    label = title or f"{model_name} ({task_type})"
    display(Markdown(f"### Метрики collect-style: {label}"))
    display(format_metrics_table(table))
    raw = dsm.get_result()[model_name].metrics
    print("raw full:", {k: (v or {}).get("full") for k, v in (raw or {}).items()})
    return table
