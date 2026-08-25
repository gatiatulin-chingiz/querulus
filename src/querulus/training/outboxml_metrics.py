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


def _model_config_without_row_filter(model_config: Any) -> Any:
    """Копия model_config без ``data_filter_condition`` (для score на полном index)."""
    if getattr(model_config, "data_filter_condition", None) in (None, ""):
        return model_config
    if hasattr(model_config, "model_copy"):
        return model_config.model_copy(update={"data_filter_condition": None})
    if hasattr(model_config, "copy"):
        try:
            return model_config.copy(update={"data_filter_condition": None})
        except TypeError:
            pass
    raise TypeError(
        "Не удалось снять data_filter_condition с model_config; "
        "нужен pydantic model_copy/copy(update=...)."
    )


def prepare_dsm_features(
    dsm: Any,
    model_name: str,
    data: pd.DataFrame,
    *,
    ignore_row_filter: bool = False,
) -> pd.DataFrame:
    """Признаки DSM для predict; ``ignore_row_filter=True`` — без query фильтра обучения.

    Severity в OutBoxML обычно с ``TARGET_SEV > 0``: без ``ignore_row_filter``
    на полном Test остаются только позитивы → финэффект/F1 ломаются.
    """
    from outboxml.core.prepared_datasets import PrepareDataset
    from outboxml.datasets_manager import DataPreprocessor

    result = dsm.get_result()[model_name]
    model_config = result.model_config
    if ignore_row_filter:
        model_config = _model_config_without_row_filter(model_config)

    preproc = DataPreprocessor(
        prepare_dataset_interface_dict={
            model_name: PrepareDataset(
                model_config=model_config, check_prepared=False
            )
        },
        dataset=data.copy(),
        data_config=dsm.data_config,
        prepare_engine="pandas",
    )
    subset = preproc.get_subset(model_name, from_pickle=False)
    num = list(result.data_subset.features_numerical or [])
    cat = list(result.data_subset.features_categorical or [])
    cols = [c for c in num + cat if c in subset.X.columns]
    return subset.X[cols]


def predict_dsm_series(
    dsm: Any,
    model_name: str,
    data: pd.DataFrame,
    *,
    task_type: TaskKind,
    calibrator: Any | None = None,
    ignore_row_filter: bool = False,
) -> pd.Series:
    """Предсказания DSM → Series на index подготовленного X."""
    X = prepare_dsm_features(
        dsm, model_name, data, ignore_row_filter=ignore_row_filter
    )
    estimator = unwrap_estimator(dsm.get_result()[model_name].model)
    if task_type == "classification":
        if calibrator is not None:
            scores = np.asarray(calibrator.predict_proba(X)[:, 1], dtype=float)
        else:
            scores = _predict_scores(estimator, X, task_type="classification")
    else:
        scores = _predict_scores(estimator, X, task_type="regression")
    return pd.Series(scores, index=X.index, dtype=float)


def _is_categorical_series(series: pd.Series) -> bool:
    return isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_categorical_dtype(
        series.dtype
    )


def _model_feature_names(X: pd.DataFrame, estimator: Any) -> list[str]:
    """Порядок признаков из модели или исходного кадра для других estimator."""
    names = list(getattr(estimator, "feature_names_", None) or [])
    if names and all(name in X.columns for name in names):
        return names
    return list(X.columns)


def _model_cat_feature_names(
    feature_names: list[str],
    estimator: Any,
) -> list[str]:
    """Имена cat-колонок строго по схеме обученной CatBoost-модели.

    ``subset.features_categorical`` сюда не мешаем: там могут быть Float-фичи
    модели (напр. APPLICANT_AGE) → stringify даёт
    ``Float in model but marked different in the dataset``.
    """
    getter = getattr(estimator, "_get_cat_feature_indices", None)
    if not callable(getter):
        return []
    try:
        indices = list(getter())
    except Exception:
        return []
    names: list[str] = []
    for idx in indices:
        if isinstance(idx, (int, np.integer)) and 0 <= int(idx) < len(feature_names):
            names.append(feature_names[int(idx)])
    return names


def _frame_for_catboost_predict(X: pd.DataFrame, estimator: Any) -> pd.DataFrame:
    """Подготовка кадра под CatBoost predict.

    - Только cat-индексы модели → string (не float).
    - Остальные category/object → numeric (совпадение с Float в модели).
    """
    from querulus.training.pipeline import _stringify_categorical_columns

    feature_names = _model_feature_names(X, estimator)
    out = X.loc[:, feature_names].copy()
    cat_names = _model_cat_feature_names(feature_names, estimator)
    cat_set = set(cat_names)

    for column in out.columns:
        if column in cat_set:
            continue
        series = out[column]
        if (
            _is_categorical_series(series)
            or pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
        ):
            out[column] = pd.to_numeric(series, errors="coerce")

    if cat_names:
        out = _stringify_categorical_columns(out, cat_names)
    return out


def _predict_scores(
    estimator: Any,
    X: pd.DataFrame,
    *,
    task_type: TaskKind,
) -> np.ndarray:
    frame = _frame_for_catboost_predict(X, estimator)
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

    split_metrics: dict[str, dict[str, float]] = {}
    for split_name, x_split, y_split in (
        ("train", subset.X_train, subset.y_train),
        ("test", subset.X_test, subset.y_test),
    ):
        scores = _predict_scores(estimator, x_split, task_type=task_type)
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
