"""Метрики DSM в стиле collect, без правок outboxml и без modeldiagnostics.

OutBoxML ``BaseMetrics`` отдаёт урезанный набор (clf: f1/precision/recall/gini;
reg: mae/rmse/r2). Здесь считаем полный bundle как в ``train_loop`` / collect
и подставляем в ``result.metrics[*]['full']``.

Classification: один порог с Val (``val_threshold``), без 0.5 по умолчанию.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from querulus.fin_effect.threshold_policy import ValThresholdResult, pick_threshold_on_val
from querulus.training.build_outboxml_configs import (
    ensure_predictable_model,
    unwrap_estimator,
)
from querulus.training.collect_metrics import metrics_bundle as _collect_metrics_bundle
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


def pick_threshold_on_val_dsm(
    dsm_cf: Any,
    cf_name: str,
    dsm_rg: Any,
    rg_name: str,
    df: pd.DataFrame,
    val_index: pd.Index,
    *,
    frequency_target: str = "TARGET_FREQ",
    config: Any | None = None,
) -> ValThresholdResult:
    """Подбор порога frequency на Val для пары DSM cf/rg (OutBoxML)."""
    index = pd.Index(val_index).intersection(df.index)
    proba = predict_dsm_series(
        dsm_cf, cf_name, df.loc[index], task_type="classification"
    )
    sev = predict_dsm_series(
        dsm_rg,
        rg_name,
        df.loc[index],
        task_type="regression",
        ignore_row_filter=True,
    )
    y_true = df.loc[index, frequency_target]
    return pick_threshold_on_val(
        df,
        index,
        proba,
        sev,
        y_true,
        config=config,
    )


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
    """Имена cat-колонок строго по схеме обученной CatBoost-модели."""
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
    """Подготовка кадра под CatBoost predict."""
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
    val_threshold: float | None = None,
) -> dict[str, float]:
    """Один сплит: те же ключи, что у collect (PR-AUC, MCC, … / MAE, …)."""
    if task_type == "classification" and val_threshold is None:
        raise ValueError("classification: нужен val_threshold (подбор на Val)")
    threshold = None if task_type == "regression" else float(val_threshold)
    return _collect_metrics_bundle(
        y_true, y_score, task_type=task_type, threshold=threshold
    )


def enrich_dsm_model_metrics(
    dsm: Any,
    model_name: str,
    *,
    task_type: TaskKind,
    val_threshold: float | None = None,
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
        bundle = collect_style_metrics_bundle(
            y_split,
            scores,
            task_type=task_type,
            val_threshold=val_threshold,
        )
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


def metrics_bundle_on_index(
    dsm: Any,
    model_name: str,
    df: pd.DataFrame,
    index: pd.Index,
    *,
    task_type: TaskKind,
    val_threshold: float | None = None,
    ignore_row_filter: bool = False,
) -> tuple[dict[str, float], int]:
    """Collect-style метрики на произвольном index (модель уже обучена)."""
    result = dsm.get_result()[model_name]
    target = result.model_config.column_target
    preds = predict_dsm_series(
        dsm,
        model_name,
        df.loc[index],
        task_type=task_type,
        ignore_row_filter=ignore_row_filter,
    )
    y_true = df.loc[preds.index, target]
    bundle = collect_style_metrics_bundle(
        y_true,
        preds.to_numpy(),
        task_type=task_type,
        val_threshold=val_threshold,
    )
    return bundle, int(len(preds))


def display_dsm_collect_metrics_cross_test(
    dsm: Any,
    model_name: str,
    df: pd.DataFrame,
    *,
    test_slices: dict[str, pd.Index],
    task_type: TaskKind,
    val_threshold: float | None = None,
    title: str | None = None,
    ignore_row_filter: bool = False,
) -> pd.DataFrame:
    """Одна модель (parity train): метрики train + несколько test-срезов."""
    from IPython.display import Markdown, display

    result = dsm.get_result()[model_name]
    result.model = ensure_predictable_model(result.model)
    estimator = unwrap_estimator(result.model)
    subset = result.data_subset

    bundles: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}

    train_scores = _predict_scores(estimator, subset.X_train, task_type=task_type)
    bundles["train"] = collect_style_metrics_bundle(
        subset.y_train,
        train_scores,
        task_type=task_type,
        val_threshold=val_threshold,
    )
    counts["train"] = int(len(subset.y_train))

    for slice_name, index in test_slices.items():
        bundle, n_rows = metrics_bundle_on_index(
            dsm,
            model_name,
            df,
            index,
            task_type=task_type,
            val_threshold=val_threshold,
            ignore_row_filter=ignore_row_filter,
        )
        bundles[slice_name] = bundle
        counts[slice_name] = n_rows

    table = _model_metrics_table(bundles)
    if "metric" in table.columns:
        table = table.loc[table["metric"].astype(str) != "ece"].reset_index(drop=True)

    thr_note = (
        f", порог Val={val_threshold:.2f}"
        if task_type == "classification" and val_threshold is not None
        else ""
    )
    label = title or f"{model_name} ({task_type}) cross-test"
    display(Markdown(f"### Метрики collect-style: {label}{thr_note}"))
    display(
        Markdown(
            "Строки после фильтра модели (severity: `TARGET_SEV > 0`): "
            + ", ".join(f"**{name}**={counts[name]:,}" for name in bundles)
        )
    )
    display(format_metrics_table(table))
    return table


def display_dsm_collect_metrics(
    dsm: Any,
    model_name: str,
    *,
    task_type: TaskKind,
    val_threshold: float | None = None,
    title: str | None = None,
) -> pd.DataFrame:
    """Enrich + человекочитаемая таблица (для ноутбука)."""
    from IPython.display import Markdown, display

    table = enrich_dsm_model_metrics(
        dsm, model_name, task_type=task_type, val_threshold=val_threshold
    )
    if "metric" in table.columns:
        table = table.loc[table["metric"].astype(str) != "ece"].reset_index(drop=True)
    thr_note = (
        f", порог Val={val_threshold:.2f}"
        if task_type == "classification" and val_threshold is not None
        else ""
    )
    label = title or f"{model_name} ({task_type})"
    display(Markdown(f"### Метрики collect-style: {label}{thr_note}"))
    display(format_metrics_table(table))
    raw = dsm.get_result()[model_name].metrics
    print(
        "raw full:",
        {
            k: {
                mk: mv
                for mk, mv in ((v or {}).get("full") or {}).items()
                if mk != "ece"
            }
            for k, v in (raw or {}).items()
        },
    )
    return table
