"""Обратная элиминация по метрике после noise-cut.

Сортируем фичи по importance ↓, снимаем с конца по одной до 1,
на каждом размере учим модель на train / метрика на Val.
Классификация — max PR-AUC; регрессия — min MAE.
При равенстве метрики оставляем более узкий набор.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error

TaskType = Literal["classification", "regression"]


@dataclass(frozen=True)
class BackwardElimStep:
    """Одна точка: размер пула → метрика на Val."""

    n_features: int
    metric: float
    features: tuple[str, ...]
    dropped_feature: str | None


@dataclass(frozen=True)
class BackwardElimResult:
    """Итог обратной элиминации."""

    selected_features: tuple[str, ...]
    best_metric: float
    metric_name: str
    ordered_features: tuple[str, ...]
    history: tuple[BackwardElimStep, ...]


def _cat_names(
    features: list[str],
    mvp_types: dict[str, tuple[str, ...]] | None,
) -> list[str]:
    if not mvp_types:
        return []
    cats = set(mvp_types.get("CATEGORIAL", ())) | set(mvp_types.get("BINARY", ()))
    return [name for name in features if name in cats]


def _prepare_xy(
    df: pd.DataFrame,
    features: list[str],
    target_column: str,
    train_index: pd.Index,
    eval_index: pd.Index,
    *,
    positive_target: bool,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    x_train = df.loc[train_index, features].copy()
    x_eval = df.loc[eval_index, features].copy()
    y_train = df.loc[train_index, target_column]
    y_eval = df.loc[eval_index, target_column]
    if positive_target:
        tr_ok = y_train > 0
        ev_ok = y_eval > 0
        x_train, y_train = x_train.loc[tr_ok], y_train.loc[tr_ok]
        x_eval, y_eval = x_eval.loc[ev_ok], y_eval.loc[ev_ok]
    if x_train.empty or x_eval.empty:
        raise ValueError("Пустой train/eval для backward elimination")
    return x_train, y_train, x_eval, y_eval


def _order_by_importance(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_eval: pd.DataFrame,
    y_eval: pd.Series,
    features: list[str],
    *,
    task_type: TaskType,
    cat_features: list[str],
    random_state: int,
    iterations: int,
    early_stopping_rounds: int,
) -> list[str]:
    """Порядок: важнее → раньше; с конца будем отрезать слабые."""
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    for col in cat_features:
        x_train[col] = x_train[col].astype(str)
        x_eval[col] = x_eval[col].astype(str)

    if task_type == "classification":
        y_tr = y_train.astype(int)
        y_ev = y_eval.astype(int)
        model = CatBoostClassifier(
            iterations=iterations,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
            auto_class_weights="Balanced",
            logging_level="Silent",
        )
    else:
        y_tr, y_ev = y_train, y_eval
        model = CatBoostRegressor(
            iterations=iterations,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
            logging_level="Silent",
        )

    train_pool = Pool(x_train[features], y_tr, cat_features=cat_features, feature_names=features)
    eval_pool = Pool(x_eval[features], y_ev, cat_features=cat_features, feature_names=features)
    model.fit(train_pool, eval_set=eval_pool, plot=False)
    scores = np.asarray(model.get_feature_importance(), dtype=float)
    ranked = (
        pd.DataFrame({"feature": features, "importance": scores})
        .sort_values(["importance", "feature"], ascending=[False, True])
    )
    return ranked["feature"].tolist()


def _eval_subset(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_eval: pd.DataFrame,
    y_eval: pd.Series,
    features: list[str],
    *,
    task_type: TaskType,
    mvp_types: dict[str, tuple[str, ...]] | None,
    random_state: int,
    iterations: int,
    early_stopping_rounds: int,
) -> float:
    """Метрика подмножества на Val: PR-AUC или MAE."""
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    cats = _cat_names(features, mvp_types)
    xt = x_train[features].copy()
    xe = x_eval[features].copy()
    for col in cats:
        xt[col] = xt[col].astype(str)
        xe[col] = xe[col].astype(str)

    if task_type == "classification":
        y_tr = y_train.astype(int)
        y_ev = y_eval.astype(int)
        model = CatBoostClassifier(
            iterations=iterations,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
            auto_class_weights="Balanced",
            logging_level="Silent",
        )
        train_pool = Pool(xt, y_tr, cat_features=cats, feature_names=features)
        eval_pool = Pool(xe, y_ev, cat_features=cats, feature_names=features)
        model.fit(train_pool, eval_set=eval_pool, plot=False)
        proba = model.predict_proba(eval_pool)[:, 1]
        return float(average_precision_score(y_ev, proba))

    model = CatBoostRegressor(
        iterations=iterations,
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
        logging_level="Silent",
    )
    train_pool = Pool(xt, y_train, cat_features=cats, feature_names=features)
    eval_pool = Pool(xe, y_eval, cat_features=cats, feature_names=features)
    model.fit(train_pool, eval_set=eval_pool, plot=False)
    pred = np.asarray(model.predict(eval_pool), dtype=float)
    return float(mean_absolute_error(y_eval, pred))


def backward_eliminate_by_metric(
    df: pd.DataFrame,
    *,
    features: list[str] | tuple[str, ...],
    target_column: str,
    train_index: pd.Index,
    eval_index: pd.Index,
    task_type: TaskType,
    mvp_types: dict[str, tuple[str, ...]] | None = None,
    positive_target: bool = False,
    random_state: int = 0,
    iterations: int = 100,
    early_stopping_rounds: int = 50,
    importance_order: list[str] | tuple[str, ...] | None = None,
) -> BackwardElimResult:
    """Снять фичи с конца (слабые) до 1; выбрать лучший набор по метрике."""
    feature_list = [f for f in features if f in df.columns]
    if not feature_list:
        raise ValueError("Пустой список признаков для backward elimination")
    if target_column not in df.columns:
        raise ValueError(f"Нет таргета {target_column}")

    x_train, y_train, x_eval, y_eval = _prepare_xy(
        df,
        feature_list,
        target_column,
        train_index,
        eval_index,
        positive_target=positive_target,
    )
    cats_all = _cat_names(feature_list, mvp_types)

    if importance_order is not None:
        ordered = [f for f in importance_order if f in feature_list]
        missing = [f for f in feature_list if f not in ordered]
        ordered = ordered + missing
    else:
        ordered = _order_by_importance(
            x_train,
            y_train,
            x_eval,
            y_eval,
            feature_list,
            task_type=task_type,
            cat_features=cats_all,
            random_state=random_state,
            iterations=iterations,
            early_stopping_rounds=early_stopping_rounds,
        )

    metric_name = "pr_auc" if task_type == "classification" else "mae"
    higher_is_better = task_type == "classification"
    history: list[BackwardElimStep] = []
    current = list(ordered)
    dropped_prev: str | None = None
    best_features = list(current)
    best_metric: float | None = None

    while current:
        metric = _eval_subset(
            x_train,
            y_train,
            x_eval,
            y_eval,
            current,
            task_type=task_type,
            mvp_types=mvp_types,
            random_state=random_state,
            iterations=iterations,
            early_stopping_rounds=early_stopping_rounds,
        )
        history.append(
            BackwardElimStep(
                n_features=len(current),
                metric=metric,
                features=tuple(current),
                dropped_feature=dropped_prev,
            )
        )
        # Строго лучше ИЛИ то же при меньшем n → обновляем (предпочитаем узкий набор)
        if best_metric is None:
            best_metric = metric
            best_features = list(current)
        elif higher_is_better and metric > best_metric:
            best_metric = metric
            best_features = list(current)
        elif (not higher_is_better) and metric < best_metric:
            best_metric = metric
            best_features = list(current)
        elif metric == best_metric and len(current) < len(best_features):
            best_features = list(current)

        if len(current) == 1:
            break
        dropped_prev = current[-1]
        current = current[:-1]

    return BackwardElimResult(
        selected_features=tuple(best_features),
        best_metric=float(best_metric if best_metric is not None else float("nan")),
        metric_name=metric_name,
        ordered_features=tuple(ordered),
        history=tuple(history),
    )
