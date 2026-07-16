"""Обучение severity: log-таргет, веса, predict с обратным преобразованием."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from querulus.training.config import TrainingConfig
from querulus.training.pipeline import TrainingArtifacts, _require_catboost

SeverityTargetTransform = Literal["raw", "log1p"]
SeveritySampleWeight = Literal["none", "sqrt", "linear"]


def severity_train_target(
    y: np.ndarray | pd.Series,
    transform: SeverityTargetTransform,
) -> np.ndarray:
    """Таргет для fit CatBoost severity."""
    arr = np.asarray(y, dtype=float)
    if transform == "log1p":
        return np.log1p(np.clip(arr, a_min=0.0, a_max=None))
    return arr


def severity_sample_weights(
    y: np.ndarray | pd.Series,
    mode: SeveritySampleWeight,
) -> np.ndarray | None:
    """Веса обучения: сильнее штраф на крупные суммы (по raw y)."""
    if mode == "none":
        return None
    arr = np.clip(np.asarray(y, dtype=float), a_min=0.0, a_max=None)
    if mode == "sqrt":
        weights = np.sqrt(arr)
    elif mode == "linear":
        weights = arr
    else:
        raise ValueError(f"Unknown severity sample weight mode: {mode!r}")
    weights = np.where(weights > 0, weights, 1.0)
    return weights


def severity_predict(
    model: object,
    features: pd.DataFrame,
    cat_features: list[str],
    *,
    transform: SeverityTargetTransform = "raw",
) -> np.ndarray:
    """Предсказание severity с учётом log1p-таргета."""
    cat_features = [column for column in cat_features if column in features.columns]
    if cat_features:
        from catboost import Pool

        pool = Pool(features, cat_features=cat_features)
        raw = np.asarray(model.predict(pool), dtype=float)
    else:
        raw = np.asarray(model.predict(features), dtype=float)
    if transform == "log1p":
        return np.clip(np.expm1(raw), a_min=0.0, a_max=None)
    return raw


def fit_severity_model(
    training: TrainingArtifacts,
    config: TrainingConfig,
    *,
    transform: SeverityTargetTransform = "raw",
    sample_weight: SeveritySampleWeight = "none",
    train_index: pd.Index | None = None,
    eval_index: pd.Index | None = None,
) -> object:
    """Обучить severity на сплите ``training`` (опционально подмножество индексов)."""
    if training.severity_split is None:
        raise ValueError("severity_split отсутствует в TrainingArtifacts")
    _, CatBoostRegressor, Pool, *_ = _require_catboost()

    split = training.severity_split
    features = training.severity_features
    cat_features = training.severity_categorical_features

    x_train = split.x_train
    y_train = split.y_train
    x_test = split.x_test
    y_test = split.y_test
    if train_index is not None:
        idx = x_train.index.intersection(train_index)
        x_train = x_train.loc[idx]
        y_train = y_train.loc[idx]
    if eval_index is not None:
        idx = x_test.index.intersection(eval_index)
        x_test = x_test.loc[idx]
        y_test = y_test.loc[idx]

    y_train_arr = np.asarray(y_train, dtype=float)
    y_test_arr = np.asarray(y_test, dtype=float)
    w_train = severity_sample_weights(y_train_arr, sample_weight)

    train_pool = Pool(
        x_train[features],
        severity_train_target(y_train_arr, transform),
        cat_features=cat_features,
        feature_names=features,
        weight=w_train,
    )
    fit_kwargs: dict[str, object] = {"plot": False}
    if len(x_test) > 0:
        test_pool = Pool(
            x_test[features],
            severity_train_target(y_test_arr, transform),
            cat_features=cat_features,
            feature_names=features,
        )
        fit_kwargs["eval_set"] = test_pool

    model = CatBoostRegressor(
        iterations=config.severity_iterations,
        random_state=config.severity_random_state,
        **config.severity_regressor_params,
    )
    model.fit(train_pool, **fit_kwargs)
    return model
