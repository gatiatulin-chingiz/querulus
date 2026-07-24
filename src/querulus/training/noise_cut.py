"""Отсечение признаков слабее шумовой фичи после SHAP feature selection.

После отбора: добавляем ``FE_NOISE_UNIFORM``, коротко учим CatBoost,
режем всё с importance ≤ шума (и сам шум). Если шум последний — только его.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

NOISE_FEATURE_NAME = "FE_NOISE_UNIFORM"

TaskType = Literal["classification", "regression"]


@dataclass(frozen=True)
class NoiseCutResult:
    """Результат noise-cut по отобранным признакам."""

    kept_features: tuple[str, ...]
    dropped_below_noise: tuple[str, ...]
    noise_feature: str
    noise_rank: int
    noise_was_last: bool
    importances: pd.DataFrame


def _cat_names(
    features: list[str],
    mvp_types: dict[str, tuple[str, ...]] | None,
) -> list[str]:
    if not mvp_types:
        return []
    cats = set(mvp_types.get("CATEGORIAL", ())) | set(mvp_types.get("BINARY", ()))
    return [name for name in features if name in cats]


def filter_features_by_noise(
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
    noise_column: str = NOISE_FEATURE_NAME,
) -> NoiseCutResult:
    """Оставить фичи строго важнее шума; шум в итог не входит."""
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    feature_list = [f for f in features if f in df.columns]
    if not feature_list:
        raise ValueError("Пустой список признаков для noise-cut")
    if target_column not in df.columns:
        raise ValueError(f"Нет таргета {target_column}")
    if noise_column in feature_list:
        raise ValueError(f"Колонка шума уже в пуле: {noise_column}")

    x_train = df.loc[train_index, feature_list].copy()
    x_eval = df.loc[eval_index, feature_list].copy()
    y_train = df.loc[train_index, target_column]
    y_eval = df.loc[eval_index, target_column]

    if positive_target:
        tr_ok = y_train > 0
        ev_ok = y_eval > 0
        x_train, y_train = x_train.loc[tr_ok], y_train.loc[tr_ok]
        x_eval, y_eval = x_eval.loc[ev_ok], y_eval.loc[ev_ok]

    if x_train.empty or x_eval.empty:
        raise ValueError("Пустой train/eval после фильтра для noise-cut")

    rng = np.random.default_rng(random_state)
    x_train[noise_column] = rng.uniform(0.0, 1.0, size=len(x_train))
    x_eval[noise_column] = rng.uniform(0.0, 1.0, size=len(x_eval))

    with_noise = [*feature_list, noise_column]
    cat_features = _cat_names(feature_list, mvp_types)
    for col in cat_features:
        x_train[col] = x_train[col].astype(str)
        x_eval[col] = x_eval[col].astype(str)

    if task_type == "classification":
        y_train = y_train.astype(int)
        y_eval = y_eval.astype(int)
        model = CatBoostClassifier(
            iterations=iterations,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
            auto_class_weights="Balanced",
            logging_level="Silent",
        )
    else:
        model = CatBoostRegressor(
            iterations=iterations,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
            logging_level="Silent",
        )

    train_pool = Pool(
        x_train[with_noise],
        y_train,
        cat_features=cat_features,
        feature_names=with_noise,
    )
    eval_pool = Pool(
        x_eval[with_noise],
        y_eval,
        cat_features=cat_features,
        feature_names=with_noise,
    )
    model.fit(train_pool, eval_set=eval_pool, plot=False)

    # Ранг по PredictionValuesChange: выше = важнее.
    importances = (
        pd.DataFrame(
            {
                "feature": with_noise,
                "importance": np.asarray(model.get_feature_importance(), dtype=float),
            }
        )
        .sort_values(["importance", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )
    importances["rank"] = np.arange(1, len(importances) + 1)

    noise_rows = importances.index[importances["feature"] == noise_column]
    if len(noise_rows) != 1:
        raise RuntimeError(f"Шум {noise_column} не найден в importance")
    noise_pos = int(noise_rows[0])
    noise_rank = int(importances.loc[noise_pos, "rank"])
    noise_was_last = noise_pos == len(importances) - 1

    if noise_was_last:
        kept = tuple(feature_list)
        dropped: tuple[str, ...] = ()
    else:
        above = set(importances.iloc[:noise_pos]["feature"].tolist())
        kept = tuple(f for f in feature_list if f in above)
        dropped = tuple(f for f in feature_list if f not in above)

    return NoiseCutResult(
        kept_features=kept,
        dropped_below_noise=dropped,
        noise_feature=noise_column,
        noise_rank=noise_rank,
        noise_was_last=noise_was_last,
        importances=importances,
    )
