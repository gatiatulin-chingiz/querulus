"""Пайплайн обучения моделей querulus."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
import importlib
import io
from pathlib import Path
import sys

import pandas as pd

from querulus import PROJECT_ROOT
from querulus.training.config import TrainingConfig


@dataclass
class DatasetSplit:
    """Train/test разбиение для одной цели."""

    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series


@dataclass
class TrainingArtifacts:
    """Результаты обучения моделей."""

    frequency_model: object
    severity_model: object
    metrics: dict[str, dict[str, float]]
    metrics_table: pd.DataFrame
    frequency_diagnostics: object
    severity_diagnostics: object
    feature_names: list[str]
    categorical_features: list[str]
    frequency_features: list[str]
    severity_features: list[str]
    frequency_categorical_features: list[str]
    severity_categorical_features: list[str]
    frequency_importance: pd.DataFrame
    severity_importance: pd.DataFrame


def _require_catboost():
    """Импортировать CatBoost только при запуске обучения."""
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    except ImportError as exc:
        raise ImportError(
            "Для обучения нужен catboost. Установите зависимости окружения проекта."
        ) from exc
    return CatBoostClassifier, CatBoostRegressor, Pool


def _require_model_diagnostics(config: TrainingConfig):
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


def _mvp_features(df: pd.DataFrame, config: TrainingConfig) -> tuple[list[str], list[str]]:
    """Определить типы признаков через AutoMVP."""
    try:
        from querulus.AutoMVP import MVP
    except Exception as exc:
        raise ImportError(
            "Не удалось импортировать querulus.AutoMVP.MVP. "
            "Проверьте, что AutoMVP.py является валидным Python-модулем."
        ) from exc

    mvp = MVP(df, print_col_type=False)
    with contextlib.redirect_stdout(io.StringIO()):
        mvp.value_type()
    other_cols = [config.date_column, *config.drop_columns]
    input_types = {key: list(value) for key, value in config.mvp_input_types.items()}
    mvp.correct_types(input_types, other_cols)
    types = mvp.types_dict
    features = [
        column
        for column in types["NUMERIC"] + types["CATEGORIAL"] + types["BINARY"]
        if column in df.columns and column not in config.drop_columns
    ]
    categorical = [
        column
        for column in types["CATEGORIAL"] + types["BINARY"]
        if column in features
    ]
    return features, categorical


def _prepare_catboost_frame(df: pd.DataFrame, cat_features: list[str]) -> pd.DataFrame:
    """Привести категориальные признаки к строкам, чтобы CatBoost принял данные."""
    result = df.copy()
    for column in cat_features:
        if column in result.columns:
            result[column] = result[column].fillna("__missing__").astype(str)
    return result


def _select_model_features(
    available_features: list[str],
    available_cat_features: list[str],
    requested_features: tuple[str, ...] | None,
    model_name: str,
) -> tuple[list[str], list[str]]:
    """Выбрать признаки для конкретной модели или использовать все MVP-признаки."""
    if requested_features is None:
        features = available_features
    else:
        missing = [column for column in requested_features if column not in available_features]
        if missing:
            raise ValueError(
                f"Для модели {model_name!r} указаны неизвестные признаки: {missing}"
            )
        features = list(requested_features)
    cat_features = [column for column in available_cat_features if column in features]
    return features, cat_features


def _split_by_date(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    config: TrainingConfig,
    *,
    target_range: tuple[float, float] | None = None,
) -> DatasetSplit:
    """Разбить датасет на train/test по периоду."""
    data = df.copy()
    data[config.date_column] = pd.to_datetime(data[config.date_column])
    if target_range is not None:
        data = data[data[target].between(*target_range)]

    train_mask = data[config.date_column].between(*config.train_period)
    test_mask = data[config.date_column].between(*config.test_period)
    return DatasetSplit(
        x_train=data.loc[train_mask, features],
        y_train=data.loc[train_mask, target],
        x_test=data.loc[test_mask, features],
        y_test=data.loc[test_mask, target],
    )


def _importance_frame(model: object, feature_names: list[str]) -> pd.DataFrame:
    """Сформировать таблицу важности признаков CatBoost."""
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.get_feature_importance(),
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _diagnostics_metrics(
    ModelDiagnostics,
    split: DatasetSplit,
    model: object,
    features: list[str],
    cat_features: list[str],
    task_type: str,
) -> tuple[object, dict[str, dict[str, float]]]:
    """Получить метрики через ModelDiagnostics."""
    diagnostics = ModelDiagnostics(
        X_train=split.x_train,
        y_train=split.y_train,
        X_test=split.x_test,
        y_test=split.y_test,
        model=model,
        features=features,
        cat_features=cat_features,
        task_type=task_type,
    )
    train_metrics, test_metrics = diagnostics.compute_metrics(print_metrics=False)
    return diagnostics, {"train": train_metrics, "test": test_metrics}


def _metrics_table(metrics: dict[str, dict[str, dict[str, float]]]) -> pd.DataFrame:
    """Вернуть метрики ModelDiagnostics в табличном виде для ноутбука."""
    rows = [
        {
            "metric": metric,
            f"{model_name}_{sample}": value,
        }
        for model_name, model_metrics in metrics.items()
        for sample, sample_metrics in model_metrics.items()
        for metric, value in sample_metrics.items()
    ]
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows).groupby("metric", as_index=False).first()
    ordered_columns = [
        "metric",
        "frequency_train",
        "frequency_test",
        "severity_train",
        "severity_test",
    ]
    existing_columns = [column for column in ordered_columns if column in table.columns]
    extra_columns = [column for column in table.columns if column not in existing_columns]
    return table[existing_columns + extra_columns].sort_values("metric").reset_index(drop=True)


def train_models(df: pd.DataFrame, config: TrainingConfig | None = None) -> TrainingArtifacts:
    """Обучить модели частоты (`TARGET_2`) и тяжести (`TARGET_3_SEV`)."""
    config = config or TrainingConfig()
    features, cat_features = _mvp_features(df, config)
    data = _prepare_catboost_frame(df, cat_features)
    frequency_features, frequency_cat_features = _select_model_features(
        features,
        cat_features,
        config.frequency_features,
        "frequency",
    )
    severity_features, severity_cat_features = _select_model_features(
        features,
        cat_features,
        config.severity_features,
        "severity",
    )

    CatBoostClassifier, CatBoostRegressor, Pool = _require_catboost()
    ModelDiagnostics = _require_model_diagnostics(config)
    frequency_cat_feature_indices = [
        frequency_features.index(column) for column in frequency_cat_features
    ]
    severity_cat_feature_indices = [
        severity_features.index(column) for column in severity_cat_features
    ]

    frequency_split = _split_by_date(data, config.frequency_target, frequency_features, config)
    frequency_pool = Pool(
        frequency_split.x_train,
        frequency_split.y_train.astype(int),
        cat_features=frequency_cat_feature_indices,
        feature_names=frequency_features,
    )
    frequency_eval_pool = Pool(
        frequency_split.x_test,
        frequency_split.y_test.astype(int),
        cat_features=frequency_cat_feature_indices,
        feature_names=frequency_features,
    )
    frequency_model = CatBoostClassifier(
        iterations=config.iterations,
        random_state=config.random_state,
        auto_class_weights="Balanced",
        verbose=250,
    )
    frequency_model.fit(
        frequency_pool,
        eval_set=frequency_eval_pool,
        plot=False,
    )

    severity_split = _split_by_date(
        data,
        config.severity_target,
        severity_features,
        config,
        target_range=config.severity_range,
    )
    severity_pool = Pool(
        severity_split.x_train,
        severity_split.y_train,
        cat_features=severity_cat_feature_indices,
        feature_names=severity_features,
    )
    severity_eval_pool = Pool(
        severity_split.x_test,
        severity_split.y_test,
        cat_features=severity_cat_feature_indices,
        feature_names=severity_features,
    )
    severity_model = CatBoostRegressor(
        iterations=config.iterations,
        random_state=config.random_state,
        verbose=250,
    )
    severity_model.fit(
        severity_pool,
        eval_set=severity_eval_pool,
        plot=False,
    )

    frequency_diagnostics, frequency_metrics = _diagnostics_metrics(
        ModelDiagnostics,
        frequency_split,
        frequency_model,
        frequency_features,
        frequency_cat_features,
        "classification",
    )
    severity_diagnostics, severity_metrics = _diagnostics_metrics(
        ModelDiagnostics,
        severity_split,
        severity_model,
        severity_features,
        severity_cat_features,
        "regression",
    )
    metrics = {"frequency": frequency_metrics, "severity": severity_metrics}

    return TrainingArtifacts(
        frequency_model=frequency_model,
        severity_model=severity_model,
        metrics=metrics,
        metrics_table=_metrics_table(metrics),
        frequency_diagnostics=frequency_diagnostics,
        severity_diagnostics=severity_diagnostics,
        feature_names=features,
        categorical_features=cat_features,
        frequency_features=frequency_features,
        severity_features=severity_features,
        frequency_categorical_features=frequency_cat_features,
        severity_categorical_features=severity_cat_features,
        frequency_importance=_importance_frame(frequency_model, frequency_features),
        severity_importance=_importance_frame(severity_model, severity_features),
    )
