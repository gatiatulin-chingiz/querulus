"""Пайплайн обучения моделей querulus."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
import importlib
import io
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from querulus import PROJECT_ROOT
from querulus.features.config import is_fe_categorical
from querulus.training.config import TrainingConfig, resolve_features_config
from querulus.training.severity_training import (
    severity_predict,
    severity_sample_weights,
    severity_train_target,
)

logger = logging.getLogger("querulus.training")


@dataclass
class DatasetSplit:
    """Train/test разбиение для одной цели."""

    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series


@dataclass
class ModelTrainingReport:
    """Сводка по обучению одной модели."""

    model: str
    target: str
    train_period: tuple[str, str]
    test_period: tuple[str, str]
    target_filter: str | None
    train_rows: int
    test_rows: int
    train_target_mean: float | None
    test_target_mean: float | None
    features: list[str]
    cat_features: list[str]
    hyperparameters: dict[str, object]


@dataclass
class TrainingSummary:
    """Сводка по пайплайну обучения frequency + severity."""

    date_column: str
    mvp_feature_count: int
    mvp_categorical_count: int
    frequency: ModelTrainingReport
    severity: ModelTrainingReport


@dataclass
class TrainingArtifacts:
    """Результаты обучения моделей."""

    frequency_model: object
    severity_model: object
    metrics: dict[str, dict[str, dict[str, float]]]
    frequency_metrics_table: pd.DataFrame
    severity_metrics_table: pd.DataFrame
    frequency_diagnostics: object
    severity_diagnostics: object
    summary: TrainingSummary
    feature_names: list[str]
    categorical_features: list[str]
    frequency_features: list[str]
    severity_features: list[str]
    frequency_categorical_features: list[str]
    severity_categorical_features: list[str]
    frequency_importance: pd.DataFrame
    severity_importance: pd.DataFrame
    frequency_split: DatasetSplit | None = None
    severity_split: DatasetSplit | None = None
    feature_frame: pd.DataFrame | None = None
    frequency_calibrator: object | None = None
    frequency_feature_selection_summary: dict[str, object] | None = None
    severity_feature_selection_summary: dict[str, object] | None = None
    severity_target_transform: str = "raw"


def _require_catboost():
    """Импортировать CatBoost только при запуске обучения."""
    try:
        from catboost import (
            CatBoostClassifier,
            CatBoostRegressor,
            EFeaturesSelectionAlgorithm,
            EShapCalcType,
            Pool,
        )
    except ImportError as exc:
        raise ImportError(
            "Для обучения нужен catboost. Установите зависимости окружения проекта."
        ) from exc
    return CatBoostClassifier, CatBoostRegressor, Pool, EFeaturesSelectionAlgorithm, EShapCalcType


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


def _stringify_categorical_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Привести категориальные признаки к строкам, как в model_learn.py."""
    result = df.copy()
    for column in columns:
        if column not in result.columns:
            continue
        try:
            result[column] = result[column].apply(lambda value: int(float(value))).astype(str)
        except (ValueError, TypeError):
            result[column] = result[column].astype(str)
    return result


def _fe_categorical_in_frame(df: pd.DataFrame) -> list[str]:
    """Категориальные FE_* колонки, присутствующие во фрейме."""
    return [column for column in df.columns if is_fe_categorical(column)]


def _merge_fe_categorical_types(input_types: dict[str, list[str]], fe_cat: list[str]) -> dict[str, list[str]]:
    """Добавить FE-бакеты в CATEGORIAL и убрать из NUMERIC."""
    merged = {key: list(value) for key, value in input_types.items()}
    categorical = list(dict.fromkeys(merged.get("CATEGORIAL", []) + fe_cat))
    numeric = [column for column in merged.get("NUMERIC", []) if column not in fe_cat]
    merged["CATEGORIAL"] = categorical
    merged["NUMERIC"] = numeric
    return merged


def _apply_mvp_types(
    df: pd.DataFrame,
    config: TrainingConfig,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """value_type → stringify cat → correct_types с учётом FE-бакетов."""
    try:
        from querulus.AutoMVP import MVP
    except Exception as exc:
        raise ImportError(
            "Не удалось импортировать querulus.AutoMVP.MVP. "
            "Проверьте, что AutoMVP.py является валидным Python-модулем."
        ) from exc

    fe_cat = _fe_categorical_in_frame(df)
    mvp = MVP(df, print_col_type=False, cutoff_nan=config.mvp_cutoff_nan)
    with contextlib.redirect_stdout(io.StringIO()):
        mvp.value_type()

    initial_categorical = list(
        dict.fromkeys(mvp.types_dict["BINARY"] + mvp.types_dict["CATEGORIAL"] + fe_cat)
    )
    data = _stringify_categorical_columns(df, initial_categorical)

    other_cols = [config.date_column, *config.drop_columns]
    input_types = _merge_fe_categorical_types(
        {key: list(value) for key, value in config.mvp_input_types.items()},
        fe_cat,
    )
    mvp.correct_types(input_types, other_cols)
    types = {key: list(value) for key, value in mvp.types_dict.items()}
    return data, types


def resolve_mvp_types(df: pd.DataFrame, config: TrainingConfig) -> dict[str, list[str]]:
    """Словарь типов признаков после value_type + correct_types (как mvp.types_dict в model_learn)."""
    _, types = _apply_mvp_types(df, config)
    return types


def _mvp_features(df: pd.DataFrame, config: TrainingConfig) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Определить типы признаков через AutoMVP (порядок шагов как в model_learn.py)."""
    data, types = _apply_mvp_types(df, config)
    fe_cat = set(_fe_categorical_in_frame(df))
    features = [
        column
        for column in types["BINARY"] + types["CATEGORIAL"] + types["NUMERIC"]
        if column in data.columns and column not in config.drop_columns
    ]
    categorical = list(
        dict.fromkeys(
            [
                column
                for column in types["CATEGORIAL"] + types["BINARY"]
                if column in features
            ]
            + [column for column in fe_cat if column in features]
        )
    )
    return data, features, categorical


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
    positive_target: bool = False,
    full_frame: bool = False,
) -> DatasetSplit:
    """Разбить датасет на train/test по периоду.

    ``target_range`` — ``between(low, high)``.
    ``positive_target`` — оставить только ``target > 0`` (если ``target_range`` не задан).
    """
    data = df.copy()
    data[config.date_column] = pd.to_datetime(data[config.date_column])
    if target_range is not None:
        data = data[data[target].between(*target_range)]
    elif positive_target:
        data = data[pd.to_numeric(data[target], errors="coerce") > 0]

    train_mask = data[config.date_column].between(*config.train_period)
    test_mask = data[config.date_column].between(*config.test_period)
    if full_frame:
        x_train = data.loc[train_mask]
        x_test = data.loc[test_mask]
    else:
        x_train = data.loc[train_mask, features]
        x_test = data.loc[test_mask, features]
    return DatasetSplit(
        x_train=x_train,
        y_train=data.loc[train_mask, target],
        x_test=x_test,
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


def _check_frequency_leakage(
    model: object,
    feature_names: list[str],
    *,
    leak_level: float,
) -> pd.DataFrame | None:
    """Предупредить, если один признак доминирует в importance (как AutoMVP.show_importances)."""
    importances = model.get_feature_importance()
    frame = pd.DataFrame({"feature": feature_names, "importance": importances})
    suspicious = frame[frame["importance"] > leak_level]
    if not suspicious.empty:
        logger.warning(
            "Возможная утечка frequency: признаки с importance > %s:\n%s",
            leak_level,
            suspicious.to_string(index=False),
        )
        return suspicious
    return None


def _select_features_by_shap(
    *,
    model: object,
    train_pool: object,
    test_pool: object,
    feature_count: int,
    num_features_to_select: int,
    EFeaturesSelectionAlgorithm: type,
    EShapCalcType: type,
) -> tuple[list[str], dict[str, object]]:
    """Отбор признаков CatBoost RecursiveByShapValues (freq/sev)."""
    summary = model.select_features(
        train_pool,
        eval_set=test_pool,
        features_for_select=f"0-{feature_count - 1}",
        num_features_to_select=num_features_to_select,
        steps=feature_count,
        algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
        shap_calc_type=EShapCalcType.Regular,
        train_final_model=False,
        logging_level="Silent",
        plot=False,
    )
    selected = list(summary["selected_features_names"])
    return selected, summary


def _select_frequency_features(
    config: TrainingConfig,
    train_pool: object,
    test_pool: object,
    feature_count: int,
    CatBoostClassifier: type,
    EFeaturesSelectionAlgorithm: type,
    EShapCalcType: type,
) -> tuple[list[str], dict[str, object]]:
    """Отбор frequency-признаков (обёртка над RecursiveByShapValues)."""
    selector = CatBoostClassifier(
        iterations=config.frequency_select_iterations,
        early_stopping_rounds=config.frequency_select_early_stopping_rounds,
        random_state=config.frequency_random_state,
        auto_class_weights="Balanced",
        logging_level="Silent",
    )
    return _select_features_by_shap(
        model=selector,
        train_pool=train_pool,
        test_pool=test_pool,
        feature_count=feature_count,
        num_features_to_select=config.frequency_num_features_to_select,
        EFeaturesSelectionAlgorithm=EFeaturesSelectionAlgorithm,
        EShapCalcType=EShapCalcType,
    )


def _select_severity_features(
    config: TrainingConfig,
    train_pool: object,
    test_pool: object,
    feature_count: int,
    CatBoostRegressor: type,
    EFeaturesSelectionAlgorithm: type,
    EShapCalcType: type,
) -> tuple[list[str], dict[str, object]]:
    """Отбор severity-признаков (обёртка над RecursiveByShapValues)."""
    selector = CatBoostRegressor(
        iterations=config.severity_select_iterations,
        early_stopping_rounds=config.severity_select_early_stopping_rounds,
        random_state=config.severity_random_state,
        logging_level="Silent",
    )
    return _select_features_by_shap(
        model=selector,
        train_pool=train_pool,
        test_pool=test_pool,
        feature_count=feature_count,
        num_features_to_select=config.severity_num_features_to_select,
        EFeaturesSelectionAlgorithm=EFeaturesSelectionAlgorithm,
        EShapCalcType=EShapCalcType,
    )


def _fit_frequency_calibrator(
    model: object,
    x_cal: pd.DataFrame,
    y_cal: pd.Series,
    *,
    method: str,
) -> object:
    """Пост-калибровка на отдельном Cal-set (не на train)."""
    from querulus.training.calibration import fit_probability_calibrator

    return fit_probability_calibrator(model, x_cal, y_cal, method=method)


def frequency_predict_proba(
    training: "TrainingArtifacts",
    features: pd.DataFrame,
) -> np.ndarray:
    """Вероятность класса 1 с учётом калибратора, если он обучен."""
    cat_features = [
        column
        for column in training.frequency_categorical_features
        if column in features.columns
    ]
    if training.frequency_calibrator is not None:
        return np.asarray(
            training.frequency_calibrator.predict_proba(features)[:, 1],
            dtype=float,
        )
    if cat_features:
        from catboost import Pool

        pool = Pool(features, cat_features=cat_features)
        return np.asarray(training.frequency_model.predict_proba(pool)[:, 1], dtype=float)
    return np.asarray(training.frequency_model.predict_proba(features)[:, 1], dtype=float)


def _diagnostics_metrics(
    ModelDiagnostics,
    split: DatasetSplit,
    diagnostics_split: DatasetSplit,
    model: object,
    features: list[str],
    cat_features: list[str],
    task_type: str,
) -> tuple[object, dict[str, dict[str, float]]]:
    """Получить метрики через ModelDiagnostics."""
    diagnostics = ModelDiagnostics(
        X_train=diagnostics_split.x_train,
        y_train=diagnostics_split.y_train,
        X_test=diagnostics_split.x_test,
        y_test=diagnostics_split.y_test,
        model=model,
        features=features,
        cat_features=cat_features,
        task_type=task_type,
    )
    train_metrics, test_metrics = diagnostics.compute_metrics(print_metrics=False)
    return diagnostics, {"train": train_metrics, "test": test_metrics}


def _model_metrics_table(model_metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Собрать train/test метрики одной модели в отдельную таблицу."""
    train_metrics = model_metrics.get("train", {})
    test_metrics = model_metrics.get("test", {})
    metric_names = sorted(set(train_metrics) | set(test_metrics))
    if not metric_names:
        return pd.DataFrame(columns=["metric", "train", "test"])
    return pd.DataFrame(
        {
            "metric": metric_names,
            "train": [train_metrics.get(metric) for metric in metric_names],
            "test": [test_metrics.get(metric) for metric in metric_names],
        }
    )


def _format_metric_value(value: float | int | None) -> str:
    """Форматировать одно значение метрики для читаемого вывода."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, (int, bool)) or float(value).is_integer():
        return f"{int(value)}"
    numeric = float(value)
    if abs(numeric) >= 100:
        return f"{numeric:,.2f}"
    if abs(numeric) >= 1:
        return f"{numeric:.4f}"
    return f"{numeric:.6f}"


def _target_mean(series: pd.Series) -> float | None:
    """Среднее значение таргета для отчёта."""
    if series.empty:
        return None
    return float(series.mean())


def _build_model_report(
    model_name: str,
    target: str,
    split: DatasetSplit,
    features: list[str],
    cat_features: list[str],
    config: TrainingConfig,
    *,
    target_filter: str | None = None,
    hyperparameters: dict[str, object] | None = None,
) -> ModelTrainingReport:
    """Собрать сводку по одной модели."""
    params = hyperparameters or {}
    return ModelTrainingReport(
        model=model_name,
        target=target,
        train_period=config.train_period,
        test_period=config.test_period,
        target_filter=target_filter,
        train_rows=len(split.y_train),
        test_rows=len(split.y_test),
        train_target_mean=_target_mean(split.y_train),
        test_target_mean=_target_mean(split.y_test),
        features=features,
        cat_features=cat_features,
        hyperparameters=params,
    )


def _format_period(period: tuple[str, str]) -> str:
    return f"{period[0]} .. {period[1]}"


def _format_feature_list(features: list[str]) -> str:
    if not features:
        return "—"
    return ", ".join(features)


def format_training_summary(summary: TrainingSummary) -> pd.DataFrame:
    """Таблица ключевых параметров обучения для ноутбука."""
    rows: list[dict[str, str]] = [
        {
            "model": "mvp",
            "parameter": "mvp_features",
            "value": str(summary.mvp_feature_count),
        },
        {
            "model": "mvp",
            "parameter": "mvp_categorical_features",
            "value": str(summary.mvp_categorical_count),
        },
        {
            "model": "mvp",
            "parameter": "date_column",
            "value": summary.date_column,
        },
    ]
    for report in (summary.frequency, summary.severity):
        train_share = report.train_rows / (report.train_rows + report.test_rows) * 100
        test_share = report.test_rows / (report.train_rows + report.test_rows) * 100
        rows.extend(
            [
                {"model": report.model, "parameter": "target", "value": report.target},
                {
                    "model": report.model,
                    "parameter": "train_period",
                    "value": _format_period(report.train_period),
                },
                {
                    "model": report.model,
                    "parameter": "test_period",
                    "value": _format_period(report.test_period),
                },
                {
                    "model": report.model,
                    "parameter": "target_filter",
                    "value": report.target_filter or "—",
                },
                {
                    "model": report.model,
                    "parameter": "train_rows",
                    "value": str(report.train_rows),
                },
                {
                    "model": report.model,
                    "parameter": "test_rows",
                    "value": str(report.test_rows),
                },
                {
                    "model": report.model,
                    "parameter": "train_share_pct",
                    "value": f"{train_share:.2f}",
                },
                {
                    "model": report.model,
                    "parameter": "test_share_pct",
                    "value": f"{test_share:.2f}",
                },
                {
                    "model": report.model,
                    "parameter": "train_target_mean",
                    "value": "—" if report.train_target_mean is None else f"{report.train_target_mean:.6f}",
                },
                {
                    "model": report.model,
                    "parameter": "test_target_mean",
                    "value": "—" if report.test_target_mean is None else f"{report.test_target_mean:.6f}",
                },
                {
                    "model": report.model,
                    "parameter": "features_count",
                    "value": str(len(report.features)),
                },
                {
                    "model": report.model,
                    "parameter": "features",
                    "value": _format_feature_list(report.features),
                },
                {
                    "model": report.model,
                    "parameter": "cat_features_count",
                    "value": str(len(report.cat_features)),
                },
                {
                    "model": report.model,
                    "parameter": "cat_features",
                    "value": _format_feature_list(report.cat_features),
                },
            ]
        )
        for key, value in report.hyperparameters.items():
            rows.append(
                {
                    "model": report.model,
                    "parameter": key,
                    "value": str(value),
                }
            )
    return pd.DataFrame(rows)


def format_features_table(features: list[str], cat_features: list[str]) -> pd.DataFrame:
    """Таблица признаков модели с типом для CatBoost."""
    cat_set = set(cat_features)
    return pd.DataFrame(
        {
            "feature": features,
            "type": ["categorical" if feature in cat_set else "numeric" for feature in features],
        }
    )


def log_training_summary(summary: TrainingSummary) -> None:
    """Вывести сводку обучения в лог."""
    import logging

    logger = logging.getLogger("querulus.training")
    table = format_training_summary(summary)
    logger.info("Training summary:\n%s", table.to_string(index=False))


def format_metrics_table(table: pd.DataFrame) -> pd.DataFrame:
    """Вернуть копию таблицы метрик с форматированными train/test колонками."""
    if table.empty:
        return table.copy()
    formatted = table.copy()
    for column in ("train", "test"):
        if column in formatted.columns:
            formatted[column] = formatted[column].map(_format_metric_value)
    return formatted


def train_models(df: pd.DataFrame, config: TrainingConfig | None = None) -> TrainingArtifacts:
    """Обучить модели частоты и тяжести (таргеты из TrainingConfig)."""
    config = resolve_features_config(config or TrainingConfig())
    data, features, cat_features = _mvp_features(df, config)
    if config.frequency_target in data.columns:
        data[config.frequency_target] = data[config.frequency_target].astype(int)

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

    CatBoostClassifier, CatBoostRegressor, Pool, EFeaturesSelectionAlgorithm, EShapCalcType = (
        _require_catboost()
    )
    ModelDiagnostics = _require_model_diagnostics(config)

    frequency_split = _split_by_date(data, config.frequency_target, frequency_features, config)
    frequency_diag_split = _split_by_date(
        data,
        config.frequency_target,
        frequency_features,
        config,
        full_frame=True,
    )
    frequency_test_pool = Pool(
        frequency_split.x_test,
        frequency_split.y_test.astype(int),
        cat_features=frequency_cat_features,
        feature_names=frequency_features,
    )
    frequency_train_pool = Pool(
        frequency_split.x_train,
        frequency_split.y_train.astype(int),
        cat_features=frequency_cat_features,
        feature_names=frequency_features,
    )
    frequency_hyperparameters = {
        "iterations": config.frequency_iterations,
        "random_state": config.frequency_random_state,
        **config.frequency_classifier_params,
    }
    frequency_feature_selection_summary: dict[str, object] | None = None

    if config.frequency_select_features and len(frequency_features) > config.frequency_num_features_to_select:
        frequency_features, frequency_feature_selection_summary = _select_frequency_features(
            config,
            frequency_train_pool,
            frequency_test_pool,
            len(frequency_features),
            CatBoostClassifier,
            EFeaturesSelectionAlgorithm,
            EShapCalcType,
        )
        frequency_cat_features = [
            column for column in frequency_cat_features if column in frequency_features
        ]
        frequency_hyperparameters["num_features_to_select"] = config.frequency_num_features_to_select
        frequency_hyperparameters["feature_selection"] = "RecursiveByShapValues"
        frequency_train_pool = Pool(
            frequency_split.x_train[frequency_features],
            frequency_split.y_train.astype(int),
            cat_features=frequency_cat_features,
            feature_names=frequency_features,
        )
        frequency_test_pool = Pool(
            frequency_split.x_test[frequency_features],
            frequency_split.y_test.astype(int),
            cat_features=frequency_cat_features,
            feature_names=frequency_features,
        )

    severity_hyperparameters = {
        "iterations": config.severity_iterations,
        "random_state": config.severity_random_state,
        **config.severity_regressor_params,
    }
    if config.severity_range is None:
        severity_target_filter = f"{config.severity_target} > 0"
    else:
        severity_target_filter = (
            f"{config.severity_target} in "
            f"[{config.severity_range[0]}, {config.severity_range[1]}]"
        )

    frequency_model = CatBoostClassifier(
        iterations=config.frequency_iterations,
        random_state=config.frequency_random_state,
        **config.frequency_classifier_params,
    )
    frequency_model.fit(
        frequency_train_pool,
        eval_set=frequency_test_pool,
        plot=False,
    )
    _check_frequency_leakage(
        frequency_model,
        frequency_features,
        leak_level=config.frequency_leak_importance_level,
    )

    frequency_calibrator = None
    if config.frequency_calibration_enabled:
        frequency_calibrator = _fit_frequency_calibrator(
            frequency_model,
            frequency_split.x_train[frequency_features],
            frequency_split.y_train,
            method=config.frequency_calibration_method,
        )

    severity_split = _split_by_date(
        data,
        config.severity_target,
        severity_features,
        config,
        target_range=config.severity_range,
        positive_target=config.severity_range is None,
    )
    severity_diag_split = _split_by_date(
        data,
        config.severity_target,
        severity_features,
        config,
        target_range=config.severity_range,
        positive_target=config.severity_range is None,
        full_frame=True,
    )
    severity_train_pool = Pool(
        severity_split.x_train[severity_features],
        severity_train_target(
            severity_split.y_train, config.severity_target_transform
        ),
        cat_features=severity_cat_features,
        feature_names=severity_features,
        weight=severity_sample_weights(
            severity_split.y_train, config.severity_sample_weight
        ),
    )
    severity_test_pool = Pool(
        severity_split.x_test[severity_features],
        severity_train_target(severity_split.y_test, config.severity_target_transform),
        cat_features=severity_cat_features,
        feature_names=severity_features,
    )
    severity_feature_selection_summary: dict[str, object] | None = None
    if (
        config.severity_select_features
        and len(severity_features) > config.severity_num_features_to_select
    ):
        severity_features, severity_feature_selection_summary = _select_severity_features(
            config,
            severity_train_pool,
            severity_test_pool,
            len(severity_features),
            CatBoostRegressor,
            EFeaturesSelectionAlgorithm,
            EShapCalcType,
        )
        severity_cat_features = [
            column for column in severity_cat_features if column in severity_features
        ]
        severity_hyperparameters["num_features_to_select"] = config.severity_num_features_to_select
        severity_hyperparameters["feature_selection"] = "RecursiveByShapValues"
        severity_train_pool = Pool(
            severity_split.x_train[severity_features],
            severity_train_target(
                severity_split.y_train, config.severity_target_transform
            ),
            cat_features=severity_cat_features,
            feature_names=severity_features,
            weight=severity_sample_weights(
                severity_split.y_train, config.severity_sample_weight
            ),
        )
        severity_test_pool = Pool(
            severity_split.x_test[severity_features],
            severity_train_target(
                severity_split.y_test, config.severity_target_transform
            ),
            cat_features=severity_cat_features,
            feature_names=severity_features,
        )

    severity_model = CatBoostRegressor(
        iterations=config.severity_iterations,
        random_state=config.severity_random_state,
        **config.severity_regressor_params,
    )
    severity_model.fit(
        severity_train_pool,
        eval_set=severity_test_pool,
        plot=False,
    )

    frequency_diagnostics, frequency_metrics = _diagnostics_metrics(
        ModelDiagnostics,
        frequency_split,
        frequency_diag_split,
        frequency_model,
        frequency_features,
        frequency_cat_features,
        "classification",
    )
    severity_diagnostics, severity_metrics = _diagnostics_metrics(
        ModelDiagnostics,
        severity_split,
        severity_diag_split,
        severity_model,
        severity_features,
        severity_cat_features,
        "regression",
    )
    metrics = {"frequency": frequency_metrics, "severity": severity_metrics}

    summary = TrainingSummary(
        date_column=config.date_column,
        mvp_feature_count=len(features),
        mvp_categorical_count=len(cat_features),
        frequency=_build_model_report(
            "frequency",
            config.frequency_target,
            frequency_split,
            frequency_features,
            frequency_cat_features,
            config,
            hyperparameters=frequency_hyperparameters,
        ),
        severity=_build_model_report(
            "severity",
            config.severity_target,
            severity_split,
            severity_features,
            severity_cat_features,
            config,
            target_filter=severity_target_filter,
            hyperparameters=severity_hyperparameters,
        ),
    )
    log_training_summary(summary)

    return TrainingArtifacts(
        frequency_model=frequency_model,
        severity_model=severity_model,
        metrics=metrics,
        frequency_metrics_table=_model_metrics_table(frequency_metrics),
        severity_metrics_table=_model_metrics_table(severity_metrics),
        frequency_diagnostics=frequency_diagnostics,
        severity_diagnostics=severity_diagnostics,
        frequency_split=frequency_split,
        severity_split=severity_split,
        summary=summary,
        feature_names=features,
        categorical_features=cat_features,
        frequency_features=frequency_features,
        severity_features=severity_features,
        frequency_categorical_features=frequency_cat_features,
        severity_categorical_features=severity_cat_features,
        frequency_importance=_importance_frame(frequency_model, frequency_features),
        severity_importance=_importance_frame(severity_model, severity_features),
        feature_frame=data,
        frequency_calibrator=frequency_calibrator,
        frequency_feature_selection_summary=frequency_feature_selection_summary,
        severity_feature_selection_summary=severity_feature_selection_summary,
        severity_target_transform=config.severity_target_transform,
    )
