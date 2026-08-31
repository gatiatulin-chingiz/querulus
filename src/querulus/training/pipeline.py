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

from querulus.training.catboost_fit import catboost_fit_stats

logger = logging.getLogger("querulus.training")


@dataclass
class DatasetSplit:
    """Train / Val / Test разбиение для одной цели.

    ``x_test`` / ``y_test`` — holdout Test (или legacy eval+test на одном окне).
    ``x_val`` / ``y_val`` — Val (только при ``use_train_val_test_split``).
    """

    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    x_val: pd.DataFrame | None = None
    y_val: pd.Series | None = None

    @property
    def has_val(self) -> bool:
        return self.x_val is not None and len(self.x_val) > 0


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
    severity_calibrator: object | None = None
    val_threshold: float | None = None
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


def _make_pool(
    features: pd.DataFrame,
    label: pd.Series | np.ndarray | None = None,
    *,
    cat_features: list[str],
    feature_names: list[str] | None = None,
    weight: pd.Series | np.ndarray | None = None,
):
    """Pool с гарантированным stringify cat-колонок (защита от float 1.0)."""
    from catboost import Pool

    names = feature_names or list(features.columns)
    data = _stringify_categorical_columns(features[names], cat_features)
    kwargs: dict[str, object] = {
        "data": data,
        "cat_features": cat_features,
        "feature_names": names,
    }
    if label is not None:
        kwargs["label"] = label
    if weight is not None:
        kwargs["weight"] = weight
    return Pool(**kwargs)


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
    """value_type → stringify cat → correct_types → stringify финальных cat."""
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

    # correct_types мог добавить в cat колонки, ещё оставшиеся float (0.0/1.0)
    final_categorical = list(
        dict.fromkeys(types.get("BINARY", []) + types.get("CATEGORIAL", []) + fe_cat)
    )
    data = _stringify_categorical_columns(data, final_categorical)
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
    # Selected-фичи из конфига: если колонка есть в df, не теряем её из-за TO_DROP AutoMVP.
    requested = list(
        dict.fromkeys(
            [
                *(config.frequency_features or ()),
                *(config.severity_features or ()),
            ]
        )
    )
    for column in requested:
        if (
            column in data.columns
            and column not in features
            and column not in config.drop_columns
        ):
            features.append(column)
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
    # Явные frequency/severity из конфига могут ссылаться на колонки из TO_DROP
    # AutoMVP (напр. LOSS_UNIT_ZONE). Такие колонки добавляем в пул и помечаем
    # categorical по dtype, иначе CatBoost получает строки в num_feature.
    for column in requested:
        if column not in categorical and column in features:
            series = data[column]
            if (
                column in fe_cat
                or pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
                or pd.api.types.is_bool_dtype(series)
            ):
                categorical.append(column)
    data = _stringify_categorical_columns(data, categorical)
    return data, features, categorical


def _select_model_features(
    available_features: list[str],
    available_cat_features: list[str],
    requested_features: tuple[str, ...] | None,
    model_name: str,
    *,
    data_columns: set[str] | frozenset[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Выбрать признаки для конкретной модели или использовать все MVP-признаки."""
    if requested_features is None:
        features = available_features
    else:
        missing = [column for column in requested_features if column not in available_features]
        if missing:
            cols = data_columns or set()
            absent = [column for column in missing if column not in cols]
            other = [column for column in missing if column not in absent]
            details: list[str] = []
            if absent:
                details.append(
                    f"нет в df ({len(absent)}): {absent} "
                    f"— для synthetic пересоберите parquet "
                    f"(python -m querulus.synthetic_dataset)"
                )
            if other:
                details.append(f"не попали в MVP-пул ({len(other)}): {other}")
            raise ValueError(
                f"Для модели {model_name!r} неизвестные признаки. " + "; ".join(details)
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
    """Разбить датасет на train [/ val] / test по периодам config.

    ``use_train_val_test_split=True``: train_period + val_period + test_period (holdout).
    Иначе legacy: train_period + test_period (test = eval и holdout).
    """
    data = df.copy()
    data[config.date_column] = pd.to_datetime(data[config.date_column])
    if target_range is not None:
        data = data[data[target].between(*target_range)]
    elif positive_target:
        data = data[pd.to_numeric(data[target], errors="coerce") > 0]

    train_mask = data[config.date_column].between(*config.train_period)
    test_mask = data[config.date_column].between(*config.test_period)
    val_mask = None
    if config.use_train_val_test_split:
        if config.val_period is None:
            raise ValueError("use_train_val_test_split=True требует val_period")
        val_mask = data[config.date_column].between(*config.val_period)

    def _xy(mask: pd.Series, *, frame: bool) -> tuple[pd.DataFrame, pd.Series]:
        if frame:
            x_part = data.loc[mask]
        else:
            x_part = data.loc[mask, features]
        return x_part, data.loc[mask, target]

    x_train, y_train = _xy(train_mask, frame=full_frame)
    x_test, y_test = _xy(test_mask, frame=full_frame)
    x_val, y_val = (None, None)
    if val_mask is not None:
        x_val, y_val = _xy(val_mask, frame=full_frame)
    return DatasetSplit(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_val=x_val,
        y_val=y_val,
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
    task_label: str = "features",
    iterations_hint: int | None = None,
    n_progress_steps: int = 20,
) -> tuple[list[str], dict[str, object]]:
    """Отбор признаков CatBoost RecursiveByShapValues с прогрессом по батчам.

    Один вызов ``select_features`` с ``steps≈n_features`` молчит минутами.
    Здесь режем elimination на ~``n_progress_steps`` батчей (``steps=1``) + tqdm/ETA.
    """
    import time

    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None

    feature_names = list(train_pool.get_feature_names())
    if len(feature_names) != feature_count:
        feature_count = len(feature_names)

    n_select = min(int(num_features_to_select), feature_count)
    if n_select >= feature_count:
        print(f"[FS] {task_label}: already ≤{n_select} features, skip")
        return feature_names, {"selected_features_names": feature_names}

    n_elim = feature_count - n_select
    n_batches = max(1, min(int(n_progress_steps), n_elim))
    batch_size = max(1, (n_elim + n_batches - 1) // n_batches)
    iters = iterations_hint
    if iters is None:
        try:
            iters = model.get_param("iterations")
        except Exception:
            iters = "?"

    print(
        f"[FS] {task_label}: {feature_count} → {n_select} "
        f"(drop {n_elim} in ~{n_batches} steps, ≤{batch_size}/step, "
        f"iters≈{iters}/step)"
    )
    started = time.perf_counter()
    remaining = feature_names
    summary: dict[str, object] = {"selected_features_names": remaining}
    dropped_so_far = 0
    pbar = (
        tqdm(total=n_elim, desc=f"[FS] {task_label}", unit="feat", leave=True)
        if tqdm is not None
        else None
    )

    while len(remaining) > n_select:
        drop_now = min(batch_size, len(remaining) - n_select)
        keep = len(remaining) - drop_now
        step_t0 = time.perf_counter()
        summary = model.select_features(
            train_pool,
            eval_set=test_pool,
            features_for_select=remaining,
            num_features_to_select=keep,
            steps=1,
            algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
            shap_calc_type=EShapCalcType.Regular,
            train_final_model=False,
            logging_level="Silent",
            plot=False,
        )
        remaining = list(summary["selected_features_names"])
        dropped_so_far += drop_now
        step_sec = time.perf_counter() - step_t0
        if pbar is not None:
            pbar.update(drop_now)
            pbar.set_postfix_str(
                f"kept={len(remaining)} step={step_sec:.0f}s"
            )
        else:
            elapsed = time.perf_counter() - started
            rate = dropped_so_far / elapsed if elapsed > 0 else 0.0
            eta = (n_elim - dropped_so_far) / rate if rate > 0 else float("nan")
            print(
                f"[FS] {task_label}: dropped {dropped_so_far}/{n_elim}, "
                f"kept={len(remaining)}, step={step_sec:.0f}s, "
                f"elapsed={elapsed / 60:.1f}m, ETA={eta / 60:.1f}m"
            )

    if pbar is not None:
        pbar.close()
    elapsed = time.perf_counter() - started
    print(
        f"[FS] {task_label}: DONE in {elapsed / 60:.1f} min, "
        f"kept {len(remaining)} features"
    )
    return remaining, summary


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
        task_label="frequency",
        iterations_hint=config.frequency_select_iterations,
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
        task_label="severity",
        iterations_hint=config.severity_select_iterations,
    )


def _fit_frequency_calibrator(
    model: object,
    x_cal: pd.DataFrame,
    y_cal: pd.Series,
    *,
    method: str,
    balance: bool = True,
) -> object:
    """Пост-калибровка на отдельном Cal-set (не на train)."""
    from querulus.training.calibration import fit_probability_calibrator

    return fit_probability_calibrator(
        model, x_cal, y_cal, method=method, balance=balance
    )


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
        pool = _make_pool(features, cat_features=cat_features)
        return np.asarray(training.frequency_model.predict_proba(pool)[:, 1], dtype=float)
    return np.asarray(training.frequency_model.predict_proba(features)[:, 1], dtype=float)


def frequency_metrics_table_at_threshold(
    df: pd.DataFrame,
    training: "TrainingArtifacts",
    split_indices: dict[str, pd.Index],
    *,
    threshold: float,
    target_column: str,
) -> pd.DataFrame:
    """Classification-метрики frequency на train/val/test при заданном пороге.

    Proba — как в fin-effect / проде: ``frequency_predict_proba`` (калибратор, если есть).
    Строки признаков — из ``training.feature_frame`` (как при fit).
    """
    from querulus.training.hpo import _fold_metrics_bundle

    feature_frame = training.feature_frame
    if feature_frame is None:
        raise ValueError("training.feature_frame должен быть заполнен")
    if target_column not in df.columns:
        raise ValueError(f"Нет колонки таргета: {target_column}")

    freq_features = list(training.frequency_features)
    bundles: dict[str, dict[str, float]] = {}
    for split_name in ("train", "val", "test"):
        index = split_indices.get(split_name)
        if index is None or len(index) == 0:
            bundles[split_name] = {}
            continue
        index = pd.Index(index).intersection(feature_frame.index).intersection(df.index)
        if len(index) == 0:
            bundles[split_name] = {}
            continue
        features = feature_frame.loc[index, freq_features]
        y_true = df.loc[index, target_column].astype(int)
        proba = frequency_predict_proba(training, features)
        bundles[split_name] = _fold_metrics_bundle(
            y_true,
            proba,
            task_type="classification",
            threshold=float(threshold),
        )
    return _model_metrics_table(bundles)


def _metrics_table_to_bundles(table: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Обратное преобразование ``_model_metrics_table`` → split → metric → value."""
    bundles: dict[str, dict[str, float]] = {}
    if table.empty or "metric" not in table.columns:
        return bundles
    for split in ("train", "val", "test"):
        if split not in table.columns:
            continue
        bundles[split] = {}
        for _, row in table.iterrows():
            value = row.get(split)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            bundles[split][str(row["metric"])] = float(value)
    return bundles


def _apply_val_threshold_policy(
    data: pd.DataFrame,
    artifacts: TrainingArtifacts,
    *,
    frequency_target: str,
    config: TrainingConfig,
) -> TrainingArtifacts:
    """Подбор порога на Val и пересчёт frequency-метрик на ``val_threshold``."""
    split = artifacts.frequency_split
    if not config.use_train_val_test_split:
        return artifacts
    if split is None or not split.has_val or split.x_val is None or len(split.x_val) == 0:
        return artifacts

    from querulus.fin_effect.threshold_policy import pick_threshold_on_val_from_training

    thr_result = pick_threshold_on_val_from_training(
        data,
        artifacts,
        split.x_val.index,
        frequency_target_column=frequency_target,
    )
    val_threshold = thr_result.threshold
    split_indices = {
        "train": split.x_train.index,
        "val": split.x_val.index,
        "test": split.x_test.index,
    }
    freq_table = frequency_metrics_table_at_threshold(
        data,
        artifacts,
        split_indices,
        threshold=val_threshold,
        target_column=frequency_target,
    )
    metrics = dict(artifacts.metrics)
    metrics["frequency"] = _metrics_table_to_bundles(freq_table)
    return TrainingArtifacts(
        frequency_model=artifacts.frequency_model,
        severity_model=artifacts.severity_model,
        metrics=metrics,
        frequency_metrics_table=freq_table,
        severity_metrics_table=artifacts.severity_metrics_table,
        frequency_diagnostics=artifacts.frequency_diagnostics,
        severity_diagnostics=artifacts.severity_diagnostics,
        summary=artifacts.summary,
        feature_names=artifacts.feature_names,
        categorical_features=artifacts.categorical_features,
        frequency_features=artifacts.frequency_features,
        severity_features=artifacts.severity_features,
        frequency_categorical_features=artifacts.frequency_categorical_features,
        severity_categorical_features=artifacts.severity_categorical_features,
        frequency_importance=artifacts.frequency_importance,
        severity_importance=artifacts.severity_importance,
        frequency_split=artifacts.frequency_split,
        severity_split=artifacts.severity_split,
        feature_frame=artifacts.feature_frame,
        frequency_calibrator=artifacts.frequency_calibrator,
        severity_calibrator=artifacts.severity_calibrator,
        val_threshold=val_threshold,
        frequency_feature_selection_summary=artifacts.frequency_feature_selection_summary,
        severity_feature_selection_summary=artifacts.severity_feature_selection_summary,
        severity_target_transform=artifacts.severity_target_transform,
    )


def _diagnostics_metrics(
    ModelDiagnostics,
    split: DatasetSplit,
    diagnostics_split: DatasetSplit,
    model: object,
    features: list[str],
    cat_features: list[str],
    task_type: str,
    *,
    severity_transform: str = "raw",
) -> tuple[object, dict[str, dict[str, float]]]:
    """Метрики train [/ val] / test через ModelDiagnostics + val при необходимости."""
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
    out: dict[str, dict[str, float]] = {
        "train": train_metrics,
        "test": test_metrics,
    }
    if split.has_val and split.x_val is not None and split.y_val is not None:
        from querulus.training.hpo import _fold_metrics_bundle

        x_val = split.x_val[features] if features[0] in split.x_val.columns else split.x_val
        y_val = split.y_val
        if task_type == "classification":
            if cat_features:
                pool = _make_pool(
                    x_val,
                    y_val.astype(int),
                    cat_features=cat_features,
                    feature_names=features,
                )
                pred = np.asarray(model.predict_proba(pool)[:, 1], dtype=float)
            else:
                pred = np.asarray(model.predict_proba(x_val)[:, 1], dtype=float)
        else:
            if cat_features:
                pool = _make_pool(
                    x_val,
                    severity_train_target(y_val, severity_transform),
                    cat_features=cat_features,
                    feature_names=features,
                )
                pred = np.asarray(model.predict(pool), dtype=float)
            else:
                pred = np.asarray(model.predict(x_val), dtype=float)
        out["val"] = _fold_metrics_bundle(
            severity_train_target(y_val, severity_transform)
            if task_type == "regression"
            else y_val,
            pred,
            task_type=task_type,
        )
    else:
        out["val"] = {}
    return diagnostics, out


def _eval_pool_for_split(
    split: DatasetSplit,
    *,
    features: list[str],
    cat_features: list[str],
    config: TrainingConfig,
    as_int: bool = False,
    severity_transform: str | None = None,
) -> object | None:
    """Eval pool для early stopping: Val при train/val/test, иначе test (legacy)."""
    if config.use_train_val_test_split and split.has_val:
        x_eval = split.x_val
        y_eval = split.y_val
    else:
        x_eval = split.x_test
        y_eval = split.y_test
    if x_eval is None or y_eval is None or len(x_eval) == 0:
        return None
    if severity_transform is not None:
        y_pool = severity_train_target(y_eval, severity_transform)
    elif as_int:
        y_pool = y_eval.astype(int)
    else:
        y_pool = y_eval
    return _make_pool(
        x_eval[features] if features and features[0] in x_eval.columns else x_eval,
        y_pool,
        cat_features=cat_features,
        feature_names=features,
    )


def _fit_catboost(
    model: object,
    train_pool: object,
    *,
    eval_pool: object | None,
    config: TrainingConfig,
) -> object:
    """Fit CatBoost: fixed tree_count без ES или legacy ES на eval."""
    if config.fit_fixed_tree_count:
        model.fit(train_pool, plot=False)
        return model
    if eval_pool is not None:
        model.fit(
            train_pool,
            eval_set=eval_pool,
            early_stopping_rounds=50,
            plot=False,
        )
        return model
    model.fit(train_pool, plot=False)
    return model


def bundles_metrics_table(
    bundles: dict[str, dict[str, float]],
    *,
    column_order: list[str] | None = None,
) -> pd.DataFrame:
    """Таблица metric × split; колонки — ключи ``bundles`` (произвольные имена срезов)."""
    if not bundles:
        return pd.DataFrame(columns=["metric"])
    columns = list(column_order) if column_order is not None else list(bundles.keys())
    metric_names = sorted({metric for split in bundles.values() for metric in split})
    if not metric_names:
        return pd.DataFrame({"metric": []})
    data: dict[str, object] = {"metric": metric_names}
    for column in columns:
        split_metrics = bundles.get(column, {})
        data[column] = [split_metrics.get(metric) for metric in metric_names]
    return pd.DataFrame(data)


def _model_metrics_table(model_metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Собрать train / val / test метрики одной модели в таблицу."""
    return bundles_metrics_table(
        model_metrics,
        column_order=["train", "val", "test"],
    )


def _format_metric_value(value: float | int | None) -> str:
    """Строковое представление метрики для таблиц (не более 2 знаков после запятой)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, (int, bool)) or float(value).is_integer():
        return f"{int(value)}"
    numeric = float(value)
    if abs(numeric) >= 100:
        return f"{numeric:,.2f}"
    return f"{numeric:.2f}"


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
                    "value": "—" if report.train_target_mean is None else f"{report.train_target_mean:.2f}",
                },
                {
                    "model": report.model,
                    "parameter": "test_target_mean",
                    "value": "—" if report.test_target_mean is None else f"{report.test_target_mean:.2f}",
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
    """Вернуть копию таблицы метрик с форматированными колонками срезов (≤2 знака)."""
    if table.empty:
        return table.copy()
    formatted = table.copy()
    skip = {"metric", "task", "split", "stack"}
    for column in formatted.columns:
        if column in skip:
            continue
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
        data_columns=set(data.columns),
    )
    severity_features, severity_cat_features = _select_model_features(
        features,
        cat_features,
        config.severity_features,
        "severity",
        data_columns=set(data.columns),
    )

    CatBoostClassifier, CatBoostRegressor, _, EFeaturesSelectionAlgorithm, EShapCalcType = (
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
    frequency_eval_pool = _eval_pool_for_split(
        frequency_split,
        features=frequency_features,
        cat_features=frequency_cat_features,
        config=config,
        as_int=True,
    )
    if frequency_eval_pool is None:
        frequency_eval_pool = _make_pool(
            frequency_split.x_test,
            frequency_split.y_test.astype(int),
            cat_features=frequency_cat_features,
            feature_names=frequency_features,
        )
    frequency_train_pool = _make_pool(
        frequency_split.x_train,
        frequency_split.y_train.astype(int),
        cat_features=frequency_cat_features,
        feature_names=frequency_features,
    )
    frequency_hyperparameters = {
        "iterations_cap": config.frequency_iterations,
        "iterations": config.frequency_iterations,
        "random_state": config.frequency_random_state,
        **config.frequency_classifier_params,
    }
    frequency_feature_selection_summary: dict[str, object] | None = None

    if config.frequency_select_features and len(frequency_features) > config.frequency_num_features_to_select:
        frequency_features, frequency_feature_selection_summary = _select_frequency_features(
            config,
            frequency_train_pool,
            frequency_eval_pool,
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
        frequency_train_pool = _make_pool(
            frequency_split.x_train[frequency_features],
            frequency_split.y_train.astype(int),
            cat_features=frequency_cat_features,
            feature_names=frequency_features,
        )
        frequency_eval_pool = _eval_pool_for_split(
            frequency_split,
            features=frequency_features,
            cat_features=frequency_cat_features,
            config=config,
            as_int=True,
        )
        if frequency_eval_pool is None:
            frequency_eval_pool = _make_pool(
                frequency_split.x_test[frequency_features],
                frequency_split.y_test.astype(int),
                cat_features=frequency_cat_features,
                feature_names=frequency_features,
            )

    severity_hyperparameters = {
        "iterations_cap": config.severity_iterations,
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
    frequency_model = _fit_catboost(
        frequency_model,
        frequency_train_pool,
        eval_pool=frequency_eval_pool,
        config=config,
    )
    frequency_hyperparameters.update(
        catboost_fit_stats(
            frequency_model, iterations_cap=config.frequency_iterations
        )
    )
    frequency_hyperparameters["iterations"] = frequency_hyperparameters["tree_count"]
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
            balance=config.frequency_calibration_balance,
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
    severity_train_pool = _make_pool(
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
    severity_eval_pool = _eval_pool_for_split(
        severity_split,
        features=severity_features,
        cat_features=severity_cat_features,
        config=config,
        severity_transform=config.severity_target_transform,
    )
    if severity_eval_pool is None:
        severity_eval_pool = _make_pool(
            severity_split.x_test[severity_features],
            severity_train_target(
                severity_split.y_test, config.severity_target_transform
            ),
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
            severity_eval_pool,
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
        severity_train_pool = _make_pool(
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
        severity_eval_pool = _eval_pool_for_split(
            severity_split,
            features=severity_features,
            cat_features=severity_cat_features,
            config=config,
            severity_transform=config.severity_target_transform,
        )
        if severity_eval_pool is None:
            severity_eval_pool = _make_pool(
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
    severity_model = _fit_catboost(
        severity_model,
        severity_train_pool,
        eval_pool=severity_eval_pool,
        config=config,
    )
    severity_hyperparameters.update(
        catboost_fit_stats(severity_model, iterations_cap=config.severity_iterations)
    )
    severity_hyperparameters["iterations"] = severity_hyperparameters["tree_count"]

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
        severity_transform=config.severity_target_transform,
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

    return _apply_val_threshold_policy(
        data,
        TrainingArtifacts(
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
            severity_calibrator=None,
            frequency_feature_selection_summary=frequency_feature_selection_summary,
            severity_feature_selection_summary=severity_feature_selection_summary,
            severity_target_transform=config.severity_target_transform,
        ),
        frequency_target=config.frequency_target,
        config=config,
    )
