"""
Сервис скоринга: классификация + регрессия (FastAPI).

Назначение
    Принять общий вектор признаков, посчитать бинарную классификацию с порогом,
    регрессию суммы, вернуть итог «классификация × регрессия»
    Маска по классу и IS_LAWYER (см. ``_masked_predictions``).

Точки входа HTTP
    GET  /api/health  — проверка живости, ``health``, ``version``; опционально ``git_commit``.
    POST /api/predict — тело JSON по схеме ``ServiceRequest`` (mldataworker):
        - main_model (str): stem файла ``<main_model>.pickle`` основной модели
          (ансамбль: классификация + регрессия в одном pickle);
        - main_request (list[dict]): единичный вектор признаков для основной модели;
        - second_model (str | null): stem фоновой модели ``<second_model>.pickle``;
        - second_request (list[dict] | null): единичный вектор признаков для фоновой модели;

Файлы моделей
    Лежат в ``config.prod_models_path`` (см. ``_prod_models_dir``). Имя файла:
    ``<stem>.pickle``, где ``stem`` — значение ``main_model`` / ``second_model`` из запроса.
    Внутри pickle — список словарей с ключами ``model_config``/``model``.
    В этом сервисе ожидается 2 элемента: [0] — классификация, [1] — регрессия.

Порог классификации (число float)
    1) Если во входном векторе есть поле ``'THRESHOLD'`` — используется оно.
    2) Иначе ``config.classification_threshold``.
    3) Иначе ``DEFAULT_CLASSIFICATION_THRESHOLD`` (0.6).

Ответ POST (верхний уровень)
    oisuu_responce — короткий блок для интеграций;
    main_response — usage_model, result, df (поля oisuu не дублируются; порядок ключей
    в result совпадает с порядком одноимённых метрик в oisuu);
    second_response — всегда пустой dict (резерв под старый контракт).

Зависимости
    config (проектный), mldataworker (prepare_dataset, ModelConfig, ResultPickle),
    fastapi, pandas, numpy.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from mldataworker.core.data_prepare import prepare_dataset
from mldataworker.core.pydantic_models import ModelConfig, ServiceRequest
from mldataworker.core.utils import ResultPickle

import uvicorn

from integration import config

# HTTP-приложение FastAPI
app = FastAPI()

# Версия API в oisuu_responce["version"] и GET /api/health (см. CHANGELOG.md)
version = "1.2.0"

# Порог классификации, если не задан во входном векторе
DEFAULT_CLASSIFICATION_THRESHOLD = 0.6

# Имена колонок, которые должны присутствовать в main_request для классификации
CLASSIFICATION_FEATURES: List[str] = [
    "FILIAL",
    "EVENT_CREATED_BY_GIBDD_FLAG",
    "VICTIM_MAX_WEIGHT",
    "RECIEVE_METHOD",
    "APPLICANT_FORM",
    "APPLICANT_AGE",
    "VICTIM_VEHICLE_CATEGORY",
    "GUILTY_CAPACITY_ENGINE",
    "VICTIM_VEHICLE_AGE",
    "EVENT_YEAR"
]

# Имена колонок, которые должны присутствовать в main_request для регрессии
REGRESSION_FEATURES: List[str] = [
    "AMOUNT_REPAIR",
    "LOSS_UNIT_ZONE",
    "VICTIM_VEHICLE_COUNTRY",
    "APPLY_DELAY"
]

# В этом файле принята стратегия "fail-fast" для входного вектора:
# любая проблема (отсутствующие колонки, невалидные даты, нарушение инвариантов
# нормализации) должна завершить запрос до передачи данных в mldataworker.

# reviewed
def validate_no_duplicate_columns(df: pd.DataFrame) -> None:
    """A6: запрещаем дубли имён колонок."""
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Дубли имён колонок во входном векторе: {dupes}")

# reviewed
def validate_preprocessing_invariants(df: pd.DataFrame) -> None:
    """
    Runtime-проверки инвариантов предобработки (B1/B2).

    Проверяет:
        - B1: после нормализации в df не осталось пустых строк '' (они должны стать NaN);
        - B2: все строковые значения приведены к UPPER().
    """
    empty_cols: List[str] = []
    non_upper_cols: List[str] = []

    for col in df.columns:
        s = df[col]
        if s.dtype != object:
            continue

        # B1: запрещаем пустые строки в object-колонках.
        if s.map(lambda v: isinstance(v, str) and v == "").any():
            empty_cols.append(col)

        # B2: запрещаем строковые значения, которые не upper().
        def _is_not_upper(v: Any) -> bool:
            return isinstance(v, str) and v != v.upper()

        if s.map(_is_not_upper).any():
            non_upper_cols.append(col)

    if empty_cols:
        raise ValueError(f"B1 нарушен: после предобработки остались пустые строки '' в колонках: {empty_cols}")
    if non_upper_cols:
        raise ValueError(f"B2 нарушен: после предобработки есть не-UPPER значения в колонках: {non_upper_cols}")

# reviewed
def validate_exact_columns_order(df: pd.DataFrame, expected_columns: List[str], role: str) -> None:
    """
    A3/A5 (после preprocess+enrich): проверка точного состава и порядка колонок
    для датафрейма, который будет подан в `prepare_dataset`.
    """
    actual = list(df.columns)
    if actual != expected_columns:
        missing = [c for c in expected_columns if c not in df.columns]
        extra = [c for c in df.columns if c not in expected_columns]
        raise ValueError(
            f"Неверный состав/порядок колонок для {role}. "
            f"missing={missing} extra={extra} expected={expected_columns} actual={actual}"
        )

# reviewed
def validate_derived_dates_strict(df: pd.DataFrame) -> None:
    """
    C1/C2/C3: строгая проверка дат и derived-полей после enrich.

    Требования:
        - `EVENT_DATE` и `PAYMENT_ORDER_DATE_TIME` парсятся в дату (не NaT);
        - `EVENT_YEAR` и `APPLY_DELAY` рассчитаны (не NaN);
        - `APPLY_DELAY` допускает значения >= 0 или -1 (как маркер "нет данных").
    """
    event_date_col = "EVENT_DATE"
    loss_date_col = "PAYMENT_ORDER_DATE_TIME"

    event_dt = pd.to_datetime(df[event_date_col], errors="coerce")
    loss_dt = pd.to_datetime(df[loss_date_col], errors="coerce")

    # Сравниваем только даты: время может отличаться и не является ошибкой.
    event_d = event_dt.dt.normalize()
    loss_d = loss_dt.dt.normalize()
    if (event_d > loss_d).any():
        raise ValueError("EVENT_DATE > PAYMENT_ORDER_DATE_TIME")
    if (event_dt.isna().any() or loss_dt.isna().any()):
        raise ValueError("EVENT_DATE or PAYMENT_ORDER_DATE_TIME is NaN")

    if (event_dt.isna()).any():
        raise ValueError("EVENT_DATE пустой")
    if (loss_dt.isna()).any():
        raise ValueError("PAYMENT_ORDER_DATE_TIME пустой")
    if "EVENT_YEAR" not in df.columns or (df["EVENT_YEAR"].isna()).any():
        raise ValueError("EVENT_YEAR не рассчитан или пустой")
    if "APPLY_DELAY" not in df.columns or (df["APPLY_DELAY"].isna()).any():
        raise ValueError("APPLY_DELAY не рассчитан или пустой")
    apply_delay = df["APPLY_DELAY"].astype(float)
    if (apply_delay < -1).any():
        raise ValueError("Некорректное значение APPLY_DELAY")

# reviewed
def preprocessing_and_validate_vector(features_values: Any) -> pd.DataFrame:
    """
    Единая точка жёсткой валидации вектора перед `prepare_dataset`.

    Делает:
        - B1/B2: нормализация ""->NaN и upper() в prepare_common_vector_dataframe();
        - B3: расчёт EVENT_YEAR/APPLY_DELAY в enrich_common_vector_dataframe();
        - A6: запрет дубликатов колонок;
        - C1/C2/C3: строгая валидность дат/derived.
    """

    # Входной вектор ожидается как list[dict]
    raw_vector = features_values

    # B1/B2: нормализуем значения:
    # - пустые строки превращаем в NaN;
    # - строки приводим к UPPER() для стабильности категориальных значений.
    df = prepare_common_vector_dataframe(raw_vector)
    # Runtime-check: убеждаемся, что B1/B2 действительно применились (fail-fast до mldataworker).
    validate_preprocessing_invariants(df)
    # B3: расчёт derived EVENT_YEAR/APPLY_DELAY из дат.
    df = enrich_common_vector_dataframe(df)

    # A6: запрет дубликатов колонок
    validate_no_duplicate_columns(df)
    # C1/C2/C3: строгая проверка дат и derived-полей (валидность, порядок дат, допустимые значения).
    validate_derived_dates_strict(df)
    return df


# reviewed
@app.on_event("startup")
def _startup_log() -> None:
    """Логирует параметры процесса при старте приложения."""
    try:
        models_dir = _prod_models_dir()
        exists = models_dir.is_dir()
        logger.info(
            "Service startup || version={} || models_dir={} || models_dir_exists={} || cwd={}",
            version,
            str(models_dir),
            exists,
            str(Path.cwd()),
        )
        if not exists:
            logger.warning(
                "Каталог моделей отсутствует на диске: {}",
                models_dir,
            )
    except Exception:
        logger.exception("Service startup failed")


# reviewed
def _prod_models_dir() -> Path:
    """
    Абсолютный путь к каталогу с файлами ``*.pickle``.

    Берётся ``config.prod_models_path`` (str или Path): если относительный — склеивается с
    ``config.base_path``.

    Возвращает:
        pathlib.Path: существующий или ожидаемый каталог моделей.
    """
    base = Path(config.base_path)
    models_path = Path(config.prod_models_path)
    if models_path.is_absolute():
        return models_path
    return (base / models_path).resolve()


# reviewed
def get_threshold_from_vector(df_common: pd.DataFrame) -> Optional[float]:
    """
    Пытается взять threshold из входного датафрейма (колонка 'THRESHOLD').

    Вход:
        df_common: общий датафрейм из main_request.

    Возвращает:
        float | None: порог для бинаризации (скаляр) или None если колонки нет/пустая.

    Примечание:
        Входной вектор единичный, поэтому берём значение из первой строки.
    """
    if "THRESHOLD" not in df_common.columns:
        logger.debug("Отсутствует THRESHOLD в векторе")
        return None
    value = pd.to_numeric(df_common["THRESHOLD"].iloc[0], errors="coerce")
    if pd.isna(value):
        logger.warning(
            "Порог из вектора: THRESHOLD не число или пусто (raw={!r})",
            df_common["THRESHOLD"].iloc[0],
        )
        return None

    return float(value)

# reviewed
def _masked_predictions(
    labels: List[float],
    raw_reg: List[float],
    df_common: pd.DataFrame,
) -> List[float]:
    """
    Маскированные значения регрессии и итога (одинаковые по контракту).

    Правила:
        - класс 1: ``regression_predictions`` и ``predictions`` = регрессия;
        - класс 0 и IS_LAWYER 0: оба поля = 0;
        - класс 0 и IS_LAWYER 1: оба поля = регрессия.

    ``classification_predictions`` маской не меняется.

    Вход:
        labels: бинарные метки классификации;
        raw_reg: сырые предсказания регрессии;
        df_common: общий вектор (колонка IS_LAWYER опциональна, по умолчанию 0).

    Возвращает:
        list[float]: значения для ``regression_predictions`` и ``predictions``.
    """
    if "IS_LAWYER" in df_common.columns:
        is_lawyer = (
            pd.to_numeric(df_common["IS_LAWYER"], errors="coerce")
            .fillna(0)
            .eq(1)
            .tolist()
        )
    else:
        is_lawyer = [False] * len(labels)

    masked: List[float] = []
    for c, r, lawyer in zip(labels, raw_reg, is_lawyer):
        reg = float(r)
        if c:
            masked.append(reg)
        elif lawyer:
            masked.append(reg)
        else:
            masked.append(0.0)
    return masked

# reviewed
def prepare_common_vector_dataframe(features_values: List[Dict]) -> pd.DataFrame:
    """
    Превращает список JSON-объектов (по одной заявке) в DataFrame.

    Вход:
        features_values: список dict с признаками; пустые строки '' заменяются на NaN.

    Возвращает:
        pandas.DataFrame: общий табличный вход для нарезки колонок под модели.
    """
    try:
        df = pd.DataFrame(features_values)
        for column in df.columns:
            # B1: заменяем пустую строку на NaN, но только для скалярных строк.
            # В некоторых колонках могут быть массивоподобные объекты; сравнение их с "" ломает pandas.replace.
            df[column] = df[column].map(lambda v: np.nan if isinstance(v, str) and v == "" else v)
   
            # Нормализуем только текстовые признаки: UPPER для значений str.
            # Важно: не трогаем числовые dtype, чтобы не превращать их в object.
            s = df[column]
            if s.dtype == object and s.map(lambda v: isinstance(v, str)).any():
                df[column] = s.map(lambda v: v.upper() if isinstance(v, str) else v)
        return df
    except Exception:
        logger.exception(
            "prepare_common_vector_dataframe failed || n_items={}",
            len(features_values) if features_values is not None else None,
        )
        raise

# reviewed
def enrich_common_vector_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Дособирает производные признаки, которых нет во входном векторе.

    Сейчас рассчитываются:
        - EVENT_YEAR = EVENT_DATE.dt.year
        - APPLY_DELAY = PAYMENT_ORDER_DATE_TIME - EVENT_DATE

    Вход:
        df: общий датафрейм из единичного вектора main_request или second_request
        (основная и фоновая модель соответственно).

    Возвращает:
        pandas.DataFrame: тот же df (in-place), для удобства — возвращаем ссылку.
    """
    event_date_col = "EVENT_DATE"
    loss_date_col = "PAYMENT_ORDER_DATE_TIME"

    event_dt = pd.to_datetime(df[event_date_col], errors="coerce")
    loss_dt = pd.to_datetime(df[loss_date_col], errors="coerce")

    df["EVENT_YEAR"] = event_dt.dt.year
    # APPLY_DELAY считаем по датам, без учёта времени.
    df["APPLY_DELAY"] = (loss_dt.dt.normalize() - event_dt.dt.normalize()).dt.days.fillna(-1).astype(int)

    # Преобразуем значения в поле APPLICANT_FORM согласно бизнес-правилам:
    # - 'Скрытый юрист' → 'Юрист с потерпевшим'
    # - 'Представитель (автоюрист)' → 'Представитель (по доверенности)'
    # - 'Представитель (не автоюрист)' → 'Выгодоприобретатель'
    # Остальные значения остаются без изменений.
    df['APPLICANT_FORM'] = df['APPLICANT_FORM'].apply(
        lambda x: (
            'Юрист с потерпевшим' if x == 'Скрытый юрист'
            else 'Представитель (по доверенности)' if x == 'Представитель (автоюрист)'
            else 'Выгодоприобретатель' if x == 'Представитель (не автоюрист)'
            else x
        )
    )
    
    return df


# reviewed
def _require_columns(df: pd.DataFrame, columns: List[str], role: str) -> pd.DataFrame:
    """
    Оставляет только перечисленные колонки в том же порядке списка ``columns``.

    Вход:
        df: исходный датафрейм;
        columns: требуемые имена колонок;
        role: подпись для сообщения об ошибке (человекочитаемо).

    Возвращает:
        pandas.DataFrame: копия с подмножеством колонок.

    Выбрасывает:
        ValueError: если каких-то имён нет в ``df``.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"В общем векторе нет полей для {role}: {missing}")
    return df.loc[:, columns]


# reviewed
def _classification_proba_and_labels(
    model: Any, t: pd.DataFrame, threshold: float
) -> tuple[Optional[List[float]], List[float]]:
    """
    Считает список вероятностей (если доступно) и бинарные метки 0.0/1.0 по порогу.

    Вход:
        model: модель классификации;
        t: признаки после prepare_dataset;
        threshold: порог сравнения с вероятностью класса 1.

    Возвращает:
        tuple:
            - первый элемент: список вероятностей P(класс 1) или None, если модель отдаёт
              только жёсткие метки без proba;
            - второй элемент: список меток после порога (float 0/1) либо сырые predict,
              если это не вероятности.
    """
    try:
        scores = model.predict(t)
    except Exception:
        logger.exception(
            "_classification_proba_and_labels: model.predict (classification) failed || t_shape={}",
            t.shape,
        )
        raise

    # `predict()` может вернуть numpy array или pandas Series — приводим к JSON-совместимому list[float].
    if isinstance(scores, pd.Series):
        scores_list = scores.astype(float).tolist()
    elif isinstance(scores, np.ndarray):
        scores_list = scores.astype(float).tolist()
    elif isinstance(scores, list):
        scores_list = [float(x) for x in scores]
    else:
        # На случай скалярного вывода (одна запись).
        scores_list = [float(scores)]

    labels = [1.0 if s >= threshold else 0.0 for s in scores_list]
    return scores_list, labels


# reviewed
def _round_numbers(obj: Any, ndigits: int=2) -> Any:
    """
    Рекурсивно округляет float в структурах для JSON (списки, dict, вложенность).

    Вход:
        obj: скаляр, список, словарь или None;
        ndigits: число знаков после запятой (по умолчанию ROUND_DECIMALS).

    Возвращает:
        Тот же тип структуры с округлёнными float; int/str/bool не меняются; None остаётся None.
    """
    if obj is None:
        return None
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), ndigits)
    if isinstance(obj, list):
        return [_round_numbers(x, ndigits) for x in obj]
    if isinstance(obj, dict):
        return {k: _round_numbers(v, ndigits) for k, v in obj.items()}
    return obj

# reviewed
def _unwrap_mldataworker_model(model: Any) -> Any:
    """
    Достаёт «сырой» estimator из обёрток mldataworker.

    По текущему контракту модели могут быть обёрнуты и реальный estimator находится в:
        model.model.model
    """
    inner = model
    for _ in range(2):
        if hasattr(inner, "model"):
            inner = getattr(inner, "model")
        else:
            break
    return inner


def _shap_top_features_comment(
    model: Any,
    t: pd.DataFrame,
    *,
    prefix: str,
    top_n: int = 5,
) -> str:
    """
    Считает SHAP для единичного запроса и возвращает короткую строку для CPMComment.

    Если SHAP недоступен или падает — возвращает строку с причиной (без raising),
    чтобы не ломать scoring.
    """
    try:
        import shap  # type: ignore
    except Exception as e:
        return f"{prefix}: SHAP unavailable ({type(e).__name__})"

    try:
        base_model = _unwrap_mldataworker_model(model)
        feature_names = list(t.columns)
        proba_prefix = ""
        base_raw_for_pct: Optional[float] = None

        def _is_catboost_model(m: Any) -> bool:
            mod = getattr(m, "__module__", "") or ""
            name = getattr(m, "__class__", type(m)).__name__
            return mod.startswith("catboost") or name.startswith("CatBoost")

        # CatBoost: считаем нативно (стабильный порядок фич), иначе TreeExplainer.
        if _is_catboost_model(base_model):
            if not hasattr(base_model, "get_feature_importance"):
                raise ValueError("CatBoost model without get_feature_importance()")
            try:
                from catboost import Pool  # type: ignore
            except Exception:
                raise

            # Важно: `get_cat_feature_indices()` возвращает индексы в порядке фичей модели.
            # Порядок/набор колонок в `t` может отличаться, поэтому маппим categorical по именам.
            cat_idxs: list[int] = []
            try:
                model_feature_names = list(getattr(base_model, "feature_names_", None) or [])
                if not model_feature_names and hasattr(base_model, "get_feature_names"):
                    model_feature_names = list(base_model.get_feature_names() or [])
                model_cat_idxs = []
                if hasattr(base_model, "get_cat_feature_indices"):
                    model_cat_idxs = list(base_model.get_cat_feature_indices() or [])
                cat_names = [model_feature_names[i] for i in model_cat_idxs if i < len(model_feature_names)]
                name_to_idx = {name: i for i, name in enumerate(feature_names)}
                cat_idxs = [name_to_idx[n] for n in cat_names if n in name_to_idx]
            except Exception:
                cat_idxs = []

            df_pool = t.copy()
            for i, col in enumerate(feature_names):
                if i in cat_idxs:
                    df_pool[col] = df_pool[col].astype("string").fillna("NA")
                else:
                    df_pool[col] = pd.to_numeric(df_pool[col], errors="coerce").fillna(0.0)

            pool = Pool(df_pool, feature_names=feature_names, cat_features=cat_idxs)
            values = base_model.get_feature_importance(pool, type="ShapValues")
            expected_value = None
        else:
            # Базовая логика: TreeExplainer для tree-based моделей.
            try:
                tree_explainer = shap.TreeExplainer(
                    base_model,
                    feature_perturbation="tree_path_dependent",
                )
                values = tree_explainer.shap_values(t)
                expected_value = getattr(tree_explainer, "expected_value", None)
            except Exception as e:
                if type(e).__name__ == "CatBoostError" and hasattr(base_model, "get_feature_importance"):
                    # CatBoost может прокинуть ошибку даже в TreeExplainer, пробуем нативный путь.
                    from catboost import Pool  # type: ignore

                    cat_idxs = []
                    try:
                        model_feature_names = list(getattr(base_model, "feature_names_", None) or [])
                        if not model_feature_names and hasattr(base_model, "get_feature_names"):
                            model_feature_names = list(base_model.get_feature_names() or [])
                        model_cat_idxs = []
                        if hasattr(base_model, "get_cat_feature_indices"):
                            model_cat_idxs = list(base_model.get_cat_feature_indices() or [])
                        cat_names = [model_feature_names[i] for i in model_cat_idxs if i < len(model_feature_names)]
                        name_to_idx = {name: i for i, name in enumerate(feature_names)}
                        cat_idxs = [name_to_idx[n] for n in cat_names if n in name_to_idx]
                    except Exception:
                        cat_idxs = []

                    df_pool = t.copy()
                    for i, col in enumerate(feature_names):
                        if i in cat_idxs:
                            df_pool[col] = df_pool[col].astype("string").fillna("NA")
                        else:
                            df_pool[col] = pd.to_numeric(df_pool[col], errors="coerce").fillna(0.0)

                    pool = Pool(df_pool, feature_names=feature_names, cat_features=cat_idxs)
                    values = base_model.get_feature_importance(pool, type="ShapValues")
                    expected_value = None
                else:
                    raise

        # Нормализуем форму значений к (n_rows, n_features).
        # Для классификации некоторые реализации возвращают list[array] по классам.
        if isinstance(values, list) and values:
            v = np.asarray(values[0])
        else:
            v = np.asarray(values)
        # CatBoost ShapValues: последний столбец — base value, убираем его.
        if v.ndim == 3 and v.shape[1] == len(feature_names) + 1:
            # Классификация: (n_rows, n_features+1, n_classes). Берём класс 1, если он есть.
            class_idx = 1 if v.shape[2] > 1 else 0
            # Для вероятности (бинарная классификация) берём raw-score класса 1.
            if prefix.startswith("shap_clf"):
                base_raw = float(v[0, -1, class_idx])
                raw = base_raw + float(np.sum(v[0, :-1, class_idx]))
                p = 1.0 / (1.0 + float(np.exp(-raw)))
                proba_prefix = f"p={p:.4f} | "
                base_raw_for_pct = base_raw
            v = v[:, :, class_idx]
        if v.ndim == 2 and v.shape[1] == len(feature_names) + 1:
            if prefix.startswith("shap_clf"):
                base_raw = float(v[0, -1])
                raw = base_raw + float(np.sum(v[0, :-1]))
                p = 1.0 / (1.0 + float(np.exp(-raw)))
                proba_prefix = f"p={p:.4f} | "
                base_raw_for_pct = base_raw
            v = v[:, :-1]
        # TreeExplainer ветка: expected_value может быть доступен отдельно.
        if not proba_prefix and prefix.startswith("shap_clf") and expected_value is not None and v.ndim == 2:
            try:
                ev = expected_value
                if isinstance(ev, (list, tuple, np.ndarray)):
                    class_idx = 1 if len(ev) > 1 else 0
                    ev = ev[class_idx]
                base_raw = float(ev)
                raw = base_raw + float(np.sum(v[0]))
                p = 1.0 / (1.0 + float(np.exp(-raw)))
                proba_prefix = f"p={p:.4f} | "
                base_raw_for_pct = base_raw
            except Exception:
                proba_prefix = ""
        if v.ndim == 3:
            v = v[:, :, 0]
        if v.ndim != 2 or v.shape[0] < 1:
            return f"{prefix}: SHAP empty"

        row = v[0]
        if len(feature_names) != row.shape[0]:
            feature_names = feature_names[: row.shape[0]]

        if prefix.startswith("shap_clf") and base_raw_for_pct is not None:
            base_p = 1.0 / (1.0 + float(np.exp(-base_raw_for_pct)))
            # Относительное изменение вероятности от вклада одного признака.
            # Это не аддитивная декомпозиция в probability-space, но удобно как "impact %".
            rel = []
            for c in row:
                p_i = 1.0 / (1.0 + float(np.exp(-(base_raw_for_pct + float(c)))))
                if base_p == 0:
                    rel.append(0.0)
                else:
                    rel.append((p_i / base_p - 1.0) * 100.0)
            rel_arr = np.asarray(rel, dtype=float)
            idx = np.argsort(np.abs(rel_arr))[::-1][:top_n]
            parts = [f"{feature_names[i]}={rel_arr[i]:+.2f}%" for i in idx]
            return f"{prefix}: " + proba_prefix + ", ".join(parts)

        idx = np.argsort(np.abs(row))[::-1][:top_n]
        parts = [f"{feature_names[i]}={row[i]:+.4f}" for i in idx]
        return f"{prefix}: " + proba_prefix + ", ".join(parts)
    except Exception as e:
        msg = str(e).replace("\n", " ").strip()
        if len(msg) > 120:
            msg = msg[:120] + "..."
        return f"{prefix}: SHAP error ({type(e).__name__}: {msg})"


# reviewed
def _df_to_json(df: pd.DataFrame) -> List[Dict]:
    """
    Преобразует DataFrame в список записей для JSON; ключи в каждой записи по алфавиту.

    Вход:
        df: таблица (строки — заявки).

    Возвращает:
        list[dict]: записи в формате orient='records'; порядок ключей в каждом dict — sorted.
    """
    rows = json.loads(df.to_json(orient="records", force_ascii=False))
    return [{k: row[k] for k in sorted(row.keys())} for row in rows]


# reviewed
def _get_or_load_group(group_name: str) -> Any:
    """
    Загружает pickle сборки моделей с диска.

    Вход:
        group_name: stem имени файла ``{group_name}.pickle``.

    Возвращает:
        list | любой объект из pickle: как сохранено при обучении (обычно список с одним блоком
        model_config + model).

    Ожидаемая структура для этого сервиса:
        group[0] — классификация (model_config + model)
        group[1] — регрессия (model_config + model)
    """
    path = _prod_models_dir() / f"{group_name}.pickle"
    if not path.is_file():
        logger.error("Pickle не найден || path={}", str(path))
    try:
        with open(path, "rb") as f:
            group = pickle.load(f)
    except Exception:
        logger.exception(
            "Загрузка pickle || group={} || path={}",
            group_name,
            str(path),
        )
        raise
    logger.info(
        "Pickle loaded || main_model={} || path={}",
        group_name,
        str(path),
    )
    return group


# reviewed
def _prepare_matrix(
    model_result: Dict[str, Any], df: pd.DataFrame
) -> tuple[Any, pd.DataFrame]:
    """
    Валидирует конфиг и строит матрицу признаков для ``predict`` через prepare_dataset.

    Вход:
        model_result: словарь из pickle (ключи ``model_config`` и ``model``);
        df: входные признаки (уже нарезанные под задачу).

    Возвращает:
        tuple:
            - обученная модель (обёртка);
            - pandas.DataFrame ``t``: трансформированные признаки для инференса.
    """
    try:
        if prepare_dataset is None or ModelConfig is None:
            raise ImportError("mldataworker не доступен (prepare_dataset/ModelConfig)")
        # Конфиг модели хранится в pickle как "сырой" dict; приводим его к ModelConfig,
        # чтобы `prepare_dataset` получил то, что ожидает.
        cfg = ModelConfig.model_validate(model_result["model_config"])
        model = model_result["model"]
        # `prepare_dataset` возвращает объект с полем `.data` — матрица признаков после всех
        # трансформаций (кодирование категорий, масштабирование и т.п.), в том виде,
        # в котором модель ожидает вход на `.predict(...)`.
        matrix = prepare_dataset(
            cfg.name,
            df,
            train_ind=df.index,
            test_ind=pd.Index([]),
            model_config=cfg,
            check_prepared=False,
            calc_corr=False,
            save_data=False,
            log=False,
        ).data
    except Exception:
        logger.exception(
            "_prepare_matrix failed || cfg_name={} || df_shape={}",
            model_result.get("model_config", {}).get("name")
            if isinstance(model_result.get("model_config"), dict)
            else None,
            df.shape,
        )
        raise
    return model, matrix

# reviewed
def run_classification_model(
    model_result: Dict[str, Any],
    df: pd.DataFrame,
    threshold: float,
) -> Dict[str, Any]:
    """
    Полный проход классификации: prepare_dataset → proba/метки → отладочные таблицы.

    Вход:
        model_result: элемент[0] из pickle (классификация);
        df: только классификационные колонки;
        threshold: порог для бинарной метки.

    Возвращает:
        dict с ключами:
            - "labels": list[float] — метки после порога или сырой predict;
            - "proba": list[float] | None — P(класс 1) по строкам, если извлекается;
            - "t": pandas.DataFrame — матрица после prepare_dataset (для сборки общего df в main_predict).
    """
    try:
        # 1) Приводим "сырой" df к матрице признаков (в терминах модели).
        model, t = _prepare_matrix(model_result, df)
        # 2) Считаем скор/вероятность и бинаризуем по порогу.
        proba_list, labels = _classification_proba_and_labels(model, t, threshold)
        out = {
            "labels": labels,
            "proba": proba_list,
            "t": t,
        }
        return out
    except Exception:
        logger.exception(
            "run_classification_model: ошибка || df_shape={} || threshold={}",
            getattr(df, "shape", None),
            threshold,
        )
        raise

# reviewed
def run_regression_model(
    model_result: Dict[str, Any],
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Регрессия: матрица признаков → ``model.predict`` (маска — в ``main_predict``).

    Вход:
        model_result: элемент[1] из pickle (регрессия);
        df: только регрессионные колонки.

    Возвращает:
        dict:
            - "raw_predictions": list — вывод регрессии по строкам;
            - "t": pandas.DataFrame — матрица после prepare_dataset (для сборки общего df в main_predict).
    """
    try:
        # Подготовка матрицы признаков аналогична классификации.
        model, t = _prepare_matrix(model_result, df)
        try:
            # Сырые значения `predict`; маска — в `main_predict`.
            pred = model.predict(t)
            raw = list(pred)
        except Exception:
            logger.exception(
                "run_regression_model: model.predict failed || t_shape={}",
                getattr(t, "shape", None),
            )
            raise
        out = {
            "raw_predictions": raw,
            "t": t,
        }
        return out
    except Exception:
        logger.exception(
            "run_regression_model: ошибка || df_shape={}",
            getattr(df, "shape", None),
        )
        raise

# reviewed
def main_predict(
    group_name: str,
    features_values: List[Dict],
    second_group_name: Optional[str] = None,
    second_features_values: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Основная бизнес-логика скоринга (вызывается из POST /api/predict).

    Вход:
        group_name: из ServiceRequest.main_model (stem ``*.pickle``: классификация+регрессия);
        features_values: из ServiceRequest.main_request — единичный вектор признаков
        для основной модели (в виде списка из 1 объекта);
        second_group_name: фоновая модель;
        second_features_values: из ServiceRequest.second_request — единичный вектор признаков
        для фоновой модели (в виде списка из 1 объекта или null).

    Возвращает:
        dict с ключами:
            - "oisuu_responce": версия, порог, proba/метки классификации,
              regression_predictions и predictions (маска, см. ``_masked_predictions``);
            - "main_response": ``usage_model``, ``result``, ``df`` — те же метрики, что в oisuu.
            # В `result` выводятся следующие ключи (в указанном порядке):
            #   - classification_proba: вероятности положительного класса для каждой записи
            #   - classification_predictions: предсказанные бинарные метки (0 или 1) для каждой записи
            #   - regression_predictions: маскированная регрессия
            #   - prediction: то же, что oisuu predictions
            - "second_response": всегда {}.

    Выбрасывает:
        Исключения из загрузки файлов / prepare_dataset пробрасываются в HTTP 500 в predict_route.
    """
    # threshold берём из входного вектора, если он там есть; иначе — из config / дефолта.
    try:
        group = _get_or_load_group(group_name)
        # Контракт pickle-группы: 2 элемента (классификация, регрессия).
        # Если структура другая — это ошибка подготовки артефакта, а не входного запроса.
        clf_model, rg_model = group[0], group[1]
    except Exception:
        logger.exception("main_predict: загрузка группы моделей || group={}", group_name)
        raise

    try:
        # Fail-fast: любые несоответствия вектора (ValueError) должны оборвать запрос до mldataworker.
        df_common = preprocessing_and_validate_vector(features_values)
    except Exception:
        logger.exception("main_predict: входной вектор некорректен")
        raise

    # Приоритет порога (контракт сервиса):
    # 1) Если 'THRESHOLD' есть во входном векторе и он валидный — используем его.
    # 2) Если поле есть, но пустое/битое — DEFAULT.
    # 3) Если поля нет — config → DEFAULT.
    threshold_value = get_threshold_from_vector(df_common)
    if threshold_value is None:
        threshold_value = DEFAULT_CLASSIFICATION_THRESHOLD

    # A3/A5: после preprocess+enrich проверяем состав/порядок колонок для mldataworker
    # Здесь мы "вырезаем" ровно те признаки, которые использует классификационная модель,
    # и фиксируем порядок колонок как часть контракта.
    clf_df = _require_columns(df_common, CLASSIFICATION_FEATURES, "Классификации")
    validate_exact_columns_order(clf_df, CLASSIFICATION_FEATURES, "Классификации")
    clf = run_classification_model(
        clf_model,
        clf_df,
        threshold_value,
    )
    labels = clf["labels"]
    proba = clf["proba"]
    clf_t = clf["t"]

    # oisuu_responce: version/threshold/CPMComment; общие метрики — в prediction_metrics ниже.
    oisuu: Dict[str, Any] = {
        "version": version,
        "threshold": round(float(threshold_value), 2),
    }
    oisuu["CPMComment"] = _shap_top_features_comment(
        _unwrap_mldataworker_model(clf_model.get("model")),
        clf_t,
        prefix="shap_clf",
    )

    # Регрессия выполняется всегда вместе с классификацией.
    # second_model / second_request — заглушки под «вторую связку» и здесь не используются.

    # A3/A5: после preprocess+enrich проверяем состав/порядок колонок для mldataworker
    # Аналогично классификации: режем и фиксируем порядок признаков для регрессии.
    reg_df = _require_columns(df_common, REGRESSION_FEATURES, "Регрессии")
    validate_exact_columns_order(reg_df, REGRESSION_FEATURES, "Регрессии")
    reg = run_regression_model(
        rg_model,
        reg_df,
    )
    raw_reg = reg["raw_predictions"]
    reg_t = reg["t"]
    oisuu["CPMComment"] = (
        oisuu.get("CPMComment", "")
        + " | "
        + _shap_top_features_comment(
            _unwrap_mldataworker_model(rg_model.get("model")),
            reg_t,
            prefix="shap_reg",
        )
    ).strip()

    if len(labels) != len(raw_reg):
        logger.error(
            "main_predict: несовпадение длин clf/reg || len_labels={} || len_raw_reg={}",
            len(labels),
            len(raw_reg),
        )

    masked_reg = _masked_predictions(labels, raw_reg, df_common)

    rounded_proba = _round_numbers(proba)
    rounded_labels = _round_numbers(labels)
    rounded_masked_reg = _round_numbers(masked_reg)

    # Общие метрики для oisuu_responce и main_response["result"]; порядок ключей — эталон.
    prediction_metrics: Dict[str, Any] = {
        "classification_proba": rounded_proba,
        "classification_predictions": rounded_labels,
        "regression_predictions": rounded_masked_reg,
    }
    oisuu.update(prediction_metrics)
    oisuu["predictions"] = json.dumps(rounded_masked_reg, ensure_ascii=False)

    # prediction — тот же итог, что oisuu predictions, но списком (не JSON-строка).
    result_body: Dict[str, Any] = {
        **prediction_metrics,
        "prediction": rounded_masked_reg,
    }

    # Отладочная таблица "что реально подали в модели": объединяем матрицы признаков
    # из классификации и регрессии (после всех трансформаций).
    output_df = pd.concat([clf_t, reg_t], axis=1)
    if output_df.columns.duplicated().any():
        dupes = output_df.columns[output_df.columns.duplicated()].tolist()
        raise ValueError(f"Дубли имён колонок в output после объединения clf/reg: {dupes}")

    df_body: Dict[str, Any] = {
        "input": _df_to_json(df_common),
        "output": _df_to_json(output_df),
    }

    # main_response без копирования полей oisuu; порядок верхнего уровня: usage_model → result → df.
    main_response: Dict[str, Any] = {
        "usage_model": group_name,
        "result": result_body,
        "df": df_body,
    }

    return {
        "oisuu_responce": oisuu,
        "main_response": main_response,
        "second_response": {},
    }

# reviewed
@app.get("/api/health")
def health_route():
    """
    Проверка, что процесс жив и маршрут зарегистрирован.

    Возвращает:
        JSONResponse: ``health``, ``version``; при сборке может быть ``git_commit``
        из переменной окружения ``SERVICE_GIT_COMMIT``.
    """
    logger.debug("GET /api/health")
    body: Dict[str, Any] = {"health": True, "version": version}
    git_commit = os.environ.get("SERVICE_GIT_COMMIT", "").strip()
    if git_commit:
        body["git_commit"] = git_commit
    return JSONResponse(content=body)

# reviewed
@app.post("/api/predict")
def predict_route(service_request: ServiceRequest):
    """
    HTTP-обёртка над ``main_predict``.

    Вход:
        service_request: тело POST, разобранное в ``ServiceRequest`` (поля main_model,
        main_request, second_model, second_request).

    Возвращает:
        JSONResponse: результат скоринга, код 200.

    При любой ошибке в main_predict — HTTP 500 и поле detail с текстом исключения и traceback.
    """
    try:
        body = main_predict(
            service_request.main_model,
            service_request.main_request,
            service_request.second_model,
            service_request.second_request,
        )
        return JSONResponse(content=body, status_code=200)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception(
            "Predict failed || type={} || error={}",
            type(exc).__name__,
            str(exc),
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(exc),
                "traceback": tb.splitlines(),
            },
        )

# reviewed
if __name__ == "__main__":
    # Локальный запуск веб-сервера (для разработки); в проде обычно uvicorn/gunicorn снаружи.
    uvicorn.run(app, host="0.0.0.0", port=8080)