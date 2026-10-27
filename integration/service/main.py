"""
Сервис скоринга: классификация + регрессия (FastAPI).

Назначение
    Принять общий вектор признаков, посчитать бинарную классификацию с порогом,
    регрессию суммы, вернуть итог «классификация × регрессия»
    Маска по классу и IS_LAWYER (см. ``_masked_predictions``).

Точки входа HTTP
    GET  /api/health  — ``health``, ``main_version``, ``second_version``; опционально ``git_commit``.
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
    2) Иначе ``config.default_main_threshold``.

Ответ POST (верхний уровень)
    main_response — usage_model, version, CPMComment, result, df (legacy);
    second_response — то же для shadow 2.0.0 при second_*; иначе {}.

# Dual-track: main_ = legacy ensemble; second_ = 2.0.0 (FE + калибраторы + τ cal).

Зависимости
    config (проектный), mldataworker (prepare_dataset, ModelConfig, ResultPickle),
    fastapi, pandas, numpy.
"""
from __future__ import annotations

import json
import os
import pickle
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

import uvicorn

try:
    from integration import config
    from integration.scoring.second import score_second
    from integration.service.masking import masked_predictions
    from integration.service.shap_comment import (
        shap_top_features_comment,
        unwrap_mldataworker_model,
    )
except ImportError:
    import config
    from scoring.second import score_second
    from service.masking import masked_predictions
    from service.shap_comment import (
        shap_top_features_comment,
        unwrap_mldataworker_model,
    )

# HTTP-приложение FastAPI
app = FastAPI()

# Совместимость re-export: версия main_-трека (см. config.main_version).
version = config.main_version

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

#reviewed
def validate_no_duplicate_columns(df: pd.DataFrame) -> None:
    """A6: запрещаем дубли имён колонок."""
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Дубли имён колонок во входном векторе: {dupes}")

#reviewed
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

#reviewed
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

#reviewed
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

#reviewed
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


#reviewed
@app.on_event("startup")
def _startup_log() -> None:
    """Логирует параметры процесса при старте приложения."""
    try:
        models_dir = _prod_models_dir()
        exists = models_dir.is_dir()
        logger.info(
            "Service startup || main_version={} || second_version={} || models_dir={} || models_dir_exists={} || cwd={}",
            config.main_version,
            config.second_version,
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


#reviewed
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


#reviewed
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

#reviewed
def _masked_predictions(
    labels: List[float],
    raw_reg: List[float],
    df_common: pd.DataFrame,
) -> List[float]:
    """Обёртка над ``service.masking.masked_predictions`` (main_ и second_)."""
    return masked_predictions(labels, raw_reg, df_common)

#reviewed
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

#reviewed
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


#reviewed
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


#reviewed
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


#reviewed
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

#reviewed
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


_MAIN_GROUP_CACHE: Dict[str, Any] = {}


#reviewed
def _get_or_load_group(group_name: str) -> Any:
    """
    Загружает pickle сборки моделей с диска (кэш в процессе).

    Вход:
        group_name: stem имени файла ``{group_name}.pickle``.

    Возвращает:
        list | любой объект из pickle: как сохранено при обучении (обычно список с одним блоком
        model_config + model).

    Ожидаемая структура для этого сервиса:
        group[0] — классификация (model_config + model)
        group[1] — регрессия (model_config + model)
    """
    cached = _MAIN_GROUP_CACHE.get(group_name)
    if cached is not None:
        return cached
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
    _MAIN_GROUP_CACHE[group_name] = group
    return group


#reviewed
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

#reviewed
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

#reviewed
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

#reviewed
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
            - "main_response": ``usage_model``, ``version``, ``CPMComment``, ``result``, ``df``.
              # В `result`: classification_proba / classification_predictions /
              # regression_predictions / prediction / threshold.
            - "second_response": shadow 2.0.0 (если second_* заданы), иначе {}.

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

    # Приоритет порога main_: THRESHOLD в векторе → config.default_main_threshold.
    threshold_value = get_threshold_from_vector(df_common)
    if threshold_value is None:
        threshold_value = float(config.default_main_threshold)

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

    cpm_comment = shap_top_features_comment(
        unwrap_mldataworker_model(clf_model.get("model")),
        clf_t,
        prefix="shap_clf",
    )

    # Регрессия выполняется всегда вместе с классификацией.
    # second_model / second_request — отдельный shadow-пайплайн 2.0.0 (ниже).

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
    cpm_comment = (
        cpm_comment
        + " | "
        + shap_top_features_comment(
            unwrap_mldataworker_model(rg_model.get("model")),
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

    result_body: Dict[str, Any] = {
        "classification_proba": rounded_proba,
        "classification_predictions": rounded_labels,
        "regression_predictions": rounded_masked_reg,
        "prediction": rounded_masked_reg,
        "threshold": round(float(threshold_value), 2),
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

    main_response: Dict[str, Any] = {
        "usage_model": group_name,
        "version": config.main_version,
        "CPMComment": cpm_comment,
        "result": result_body,
        "df": df_body,
    }

    # second_ 2.0.0 (FE + калибраторы + τ cal).
    second_response: Dict[str, Any] = {}
    if second_group_name and second_features_values:
        try:
            second_response = score_second(
                second_group_name,
                second_features_values,
            )
        except Exception:
            logger.exception(
                "main_predict: second_ failed || group={}",
                second_group_name,
            )
            raise

    return {
        "main_response": main_response,
        "second_response": second_response,
    }

#reviewed
@app.get("/api/health")
def health_route():
    """
    Проверка, что процесс жив и маршрут зарегистрирован.

    Возвращает:
        JSONResponse: ``health``, ``main_version``, ``second_version``; при сборке
        может быть ``git_commit`` из ``SERVICE_GIT_COMMIT``.
    """
    logger.debug("GET /api/health")
    body: Dict[str, Any] = {
        "health": True,
        "main_version": config.main_version,
        "second_version": config.second_version,
    }
    git_commit = os.environ.get("SERVICE_GIT_COMMIT", "").strip()
    if git_commit:
        body["git_commit"] = git_commit
    return JSONResponse(content=body)

#reviewed
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

if __name__ == "__main__":
    # Локальный запуск веб-сервера (для разработки); в проде обычно uvicorn/gunicorn снаружи.
    uvicorn.run(app, host="0.0.0.0", port=8080)