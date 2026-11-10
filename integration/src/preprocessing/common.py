"""Общие шаги предобработки для main_ и second_.

Вход: ``preprocess_input_vector`` → UPPER + ``coerce_and_enrich`` + validate.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from mldataworker.core.data_prepare import prepare_dataset
from mldataworker.core.pydantic_models import ModelConfig

import config as _service_config

# Ключи уже в UPPER: remap вызывается после нормализации строк.
APPLICANT_FORM_REMAP: Dict[str, str] = {
    "СКРЫТЫЙ ЮРИСТ": "ЮРИСТ С ПОТЕРПЕВШИМ",
    "ПРЕДСТАВИТЕЛЬ (АВТОЮРИСТ)": "ПРЕДСТАВИТЕЛЬ (ПО ДОВЕРЕННОСТИ)",
    "ПРЕДСТАВИТЕЛЬ (НЕ АВТОЮРИСТ)": "ВЫГОДОПРИОБРЕТАТЕЛЬ",
}

# Brand → страна ТС (levels модели: JAPAN/RUSSIA/OTHER/…).
# Файл — плоский Other_dicts (или обёртка {"Other_dicts": {...}}).
_OTHER_DICTS_PATH: Path = (
    Path(_service_config.base_path) / "results" / "Other_dicts.json"
).resolve()


# reviewed
def _load_ts_country_by_brand(path: Path) -> Dict[str, str]:
    """Загрузить ``dict_ts_country``: ключи/значения в UPPER под B2 и replace модели."""
    if not path.is_file():
        raise FileNotFoundError(f"Нет Other_dicts.json: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    other = payload.get("Other_dicts", payload)
    raw = other.get("dict_ts_country") if isinstance(other, dict) else None
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Нет dict_ts_country в {path}")
    return {str(k).upper(): str(v).upper() for k, v in raw.items()}


TS_COUNTRY_BY_BRAND: Dict[str, str] = _load_ts_country_by_brand(_OTHER_DICTS_PATH)

# Каст только здесь; CPI/FE берут уже готовые dtype.
DATE_INPUT_COLUMNS: tuple[str, ...] = (
    "EVENT_DATE",
    "PAYMENT_ORDER_DATE_TIME",
    "LOSS_DATE_TIME",
    "POLICYHOLDER_CONTRACT_END_DATE",
)

NUMERIC_INPUT_COLUMNS: tuple[str, ...] = (
    "VALUE_BEFORE_WITH",
    "VALUE_BEFORE_WITHOUT",
    "AMOUNT_REPAIR",
    "WORK",
    "WORKS",
    "SHARE_WORK",
    "SHARE_WORK_NUM",
    "VICTIM_CAPACITY_ENGINE",
    "VICTIM_MAX_WEIGHT",
    "VICTIM_OBJECT_YEAR",
)

# Общий входной контракт main_ и second_: все колонки должны прийти разом.
# VIC_TS_COUNTRY сервис считает из VICTIM_VEHICLE_BRAND (Other_dicts.dict_ts_country).
# Серые VICTIM_TS_REGION / VICTIM_VEHICLE_COUNTRY клиент отдаёт готовыми.
INPUT_REQUIRED_COLUMNS: List[str] = [
    "INCIDENT_NUMBER",
    "LOSS_NUMBER",
    "FILIAL",
    "GUILTY_MAX_WEIGHT",
    "EVENT_CREATED_BY_GIBDD_FLAG",
    "RECIEVE_METHOD",
    "FL_PHOTO_VIDEO",
    "APPLICANT_FORM",
    "PAYMENT_RECIPIENT_TYPE",
    "LOSS_UNIT_TYPE",
    "VICTIM_VEHICLE_CATEGORY",
    "VICTIM_OBJECT_YEAR",
    "APPLICANT_AGE",
    "VICTIM_AGE",
    "NOT_NOTIFICATION",
    "IS_TOTAL",
    "VICTIM_VEHICLE_AGE",
    "REGION",
    "VICTIM_TS_REGION",
    "VICTIM_VEHICLE_BRAND",
    "VICTIM_VEHICLE_COUNTRY",
    "EVENT_DATE",
    "PAYMENT_ORDER_DATE_TIME",
    "LOSS_DATE_TIME",
    "POLICYHOLDER_CONTRACT_END_DATE",
    "VALUE_BEFORE_WITH",
    "VALUE_BEFORE_WITHOUT",
    "WORK",
    "VICTIM_CAPACITY_ENGINE",
    "VICTIM_MAX_WEIGHT",
]

# После FE: только engineered (вход уже проверен в validate_input_vector).
ENGINEERED_FEATURES: List[str] = [
    "FE_VALUE_BEFORE_WITH_REAL_2022",
    "FE_VALUE_BEFORE_WITH_BIN",
    "FE_VALUE_BEFORE_WITHOUT_REAL_2022",
    "FE_DAYS_EVENT_TO_LOSS",
    "FE_VALUE_BEFORE_RATIO",
    "FE_SHARE_WORK_TIER",
    "EVENT_YEAR",
    "FE_DAYS_TO_PH_CONTRACT_END",
    "FE_VICTIM_POWER_PER_TON",
    "FE_VALUE_BEFORE_DIFF",
    "VICTIM_OBJECT_AGE_MONTHS",
]
# Матрица модели: brand → VIC_TS_COUNTRY в coerce_and_enrich; brand в pickle не идёт.
MODEL_FEATURES: List[str] = [
    *(c for c in INPUT_REQUIRED_COLUMNS if c != "VICTIM_VEHICLE_BRAND"),
    "VIC_TS_COUNTRY",
] + ENGINEERED_FEATURES



# reviewed
def prepare_vector_upper(features_values: List[Dict]) -> pd.DataFrame:
    """Список JSON-объектов → DataFrame: ``""→NaN``, строки → UPPER.

    Числовые dtype не трогаем. Дальше — ``coerce_and_enrich``.

    Args:
        features_values: ``main_request`` / ``second_request`` (list[dict]).

    Returns:
        DataFrame одной (или нескольких) строк.

    Raises:
        Любое исключение построения/нормализации пробрасывается после лога.
    """
    try:
        df = pd.DataFrame(features_values)
        for column in df.columns:
            if df[column].dtype == object:
                df[column] = (
                    df[column]
                    .map(lambda v: np.nan if v == "" else v)
                    .map(lambda v: v.upper() if isinstance(v, str) else v)
                )
        return df
    except Exception:
        logger.exception(
            "prepare_vector_upper failed || n_items={}",
            len(features_values) if features_values is not None else None,
        )
        raise


# reviewed
def coerce_and_enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Единственное место каста dtype + derived-дат + remap формы/флагов.

    * даты из ``DATE_INPUT_COLUMNS`` → datetime (если колонка есть);
    * числа из ``NUMERIC_INPUT_COLUMNS`` → numeric;
    * ``EVENT_YEAR``, ``PAYMENT_ORDER_YEAR``, ``APPLY_DELAY`` из уже datetime-колонок;
    * remap ``APPLICANT_FORM`` по ``APPLICANT_FORM_REMAP``;
    * ``NOT_NOTIFICATION``: «Не уведомлять» → 0, иначе → 1;
    * ``VIC_TS_COUNTRY`` ← ``VICTIM_VEHICLE_BRAND`` через ``TS_COUNTRY_BY_BRAND``
      (нет бренда в словаре → ``OTHER``; клиентский ``VIC_TS_COUNTRY`` перезаписывается).

    Args:
        df: кадр после ``prepare_vector_upper`` (или эквивалент).

    Returns:
        Копия с приведёнными типами и derived-полями.
    """
    out = df.copy()
    for col in DATE_INPUT_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    for col in NUMERIC_INPUT_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["EVENT_YEAR"] = out["EVENT_DATE"].dt.year
    out["PAYMENT_ORDER_YEAR"] = out["PAYMENT_ORDER_DATE_TIME"].dt.year
    out["APPLY_DELAY"] = (
        out["PAYMENT_ORDER_DATE_TIME"] - out["EVENT_DATE"]
    ).dt.days
    if "APPLICANT_FORM" in out.columns:
        out["APPLICANT_FORM"] = out["APPLICANT_FORM"].map(
            lambda x: APPLICANT_FORM_REMAP.get(x, x) if isinstance(x, str) else x
        )
    if "NOT_NOTIFICATION" in out.columns:
        out["NOT_NOTIFICATION"] = out["NOT_NOTIFICATION"].map(
            lambda v: 0 if v == "НЕ УВЕДОМЛЯТЬ" else 1
        )
    if "VICTIM_VEHICLE_BRAND" not in out.columns:
        raise ValueError("Нет VICTIM_VEHICLE_BRAND для VIC_TS_COUNTRY")
    out["VIC_TS_COUNTRY"] = out["VICTIM_VEHICLE_BRAND"].map(
        lambda x: TS_COUNTRY_BY_BRAND.get(x, "OTHER") if isinstance(x, str) else "OTHER"
    )
    return out


# reviewed
def validate_no_duplicate_columns(df: pd.DataFrame) -> None:
    """A6: запретить дубли имён колонок во входном векторе."""
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Дубли имён колонок во входном векторе: {dupes}")


# reviewed
def validate_derived_dates_strict(df: pd.DataFrame) -> None:
    """Проверить даты по полям после ``coerce_and_enrich``.

    Смотрит ``EVENT_YEAR`` / ``APPLY_DELAY`` (уже посчитаны в enrich).

    Raises:
        ValueError: дата пустая или ``EVENT_DATE`` позже выплаты.
    """
    if df["EVENT_YEAR"].isna().any():
        raise ValueError("EVENT_DATE пустой")
    if df["APPLY_DELAY"].isna().any():
        raise ValueError("PAYMENT_ORDER_DATE_TIME пустой")
    if (df["APPLY_DELAY"] < 0).any():
        raise ValueError("EVENT_DATE > PAYMENT_ORDER_DATE_TIME")


# reviewed
def validate_preprocessed_vector(df: pd.DataFrame) -> None:
    """Общие fail-fast проверки после ``coerce_and_enrich`` для main_ и second_.

    * B2: строки в UPPER (``''`` уже стали NaN в ``prepare_vector_upper``);
    * A6: нет дублей имён колонок;
    * C1/C2/C3: даты через ``validate_derived_dates_strict``.
    """
    non_upper_cols: List[str] = []
    for col in df.columns:
        s = df[col]
        if s.dtype == object:
            if s.map(lambda v: isinstance(v, str) and v != v.upper()).any():
                non_upper_cols.append(col)
    if non_upper_cols:
        raise ValueError(
            f"B2 нарушен: после предобработки есть не-UPPER значения в колонках: {non_upper_cols}"
        )
    validate_no_duplicate_columns(df)
    validate_derived_dates_strict(df)


# reviewed
def require_columns(df: pd.DataFrame, columns: List[str], role: str) -> pd.DataFrame:
    """Проверить наличие колонок и вернуть кадр в заданном порядке.

    Одна точка для входа, post-FE и матриц перед моделью/калибратором:
    нет имени → ``ValueError``; иначе ``df[columns]``.

    Args:
        df: полный кадр.
        columns: обязательные имена в нужном порядке.
        role: подпись для ошибки ("input", "Классификации", …).

    Returns:
        Кадр с колонками ``columns`` в этом порядке.

    Raises:
        ValueError: хотя бы одной колонки нет.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Нет обязательных колонок для {role}: {missing}")
    return df.loc[:, list(columns)]


# reviewed
def validate_input_vector(df: pd.DataFrame) -> None:
    """Все обязательные фичи пришли разом + preprocess-инварианты (main_ и second_)."""
    require_columns(df, INPUT_REQUIRED_COLUMNS, "input")
    validate_preprocessed_vector(df)


# reviewed
def preprocess_input_vector(features_values: List[Dict]) -> pd.DataFrame:
    """Общий вход main_/second_: UPPER → касты/derived → validate.

    Args:
        features_values: ``main_request`` / ``second_request`` (list[dict]).

    Returns:
        DataFrame после ``coerce_and_enrich`` и ``validate_input_vector``.
    """
    df = coerce_and_enrich(prepare_vector_upper(features_values))
    validate_input_vector(df)
    return df


# reviewed
def validate_model_features(df: pd.DataFrame) -> None:
    """После FE: только engineered (вход уже проверен)."""
    require_columns(df, ENGINEERED_FEATURES, "model")


# reviewed
def prepare_matrix(
    model_result: Dict[str, Any], df: pd.DataFrame
) -> tuple[Any, pd.DataFrame]:
    """Прогнать df через ``prepare_dataset`` по ``model_config`` из pickle.

    Args:
        model_result: элемент группы ``{"model_config", "model"}``.
        df: кадр с нужными сырыми/FE колонками.

    Returns:
        Кортеж ``(model, matrix)`` — estimator и матрица для predict.

    Raises:
        Любая ошибка валидации config / prepare_dataset после лога.
    """
    try:
        # Валидируем конфиг модели и собираем матрицу для predict.
        cfg = ModelConfig.model_validate(model_result["model_config"])
        model = model_result["model"]
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
        raw_cfg = model_result.get("model_config")
        logger.exception(
            "prepare_matrix failed || cfg_name={} || df_shape={}",
            raw_cfg.get("name") if isinstance(raw_cfg, dict) else None,
            df.shape,
        )
        raise
    return model, matrix


# reviewed
def resolve_threshold(df: pd.DataFrame, default: float) -> float:
    """Порог классификации τ из вектора или fallback трека.

    Берётся первая строка колонки ``THRESHOLD``.

    Args:
        df: кадр с возможным ``THRESHOLD``.
        default: ``config.default_main_threshold`` /
            ``config.default_second_threshold``.

    Returns:
        float τ.
    """
    # Нет колонки — берём дефолт трека.
    if "THRESHOLD" not in df.columns:
        logger.debug("Нет THRESHOLD в векторе — fallback {}", default)
        return float(default)
    # Берём порог из первой строки вектора.
    raw = df["THRESHOLD"].iloc[0]
    value = pd.to_numeric(raw, errors="coerce")
    if pd.isna(value):
        logger.warning("THRESHOLD в векторе не число или пуст (raw={!r}) — fallback {}", raw, default)
        return float(default)
    return float(value)


# reviewed
def df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """DataFrame → list[dict] для JSON-ответа (ключи каждой записи отсортированы).

    Args:
        df: таблица input/output в ответе.

    Returns:
        Список записей, безопасных для ``JSONResponse``.
    """
    rows = json.loads(df.to_json(orient="records", force_ascii=False))
    return [{k: row[k] for k in sorted(row.keys())} for row in rows]
