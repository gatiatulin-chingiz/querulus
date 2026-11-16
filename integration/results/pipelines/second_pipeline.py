"""Second-пайплайн: базовые шаги main + FE/CPI; postprocess с калибровкой."""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

APPLICANT_FORM_REMAP: Dict[str, str] = {
    "СКРЫТЫЙ ЮРИСТ": "ЮРИСТ С ПОТЕРПЕВШИМ",
    "ПРЕДСТАВИТЕЛЬ (АВТОЮРИСТ)": "ПРЕДСТАВИТЕЛЬ (ПО ДОВЕРЕННОСТИ)",
    "ПРЕДСТАВИТЕЛЬ (НЕ АВТОЮРИСТ)": "ВЫГОДОПРИОБРЕТАТЕЛЬ",
}

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

INFLATION_BASE_YEAR: int = 2022

RU_CPI_LEVEL_VS_BASE: Dict[int, float] = {
    2017: 0.887278,
    2018: 0.925076,
    2019: 0.953198,
    2020: 1.0,
    2021: 1.0839,
    2022: 1.213318,
    2023: 1.303346,
    2024: 1.427424,
    2025: 1.507217,
    2026: 1.5704,
}

_base_level = float(RU_CPI_LEVEL_VS_BASE[INFLATION_BASE_YEAR])
_NORMALIZED_CPI: Dict[int, float] = {
    y: float(v) / _base_level for y, v in RU_CPI_LEVEL_VS_BASE.items()
}
del _base_level

MONETARY_COLUMNS_FOR_REAL: tuple[str, ...] = (
    "VALUE_BEFORE_WITH",
    "VALUE_BEFORE_WITHOUT",
)

SHARE_WORK_BINS: tuple[float, ...] = (0.3, 0.6)
SHARE_WORK_LABELS: tuple[str, ...] = ("low", "mid", "high")
AMOUNT_REPAIR_BINS: tuple[float, ...] = (120_000.0, 365_000.0)

# reviewed
def load_ts_country_by_brand(models_dir: Path) -> Dict[str, str]:
    """Бренд ТС → страна из ``Other_dicts.json`` (ключи и значения UPPER)."""
    path = Path(models_dir) / "Other_dicts.json"
    if not path.is_file():
        raise FileNotFoundError(f"Нет Other_dicts.json: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Пустой Other_dicts.json: {path}")
    brands_raw = payload.get("dict_ts_country")
    if not brands_raw:
        raise ValueError(f"Нет dict_ts_country в {path}")
    return {k.upper(): v.upper() for k, v in brands_raw.items()}

# reviewed
def _require_columns(df: pd.DataFrame, columns: List[str], role: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Нет обязательных колонок для {role}: {missing}")

# reviewed
def _prepare_vector_upper(features_values: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(features_values)
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = (
                df[column]
                .map(lambda v: np.nan if v == "" else v)
                .map(lambda v: v.upper() if isinstance(v, str) else v)
            )
    return df

# reviewed
def _coerce_and_enrich(
    df: pd.DataFrame,
    ts_country_by_brand: Dict[str, str],
) -> pd.DataFrame:
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
    out["APPLICANT_FORM"] = out["APPLICANT_FORM"].map(
        lambda x: APPLICANT_FORM_REMAP.get(x, x) if isinstance(x, str) else x
    )
    out["NOT_NOTIFICATION"] = out["NOT_NOTIFICATION"].map(
        lambda v: 0 if v == "НЕ УВЕДОМЛЯТЬ" else 1
    )
    out["VIC_TS_COUNTRY"] = out["VICTIM_VEHICLE_BRAND"].map(
        lambda x: ts_country_by_brand.get(x, "OTHER") if isinstance(x, str) else "OTHER"
    )
    return out

# reviewed
def _validate_derived_dates_strict(df: pd.DataFrame) -> None:
    if df["EVENT_YEAR"].isna().any():
        raise ValueError("EVENT_DATE пустой")
    if df["APPLY_DELAY"].isna().any():
        raise ValueError("PAYMENT_ORDER_DATE_TIME пустой")
    if (df["APPLY_DELAY"] < 0).any():
        raise ValueError("EVENT_DATE > PAYMENT_ORDER_DATE_TIME")

# reviewed
def resolve_cpi_year(year: float | int) -> int:
    table = RU_CPI_LEVEL_VS_BASE
    known = sorted(table)
    min_y, max_y = known[0], known[-1]
    y = int(year)
    if y in table:
        return y
    if y > max_y:
        msg = (
            f"CPI-таблица устарела: год инцидента {y} > max={max_y}. "
            f"LOCF year={max_y}."
        )
        logger.warning(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
        return max_y
    if y < min_y:
        logger.warning(
            "CPI: год инцидента {} раньше таблицы (min={}); берём earliest",
            y,
            min_y,
        )
        return min_y
    nearest = min(known, key=lambda k: abs(k - y))
    logger.warning("CPI: год {} отсутствует в таблице; ближайший {}", y, nearest)
    return nearest

# reviewed
def _real_feature_name(column: str) -> str:
    return f"FE_{column}_REAL_{INFLATION_BASE_YEAR}"

# reviewed
def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator != 0)

# reviewed
def _amount_bins(amount: pd.Series, edges: tuple[float, ...]) -> pd.Series:
    low, high = edges
    labels = [f"<{int(low)}", f"{int(low)}-{int(high)}", f">{int(high)}"]
    cut = pd.cut(
        amount,
        bins=[-np.inf, low, high, np.inf],
        labels=labels,
        right=False,
        ordered=True,
    )
    return cut.astype(object).where(cut.notna(), np.nan)

# reviewed
def _add_second_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    years = out["PAYMENT_ORDER_YEAR"]
    for col in MONETARY_COLUMNS_FOR_REAL:
        level = years.map(lambda y: _NORMALIZED_CPI[resolve_cpi_year(y)])
        out[_real_feature_name(col)] = out[col] / level

    if "SHARE_WORK" not in out.columns:
        out["SHARE_WORK"] = _safe_div(out["WORK"], out["VALUE_BEFORE_WITHOUT"])

    value_without_real = out[_real_feature_name("VALUE_BEFORE_WITHOUT")]
    value_with_real = out[_real_feature_name("VALUE_BEFORE_WITH")]

    both_ok = (value_without_real > 0) & (value_with_real >= 0)
    both_positive = (value_without_real > 0) & (value_with_real > 0)
    out["FE_VALUE_BEFORE_DIFF"] = (value_without_real - value_with_real).where(both_ok)
    out["FE_VALUE_BEFORE_RATIO"] = _safe_div(value_with_real, value_without_real).where(
        both_positive
    )
    out["FE_VALUE_BEFORE_WITH_BIN"] = _amount_bins(value_with_real, AMOUNT_REPAIR_BINS)

    out["FE_DAYS_EVENT_TO_LOSS"] = (
        out["LOSS_DATE_TIME"] - out["EVENT_DATE"]
    ).dt.days
    out["FE_DAYS_TO_PH_CONTRACT_END"] = (
        out["POLICYHOLDER_CONTRACT_END_DATE"] - out["EVENT_DATE"]
    ).dt.days

    low, high = SHARE_WORK_BINS
    cut = pd.cut(
        out["SHARE_WORK"],
        bins=[-np.inf, low, high, np.inf],
        labels=list(SHARE_WORK_LABELS),
        right=False,
        ordered=True,
    )
    out["FE_SHARE_WORK_TIER"] = cut.astype(object).where(cut.notna(), np.nan)

    out["FE_VICTIM_POWER_PER_TON"] = (
        _safe_div(out["VICTIM_CAPACITY_ENGINE"], out["VICTIM_MAX_WEIGHT"]) * 1e6
    )

    age = (
        (out["EVENT_DATE"].dt.year - out["VICTIM_OBJECT_YEAR"]) * 12
        + (out["EVENT_DATE"].dt.month - 1)
    )
    age = age.where(age.isna(), age.clip(lower=0))
    out["VICTIM_OBJECT_AGE_MONTHS"] = age.round().astype("Int64")

    out["FE_VALUE_BEFORE_DIFF"] = out["FE_VALUE_BEFORE_DIFF"].clip(lower=0)
    return out

# reviewed
def preprocess(
    features_values: List[Dict],
    ts_country_by_brand: Dict[str, str],
) -> pd.DataFrame:
    """Базовый вход + FE/CPI + проверка engineered."""
    df = _prepare_vector_upper(features_values)
    _require_columns(df, INPUT_REQUIRED_COLUMNS, "input")
    df = _coerce_and_enrich(df, ts_country_by_brand)
    _validate_derived_dates_strict(df)
    df = _add_second_features(df)
    _require_columns(df, ENGINEERED_FEATURES, "model")
    return df

# reviewed
def _apply_cf_calibrator(calibrator: Any, raw_scores: List[float]) -> List[float]:
    arr = np.asarray(
        calibrator.predict(np.asarray(raw_scores, dtype=float)), dtype=float
    )
    return [float(x) for x in arr.ravel()]

# reviewed
def _apply_rg_calibrator(calibrator: Any, raw_reg: List[float]) -> List[float]:
    arr = np.asarray(
        calibrator.predict(np.asarray(raw_reg, dtype=float)), dtype=float
    )
    return [float(max(0.0, x)) for x in arr.ravel()]

# reviewed
def _masked_predictions(
    labels: List[float],
    raw_reg: List[float],
    df_common: pd.DataFrame,
) -> List[float]:
    if "IS_LAWYER" in df_common.columns:
        is_lawyer = (
            pd.to_numeric(df_common["IS_LAWYER"], errors="coerce").fillna(0).eq(1)
        ).tolist()
    else:
        is_lawyer = [False] * len(labels)
    masked: List[float] = []
    for c, r, lawyer in zip(labels, raw_reg, is_lawyer):
        if c or lawyer:
            masked.append(r)
        else:
            masked.append(0.0)
    return masked

# reviewed
def postprocess(
    df_pre: pd.DataFrame,
    raw_clf: List[float],
    raw_reg: List[float],
    calibrators: Optional[Tuple[Any, Any]],
    threshold: float,
) -> Dict[str, Any]:
    """Калибровка CF/RG + порог + маска IS_LAWYER."""
    if not calibrators or calibrators[0] is None or calibrators[1] is None:
        raise ValueError("second_pipeline требует cf и rg калибраторы")
    cf_cal, rg_cal = calibrators
    proba = _apply_cf_calibrator(cf_cal, raw_clf)
    cal_reg = _apply_rg_calibrator(rg_cal, raw_reg)
    labels = [1.0 if s >= threshold else 0.0 for s in proba]
    regression = _masked_predictions(labels, cal_reg, df_pre)
    return {"proba": proba, "labels": labels, "regression": regression}
