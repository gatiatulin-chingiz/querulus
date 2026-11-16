"""Main-пайплайн: preprocess без FE/CPI + postprocess (маска, без калибровки)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
def preprocess(
    features_values: List[Dict],
    ts_country_by_brand: Dict[str, str],
) -> pd.DataFrame:
    """UPPER → required columns → enrich → даты (без FE/CPI)."""
    df = _prepare_vector_upper(features_values)
    _require_columns(df, INPUT_REQUIRED_COLUMNS, "input")
    df = _coerce_and_enrich(df, ts_country_by_brand)
    _validate_derived_dates_strict(df)
    return df

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
    """Без калибровки: порог + маска IS_LAWYER."""
    del calibrators
    proba = list(raw_clf)
    labels = [1.0 if s >= threshold else 0.0 for s in proba]
    regression = _masked_predictions(labels, list(raw_reg), df_pre)
    return {"proba": proba, "labels": labels, "regression": regression}
