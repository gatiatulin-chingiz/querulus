"""Pandas-фильтры и merge-хелперы (без T-SQL)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from querulus import PROJECT_ROOT

_FILTERS_PATH = PROJECT_ROOT / "configs" / "dataset_filters.json"
VICTIM_OBJECT_TYPE_COLUMN = "VICTIM_OBJECT_TYPE"
_VICTIM_OBJECT_TYPE_ALIASES = ("VictimObjectType",)


def load_dataset_filters() -> dict[str, Any]:
    """Загрузить configs/dataset_filters.json."""
    if not _FILTERS_PATH.exists():
        raise FileNotFoundError(f"Не найден конфиг фильтров: {_FILTERS_PATH}")
    return json.loads(_FILTERS_PATH.read_text(encoding="utf-8"))


def _quote_list(values: list[str]) -> str:
    return ", ".join(json.dumps(item, ensure_ascii=False) for item in values)


def _normalize_victim_object_type_column(df: pd.DataFrame) -> pd.DataFrame:
    """Привести имя колонки типа объекта потерпевшего к VICTIM_OBJECT_TYPE."""
    if VICTIM_OBJECT_TYPE_COLUMN in df.columns:
        return df
    for alias in _VICTIM_OBJECT_TYPE_ALIASES:
        if alias in df.columns:
            return df.rename(columns={alias: VICTIM_OBJECT_TYPE_COLUMN})
    return df


def ensure_victim_object_type_column(
    df: pd.DataFrame, filters: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Гарантировать колонку VICTIM_OBJECT_TYPE в итоговом датасете."""
    df = _normalize_victim_object_type_column(df)
    if VICTIM_OBJECT_TYPE_COLUMN not in df.columns:
        raise KeyError(
            f"Колонка {VICTIM_OBJECT_TYPE_COLUMN!r} отсутствует. "
            "Загрузите df_loss_object_types.parquet или включите USE_SQL=True."
        )
    df = df.copy()
    df[VICTIM_OBJECT_TYPE_COLUMN] = df[VICTIM_OBJECT_TYPE_COLUMN].astype(str)
    return df


def victim_parquet_filter_query(filters: dict[str, Any] | None = None) -> str:
    """Pandas query для victim parquet (без VictimObjectType)."""
    cfg = (filters or load_dataset_filters())["victim"]
    forms = _quote_list(cfg["refund_forms"])
    processes = _quote_list(cfg["loss_processes"])
    risk = json.dumps(cfg["risk"], ensure_ascii=False)
    date_col = cfg.get("date_column") or cfg.get("loss_date_column") or "PAYMENT_ORDER_DATE_TIME"
    date_from = json.dumps(cfg.get("date_from") or cfg.get("loss_date_from"))
    parts = [
        f"REFUND_FORM_DETAILED in [{forms}]",
        f"{date_col} >= {date_from}",
        f"LOSS_PROCESS in [{processes}]",
        f"RISK == {risk}",
    ]
    date_to = cfg.get("date_to") or cfg.get("loss_date_to")
    if date_to:
        parts.insert(2, f"{date_col} <= {json.dumps(date_to)}")
    return " and ".join(parts)


def merge_loss_object_types(df: pd.DataFrame, df_loss_types: pd.DataFrame) -> pd.DataFrame:
    """Присоединить VictimObjectType и форму ВФ к victim по LOSS_NUMBER."""
    loss_types = _normalize_victim_object_type_column(df_loss_types)
    columns = ["LOSS_NUMBER", VICTIM_OBJECT_TYPE_COLUMN]
    if "REFUND_FORM_BY_PAYMENT_ORDER" in loss_types.columns:
        columns.append("REFUND_FORM_BY_PAYMENT_ORDER")
    return df.merge(loss_types[columns].drop_duplicates("LOSS_NUMBER"), on="LOSS_NUMBER", how="left")


def select_primary_loss_per_incident(df: pd.DataFrame) -> pd.DataFrame:
    """Оставить одну строку на инцидент: первичный убыток (минимальный LOSS_NUMBER)."""
    if "LOSS_NUMBER" not in df.columns:
        raise KeyError("Для выбора первичного убытка нужна колонка LOSS_NUMBER")
    return (
        df.sort_values(["INCIDENT_NUMBER", "LOSS_NUMBER"], ascending=[True, True])
        .drop_duplicates(subset=["INCIDENT_NUMBER"], keep="first")
        .reset_index(drop=True)
    )
