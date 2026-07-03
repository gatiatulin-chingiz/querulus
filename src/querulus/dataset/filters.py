"""Фильтры датасета из configs/dataset_filters.json."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from querulus import PROJECT_ROOT

_FILTERS_PATH = PROJECT_ROOT / "configs" / "dataset_filters.json"
VICTIM_OBJECT_TYPE_COLUMN = "VICTIM_OBJECT_TYPE"
_VICTIM_OBJECT_TYPE_ALIASES = ("VictimObjectType", "Victim_Object_Type")
_VICTIM_OBJECT_TYPE_CANONICAL = "".join(
    ch for ch in VICTIM_OBJECT_TYPE_COLUMN if ch.isalnum()
).upper()


def _canonical_column_key(name: str) -> str:
    return "".join(ch for ch in name if ch.isalnum()).upper()


def _find_victim_object_type_source_column(df: pd.DataFrame) -> str | None:
    """Найти исходное имя колонки типа объекта потерпевшего в victim parquet."""
    if VICTIM_OBJECT_TYPE_COLUMN in df.columns:
        return VICTIM_OBJECT_TYPE_COLUMN
    for alias in _VICTIM_OBJECT_TYPE_ALIASES:
        if alias in df.columns:
            return alias
    matches = [
        col
        for col in df.columns
        if _canonical_column_key(col) == _VICTIM_OBJECT_TYPE_CANONICAL
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        for col in matches:
            if col in _VICTIM_OBJECT_TYPE_ALIASES:
                return col
        return matches[0]
    fuzzy = [
        col
        for col in df.columns
        if "VICTIM" in col.upper()
        and "OBJECT" in col.upper()
        and "TYPE" in col.upper()
        and "OWNER" not in col.upper()
    ]
    if len(fuzzy) == 1:
        return fuzzy[0]
    return None


def _raise_missing_victim_object_type_column(df: pd.DataFrame) -> None:
    hints = [
        col
        for col in df.columns
        if "OBJECT" in col.upper() and "TYPE" in col.upper()
    ]
    hint_text = ", ".join(hints[:15]) if hints else "нет похожих колонок"
    raise KeyError(
        f"Колонка {VICTIM_OBJECT_TYPE_COLUMN!r} не найдена в victim parquet. "
        f"Похожие колонки: {hint_text}"
    )


def load_dataset_filters() -> dict[str, Any]:
    """Загрузить configs/dataset_filters.json."""
    if not _FILTERS_PATH.exists():
        raise FileNotFoundError(f"Не найден конфиг фильтров: {_FILTERS_PATH}")
    return json.loads(_FILTERS_PATH.read_text(encoding="utf-8"))


def _quote_list(values: list[str]) -> str:
    return ", ".join(json.dumps(item, ensure_ascii=False) for item in values)


def _normalize_victim_object_type_column(df: pd.DataFrame) -> pd.DataFrame:
    """Привести имя колонки типа объекта потерпевшего к VICTIM_OBJECT_TYPE."""
    source = _find_victim_object_type_source_column(df)
    if source is None or source == VICTIM_OBJECT_TYPE_COLUMN:
        return df
    return df.rename(columns={source: VICTIM_OBJECT_TYPE_COLUMN})


def ensure_victim_object_type_column(
    df: pd.DataFrame, filters: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Гарантировать колонку VICTIM_OBJECT_TYPE в итоговом датасете."""
    df = _normalize_victim_object_type_column(df)
    if VICTIM_OBJECT_TYPE_COLUMN not in df.columns:
        cfg = (filters or load_dataset_filters())["victim"]
        df = df.copy()
        df[VICTIM_OBJECT_TYPE_COLUMN] = cfg["victim_object_type"]
    else:
        df = df.copy()
        df[VICTIM_OBJECT_TYPE_COLUMN] = df[VICTIM_OBJECT_TYPE_COLUMN].astype(str)
    return df


def victim_filter_query(
    filters: dict[str, Any] | None = None,
    *,
    include_object_type: bool = True,
) -> str:
    """Pandas query для victim после загрузки parquet."""
    cfg = (filters or load_dataset_filters())["victim"]
    forms = _quote_list(cfg["refund_forms"])
    processes = _quote_list(cfg["loss_processes"])
    risk = json.dumps(cfg["risk"], ensure_ascii=False)
    date_from = json.dumps(cfg["loss_date_from"])
    date_to = json.dumps(cfg["loss_date_to"])
    query = (
        f"REFUND_FORM_DETAILED in [{forms}]"
        f" and LOSS_DATE_TIME >= {date_from}"
        f" and LOSS_DATE_TIME <= {date_to}"
        f" and LOSS_PROCESS in [{processes}]"
        f" and RISK == {risk}"
    )
    if include_object_type:
        victim_object_type = json.dumps(cfg["victim_object_type"], ensure_ascii=False)
        query += f" and {VICTIM_OBJECT_TYPE_COLUMN} == {victim_object_type}"
    return query


def apply_victim_filters(df: pd.DataFrame, filters: dict[str, Any] | None = None) -> pd.DataFrame:
    """Отфильтровать victim по единому конфигу."""
    cfg = (filters or load_dataset_filters())["victim"]
    df = _normalize_victim_object_type_column(df)
    if VICTIM_OBJECT_TYPE_COLUMN not in df.columns:
        _raise_missing_victim_object_type_column(df)
    df = df.query(victim_filter_query(filters, include_object_type=False))
    object_type = cfg["victim_object_type"]
    df = df[df[VICTIM_OBJECT_TYPE_COLUMN].astype(str) == object_type]
    return df.reset_index(drop=True)


def claims_sql_predicate(
    *,
    icnl_alias: str = "icnl",
    loss_alias: str = "l",
    filters: dict[str, Any] | None = None,
) -> str:
    """Фрагмент SQL AND для фильтрации исков (без ведущего WHERE)."""
    cfg = (filters or load_dataset_filters())["claims_sql"]
    origins = ", ".join(f"'{value}'" for value in cfg["claim_origins"])
    excluded = ", ".join(f"'{value}'" for value in cfg["exclude_claim_items"])
    processes = ", ".join(f"'{value}'" for value in cfg["loss_processes"])
    return (
        f"({icnl_alias}.ClaimOrigin in ({origins}) or {icnl_alias}.ClaimOrigin is null)\n"
        f"      AND ({icnl_alias}.ClaimItem not in ({excluded}) or {icnl_alias}.ClaimItem is null)\n"
        f"      AND {loss_alias}.LossProcess IN ({processes})"
    )


def select_primary_loss_per_incident(df: pd.DataFrame) -> pd.DataFrame:
    """Оставить одну строку на инцидент: убыток с максимальным LOSS_NUMBER."""
    if "LOSS_NUMBER" not in df.columns:
        raise KeyError("Для выбора первичного убытка нужна колонка LOSS_NUMBER")
    return (
        df.sort_values(["INCIDENT_NUMBER", "LOSS_NUMBER"], ascending=[True, False])
        .drop_duplicates(subset=["INCIDENT_NUMBER"], keep="first")
        .reset_index(drop=True)
    )
