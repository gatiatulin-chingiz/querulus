"""Фильтры датасета из configs/dataset_filters.json."""
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
    date_from = json.dumps(cfg["loss_date_from"])
    date_to = json.dumps(cfg["loss_date_to"])
    return (
        f"REFUND_FORM_DETAILED in [{forms}]"
        f" and LOSS_DATE_TIME >= {date_from}"
        f" and LOSS_DATE_TIME <= {date_to}"
        f" and LOSS_PROCESS in [{processes}]"
        f" and RISK == {risk}"
    )


def loss_object_types_sql(filters: dict[str, Any] | None = None) -> str:
    """SQL: поля из oisuu81_t_Losses, отсутствующие в victim parquet."""
    filters_cfg = filters or load_dataset_filters()
    victim_cfg = filters_cfg["victim"]
    sql_cfg = filters_cfg["loss_object_types_sql"]
    processes = ", ".join(f"'{value}'" for value in victim_cfg["loss_processes"])
    risk = victim_cfg["risk"].replace("'", "''")
    insurance_type_group = sql_cfg["insurance_type_group"].replace("'", "''")
    return f"""
    SELECT
        l.LossNumber AS LOSS_NUMBER,
        l.VictimObjectType AS VICTIM_OBJECT_TYPE,
        l.RefundFormByPaymentOrder AS REFUND_FORM_BY_PAYMENT_ORDER
    FROM [OISUU_report].[dbo].[oisuu81_t_Losses] AS l
    WHERE l.InsuranceTypeGroup = '{insurance_type_group}'
      AND l.LossProcess IN ({processes})
      AND l.Risk = '{risk}'
    """


def merge_loss_object_types(df: pd.DataFrame, df_loss_types: pd.DataFrame) -> pd.DataFrame:
    """Присоединить поля из oisuu81_t_Losses к victim по LOSS_NUMBER."""
    loss_types = _normalize_victim_object_type_column(df_loss_types)
    columns = ["LOSS_NUMBER", VICTIM_OBJECT_TYPE_COLUMN]
    if "REFUND_FORM_BY_PAYMENT_ORDER" in loss_types.columns:
        columns.append("REFUND_FORM_BY_PAYMENT_ORDER")
    return df.merge(loss_types[columns].drop_duplicates("LOSS_NUMBER"), on="LOSS_NUMBER", how="left")


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
    """Оставить одну строку на инцидент: первичный убыток (минимальный LOSS_NUMBER)."""
    if "LOSS_NUMBER" not in df.columns:
        raise KeyError("Для выбора первичного убытка нужна колонка LOSS_NUMBER")
    return (
        df.sort_values(["INCIDENT_NUMBER", "LOSS_NUMBER"], ascending=[True, True])
        .drop_duplicates(subset=["INCIDENT_NUMBER"], keep="first")
        .reset_index(drop=True)
    )
