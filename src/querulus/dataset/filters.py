"""Фильтры датасета из configs/dataset_filters.json."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from querulus import PROJECT_ROOT

_FILTERS_PATH = PROJECT_ROOT / "configs" / "dataset_filters.json"


def load_dataset_filters() -> dict[str, Any]:
    """Загрузить configs/dataset_filters.json."""
    if not _FILTERS_PATH.exists():
        raise FileNotFoundError(f"Не найден конфиг фильтров: {_FILTERS_PATH}")
    return json.loads(_FILTERS_PATH.read_text(encoding="utf-8"))


def _quote_list(values: list[str]) -> str:
    return ", ".join(json.dumps(item, ensure_ascii=False) for item in values)


def victim_filter_query(filters: dict[str, Any] | None = None) -> str:
    """Pandas query для victim после загрузки parquet."""
    cfg = (filters or load_dataset_filters())["victim"]
    forms = _quote_list(cfg["refund_forms"])
    processes = _quote_list(cfg["loss_processes"])
    risk = json.dumps(cfg["risk"], ensure_ascii=False)
    date_from = json.dumps(cfg["loss_date_from"])
    date_to = json.dumps(cfg["loss_date_to"])
    victim_object_type = json.dumps(cfg["victim_object_type"], ensure_ascii=False)
    return (
        f"REFUND_FORM_DETAILED in [{forms}]"
        f" and LOSS_DATE_TIME >= {date_from}"
        f" and LOSS_DATE_TIME <= {date_to}"
        f" and LOSS_PROCESS in [{processes}]"
        f" and RISK == {risk}"
        f" and VICTIM_OBJECT_TYPE == {victim_object_type}"
    )


def apply_victim_filters(df: pd.DataFrame, filters: dict[str, Any] | None = None) -> pd.DataFrame:
    """Отфильтровать victim по единому конфигу."""
    return df.query(victim_filter_query(filters))


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
