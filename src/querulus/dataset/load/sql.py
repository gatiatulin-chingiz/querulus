"""Рендер SQL-шаблонов из dataset/sql/."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from querulus.dataset.load.io import LazyOisuuConnection, load_sql_artifact
from querulus.dataset.paths import DataPaths
from querulus.dataset.preprocess.filters import load_dataset_filters

_SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def render_sql(relative_path: str, /, **params: str) -> str:
    """Подставить ``{placeholders}`` в SQL-файл из ``dataset/sql/``."""
    template_path = _SQL_DIR / relative_path
    if not template_path.exists():
        raise FileNotFoundError(f"SQL-шаблон не найден: {template_path}")
    return template_path.read_text(encoding="utf-8").format(**params)


def _quote_sql_list(values: list[str]) -> str:
    return ", ".join(f"'{value}'" for value in values)


def claims_predicate_params(
    *,
    icnl_alias: str = "icnl",
    loss_alias: str = "l",
    filters: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Параметры для ``fragments/claims_predicate.sql``."""
    cfg = (filters or load_dataset_filters())["claims_sql"]
    return {
        "icnl_alias": icnl_alias,
        "loss_alias": loss_alias,
        "origins": _quote_sql_list(cfg["claim_origins"]),
        "excluded": _quote_sql_list(cfg["exclude_claim_items"]),
        "processes": _quote_sql_list(cfg["loss_processes"]),
    }


def render_claims_predicate(
    *,
    icnl_alias: str = "icnl",
    loss_alias: str = "l",
    filters: dict[str, Any] | None = None,
) -> str:
    """Фрагмент SQL AND для фильтрации исков."""
    return render_sql(
        "fragments/claims_predicate.sql",
        **claims_predicate_params(
            icnl_alias=icnl_alias,
            loss_alias=loss_alias,
            filters=filters,
        ),
    )


def loss_object_types_params(filters: dict[str, Any] | None = None) -> dict[str, str]:
    """Параметры для ``loss_object_types.sql``."""
    filters_cfg = filters or load_dataset_filters()
    victim_cfg = filters_cfg["victim"]
    sql_cfg = filters_cfg["loss_object_types_sql"]
    return {
        "insurance_type_group": sql_cfg["insurance_type_group"].replace("'", "''"),
        "processes": _quote_sql_list(victim_cfg["loss_processes"]),
        "risk": victim_cfg["risk"].replace("'", "''"),
    }


def pretension_surcharge_params(
    *,
    surcharge_alias: str,
    uts_alias: str,
    pretension_types: tuple[str, ...] | None = None,
    answer_types: tuple[str, ...] = ("Выплата", "Частичная выплата"),
) -> dict[str, str]:
    """Параметры для ``pretension_surcharge_by_incident.sql``."""
    type_filter = ""
    if pretension_types:
        types_list = _quote_sql_list(list(pretension_types))
        type_filter = f"\n      AND p.[PretensionType] IN ({types_list})"
    return {
        "answer_list": _quote_sql_list(list(answer_types)),
        "type_filter": type_filter,
        "surcharge_alias": surcharge_alias,
        "uts_alias": uts_alias,
    }


def load_named_sql_artifact(
    paths: DataPaths,
    conn: LazyOisuuConnection,
    directory: Path,
    name: str,
    sql_name: str,
    *,
    sql_params: dict[str, str] | None = None,
    use_sql: bool = False,
    save_checkpoint: bool = True,
    sql_reader: Callable | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Загрузить артефакт через ``render_sql``."""
    query = render_sql(sql_name, **(sql_params or {}))
    return load_sql_artifact(
        paths,
        conn,
        directory,
        name,
        query,
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
        sql_reader=sql_reader,
        columns=columns,
    )
