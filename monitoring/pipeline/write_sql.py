"""Запись датафреймов мониторинга в Motor.agrafana (MSSQL).

При первом запуске ``ensure_monitoring_tables`` создаёт только таблицы
(IF NOT EXISTS). БД Motor и schema agrafana не трогаем — они уже есть на контуре.
Дальше только INSERT новых ключей (``write_append_guarded``).
ALTER / DROP не делаем: недостающие колонки — вручную или пересоздать таблицу.

Таблицы:
  Querulus_errors, Querulus_DataDrift, Querulus_FinEffect.
"""
from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.types import NVARCHAR

from pipeline.config import SCHEMA, TABLE_DRIFT, TABLE_ERRORS, TABLE_FIN_EFFECT

logger = logging.getLogger(__name__)


def get_engine(url: str) -> Engine:
    """SQLAlchemy engine (mssql+pyodbc). fast_executemany — быстрее bulk insert."""
    return create_engine(url, fast_executemany=True)


def _mssql_dtypes(df: pd.DataFrame) -> dict[str, NVARCHAR]:
    """Строки → NVARCHAR (иначе VARCHAR и кириллица как «???»)."""
    out: dict[str, NVARCHAR] = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            out[str(col)] = NVARCHAR(None)
    return out


def _quote_ident(name: str) -> str:
    """Экранирование ] в идентификаторе MSSQL."""
    return name.replace("]", "]]")


def _ddl_statements(schema: str = SCHEMA) -> list[str]:
    """Только CREATE TABLE IF NOT EXISTS (MSSQL). Без CREATE DATABASE / SCHEMA."""
    s = _quote_ident(schema)
    t_err = _quote_ident(TABLE_ERRORS)
    t_drift = _quote_ident(TABLE_DRIFT)
    t_fin = _quote_ident(TABLE_FIN_EFFECT)
    return [
        f"""
IF OBJECT_ID(N'[{s}].[{t_err}]', N'U') IS NULL
BEGIN
    CREATE TABLE [{s}].[{t_err}] (
        [run_date] DATE NULL,
        [stem] NVARCHAR(256) NULL,
        [slot] NVARCHAR(64) NULL,
        [model_label] NVARCHAR(256) NULL,
        [model_kind] NVARCHAR(256) NULL,
        [model_version] NVARCHAR(256) NULL,
        [error_fingerprint] NVARCHAR(64) NULL,
        [error_head] NVARCHAR(MAX) NULL,
        [count] INT NULL
    );
END
""",
        f"""
IF OBJECT_ID(N'[{s}].[{t_drift}]', N'U') IS NULL
BEGIN
    CREATE TABLE [{s}].[{t_drift}] (
        [model_name] NVARCHAR(256) NULL,
        [model_version] NVARCHAR(256) NULL,
        [date] DATE NULL,
        [col] NVARCHAR(256) NULL,
        [TYPE] NVARCHAR(64) NULL,
        [PSI] FLOAT NULL,
        [KL] FLOAT NULL,
        [JS] FLOAT NULL,
        [NaN_train] FLOAT NULL,
        [NaN_test] FLOAT NULL,
        [uniq_train] FLOAT NULL,
        [uniq_test] FLOAT NULL,
        [mode_train] NVARCHAR(MAX) NULL,
        [mode_test] NVARCHAR(MAX) NULL,
        [mean_train] FLOAT NULL,
        [mean_test] FLOAT NULL,
        [dif_train] NVARCHAR(MAX) NULL,
        [dif_test] NVARCHAR(MAX) NULL,
        [stem] NVARCHAR(256) NULL,
        [track] NVARCHAR(64) NULL,
        [model_label] NVARCHAR(256) NULL
    );
END
""",
        f"""
IF OBJECT_ID(N'[{s}].[{t_fin}]', N'U') IS NULL
BEGIN
    CREATE TABLE [{s}].[{t_fin}] (
        [week_start] DATE NULL,
        [stem] NVARCHAR(256) NULL,
        [slot] NVARCHAR(64) NULL,
        [model_label] NVARCHAR(256) NULL,
        [filial] NVARCHAR(256) NULL,
        [n_incidents] INT NULL,
        [n_model] INT NULL,
        [n_control] INT NULL,
        [itt] FLOAT NULL,
        [net] FLOAT NULL,
        [coverage] FLOAT NULL,
        [formula_version] NVARCHAR(128) NULL,
        [annual_window_yult] FLOAT NULL,
        [annual_pilot_current_yult] FLOAT NULL,
        [annual_pilot_full_yult] FLOAT NULL,
        [annual_network_current_yult] FLOAT NULL,
        [annual_network_full_yult] FLOAT NULL,
        [annual_network_full_compliance_yult] FLOAT NULL,
        [agreement_share_control] FLOAT NULL,
        [agreement_share_model] FLOAT NULL,
        [agreement_share_lift_pp] FLOAT NULL,
        [pretension_share_control] FLOAT NULL,
        [pretension_share_model] FLOAT NULL,
        [pretension_share_lift_pp] FLOAT NULL,
        [fu_share_control] FLOAT NULL,
        [fu_share_model] FLOAT NULL,
        [fu_share_lift_pp] FLOAT NULL,
        [court_share_control] FLOAT NULL,
        [court_share_model] FLOAT NULL,
        [court_share_lift_pp] FLOAT NULL,
        [agr_control_pct] FLOAT NULL,
        [agr_model_pct] FLOAT NULL,
        [gap_pp] FLOAT NULL
    );
END
""",
    ]


def ensure_monitoring_tables(
    engine: Engine,
    *,
    schema: str = SCHEMA,
) -> None:
    """Создать 3 таблицы, если их ещё нет (без ALTER / без CREATE DATABASE|SCHEMA)."""
    with engine.begin() as conn:
        for stmt in _ddl_statements(schema):
            conn.execute(text(stmt))
    logger.info(
        "ensure_monitoring_tables ok || schema=%s || tables=%s,%s,%s",
        schema,
        TABLE_ERRORS,
        TABLE_DRIFT,
        TABLE_FIN_EFFECT,
    )


def _require_existing_table(engine: Engine, schema: str, table: str) -> list[str]:
    """Таблица должна существовать. Возвращает список колонок."""
    insp = inspect(engine)
    if not insp.has_table(table, schema=schema):
        raise RuntimeError(
            f"Table [{schema}].[{table}] does not exist after ensure_monitoring_tables. "
            "Check Motor permissions (CREATE TABLE on schema agrafana)."
        )
    return [c["name"] for c in insp.get_columns(table, schema=schema)]


def _align_to_table(df: pd.DataFrame, table_cols: list[str], *, schema: str, table: str) -> pd.DataFrame:
    """Оставить только колонки, которые есть в таблице. Лишние в df — отбросить (не ALTER)."""
    extra = [c for c in df.columns if c not in table_cols]
    if extra:
        logger.warning(
            "Dropping columns not present in %s.%s (no ALTER): %s",
            schema,
            table,
            ", ".join(map(str, extra)),
        )
        df = df.drop(columns=list(extra))
    if df.empty and len(df.columns) == 0:
        raise ValueError(f"No overlapping columns with {schema}.{table}")
    return df


def write_replace(
    df: pd.DataFrame,
    *,
    engine: Engine,
    table: str,
    schema: str,
) -> int:
    """Заменить *данные* в таблице: DELETE всех строк → INSERT df.

    При отсутствии таблицы — сначала ensure_monitoring_tables.
    """
    ensure_monitoring_tables(engine, schema=schema)
    table_cols = _require_existing_table(engine, schema, table)
    if df.empty:
        logger.info("Empty frame — clearing %s.%s via DELETE", schema, table)
    else:
        df = _align_to_table(df, table_cols, schema=schema, table=table)

    safe_schema = _quote_ident(schema)
    safe_table = _quote_ident(table)
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM [{safe_schema}].[{safe_table}]"))

    if df.empty:
        logger.info("Cleared %s.%s (0 rows to insert)", schema, table)
        return 0

    df.to_sql(
        table,
        con=engine,
        schema=schema,
        if_exists="append",
        index=False,
        dtype=_mssql_dtypes(df),
    )
    logger.info("Wrote %s rows to %s.%s (delete+insert)", len(df), schema, table)
    return len(df)


def write_append_guarded(
    df: pd.DataFrame,
    *,
    engine: Engine,
    table: str,
    schema: str,
    guard_cols: list[str],
) -> int:
    """INSERT только строк с новыми ключами по guard_cols.

    При отсутствии таблицы — сначала ensure_monitoring_tables.
    Типично datadrift: guard = date + model_name + track.
    """
    ensure_monitoring_tables(engine, schema=schema)
    table_cols = _require_existing_table(engine, schema, table)
    if df.empty:
        logger.info("Skip empty frame for %s.%s", schema, table)
        return 0

    df = _align_to_table(df, table_cols, schema=schema, table=table)

    for col in guard_cols:
        if col not in df.columns:
            raise ValueError(f"guard column {col!r} missing in new frame")
        if col not in table_cols:
            raise ValueError(
                f"guard column {col!r} missing in {schema}.{table} — "
                "add it outside the pipeline (no ALTER here)"
            )

    existing = pd.read_sql_table(table, con=engine, schema=schema)
    if existing.empty:
        to_write = df
    else:
        key_new = df[guard_cols].astype(str).agg("|".join, axis=1)
        key_old = existing[guard_cols].astype(str).agg("|".join, axis=1)
        to_write = df.loc[~key_new.isin(set(key_old))]

    if to_write.empty:
        logger.info(
            "Guard hit: no new keys for %s.%s on %s", schema, table, guard_cols
        )
        return 0

    to_write.to_sql(
        table,
        con=engine,
        schema=schema,
        if_exists="append",
        index=False,
        dtype=_mssql_dtypes(to_write),
    )
    logger.info("Wrote %s new rows to %s.%s (append)", len(to_write), schema, table)
    return len(to_write)
