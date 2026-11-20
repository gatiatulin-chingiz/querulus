"""Запись датафреймов мониторинга в уже существующие таблицы MSSQL (agrafana).

Важно для контура: DDL запрещён — не создаём schema/таблицы, не ALTER, не DROP.
Пайплайн только INSERT новых ключей (write_append_guarded).
write_replace (DELETE+INSERT) оставлен для ручного полного сброса, скрипты его не зовут.

Таблицы готовит DBA / init вне этого кода:
  Querulus_errors, querulus_DataDrift, Querulus_FinEffect.
"""
from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.types import NVARCHAR

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


def _require_existing_table(engine: Engine, schema: str, table: str) -> list[str]:
    """Таблица должна уже существовать. Возвращает список колонок. DDL не трогаем."""
    insp = inspect(engine)
    if not insp.has_table(table, schema=schema):
        raise RuntimeError(
            f"Table [{schema}].[{table}] does not exist. "
            "Create it outside the pipeline — writers only INSERT/DELETE rows."
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

    Структура таблицы (DDL) не меняется. Для Querulus_errors / Querulus_FinEffect.
    """
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
    """INSERT только строк с новыми ключами по guard_cols (без DDL).

    Типично datadrift: guard = date + model_name + track.
    Таблица должна существовать заранее.
    """
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
