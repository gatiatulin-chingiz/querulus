"""Write monitoring frames to schema agrafana (Motor / MSSQL)."""
from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.types import NVARCHAR

logger = logging.getLogger(__name__)


def get_engine(url: str) -> Engine:
    return create_engine(url, fast_executemany=True)


def _mssql_dtypes(df: pd.DataFrame) -> dict[str, NVARCHAR]:
    """Force Unicode string columns — pandas/pyodbc default VARCHAR → Cyrillic as ???."""
    out: dict[str, NVARCHAR] = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            out[str(col)] = NVARCHAR(None)
    return out


def ensure_schema(engine: Engine, schema: str) -> None:
    """CREATE SCHEMA if missing (MSSQL)."""
    # Identifier only — no user input beyond fixed "agrafana".
    safe = schema.replace("]", "]]")
    stmt = f"""
    IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = N'{safe}')
        EXEC(N'CREATE SCHEMA [{safe}]');
    """
    with engine.begin() as conn:
        conn.execute(text(stmt))


def _mssql_type_for_series(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "BIT"
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return "FLOAT"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "DATETIME2"
    return "NVARCHAR(MAX)"


def _ensure_missing_columns(
    engine: Engine,
    *,
    schema: str,
    table: str,
    df: pd.DataFrame,
    existing: pd.DataFrame,
) -> None:
    missing = [c for c in df.columns if c not in existing.columns]
    if not missing:
        return
    safe_schema = schema.replace("]", "]]")
    safe_table = table.replace("]", "]]")
    with engine.begin() as conn:
        for col in missing:
            safe_col = str(col).replace("]", "]]")
            col_type = _mssql_type_for_series(df[col])
            conn.execute(
                text(
                    f"ALTER TABLE [{safe_schema}].[{safe_table}] "
                    f"ADD [{safe_col}] {col_type} NULL"
                )
            )
    logger.info(
        "Added columns to %s.%s: %s",
        schema,
        table,
        ", ".join(missing),
    )


def write_replace(
    df: pd.DataFrame,
    *,
    engine: Engine,
    table: str,
    schema: str,
) -> int:
    ensure_schema(engine, schema)
    df.to_sql(
        table,
        con=engine,
        schema=schema,
        if_exists="replace",
        index=False,
        dtype=_mssql_dtypes(df),
    )
    logger.info("Wrote %s rows to %s.%s (replace)", len(df), schema, table)
    return len(df)


def write_append_guarded(
    df: pd.DataFrame,
    *,
    engine: Engine,
    table: str,
    schema: str,
    guard_cols: list[str],
) -> int:
    """Append rows unless guard key already exists in table."""
    ensure_schema(engine, schema)
    if df.empty:
        logger.info("Skip empty frame for %s.%s", schema, table)
        return 0

    try:
        existing = pd.read_sql_table(table, con=engine, schema=schema)
        insp_empty = False
    except (ValueError, Exception):
        existing = pd.DataFrame()
        insp_empty = True

    dtypes = _mssql_dtypes(df)
    if insp_empty or existing.empty:
        df.to_sql(
            table,
            con=engine,
            schema=schema,
            if_exists="append",
            index=False,
            dtype=dtypes,
        )
        logger.info("Wrote %s rows to %s.%s (create/append)", len(df), schema, table)
        return len(df)

    _ensure_missing_columns(engine, schema=schema, table=table, df=df, existing=existing)
    existing = pd.read_sql_table(table, con=engine, schema=schema)

    for col in guard_cols:
        if col not in df.columns:
            raise ValueError(f"guard column {col!r} missing in new frame")
        if col not in existing.columns:
            df.to_sql(
                table,
                con=engine,
                schema=schema,
                if_exists="append",
                index=False,
                dtype=dtypes,
            )
            return len(df)

    key_new = df[guard_cols].astype(str).agg("|".join, axis=1)
    key_old = existing[guard_cols].astype(str).agg("|".join, axis=1)
    mask = ~key_new.isin(set(key_old))
    to_write = df.loc[mask]
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
        dtype=dtypes,
    )
    logger.info("Wrote %s new rows to %s.%s (append)", len(to_write), schema, table)
    return len(to_write)
