"""Боевые подключения к источникам Querulus monitoring.

Два разных контура (не путать с Motor.agrafana — туда пишем метрики через write_sql):

1) OISUU (MSSQL) — Airflow Connection ``OISUU-DB-BIZ_monitoring``
   → env ``database_host`` / ``database_user_*``
   → ошибки MatemModelsLogs, витрина сутяжности для fin_effect.

2) Oracle RTDM (PROD8) — Airflow Connection ``sandbox_rtdm``
   → env ``SERVER_PROD8`` / ``USER_PROD8`` / …
   → таблица ``spr_uu_json_log`` (JSON-логи маршрутизации СПР).

Паттерн как в archive MONITORING_BUU + spr_minimization oracle_utils.
Разворот JSON — как в archive/save_logs_for_DataDrift_*.ipynb.
"""
from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
import pyodbc
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from pipeline.config import Settings

# ---------------------------------------------------------------------------
# Разворот JSON_FILE: в логах СПР числа часто лежат как singleton-list
# ("[1]", "[1.0]" или Python-list [1]). Для DataDrift нужны настоящие числа.
# Регулярка матчит ТОЛЬКО строку вида "[число]" целиком.
# ---------------------------------------------------------------------------
_LIST_NUM_RE = re.compile(
    r"^\s*\[\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\]\s*$"
)

# Маршрутизация Querulus в spr_uu_json_log (system_from / system_to).
OISUU_SYSTEM = "ОИСУУ"
QUERULUS_SYSTEM_TO = "SPR_UU::QUERULUS"


# =============================================================================
# OISUU (MSSQL) — чтение
# =============================================================================


def _odbc_driver(preferred: str = "") -> str:
    """Выбрать установленный ODBC Driver for SQL Server.

    На контуре обычно 17/18; локально может отличаться.
    """
    installed = [d for d in pyodbc.drivers()]
    if preferred and preferred in installed:
        return preferred
    for name in (
        "ODBC Driver 18 for SQL Server",
        "ODBC Driver 17 for SQL Server",
        "SQL Server",
    ):
        if name in installed:
            return name
    if installed:
        return installed[0]
    raise RuntimeError("No ODBC SQL Server driver installed")


def connect_oisuu(settings: Settings):
    """pyodbc → OISUU_report.

    Используют: errors (MatemModelsLogs), fin_effect (ВитринаСутяжность).
    Encrypt=no — как в BUU spr_minimization_logic (внутренний контур).
    """
    if not settings.oisuu_user or not settings.oisuu_password:
        raise ValueError(
            "OISUU credentials missing: set database_user_name / database_user_pass "
            "(Airflow Connection OISUU-DB-BIZ_monitoring)"
        )
    driver = _odbc_driver()
    return pyodbc.connect(
        f"Driver={{{driver}}};"
        f"Server={settings.oisuu_host};"
        f"Trusted_Connection=no;Encrypt=no;"
        f"UID={settings.oisuu_user};"
        f"PWD={settings.oisuu_password};"
        f"Database={settings.oisuu_database};"
    )


def oisuu_sqlalchemy_url(settings: Settings) -> str:
    """SQLAlchemy URL на OISUU (редко нужен; большинство читает через pyodbc)."""
    if not settings.oisuu_user or not settings.oisuu_password:
        raise ValueError("OISUU credentials missing")
    driver = _odbc_driver()
    odbc = (
        f"DRIVER={{{driver}}};"
        f"SERVER={settings.oisuu_host},{settings.oisuu_port};"
        f"DATABASE={settings.oisuu_database};"
        f"UID={settings.oisuu_user};PWD={settings.oisuu_password};"
        "TrustServerCertificate=yes;"
    )
    # quote_plus — спецсимволы пароля/драйвера в URL
    return "mssql+pyodbc:///?odbc_connect=" + quote_plus(odbc)


def get_oisuu_engine(settings: Settings) -> Engine:
    """Engine на OISUU (если нужен pandas/SQLAlchemy вместо сырого pyodbc)."""
    return create_engine(oisuu_sqlalchemy_url(settings), future=True)


# =============================================================================
# Oracle RTDM — чтение spr_uu_json_log
# =============================================================================


def connect_oracle(settings: Settings):
    """python-oracledb (thin) → RTDM PROD8.

    DSN = host:port/service_name (SERVICE_NAME_PROD8, обычно rtdmdb08).
    """
    import oracledb

    if not settings.oracle_user or not settings.oracle_password:
        raise ValueError(
            "Oracle credentials missing: set USER_PROD8 / PASSWORD_PROD8 "
            "(Airflow Connection sandbox_rtdm)"
        )
    dsn = oracledb.makedsn(
        settings.oracle_host,
        settings.oracle_port,
        service_name=settings.oracle_service_name,
    )
    return oracledb.connect(
        user=settings.oracle_user,
        password=settings.oracle_password,
        dsn=dsn,
    )


def get_uu_json_log_table(conn) -> str:
    """Полное имя таблицы логов: ``rtdm.spr_uu_json_log`` или ``rtdm_etl.*``.

    Схема на стендах плавает — смотрим all_tables, кэшируем на процесс.
    Приоритет: RTDM, затем RTDM_ETL. Fallback: rtdm.spr_uu_json_log.
    """
    cached = getattr(get_uu_json_log_table, "_cached", None)
    if cached is not None:
        return cached
    cur = conn.cursor()
    owners: list[str] = []
    try:
        cur.execute(
            """
            SELECT owner
            FROM all_tables
            WHERE table_name = 'SPR_UU_JSON_LOG'
              AND owner IN ('RTDM', 'RTDM_ETL')
            ORDER BY CASE owner WHEN 'RTDM' THEN 0 ELSE 1 END
            """
        )
        owners = [str(r[0]).upper() for r in cur.fetchall()]
    finally:
        cur.close()
    name = f"{owners[0].lower()}.spr_uu_json_log" if owners else "rtdm.spr_uu_json_log"
    get_uu_json_log_table._cached = name
    return name


def _read_lobs(df: pd.DataFrame) -> pd.DataFrame:
    """JSON_FILE приходит как lazy LOB — читать, пока соединение открыто.

    Иначе после conn.close() будет oracledb DPY-1001.
    """
    import oracledb

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda v: v.read() if isinstance(v, oracledb.LOB) else v
            )
    return df


def query_spr_uu_json_log(
    conn,
    *,
    system_from: str | None = None,
    system_to: str | None = None,
    min_time_stamp=None,
    limit: int | None = None,
) -> pd.DataFrame:
    """SELECT из spr_uu_json_log с фильтрами маршрутизации.

    Bind-параметры Oracle (:name). Колонки приводим к lower — Oracle отдаёт UPPER.
    """
    query = f"""
        SELECT *
        FROM {get_uu_json_log_table(conn)}
        WHERE 1 = 1
    """
    params: dict[str, Any] = {}
    if system_from is not None:
        query += "\n          AND system_from = :system_from"
        params["system_from"] = system_from
    if system_to is not None:
        query += "\n          AND system_to = :system_to"
        params["system_to"] = system_to
    if min_time_stamp is not None:
        query += "\n          AND time_stamp >= :min_time_stamp"
        params["min_time_stamp"] = min_time_stamp
    query += "\n        ORDER BY process_id DESC"
    if limit:
        # Отладка / smoke: не тащить всю историю
        query += "\n        FETCH FIRST :limit ROWS ONLY"
        params["limit"] = int(limit)
    df = pd.read_sql_query(query, conn, params=params)
    df.columns = [str(c).lower() for c in df.columns]
    return _read_lobs(df)


def query_querulus_request_logs(
    conn,
    *,
    min_time_stamp=None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Запросы ОИСУУ → SPR_UU::QUERULUS (входные фичи для DataDrift).

    Это не ошибки и не ответы модели — payload запроса с фичами в JSON_FILE.
    """
    return query_spr_uu_json_log(
        conn,
        system_from=OISUU_SYSTEM,
        system_to=QUERULUS_SYSTEM_TO,
        min_time_stamp=min_time_stamp,
        limit=limit,
    )


# =============================================================================
# Разбор JSON_FILE → плоские колонки фич
# =============================================================================


def _coerce_singleton_list_number(v: Any) -> Any:
    """``[1]`` / ``"[1]"`` / ``"[1.0]"`` → число; остальное без изменений.

    После json_normalize часть колонок остаётся строкой-списком из логов СПР.
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return v
    # Уже list из одного числа
    if isinstance(v, list) and len(v) == 1 and isinstance(v[0], (int, float)):
        return v[0]
    if isinstance(v, str):
        m = _LIST_NUM_RE.match(v)
        if not m:
            return v
        num_f = float(m.group(1))
        return int(num_f) if num_f.is_integer() else num_f
    return v


def _parse_json_cell(x: Any) -> Any:
    """Ячейка JSON_FILE → dict для ``pd.json_normalize``.

    В логах встречается dict, list[dict], битая строка, пустота.
    Берём первый dict из list; скаляры/прочие list кладём в служебные ключи.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, (dict, list)):
        obj = x
    else:
        s = str(x).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return None
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return None
    if isinstance(obj, list):
        if not obj:
            return None
        # Типичный кейс: массив из одного объекта с фичами
        if isinstance(obj[0], dict):
            return obj[0]
        return {"_list": obj}
    if isinstance(obj, dict):
        return obj
    return {"_value": obj}


def expand_json_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Разворот колонки JSON → wide-фичи (как save_logs_for_DataDrift_*.ipynb).

    Исходную JSON-колонку убираем; нормализованные поля конкатеним слева направо.
    """
    if col not in df.columns:
        raise KeyError(f"column {col!r} not in frame: {list(df.columns)}")
    parsed = df[col].map(_parse_json_cell)
    expanded = pd.json_normalize(parsed)
    # applymap: на каждой ячейке снять singleton-list числа
    expanded = expanded.applymap(_coerce_singleton_list_number)
    return pd.concat([df.drop(columns=[col]), expanded], axis=1)
