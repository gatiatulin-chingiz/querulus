"""Конфиг пайплайна: дефолты в коде, секреты только из Airflow env.

Цепочка на контуре (как MONITORING_BUU)::

    Airflow Connections / Variables  →  env_vars в K8s pod  →  load_settings()

Имена ключей совместимы с BUU:

* OISUU: ``database_host``, ``database_user_name``, … (или ``OISUU_DB_*``)
* Motor: ``MSSQL_*`` / ``motor_*`` или ``GRAFANA_SQLALCHEMY_URL``
* Oracle: ``SERVER_PROD8``, ``USER_PROD8``, ``PASSWORD_PROD8``, …

Несекретные параметры (окна, PSI, получатели) — константы ниже;
env/Variable перекрывают при необходимости.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import quote_plus


# --- целевые таблицы Motor.agrafana ------------------------------------------
SCHEMA = "agrafana"
TABLE_ERRORS = "Querulus_errors"
TABLE_DRIFT = "querulus_DataDrift"
TABLE_FIN_EFFECT = "Querulus_FinEffect"

# --- несекретные дефолты -----------------------------------------------------
DEFAULT_PSI_THRESHOLD = 0.3
DEFAULT_DRIFT_WINDOW_WEEKS = 2.0
DEFAULT_ERRORS_LOOKBACK_DAYS = 14
DEFAULT_FIN_EFFECT_STEM = "prod"
DEFAULT_FIN_EFFECT_SLOT = "main"
DEFAULT_FIN_EFFECT_VITRINA = "[OISUU_report].[dbo].[ВитринаСутяжность]"

# DataDrift: main/second — разные Hive-таблицы (как models.json).
# Менять stem/hive здесь при смене модели; env REFERENCE_HIVE_TABLE_MAIN/SECOND перекрывает.
DEFAULT_DRIFT_MODELS: tuple[dict[str, str], ...] = (
    {
        "slot": "main",
        "stem": "querulus_ansamble_2026_04_18_v1",
        "hive": "models.querulus_ansamble_2026_04_18_v1",
        "label": "querulus_ansamble_2026_04_18_v1",
    },
    {
        "slot": "second",
        "stem": "UU_querulus_2026_11_05_12_44_24",
        "hive": "models.UU_querulus_2026_11_05_12_44_24",
        "label": "UU_querulus_2026_11_05_12_44_24",
    },
)
# Опциональный срез train по PAYMENT_ORDER_DATE_TIME; пусто = без среза
DEFAULT_REFERENCE_TRAIN_START = ""
DEFAULT_REFERENCE_TRAIN_END = ""

DEFAULT_OISUU_HOST = "oisuu-db-biz.vsk.ru"
DEFAULT_OISUU_PORT = 1433
DEFAULT_OISUU_DATABASE = "OISUU_report"
DEFAULT_ORACLE_HOST = "v-rtdm-db-p8.vsk.ru"
DEFAULT_ORACLE_PORT = 1521
DEFAULT_ORACLE_SERVICE = "rtdmdb08"
DEFAULT_MSSQL_PORT = "1433"
DEFAULT_MSSQL_DATABASE = "Motor"
DEFAULT_MSSQL_DRIVER = "ODBC Driver 18 for SQL Server"

DEFAULT_EMAIL_SMTP = "internalsmtp.vsk.ru"
DEFAULT_EMAIL_PORT = 2525
DEFAULT_EMAIL_SENDER = "artificial.intelligence@vsk.ru"
DEFAULT_EMAIL_RECEIVERS = (
    "Gatyatulin@vsk.ru",
    "AVBondarev@vsk.ru",
    "Matsera@VSK.RU",
)
DEFAULT_GRAFANA_DASHBOARD_URL = (
    "https://grafana.vsk.ru/d/afkhng0p8bym8f/querulus"
)
DEFAULT_ALERT_LOOKBACK_DAYS = 7
DEFAULT_ALERT_WEEKDAY = 0  # Monday


@dataclass(frozen=True)
class Settings:
    """Снимок настроек на один запуск (prod)."""

    grafana_sqlalchemy_url: str

    oisuu_host: str
    oisuu_port: int
    oisuu_database: str
    oisuu_user: str
    oisuu_password: str

    oracle_host: str
    oracle_port: int
    oracle_user: str
    oracle_password: str
    oracle_service_name: str

    # DataDrift: список моделей (slot/stem/hive/label) — только Hive
    drift_models: tuple[dict[str, str], ...]
    reference_train_start: str  # ISO date или ""
    reference_train_end: str
    drift_window_weeks: float
    errors_lookback_days: int
    errors_model_kinds: tuple[str, ...]

    fin_effect_monitoring_path: str
    fin_effect_retro_path: str
    fin_effect_vitrina_table: str
    fin_effect_stem: str
    fin_effect_slot: str

    email_receivers: tuple[str, ...]
    grafana_dashboard_url: str
    alert_lookback_days: int

    schema: str = SCHEMA
    table_errors: str = TABLE_ERRORS
    table_drift: str = TABLE_DRIFT
    table_fin_effect: str = TABLE_FIN_EFFECT
    psi_threshold: float = DEFAULT_PSI_THRESHOLD


def _resolve_db_url() -> str:
    """SQLAlchemy URL на Motor (запись agrafana)."""
    url = os.environ.get("GRAFANA_SQLALCHEMY_URL", "").strip()
    if url:
        return url

    user = os.environ.get("MSSQL_USER", "").strip() or os.environ.get(
        "motor_user_name", ""
    ).strip()
    password = os.environ.get("MSSQL_SA_PASSWORD", "").strip() or os.environ.get(
        "motor_user_pass", ""
    ).strip()
    if not user or not password:
        raise ValueError(
            "Set GRAFANA_SQLALCHEMY_URL or MSSQL_USER+MSSQL_SA_PASSWORD "
            "(Airflow ACTUAR2_monitoring → MSSQL_* / motor_*)"
        )

    host = (
        os.environ.get("MSSQL_HOST", "").strip()
        or os.environ.get("motor_host", "").strip()
        or "localhost"
    )
    port = os.environ.get("MSSQL_PORT", DEFAULT_MSSQL_PORT).strip() or DEFAULT_MSSQL_PORT
    database = (
        os.environ.get("MSSQL_DATABASE", "").strip()
        or os.environ.get("motor_name", "").strip()
        or DEFAULT_MSSQL_DATABASE
    )
    driver = (
        os.environ.get("MSSQL_ODBC_DRIVER", DEFAULT_MSSQL_DRIVER).strip()
        or DEFAULT_MSSQL_DRIVER
    )
    odbc = (
        f"DRIVER={{{driver}}};"
        f"SERVER={host},{port};"
        f"DATABASE={database};"
        f"UID={user};PWD={password};"
        "TrustServerCertificate=yes;"
    )
    return "mssql+pyodbc:///?odbc_connect=" + quote_plus(odbc)


def _split_csv(raw: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in raw.split(",") if s.strip())


def load_settings() -> Settings:
    """Собрать Settings из env (Airflow) + дефолты констант выше."""
    # Опционально перекрыть hive-таблицы без правки кода
    models: list[dict[str, str]] = [dict(m) for m in DEFAULT_DRIFT_MODELS]
    hive_main = os.environ.get("REFERENCE_HIVE_TABLE_MAIN", "").strip()
    hive_second = os.environ.get("REFERENCE_HIVE_TABLE_SECOND", "").strip()
    for m in models:
        if m.get("slot") == "main" and hive_main:
            m["hive"] = hive_main
        if m.get("slot") == "second" and hive_second:
            m["hive"] = hive_second

    receivers_raw = os.environ.get("EMAIL_RECEIVERS", "").strip()
    receivers = (
        _split_csv(receivers_raw) if receivers_raw else DEFAULT_EMAIL_RECEIVERS
    )

    return Settings(
        grafana_sqlalchemy_url=_resolve_db_url(),
        oisuu_host=os.environ.get("database_host", "").strip()
        or os.environ.get("OISUU_DB_SERVER", "").strip()
        or DEFAULT_OISUU_HOST,
        oisuu_port=int(
            os.environ.get("database_port", "").strip()
            or os.environ.get("OISUU_DB_PORT", str(DEFAULT_OISUU_PORT))
            or str(DEFAULT_OISUU_PORT)
        ),
        oisuu_database=os.environ.get("database_name", "").strip()
        or os.environ.get("OISUU_DB_DATABASE", "").strip()
        or DEFAULT_OISUU_DATABASE,
        oisuu_user=os.environ.get("database_user_name", "").strip()
        or os.environ.get("OISUU_DB_USER", "").strip(),
        oisuu_password=os.environ.get("database_user_pass", "").strip()
        or os.environ.get("OISUU_DB_PASSWORD", "").strip(),
        oracle_host=os.environ.get("SERVER_PROD8", "").strip() or DEFAULT_ORACLE_HOST,
        oracle_port=int(
            os.environ.get("PORT_PROD8", str(DEFAULT_ORACLE_PORT))
            or str(DEFAULT_ORACLE_PORT)
        ),
        oracle_user=os.environ.get("USER_PROD8", "").strip(),
        oracle_password=os.environ.get("PASSWORD_PROD8", "").strip(),
        oracle_service_name=os.environ.get("SERVICE_NAME_PROD8", "").strip()
        or DEFAULT_ORACLE_SERVICE,
        drift_models=tuple(models),
        reference_train_start=os.environ.get(
            "REFERENCE_TRAIN_START", DEFAULT_REFERENCE_TRAIN_START
        ).strip(),
        reference_train_end=os.environ.get(
            "REFERENCE_TRAIN_END", DEFAULT_REFERENCE_TRAIN_END
        ).strip(),
        drift_window_weeks=float(
            os.environ.get("DRIFT_WINDOW_WEEKS", str(DEFAULT_DRIFT_WINDOW_WEEKS))
            or str(DEFAULT_DRIFT_WINDOW_WEEKS)
        ),
        errors_lookback_days=int(
            os.environ.get("ERRORS_LOOKBACK_DAYS", str(DEFAULT_ERRORS_LOOKBACK_DAYS))
            or str(DEFAULT_ERRORS_LOOKBACK_DAYS)
        ),
        errors_model_kinds=_split_csv(os.environ.get("ERRORS_MODEL_KINDS", "").strip()),
        fin_effect_monitoring_path=os.environ.get(
            "FIN_EFFECT_MONITORING_PATH", ""
        ).strip(),
        fin_effect_retro_path=os.environ.get("FIN_EFFECT_RETRO_PATH", "").strip(),
        fin_effect_vitrina_table=os.environ.get(
            "FIN_EFFECT_VITRINA_TABLE", DEFAULT_FIN_EFFECT_VITRINA
        ).strip()
        or DEFAULT_FIN_EFFECT_VITRINA,
        fin_effect_stem=os.environ.get("FIN_EFFECT_STEM", DEFAULT_FIN_EFFECT_STEM).strip()
        or DEFAULT_FIN_EFFECT_STEM,
        fin_effect_slot=os.environ.get("FIN_EFFECT_SLOT", DEFAULT_FIN_EFFECT_SLOT).strip()
        or DEFAULT_FIN_EFFECT_SLOT,
        email_receivers=receivers,
        grafana_dashboard_url=os.environ.get("GRAFANA_DASHBOARD_URL", "").strip()
        or DEFAULT_GRAFANA_DASHBOARD_URL,
        alert_lookback_days=int(
            os.environ.get("ALERT_LOOKBACK_DAYS", str(DEFAULT_ALERT_LOOKBACK_DAYS))
            or str(DEFAULT_ALERT_LOOKBACK_DAYS)
        ),
        psi_threshold=float(
            os.environ.get("PSI_THRESHOLD", str(DEFAULT_PSI_THRESHOLD))
            or str(DEFAULT_PSI_THRESHOLD)
        ),
    )
