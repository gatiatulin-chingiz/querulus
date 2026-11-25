"""Сборка Settings из env (Airflow на контуре / pipeline/.env локально).

Не путать с ``config.py`` — там только константы (модели, окна, таблицы).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

from pipeline.config import (
    DEFAULT_DRIFT_LOOKBACK_DAYS,
    DEFAULT_DRIFT_MODELS,
    DEFAULT_EMAIL_RECEIVERS,
    DEFAULT_ERRORS_LOOKBACK_DAYS,
    DEFAULT_ERRORS_MODEL_KINDS,
    DEFAULT_PSI_THRESHOLD,
    DEFAULT_REFERENCE_TRAIN_END,
    DEFAULT_REFERENCE_TRAIN_START,
    SCHEMA,
    TABLE_DRIFT,
    TABLE_ERRORS,
    TABLE_FIN_EFFECT,
)

_PIPELINE_DIR = Path(__file__).resolve().parent
_DOTENV_PATH = _PIPELINE_DIR / ".env"

# Как get_sql_driver_name() в MONITORING_BUU helpers
_MOTOR_ODBC_DRIVER = "ODBC Driver 18 for SQL Server"


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

    # DataDrift: список моделей (slot/stem/hive/features) — только Hive
    drift_models: tuple[dict, ...]
    reference_train_start: str  # ISO date или ""
    reference_train_end: str
    drift_lookback_days: int
    errors_lookback_days: int
    errors_model_kinds: tuple[str, ...]

    fin_effect_monitoring_path: str
    fin_effect_retro_path: str

    email_receivers: tuple[str, ...]

    schema: str = SCHEMA
    table_errors: str = TABLE_ERRORS
    table_drift: str = TABLE_DRIFT
    table_fin_effect: str = TABLE_FIN_EFFECT
    psi_threshold: float = DEFAULT_PSI_THRESHOLD


def _load_dotenv(path: Path = _DOTENV_PATH) -> None:
    """Подтянуть pipeline/.env, не перекрывая уже заданные переменные (Airflow)."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[key] = value


def _env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _env_int(*names: str, default: int | None = None) -> int:
    raw = _env(*names)
    if raw:
        return int(raw)
    if default is not None:
        return default
    raise ValueError(
        f"Missing int env: set one of {', '.join(names)} (Airflow or pipeline/.env)"
    )


def _resolve_db_url() -> str:
    """SQLAlchemy URL на Motor (agrafana) — ключи motor_* как в BUU."""
    host = _env("motor_host")
    database = _env("motor_name")
    user = _env("motor_user_name")
    password = _env("motor_user_pass")
    missing = [
        name
        for name, val in (
            ("motor_host", host),
            ("motor_name", database),
            ("motor_user_name", user),
            ("motor_user_pass", password),
        )
        if not val
    ]
    if missing:
        raise ValueError(
            "Motor connection incomplete. Set "
            f"{', '.join(missing)} (Airflow ACTUAR2_monitoring / pipeline/.env)"
        )

    odbc = (
        f"DRIVER={{{_MOTOR_ODBC_DRIVER}}};"
        f"SERVER={host};"
        f"DATABASE={database};"
        f"UID={user};PWD={password};"
        "TrustServerCertificate=yes;"
    )
    return "mssql+pyodbc:///?odbc_connect=" + quote_plus(odbc)


def _split_csv(raw: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in raw.split(",") if s.strip())


def load_settings() -> Settings:
    """Собрать Settings из env (Airflow или локальный pipeline/.env).

    stem / hive / features — только из DEFAULT_DRIFT_MODELS в config.py.
    """
    _load_dotenv()
    models: list[dict] = [dict(m) for m in DEFAULT_DRIFT_MODELS]

    receivers_raw = _env("EMAIL_RECEIVERS")
    receivers = _split_csv(receivers_raw) if receivers_raw else DEFAULT_EMAIL_RECEIVERS

    return Settings(
        grafana_sqlalchemy_url=_resolve_db_url(),
        oisuu_host=_env("database_host"),
        oisuu_port=_env_int("database_port", default=0),
        oisuu_database=_env("database_name"),
        oisuu_user=_env("database_user_name"),
        oisuu_password=_env("database_user_pass"),
        oracle_host=_env("SERVER_PROD8"),
        oracle_port=_env_int("PORT_PROD8", default=0),
        oracle_user=_env("USER_PROD8"),
        oracle_password=_env("PASSWORD_PROD8"),
        oracle_service_name=_env("SERVICE_NAME_PROD8"),
        drift_models=tuple(models),
        reference_train_start=_env("REFERENCE_TRAIN_START")
        or DEFAULT_REFERENCE_TRAIN_START,
        reference_train_end=_env("REFERENCE_TRAIN_END") or DEFAULT_REFERENCE_TRAIN_END,
        drift_lookback_days=_env_int(
            "DRIFT_LOOKBACK_DAYS", default=DEFAULT_DRIFT_LOOKBACK_DAYS
        ),
        errors_lookback_days=_env_int(
            "ERRORS_LOOKBACK_DAYS", default=DEFAULT_ERRORS_LOOKBACK_DAYS
        ),
        errors_model_kinds=_split_csv(_env("ERRORS_MODEL_KINDS"))
        or DEFAULT_ERRORS_MODEL_KINDS,
        fin_effect_monitoring_path=_env("FIN_EFFECT_MONITORING_PATH"),
        fin_effect_retro_path=_env("FIN_EFFECT_RETRO_PATH"),
        email_receivers=receivers,
        psi_threshold=float(
            _env("PSI_THRESHOLD") or str(DEFAULT_PSI_THRESHOLD)
        ),
    )
