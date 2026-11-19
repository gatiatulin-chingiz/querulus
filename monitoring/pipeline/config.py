"""Runtime config from environment (no secrets in code)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus


SCHEMA = "agrafana"
TABLE_ERRORS = "Querulus_errors"
TABLE_DRIFT = "querulus_DataDrift"
TABLE_FIN_EFFECT = "Querulus_FinEffect"


@dataclass(frozen=True)
class Settings:
    """Pipeline settings resolved once per run."""

    source_mode: str
    grafana_sqlalchemy_url: str
    synthetic_data_dir: Path
    schema: str = SCHEMA
    table_errors: str = TABLE_ERRORS
    table_drift: str = TABLE_DRIFT
    table_fin_effect: str = TABLE_FIN_EFFECT
    psi_threshold: float = 0.3


def _resolve_db_url() -> str:
    """SQLAlchemy URL for Motor/MSSQL metrics DB.

    Prefer GRAFANA_SQLALCHEMY_URL. Else build from MSSQL_* env vars
    (MSSQL_USER + MSSQL_SA_PASSWORD required).
    """
    url = os.environ.get("GRAFANA_SQLALCHEMY_URL", "").strip()
    if url:
        return url

    user = os.environ.get("MSSQL_USER", "").strip()
    password = os.environ.get("MSSQL_SA_PASSWORD", "").strip()
    if not user or not password:
        raise ValueError(
            "Set GRAFANA_SQLALCHEMY_URL or both MSSQL_USER and MSSQL_SA_PASSWORD"
        )

    host = os.environ.get("MSSQL_HOST", "localhost").strip() or "localhost"
    port = os.environ.get("MSSQL_PORT", "1433").strip() or "1433"
    database = os.environ.get("MSSQL_DATABASE", "Motor").strip() or "Motor"
    driver = (
        os.environ.get("MSSQL_ODBC_DRIVER", "ODBC Driver 18 for SQL Server").strip()
        or "ODBC Driver 18 for SQL Server"
    )
    odbc = (
        f"DRIVER={{{driver}}};"
        f"SERVER={host},{port};"
        f"DATABASE={database};"
        f"UID={user};PWD={password};"
        "TrustServerCertificate=yes;"
    )
    return "mssql+pyodbc:///?odbc_connect=" + quote_plus(odbc)


def load_settings() -> Settings:
    """Read settings from env.

    SOURCE_MODE: synthetic | prod
    GRAFANA_SQLALCHEMY_URL or MSSQL_USER + MSSQL_SA_PASSWORD (+ host/port/db)
    SYNTHETIC_DATA_DIR: directory with generated parquet/csv
    """
    mode = os.environ.get("SOURCE_MODE", "synthetic").strip().lower()
    if mode not in {"synthetic", "prod"}:
        raise ValueError(f"SOURCE_MODE must be synthetic|prod, got {mode!r}")

    data_dir = Path(
        os.environ.get(
            "SYNTHETIC_DATA_DIR",
            str(Path(__file__).resolve().parents[1] / "local" / "synthetic" / "data"),
        )
    )
    return Settings(
        source_mode=mode,
        grafana_sqlalchemy_url=_resolve_db_url(),
        synthetic_data_dir=data_dir,
        psi_threshold=float(os.environ.get("PSI_THRESHOLD", "0.3")),
    )
