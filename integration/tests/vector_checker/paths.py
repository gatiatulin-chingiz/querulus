"""Пути и поиск актуального df_for_service."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
TESTS_DIR = PACKAGE_DIR.parent
INTEGRATION_DIR = TESTS_DIR.parent
QUERULUS_ROOT = INTEGRATION_DIR.parent

WORK_DIR = PACKAGE_DIR / "work"
EXCEL_DIR = WORK_DIR / "excel"
REPORTS_DIR = WORK_DIR / "reports"
FIXTURES_DIR = PACKAGE_DIR / "fixtures" / "synthetic"

DEFAULT_QUERY_PATH = INTEGRATION_DIR / "Сутяжность.txt"
DEFAULT_QUERY_OUT = WORK_DIR / "Сутяжность_for_1c.txt"
DEFAULT_DF_CANDIDATES = (
    INTEGRATION_DIR / "results" / "df_for_service.parquet",
    QUERULUS_ROOT / "data" / "processed" / "df_for_service.parquet",
)

LOSS_NUMBER_COL = "LOSS_NUMBER"
PRED_COLS = ("preds_cf", "preds_rg")

# Фичи контракта сервиса (main.py) + ключи/поля, которые отдаёт запрос Сутяжность.
SERVICE_VECTOR_FEATURES: tuple[str, ...] = (
    "FILIAL",
    "EVENT_CREATED_BY_GIBDD_FLAG",
    "VICTIM_MAX_WEIGHT",
    "RECIEVE_METHOD",
    "APPLICANT_FORM",
    "APPLICANT_AGE",
    "VICTIM_VEHICLE_CATEGORY",
    "GUILTY_CAPACITY_ENGINE",
    "VICTIM_VEHICLE_AGE",
    "EVENT_YEAR",
    "AMOUNT_REPAIR",
    "LOSS_UNIT_ZONE",
    "VICTIM_VEHICLE_COUNTRY",
    "APPLY_DELAY",
    "EVENT_DATE",
    "PAYMENT_ORDER_DATE_TIME",
    "INCIDENT_NUMBER",
    "VICTIM_VEHICLE_BRAND",
    "VALUE_BEFORE_WITH",
    "VALUE_BEFORE_WITHOUT",
    "REGION",
)


def ensure_work_dirs() -> None:
    """Создать work/excel/reports при необходимости."""
    for path in (WORK_DIR, EXCEL_DIR, REPORTS_DIR, FIXTURES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def resolve_df_for_service(explicit: str | Path | None = None) -> Path:
    """Найти актуальный df_for_service.parquet (явный путь / env / candidates / mtime)."""
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"df_for_service не найден: {path}")
        return path.resolve()

    env_path = os.environ.get("OUTBOXML_TRAIN_DF_PATH", "").strip()
    if env_path:
        path = Path(env_path)
        if path.is_file():
            return path.resolve()

    try:
        import config as integration_config

        cfg_path = Path(getattr(integration_config, "outboxml_train_df_path", "") or "")
        if cfg_path.is_file():
            return cfg_path.resolve()
    except Exception:
        pass

    existing = [p for p in DEFAULT_DF_CANDIDATES if p.is_file()]
    if existing:
        return max(existing, key=lambda p: p.stat().st_mtime).resolve()

    globs = list(INTEGRATION_DIR.glob("**/df_for_service*.parquet"))
    globs += list((QUERULUS_ROOT / "data").glob("**/df_for_service*.parquet"))
    if globs:
        return max(globs, key=lambda p: p.stat().st_mtime).resolve()

    raise FileNotFoundError(
        "Не найден df_for_service.parquet. Укажите --df или положите файл в "
        f"{INTEGRATION_DIR / 'results' / 'df_for_service.parquet'}"
    )
