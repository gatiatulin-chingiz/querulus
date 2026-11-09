"""Пути и URL пайплайна lawyer_service (env из корня querulus)."""

from __future__ import annotations

import os
from pathlib import Path

# lawyer_service/pipeline/config.py → querulus/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent.parent  # lawyer_service/
ARTIFACTS = ROOT / "artifacts"

DEFAULT_INPUT = ROOT / "Представители автоюристов_разложенные.xlsx"
DEFAULT_DF_FOR_SERVICE = (
    PROJECT_ROOT / "integration" / "results" / "df_for_service.parquet"
)
DEFAULT_PAYLOAD_TEMPLATE = PROJECT_ROOT / "integration" / "payload.json"


def _parse_env_file(path: Path, *, override: bool) -> None:
    """Простой KEY=VAL парсер без зависимости от environs/dotenv."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value


def _load_root_env() -> None:
    """Подтянуть env_template / .env из корня querulus (не из lawyer_service/)."""
    _parse_env_file(PROJECT_ROOT / "env_template", override=False)
    _parse_env_file(PROJECT_ROOT / ".env", override=True)


_load_root_env()

PREDICT_URL = os.environ.get(
    "LAWYER_PREDICT_URL",
    "http://perf-pydsp-test1.vsk.ru:10655/api/predict",
)
AHUNTER_BASE = os.environ.get(
    "LAWYER_AHUNTER_BASE",
    "http://v-ahunter-lb-pb.vsk.ru",
)
AHUNTER_USER_ID = os.environ.get("LAWYER_AHUNTER_USER_ID", "aRTDM")

MAIN_MODEL = os.environ.get(
    "LAWYER_MAIN_MODEL",
    "querulus_ansamble_2026_04_18_v1",
)
SECOND_MODEL = os.environ.get(
    "LAWYER_SECOND_MODEL",
    "UU_querulus_2026_10_27_20_12_44",
)

REQUEST_DELAY_SEC = float(os.environ.get("LAWYER_REQUEST_DELAY", "1.0"))
AHUNTER_DELAY_SEC = float(os.environ.get("LAWYER_AHUNTER_DELAY", "0.1"))
MAX_RETRIES = int(os.environ.get("LAWYER_MAX_RETRIES", "3"))
BASELINE_SCAN_LIMIT = int(os.environ.get("LAWYER_BASELINE_SCAN_LIMIT", "30"))
