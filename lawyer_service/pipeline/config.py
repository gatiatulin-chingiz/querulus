"""Пути и URL пайплайна lawyer_service (env-переопределения)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

DEFAULT_INPUT = ROOT / "Представители автоюристов_разложенные.xlsx"
DEFAULT_DF_FOR_SERVICE = (
    ROOT.parent / "integration" / "results" / "df_for_service.parquet"
)
DEFAULT_PAYLOAD_TEMPLATE = ROOT.parent / "integration" / "payload.json"

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
