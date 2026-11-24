"""Ошибки модели: OISUU MatemModelsLogs → Motor.agrafana.Querulus_errors.

Поток:
  1) SELECT из oisuu81_t_MatemModelsLogs (РезультатПроверки = -1)
  2) нормализация колонок + fingerprint traceback
  3) дневная агрегация count по модели × fingerprint
  4) append в Motor (guard: run_date/stem/slot/model_version/error_fingerprint)

Synthetic — только в local/synthetic/run_errors.py.
"""
from __future__ import annotations

import hashlib
import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

_PIPELINE_ROOT = Path(__file__).resolve().parent
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))
import _bootstrap_path  # noqa: F401 — monorepo + standalone GitLab clone

from pipeline.settings import Settings, load_settings
from pipeline.db import connect_oisuu
from pipeline.write_sql import get_engine, write_append_guarded

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("errors")

# --- нормализация текста ошибки для fingerprint (пути/адреса не должны плодить ключи) ---
_RE_FILE_LINE = re.compile(r'File "[^"]+", line \d+', re.IGNORECASE)
_RE_LINE_NO = re.compile(r", line \d+")
_RE_HEX_ADDR = re.compile(r"0x[0-9a-fA-F]+")
_RE_WS = re.compile(r"\s+")

# Допустимые наборы колонок сырья (русские имена OISUU или уже нормализованные)
_REQUIRED_ANY = (
    {"ДатаПроверки", "ОписаниеОшибки", "ВидМодели"},
    {"checked_at", "error_text", "model_kind"},
    {"run_date", "error_text", "model_kind"},
)


def traceback_head(text: str, max_len: int = 180) -> str:
    """Короткий «заголовок» ошибки для Grafana (последняя осмысленная строка)."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return "Ответа нет."
    raw = str(text).strip()
    if not raw:
        return "Ответа нет."
    lines = [ln.strip() for ln in raw.replace("\r\n", "\n").split("\n") if ln.strip()]
    head = lines[-1] if lines else "Ответа нет."
    # Ищем снизу строку с Error/Exception или «тип: сообщение»
    for ln in reversed(lines):
        if re.search(r"Error|Exception|Traceback", ln, re.IGNORECASE) or ":" in ln:
            head = ln
            break
    return head[:max_len]


def traceback_fingerprint(text: str) -> str:
    """Стабильный короткий hash traceback без абсолютных путей и адресов памяти."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return "empty"
    raw = str(text)
    norm = _RE_FILE_LINE.sub('File "<path>", line N', raw)
    norm = _RE_LINE_NO.sub(", line N", norm)
    norm = _RE_HEX_ADDR.sub("0xADDR", norm)
    norm = _RE_WS.sub(" ", norm).strip().lower()
    if not norm:
        return "empty"
    return hashlib.sha1(norm.encode("utf-8", errors="replace")).hexdigest()[:12]


def load_errors_frame(settings: Settings) -> pd.DataFrame:
    """Боевая выгрузка ошибок из OISUU.

    ``РезультатПроверки = -1`` — сбой/ошибка модели (не успешный score).
    Окно: ``ERRORS_LOOKBACK_DAYS``. Опционально ``ERRORS_MODEL_KINDS`` (ВидМодели IN …).
    """
    since = (datetime.now() - timedelta(days=settings.errors_lookback_days)).strftime(
        "%Y-%m-%d"
    )
    sql = """
        SELECT
            ДатаПроверки,
            LossNumber,
            ОписаниеОшибки,
            ВерсияМодели,
            Комментарий,
            Filial,
            ВидМодели,
            Подразделение,
            LossProcess,
            РезультатПроверки
        FROM [OISUU_report].[dbo].[oisuu81_t_MatemModelsLogs] WITH (NOLOCK)
        WHERE РезультатПроверки = -1
          AND ДатаПроверки >= ?
    """
    params: list = [since]
    if settings.errors_model_kinds:
        placeholders = ",".join("?" for _ in settings.errors_model_kinds)
        sql += f"\n          AND ВидМодели IN ({placeholders})"
        params.extend(settings.errors_model_kinds)

    conn = connect_oisuu(settings)
    try:
        df = pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()

    cols = set(df.columns)
    if not any(req.issubset(cols) for req in _REQUIRED_ANY):
        raise ValueError(
            "errors frame must look like MatemModelsLogs "
            "(ДатаПроверки/ОписаниеОшибки/ВидМодели); "
            f"got columns={list(df.columns)[:20]}"
        )
    logger.info(
        "errors loaded || rows=%s || since=%s || kinds=%s",
        len(df),
        since,
        settings.errors_model_kinds or "(all)",
    )
    return df


def _normalize_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Русские колонки OISUU → канон pipeline + head/fingerprint."""
    work = df.copy()
    rename = {
        "ДатаПроверки": "checked_at",
        "LossNumber": "loss_number",
        "ОписаниеОшибки": "error_text",
        "ВерсияМодели": "model_version",
        "Комментарий": "comment",
        "Filial": "filial",
        "ВидМодели": "model_kind",
        "Подразделение": "subdivision",
        "LossProcess": "loss_process",
        "РезультатПроверки": "check_result",
    }
    work = work.rename(columns={k: v for k, v in rename.items() if k in work.columns})
    if "checked_at" not in work.columns and "run_date" in work.columns:
        work["checked_at"] = work["run_date"]
    if "checked_at" not in work.columns:
        raise ValueError("errors frame needs checked_at or ДатаПроверки / run_date")
    work["checked_at"] = pd.to_datetime(work["checked_at"], errors="coerce")
    work["run_date"] = work["checked_at"].dt.date
    if "error_text" not in work.columns:
        raise ValueError("errors frame needs error_text or ОписаниеОшибки")
    work["error_text"] = work["error_text"].fillna("").astype(str)
    work.loc[work["error_text"].str.strip() == "", "error_text"] = "Ответа нет."
    # stem/slot/label — для Grafana-фильтра «Модель»; без них берём ВидМодели
    if "model_kind" not in work.columns:
        work["model_kind"] = work["stem"] if "stem" in work.columns else "unknown"
    work["model_kind"] = work["model_kind"].fillna("unknown").astype(str)
    if "stem" not in work.columns:
        work["stem"] = work["model_kind"]
    if "model_label" not in work.columns:
        work["model_label"] = work["stem"].astype(str)
    if "slot" not in work.columns:
        work["slot"] = ""
    if "model_version" not in work.columns:
        work["model_version"] = ""
    work["error_head"] = work["error_text"].map(traceback_head)
    work["error_fingerprint"] = work["error_text"].map(traceback_fingerprint)
    return work


def aggregate_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Дневные счётчики: модель × fingerprint (зерно таблицы Querulus_errors)."""
    work = _normalize_raw(df)
    return (
        work.groupby(
            [
                "run_date",
                "stem",
                "slot",
                "model_label",
                "model_kind",
                "model_version",
                "error_fingerprint",
                "error_head",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .sort_values(["run_date", "count"], ascending=[True, False])
        .reset_index(drop=True)
    )


def main() -> int:
    """Точка входа make automl / python errors.py."""
    settings = load_settings()
    logger.info("errors ← OISUU MatemModelsLogs → %s.%s", settings.schema, settings.table_errors)
    raw = load_errors_frame(settings)
    daily = aggregate_errors(raw)
    engine = get_engine(settings.grafana_sqlalchemy_url)
    # guard: повторный прогон того же дня не дублирует строки
    n = write_append_guarded(
        daily,
        engine=engine,
        table=settings.table_errors,
        schema=settings.schema,
        guard_cols=[
            "run_date",
            "stem",
            "slot",
            "model_version",
            "error_fingerprint",
        ],
    )
    logger.info("errors done || rows=%s", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
