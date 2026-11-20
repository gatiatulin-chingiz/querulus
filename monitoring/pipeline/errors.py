"""Errors monitoring: MatemModelsLogs → Querulus_errors (MSSQL agrafana)."""
from __future__ import annotations

import hashlib
import logging
import re
import sys
from pathlib import Path

import pandas as pd

_PIPELINE_ROOT = Path(__file__).resolve().parent
_MONITORING_ROOT = _PIPELINE_ROOT.parent
if str(_MONITORING_ROOT) not in sys.path:
    sys.path.insert(0, str(_MONITORING_ROOT))

from pipeline.config import Settings, load_settings
from pipeline.write_sql import get_engine, write_append_guarded

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("errors")

_RE_FILE_LINE = re.compile(r'File "[^"]+", line \d+', re.IGNORECASE)
_RE_LINE_NO = re.compile(r", line \d+")
_RE_HEX_ADDR = re.compile(r"0x[0-9a-fA-F]+")
_RE_WS = re.compile(r"\s+")

_REQUIRED_ANY = (
    {"ДатаПроверки", "ОписаниеОшибки", "ВидМодели"},
    {"checked_at", "error_text", "model_kind"},
    {"run_date", "error_text", "model_kind"},
)


def traceback_head(text: str, max_len: int = 180) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return "Ответа нет."
    raw = str(text).strip()
    if not raw:
        return "Ответа нет."
    lines = [ln.strip() for ln in raw.replace("\r\n", "\n").split("\n") if ln.strip()]
    head = lines[-1] if lines else "Ответа нет."
    for ln in reversed(lines):
        if re.search(r"Error|Exception|Traceback", ln, re.IGNORECASE) or ":" in ln:
            head = ln
            break
    return head[:max_len]


def traceback_fingerprint(text: str) -> str:
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


def _load_synthetic(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "errors.parquet"
    if not path.is_file():
        path = data_dir / "errors.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Synthetic errors not found under {data_dir}. "
            "Run local/synthetic/generate.py"
        )
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    cols = set(df.columns)
    if not any(req.issubset(cols) for req in _REQUIRED_ANY):
        raise ValueError(
            "errors frame must look like MatemModelsLogs "
            "(ДатаПроверки/ОписаниеОшибки/ВидМодели) or "
            "(checked_at|run_date, error_text, model_kind)"
        )
    if "РезультатПроверки" in df.columns:
        df = df.loc[df["РезультатПроверки"] == -1].copy()
    if "check_result" in df.columns:
        df = df.loc[df["check_result"] == -1].copy()
    return df


def _load_prod(settings: Settings) -> pd.DataFrame:
    raise NotImplementedError(
        "prod errors: SELECT from oisuu81_t_MatemModelsLogs "
        f"where РезультатПроверки = -1. SOURCE_MODE={settings.source_mode}"
    )


def load_errors_frame(settings: Settings) -> pd.DataFrame:
    if settings.source_mode == "synthetic":
        return _load_synthetic(settings.synthetic_data_dir)
    return _load_prod(settings)


def _normalize_raw(df: pd.DataFrame) -> pd.DataFrame:
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
    """Daily counts by model × traceback fingerprint."""
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
    settings = load_settings()
    logger.info("SOURCE_MODE=%s", settings.source_mode)
    raw = load_errors_frame(settings)
    daily = aggregate_errors(raw)
    engine = get_engine(settings.grafana_sqlalchemy_url)
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
