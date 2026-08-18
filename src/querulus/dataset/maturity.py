"""Вызревание таргета: горизонт после поручения на выплату и незакрытый суд."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from querulus.dataset.constants import RENAME_DICT
from querulus.dataset.filters import load_dataset_filters
from querulus.dataset.io import read_artifact
from querulus.dataset.paths import DataPaths
from querulus.dataset.steps.targets import _CLAIM_PERIOD_COL, _is_void_claim_instance

logger = logging.getLogger("querulus.dataset")

CLAIMS_ARTIFACT = "target_3_claims.parquet"
MATURITY_REPORT_NAME = "target_maturity_report.json"
_T0_COLUMN = "PAYMENT_ORDER_DATE_TIME"
_CLAIM_EVENT_DATE_CANDIDATES = (
    _CLAIM_PERIOD_COL,
    "RECOVEREDVALUEPERIOD",
    "COURTWORKOVERDATE",
    "INCOMING_CLAIM_GET_DATE",
    "INCOMINGCLAIMGETDATE",
)


def _maturity_config(filters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Секция target_maturity из dataset_filters.json."""
    cfg = (filters or load_dataset_filters()).get("target_maturity") or {}
    return cfg


def _normalize_claims_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Сырой SQL parquet → UPPER + RENAME_DICT (как в build_targets)."""
    out = df.rename(columns=RENAME_DICT)
    out.columns = out.columns.str.upper()
    out = out.rename(columns=RENAME_DICT)
    if "INCIDENTNUMBER" in out.columns and "INCIDENT_NUMBER" not in out.columns:
        out["INCIDENT_NUMBER"] = out["INCIDENTNUMBER"]
    if "INCOMINGCLAIMNUMBER" in out.columns and "INCOMING_CLAIM_NUMBER" not in out.columns:
        out["INCOMING_CLAIM_NUMBER"] = out["INCOMINGCLAIMNUMBER"]
    if "INCOMINGCLAIMGETDATE" in out.columns and "INCOMING_CLAIM_GET_DATE" not in out.columns:
        out["INCOMING_CLAIM_GET_DATE"] = out["INCOMINGCLAIMGETDATE"]
    return out


def _incident_key(series: pd.Series) -> pd.Series:
    """Сравнимый ключ инцидента (int-like без '.0', иначе строка)."""
    num = pd.to_numeric(series, errors="coerce")
    out = series.astype(str).str.strip()
    int_like = num.notna() & ((num % 1) == 0)
    out = out.copy()
    out.loc[int_like] = num.loc[int_like].astype("int64").astype(str)
    return out


def _first_present(columns: pd.Index, *names: str) -> str | None:
    upper = {str(col).upper(): col for col in columns}
    for name in names:
        if name.upper() in upper:
            return upper[name.upper()]
    return None


def resolve_snapshot_date(
    claims: pd.DataFrame,
    *,
    snapshot_date: str | None,
) -> pd.Timestamp:
    """S: явная дата снимка или max дат событий в исках."""
    if snapshot_date:
        return pd.Timestamp(snapshot_date).normalize()
    dates: list[pd.Series] = []
    for name in _CLAIM_EVENT_DATE_CANDIDATES:
        col = _first_present(claims.columns, name)
        if col is None:
            continue
        series = pd.to_datetime(claims[col], errors="coerce")
        if series.notna().any():
            dates.append(series)
    if not dates:
        raise ValueError(
            "Не задан snapshot_date и в исках нет дат для S. "
            "Укажите target_maturity.snapshot_date в dataset_filters.json."
        )
    snapshot = pd.concat(dates, axis=0).max()
    return pd.Timestamp(snapshot).normalize()


def incidents_with_open_court(claims: pd.DataFrame) -> pd.Index:
    """Инциденты, у которых последняя инстанция иска ещё не принята.

    По каждому (убыток, иск) берётся последняя строка по ClaimedValuePeriod,
    включая «Не принято» / пустые взыскания. Если она void — процесс не закрыт.
    Нет иска — не «суд идёт».
    """
    incident_col = _first_present(claims.columns, "INCIDENT_NUMBER")
    claim_col = _first_present(claims.columns, "INCOMING_CLAIM_NUMBER")
    period_col = _first_present(claims.columns, _CLAIM_PERIOD_COL)
    if incident_col is None or claim_col is None or period_col is None:
        raise KeyError(
            "Для незакрытого суда нужны INCIDENT_NUMBER, INCOMING_CLAIM_NUMBER, "
            f"{_CLAIM_PERIOD_COL}"
        )
    work = claims.loc[
        claims[incident_col].notna() & claims[claim_col].notna(),
        :,
    ].copy()
    if work.empty:
        return pd.Index([])
    work["_period"] = pd.to_datetime(work[period_col], errors="coerce")
    work["_void"] = _is_void_claim_instance(work)
    last = (
        work.sort_values(
            [incident_col, claim_col, "_period"],
            ascending=[True, True, True],
            na_position="first",
        )
        .drop_duplicates([incident_col, claim_col], keep="last")
    )
    open_mask = last["_void"] | last["_period"].isna()
    return pd.Index(last.loc[open_mask, incident_col].dropna().unique())


def _lag_days_percentiles(
    df: pd.DataFrame,
    claims: pd.DataFrame,
    *,
    t0_column: str,
) -> dict[str, float | None]:
    """Лаг T0psr → последняя дата иска среди TARGET_FREQ=1 (для подбора H)."""
    empty = {"p50": None, "p90": None, "p95": None, "n": 0}
    if "TARGET_FREQ" not in df.columns or t0_column not in df.columns:
        return empty
    incident_col = _first_present(claims.columns, "INCIDENT_NUMBER")
    period_col = _first_present(claims.columns, _CLAIM_PERIOD_COL)
    if incident_col is None or period_col is None:
        return empty
    positives = df.loc[
        df["TARGET_FREQ"].fillna(0).astype(int).eq(1), ["INCIDENT_NUMBER", t0_column]
    ].copy()
    if positives.empty:
        return empty
    last_event = (
        claims.assign(_period=pd.to_datetime(claims[period_col], errors="coerce"))
        .groupby(incident_col, as_index=True)["_period"]
        .max()
    )
    last_event.index = _incident_key(pd.Series(last_event.index)).to_numpy()
    positives["_inc"] = _incident_key(positives["INCIDENT_NUMBER"])
    aligned = positives.merge(
        last_event.rename("last_event"),
        how="inner",
        left_on="_inc",
        right_index=True,
    )
    t0 = pd.to_datetime(aligned[t0_column], errors="coerce")
    lag = (aligned["last_event"] - t0).dt.days
    lag = lag[lag.notna() & (lag >= 0)]
    if lag.empty:
        return empty
    return {
        "p50": float(lag.quantile(0.50)),
        "p90": float(lag.quantile(0.90)),
        "p95": float(lag.quantile(0.95)),
        "n": int(lag.shape[0]),
    }


def apply_target_maturity(
    df: pd.DataFrame,
    claims: pd.DataFrame,
    *,
    filters: dict[str, Any] | None = None,
    report_path: Path | str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Оставить строки с T0psr+H ≤ S и без незакрытого суда.

    Один горизонт H на freq и sev. Хвост моложе S−H в датасет не входит.
    """
    cfg = _maturity_config(filters)
    if not cfg.get("enabled", True):
        report = {"enabled": False, "n_before": int(len(df)), "n_after": int(len(df))}
        return df, report

    t0_column = str(cfg.get("t0_column") or _T0_COLUMN)
    horizon_months = int(cfg.get("horizon_months") or 24)
    if t0_column not in df.columns:
        raise KeyError(f"Для вызревания нужна колонка {t0_column}")
    if "INCIDENT_NUMBER" not in df.columns:
        raise KeyError("Для вызревания нужна колонка INCIDENT_NUMBER")

    claims_n = _normalize_claims_columns(claims)
    snapshot = resolve_snapshot_date(
        claims_n, snapshot_date=cfg.get("snapshot_date") or None
    )
    t0 = pd.to_datetime(df[t0_column], errors="coerce").dt.normalize()
    horizon_end = t0 + pd.DateOffset(months=horizon_months)
    missing_t0 = t0.isna()
    mature = (~missing_t0) & (horizon_end <= snapshot)

    open_ids = set(_incident_key(pd.Series(incidents_with_open_court(claims_n))).tolist())
    open_court = _incident_key(df["INCIDENT_NUMBER"]).isin(open_ids)

    keep = mature & ~open_court
    n_before = int(len(df))
    out = df.loc[keep].reset_index(drop=True)
    lags = _lag_days_percentiles(df, claims_n, t0_column=t0_column)
    mature_until = (snapshot - pd.DateOffset(months=horizon_months)).normalize()
    report: dict[str, Any] = {
        "enabled": True,
        "t0_column": t0_column,
        "horizon_months": horizon_months,
        "snapshot_date": str(snapshot.date()),
        "mature_until": str(pd.Timestamp(mature_until).date()),
        "n_before": n_before,
        "n_after": int(len(out)),
        "n_dropped_t0_missing": int(missing_t0.sum()),
        "n_dropped_horizon": int((~missing_t0 & ~mature).sum()),
        "n_dropped_open_court": int(open_court.sum()),
        "n_open_court_in_horizon": int((mature & open_court).sum()),
        "lag_days_target_freq_1": lags,
    }
    logger.info(
        "maturity: S=%s H=%sм mature_until=%s rows %s → %s "
        "(horizon=%s open_court=%s t0_na=%s) lag_days p50/p90/p95=%s/%s/%s n=%s",
        report["snapshot_date"],
        horizon_months,
        report["mature_until"],
        n_before,
        report["n_after"],
        report["n_dropped_horizon"],
        report["n_dropped_open_court"],
        report["n_dropped_t0_missing"],
        lags["p50"],
        lags["p90"],
        lags["p95"],
        lags["n"],
    )
    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("maturity report: %s", path)
    return out, report


def apply_target_maturity_from_paths(
    df: pd.DataFrame,
    paths: DataPaths,
    *,
    filters: dict[str, Any] | None = None,
    save_report: bool = True,
) -> pd.DataFrame:
    """Вызревание по ``target_3_claims.parquet`` (resume / повторный прогон)."""
    cfg = _maturity_config(filters)
    if not cfg.get("enabled", True):
        return df
    claims = read_artifact(paths, paths.raw_dir, CLAIMS_ARTIFACT)
    report_path = paths.processed_dir / MATURITY_REPORT_NAME if save_report else None
    out, _ = apply_target_maturity(
        df, claims, filters=filters, report_path=report_path
    )
    return out
