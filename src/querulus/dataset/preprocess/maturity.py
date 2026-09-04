"""Вызревание таргета: закрытый ПСР (ветка A) или горизонт без суда (ветка B)."""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from querulus.dataset.constants import RENAME_DICT
from querulus.dataset.load.io import read_artifact
from querulus.dataset.paths import DataPaths
from querulus.dataset.preprocess.filters import load_dataset_filters
from querulus.dataset.preprocess.targets import CLAIM_PERIOD_COL, is_void_claim_instance

logger = logging.getLogger("querulus.dataset")

CLAIMS_ARTIFACT = "target_3_claims.parquet"
MATURITY_REPORT_NAME = "target_maturity_report.json"
_T0_COLUMN = "PAYMENT_ORDER_DATE_TIME"
LAG_DAYS_PERCENTILES: tuple[int, ...] = (50, 60, 70, 80, 90, 95, 99)


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
    filters: dict[str, Any] | None = None,
) -> tuple[pd.Timestamp, str]:
    """S: дата среза — явная в конфиге или дата запуска пайплайна."""
    cfg = _maturity_config(filters)
    explicit = cfg.get("snapshot_date")
    if explicit:
        return pd.Timestamp(explicit).normalize(), "target_maturity.snapshot_date"
    return pd.Timestamp(date.today()).normalize(), "run_date"


def _horizon_months_no_court(cfg: dict[str, Any]) -> int:
    """Горизонт ветки B (месяцы после T0 при отсутствии закрытого ПСР)."""
    if cfg.get("horizon_months_no_court") is not None:
        return int(cfg["horizon_months_no_court"])
    if cfg.get("horizon_months") is not None:
        return int(cfg["horizon_months"])
    return 36


def _last_instance_per_claim(claims: pd.DataFrame) -> pd.DataFrame:
    """Последняя инстанция каждого иска на инциденте (как для open/closed court)."""
    incident_col = _first_present(claims.columns, "INCIDENT_NUMBER")
    claim_col = _first_present(claims.columns, "INCOMING_CLAIM_NUMBER")
    period_col = _first_present(claims.columns, CLAIM_PERIOD_COL)
    if incident_col is None or claim_col is None or period_col is None:
        raise KeyError(
            "Для maturity по искам нужны INCIDENT_NUMBER, INCOMING_CLAIM_NUMBER, "
            f"{CLAIM_PERIOD_COL}"
        )
    work = claims.loc[
        claims[incident_col].notna() & claims[claim_col].notna(),
        :,
    ].copy()
    if work.empty:
        return work
    work["_period"] = pd.to_datetime(work[period_col], errors="coerce")
    work["_void"] = is_void_claim_instance(work)
    return (
        work.sort_values(
            [incident_col, claim_col, "_period"],
            ascending=[True, True, True],
            na_position="first",
        )
        .drop_duplicates([incident_col, claim_col], keep="last")
    )


def incidents_with_open_court(claims: pd.DataFrame) -> pd.Index:
    """Инциденты, у которых хотя бы один иск с незакрытой последней инстанцией."""
    last = _last_instance_per_claim(claims)
    if last.empty:
        return pd.Index([])
    incident_col = _first_present(last.columns, "INCIDENT_NUMBER")
    if incident_col is None:
        return pd.Index([])
    open_mask = last["_void"] | last["_period"].isna()
    return pd.Index(last.loc[open_mask, incident_col].dropna().unique())


def incidents_with_closed_claim(claims: pd.DataFrame) -> pd.Index:
    """Инциденты с ≥1 принятой последней инстанцией иска (суд или ФU)."""
    last = _last_instance_per_claim(claims)
    if last.empty:
        return pd.Index([])
    incident_col = _first_present(last.columns, "INCIDENT_NUMBER")
    if incident_col is None:
        return pd.Index([])
    closed_mask = (~last["_void"]) & last["_period"].notna()
    return pd.Index(last.loc[closed_mask, incident_col].dropna().unique())


def _branch_a_mask(
    df: pd.DataFrame,
    claims: pd.DataFrame,
) -> pd.Series:
    """Ветка A: закрытый ПСР — принятый суд/ФU или доплата по претензии в снимке данных."""
    closed_ids = set(_incident_key(pd.Series(incidents_with_closed_claim(claims))).tolist())
    inc_key = _incident_key(df["INCIDENT_NUMBER"])
    branch_a = inc_key.isin(closed_ids)
    if "TARGET_FREQ_PRET_AMOUNT" in df.columns:
        branch_a = branch_a | (pd.to_numeric(df["TARGET_FREQ_PRET_AMOUNT"], errors="coerce").fillna(0) > 0)
    if "TARGET_FREQ_CLAIMS_AMOUNT" in df.columns:
        branch_a = branch_a | (pd.to_numeric(df["TARGET_FREQ_CLAIMS_AMOUNT"], errors="coerce").fillna(0) > 0)
    return branch_a


def _empty_lag_report() -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {f"p{p}": None for p in LAG_DAYS_PERCENTILES}
    out["n"] = 0
    return out


def _lag_days_percentiles(
    df: pd.DataFrame,
    claims: pd.DataFrame,
    *,
    t0_column: str,
) -> dict[str, float | int | None]:
    """Лаг T0 → последняя дата иска среди TARGET_FREQ=1 (ориентир для H)."""
    empty = _empty_lag_report()
    if "TARGET_FREQ" not in df.columns or t0_column not in df.columns:
        return empty
    incident_col = _first_present(claims.columns, "INCIDENT_NUMBER")
    period_col = _first_present(claims.columns, CLAIM_PERIOD_COL)
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
    out: dict[str, float | int | None] = {
        f"p{p}": float(lag.quantile(p / 100.0)) for p in LAG_DAYS_PERCENTILES
    }
    out["n"] = int(lag.shape[0])
    return out


def apply_target_maturity(
    df: pd.DataFrame,
    claims: pd.DataFrame,
    *,
    filters: dict[str, Any] | None = None,
    report_path: Path | str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Отбор строк для обучения по веткам A / B и незакрытому суду.

    Приоритет:
    1) open court → исключить;
    2) ветка A — закрытый ПСР (суд/ФU/претензия с доплатой в данных);
    3) ветка B — T0 + horizon_months_no_court ≤ S (дата запуска);
    4) иначе исключить.

    ``TARGET_FREQ`` уже отражает все претензии/иски в снимке: если вторая
    претензия с выплатой есть в выгрузке, freq=1; если ещё нет — freq=0.
    """
    cfg = _maturity_config(filters)
    if not cfg.get("enabled", True):
        report = {"enabled": False, "n_before": int(len(df)), "n_after": int(len(df))}
        return df, report

    t0_column = str(cfg.get("t0_column") or _T0_COLUMN)
    horizon_months = _horizon_months_no_court(cfg)
    if t0_column not in df.columns:
        raise KeyError(f"Для вызревания нужна колонка {t0_column}")
    if "INCIDENT_NUMBER" not in df.columns:
        raise KeyError("Для вызревания нужна колонка INCIDENT_NUMBER")

    claims_n = _normalize_claims_columns(claims)
    snapshot, snapshot_source = resolve_snapshot_date(filters)
    t0 = pd.to_datetime(df[t0_column], errors="coerce").dt.normalize()
    missing_t0 = t0.isna()
    horizon_end = t0 + pd.DateOffset(months=horizon_months)
    branch_b = (~missing_t0) & (horizon_end <= snapshot)

    open_ids = set(_incident_key(pd.Series(incidents_with_open_court(claims_n))).tolist())
    open_court = _incident_key(df["INCIDENT_NUMBER"]).isin(open_ids)

    branch_a = _branch_a_mask(df, claims_n)
    eligible = ~missing_t0 & ~open_court & (branch_a | branch_b)
    keep = eligible

    n_before = int(len(df))
    out = df.loc[keep].reset_index(drop=True)
    lags = _lag_days_percentiles(df, claims_n, t0_column=t0_column)
    mature_until = (snapshot - pd.DateOffset(months=horizon_months)).normalize()

    branch_a_eligible = branch_a & ~missing_t0 & ~open_court
    branch_b_only = branch_b & ~branch_a & ~missing_t0 & ~open_court

    report: dict[str, Any] = {
        "enabled": True,
        "policy": "branch_a_closed_psr_or_branch_b_horizon",
        "t0_column": t0_column,
        "horizon_months_no_court": horizon_months,
        "snapshot_date": str(snapshot.date()),
        "snapshot_source": snapshot_source,
        "mature_until": str(pd.Timestamp(mature_until).date()),
        "n_before": n_before,
        "n_after": int(len(out)),
        "n_dropped_t0_missing": int(missing_t0.sum()),
        "n_dropped_open_court": int((~missing_t0 & open_court).sum()),
        "n_open_court_would_pass_horizon": int((branch_b & open_court & ~missing_t0).sum()),
        "n_kept_branch_a": int(branch_a_eligible.sum()),
        "n_kept_branch_b_only": int(branch_b_only.sum()),
        "n_dropped_horizon": int((~missing_t0 & ~open_court & ~branch_a & ~branch_b).sum()),
        "lag_days_target_freq_1": lags,
    }
    logger.info(
        "maturity: S=%s (%s) B_horizon=%sм mature_until=%s rows %s → %s "
        "(open_court=%s branch_a=%s branch_b_only=%s horizon=%s t0_na=%s) lag n=%s",
        report["snapshot_date"],
        snapshot_source,
        horizon_months,
        report["mature_until"],
        n_before,
        report["n_after"],
        report["n_dropped_open_court"],
        report["n_kept_branch_a"],
        report["n_kept_branch_b_only"],
        report["n_dropped_horizon"],
        report["n_dropped_t0_missing"],
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
