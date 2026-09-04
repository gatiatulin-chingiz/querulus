"""Вызревание таргета: тишина без ПСР или охлаждение после последнего события."""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from querulus.dataset.constants import RENAME_DICT
from querulus.dataset.load.io import read_artifact, read_parquet_path
from querulus.dataset.paths import DataPaths
from querulus.dataset.preprocess.filters import load_dataset_filters
from querulus.dataset.preprocess.targets import CLAIM_PERIOD_COL, is_void_claim_instance

logger = logging.getLogger("querulus.dataset")

CLAIMS_ARTIFACT = "target_3_claims.parquet"
PRETENSIONS_ARTIFACT = "df_pretensions.parquet"
MATURITY_REPORT_NAME = "target_maturity_report.json"
_T0_COLUMN = "PAYMENT_ORDER_DATE_TIME"
LAG_DAYS_PERCENTILES: tuple[int, ...] = (50, 60, 70, 80, 90, 95, 99)

# Silent: претензии вызревают быстрее, ФУ/суд — дольше; keep = max(оба) без событий.
DEFAULT_SILENT_PRETENSION_MONTHS = 6
DEFAULT_SILENT_FU_COURT_MONTHS = 24
DEFAULT_COOLOFF_MONTHS = 24  # после последнего события ПСР


def _maturity_config(filters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Секция target_maturity из dataset_filters.json."""
    cfg = (filters or load_dataset_filters()).get("target_maturity") or {}
    return cfg


def with_maturity_enabled(
    filters: dict[str, Any] | None,
    enabled: bool | None,
) -> dict[str, Any] | None:
    """Копия filters с override ``target_maturity.enabled`` (None → без изменений)."""
    if enabled is None:
        return filters
    base = dict(filters or load_dataset_filters())
    mat = dict(base.get("target_maturity") or {})
    mat["enabled"] = bool(enabled)
    base["target_maturity"] = mat
    return base


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


def _normalize_pretensions_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns=RENAME_DICT)
    out.columns = out.columns.str.upper()
    out = out.rename(columns=RENAME_DICT)
    if "INCIDENTNUMBER" in out.columns and "INCIDENT_NUMBER" not in out.columns:
        out["INCIDENT_NUMBER"] = out["INCIDENTNUMBER"]
    if "PRETENSIONGETDATE" in out.columns and "PRETENSION_GET_DATE" not in out.columns:
        out["PRETENSION_GET_DATE"] = out["PRETENSIONGETDATE"]
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


def _silent_pretension_months(cfg: dict[str, Any]) -> int:
    if cfg.get("silent_horizon_months_pretension") is not None:
        return int(cfg["silent_horizon_months_pretension"])
    return DEFAULT_SILENT_PRETENSION_MONTHS


def _silent_fu_court_months(cfg: dict[str, Any]) -> int:
    """Горизонт тишины по ФУ/суду (обычно 24); им же задаётся keep без ПСР."""
    if cfg.get("silent_horizon_months_fu_court") is not None:
        return int(cfg["silent_horizon_months_fu_court"])
    # legacy: один общий ключ = горизонт ФУ/суда
    if cfg.get("silent_horizon_months") is not None:
        return int(cfg["silent_horizon_months"])
    if cfg.get("horizon_months_no_court") is not None:
        return int(cfg["horizon_months_no_court"])
    if cfg.get("horizon_months") is not None:
        return int(cfg["horizon_months"])
    return DEFAULT_SILENT_FU_COURT_MONTHS


def _cooloff_months(cfg: dict[str, Any]) -> int:
    """Месяцы после последнего события претензия/ФУ/суд → кейс закрыт."""
    if cfg.get("cooloff_months_after_psr") is not None:
        return int(cfg["cooloff_months_after_psr"])
    return DEFAULT_COOLOFF_MONTHS


def _drop_psr_before_t0(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("drop_psr_before_t0", True))


def _last_instance_per_claim(claims: pd.DataFrame) -> pd.DataFrame:
    """Последняя инстанция каждого иска на инциденте."""
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
    """Инциденты с ≥1 принятой последней инстанцией иска (суд или ФУ)."""
    last = _last_instance_per_claim(claims)
    if last.empty:
        return pd.Index([])
    incident_col = _first_present(last.columns, "INCIDENT_NUMBER")
    if incident_col is None:
        return pd.Index([])
    closed_mask = (~last["_void"]) & last["_period"].notna()
    return pd.Index(last.loc[closed_mask, incident_col].dropna().unique())


def _pretension_agg_dates(
    pretensions: pd.DataFrame | None,
    *,
    how: str,
) -> pd.Series:
    """min/max PRETENSION_GET_DATE по инциденту."""
    empty = pd.Series(dtype="datetime64[ns]")
    if pretensions is None or pretensions.empty:
        return empty
    work = _normalize_pretensions_columns(pretensions)
    get_col = _first_present(
        work.columns, "PRETENSION_GET_DATE", "PRETENSION_DATE"
    )
    if "INCIDENT_NUMBER" not in work.columns or get_col is None:
        return empty
    work = work.loc[work["INCIDENT_NUMBER"].notna()].copy()
    work["_inc"] = _incident_key(work["INCIDENT_NUMBER"])
    work["_d"] = pd.to_datetime(work[get_col], errors="coerce")
    work = work.loc[work["_d"].notna()]
    if work.empty:
        return empty
    if how == "min":
        return work.groupby("_inc", sort=False)["_d"].min()
    return work.groupby("_inc", sort=False)["_d"].max()


def _claim_event_agg_dates(claims: pd.DataFrame, *, how: str) -> pd.Series:
    """min/max даты ФУ/суда по инциденту (period / get_date / court over)."""
    empty = pd.Series(dtype="datetime64[ns]")
    if claims.empty:
        return empty
    incident_col = _first_present(claims.columns, "INCIDENT_NUMBER")
    if incident_col is None:
        return empty
    work = claims.loc[claims[incident_col].notna()].copy()
    work["_inc"] = _incident_key(work[incident_col])
    date_parts: list[pd.Series] = []
    for name in (CLAIM_PERIOD_COL, "INCOMING_CLAIM_GET_DATE", "COURTWORKOVERDATE"):
        col = _first_present(work.columns, name)
        if col is not None:
            date_parts.append(pd.to_datetime(work[col], errors="coerce"))
    if not date_parts:
        return empty
    stacked = date_parts[0]
    for part in date_parts[1:]:
        frame = pd.concat([stacked, part], axis=1)
        stacked = frame.min(axis=1) if how == "min" else frame.max(axis=1)
    work["_d"] = stacked
    work = work.loc[work["_d"].notna()]
    if work.empty:
        return empty
    if how == "min":
        return work.groupby("_inc", sort=False)["_d"].min()
    return work.groupby("_inc", sort=False)["_d"].max()


def _psr_event_dates(
    claims: pd.DataFrame,
    pretensions: pd.DataFrame | None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """has_psr, first_event, last_event по ключу инцидента."""
    pret_last = _pretension_agg_dates(pretensions, how="max")
    pret_first = _pretension_agg_dates(pretensions, how="min")
    claim_last = _claim_event_agg_dates(claims, how="max")
    claim_first = _claim_event_agg_dates(claims, how="min")

    last_parts = [s for s in (pret_last, claim_last) if len(s)]
    first_parts = [s for s in (pret_first, claim_first) if len(s)]
    empty_dt = pd.Series(dtype="datetime64[ns]")
    empty_bool = pd.Series(dtype=bool)
    if not last_parts:
        return empty_bool, empty_dt, empty_dt

    last = pd.concat(last_parts, axis=1).max(axis=1).dropna()
    first = (
        pd.concat(first_parts, axis=1).min(axis=1).dropna()
        if first_parts
        else empty_dt
    )
    # выровнять first по индексу last
    first = first.reindex(last.index)
    has_psr = pd.Series(True, index=last.index)
    return has_psr, first, last


def _map_incident_series(
    inc_key: pd.Series,
    by_inc: pd.Series,
    *,
    fill_bool: bool | None = None,
) -> pd.Series:
    """Сопоставить Series с index=incident key на строки df."""
    if by_inc.empty:
        if fill_bool is not None:
            return pd.Series(fill_bool, index=inc_key.index)
        return pd.Series(pd.NaT, index=inc_key.index)
    mapped = inc_key.map(by_inc)
    if fill_bool is not None:
        return mapped.fillna(fill_bool).astype(bool)
    return mapped


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
    """Лаг T0 → первая дата иска (CLAIMEDVALUEPERIOD.min) среди TARGET_FREQ=1."""
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
    first_event = (
        claims.assign(_period=pd.to_datetime(claims[period_col], errors="coerce"))
        .groupby(incident_col, as_index=True)["_period"]
        .min()
    )
    first_event.index = _incident_key(pd.Series(first_event.index)).to_numpy()
    positives["_inc"] = _incident_key(positives["INCIDENT_NUMBER"])
    aligned = positives.merge(
        first_event.rename("first_event"),
        how="inner",
        left_on="_inc",
        right_index=True,
    )
    t0 = pd.to_datetime(aligned[t0_column], errors="coerce")
    lag = (aligned["first_event"] - t0).dt.days
    lag = lag[lag.notna() & (lag >= 0)]
    if lag.empty:
        return empty
    out: dict[str, float | int | None] = {
        f"p{p}": float(lag.quantile(p / 100.0)) for p in LAG_DAYS_PERCENTILES
    }
    out["n"] = int(lag.shape[0])
    return out


def try_load_pretensions(paths: DataPaths) -> pd.DataFrame | None:
    """Опционально загрузить df_pretensions.parquet (без падения)."""
    path = paths.resolve_artifact(paths.raw_dir, PRETENSIONS_ARTIFACT)
    if path is None or not path.is_file():
        logger.warning(
            "Нет %s — cooloff по претензиям только через claims; "
            "тишина без претензий может быть неточной",
            PRETENSIONS_ARTIFACT,
        )
        return None
    return read_parquet_path(path, artifact=PRETENSIONS_ARTIFACT)


def apply_target_maturity(
    df: pd.DataFrame,
    claims: pd.DataFrame,
    *,
    pretensions: pd.DataFrame | None = None,
    filters: dict[str, Any] | None = None,
    report_path: Path | str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Отбор строк для обучения по политике закрытия.

    1) ``enabled=false`` → без фильтра (отчёт с enabled=False);
    2) open court → исключить;
    3) событие претензия/ФУ/суд раньше T0 → исключить (если drop_psr_before_t0);
    4) **тишина**: нет ПСР и прошли оба горизонта после T0
       (претензии ``silent_horizon_months_pretension``, ФУ/суд
       ``silent_horizon_months_fu_court``) → оставить;
    5) **охлаждение**: было ПСР и last_event + cooloff ≤ S → оставить;
    6) иначе исключить (ещё вызревает).
    """
    cfg = _maturity_config(filters)
    n_before = int(len(df))
    if not cfg.get("enabled", True):
        report: dict[str, Any] = {
            "enabled": False,
            "policy": "disabled",
            "n_before": n_before,
            "n_after": n_before,
        }
        logger.info("maturity: disabled, rows unchanged (%s)", n_before)
        if report_path is not None:
            path = Path(report_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return df, report

    t0_column = str(cfg.get("t0_column") or _T0_COLUMN)
    pret_months = _silent_pretension_months(cfg)
    fu_court_months = _silent_fu_court_months(cfg)
    cooloff_months = _cooloff_months(cfg)
    drop_before = _drop_psr_before_t0(cfg)
    if t0_column not in df.columns:
        raise KeyError(f"Для вызревания нужна колонка {t0_column}")
    if "INCIDENT_NUMBER" not in df.columns:
        raise KeyError("Для вызревания нужна колонка INCIDENT_NUMBER")

    claims_n = _normalize_claims_columns(claims)
    snapshot, snapshot_source = resolve_snapshot_date(filters)
    t0 = pd.to_datetime(df[t0_column], errors="coerce").dt.normalize()
    missing_t0 = t0.isna()
    inc_key = _incident_key(df["INCIDENT_NUMBER"])

    open_ids = set(_incident_key(pd.Series(incidents_with_open_court(claims_n))).tolist())
    open_court = inc_key.isin(open_ids)

    has_psr_by_inc, first_by_inc, last_by_inc = _psr_event_dates(claims_n, pretensions)
    has_psr = _map_incident_series(inc_key, has_psr_by_inc, fill_bool=False)
    first_event = pd.to_datetime(
        _map_incident_series(inc_key, first_by_inc),
        errors="coerce",
    )
    last_event = pd.to_datetime(
        _map_incident_series(inc_key, last_by_inc),
        errors="coerce",
    )

    psr_before_t0 = (
        has_psr
        & first_event.notna()
        & (~missing_t0)
        & (first_event.dt.normalize() < t0)
    )
    if not drop_before:
        psr_before_t0 = pd.Series(False, index=df.index)

    # Silent: оба горизонта должны истечь (эффективно = max = ФУ/суд).
    silent_end_pret = t0 + pd.DateOffset(months=pret_months)
    silent_end_court = t0 + pd.DateOffset(months=fu_court_months)
    cooloff_end = last_event + pd.DateOffset(months=cooloff_months)

    branch_silent = (
        (~missing_t0)
        & (~has_psr)
        & (silent_end_pret <= snapshot)
        & (silent_end_court <= snapshot)
    )
    branch_cooloff = (
        (~missing_t0)
        & has_psr
        & last_event.notna()
        & (cooloff_end <= snapshot)
        & (~psr_before_t0)
    )
    eligible = (
        ~missing_t0
        & ~open_court
        & ~psr_before_t0
        & (branch_silent | branch_cooloff)
    )
    keep = eligible

    out = df.loc[keep].reset_index(drop=True)
    lags = _lag_days_percentiles(df.loc[keep], claims_n, t0_column=t0_column)
    pret_until = (snapshot - pd.DateOffset(months=pret_months)).normalize()
    silent_until = (snapshot - pd.DateOffset(months=fu_court_months)).normalize()
    cooloff_until = (snapshot - pd.DateOffset(months=cooloff_months)).normalize()

    silent_kept = branch_silent & ~open_court & ~missing_t0 & ~psr_before_t0
    cooloff_kept = branch_cooloff & ~open_court & ~missing_t0
    still_maturing = (
        ~missing_t0
        & ~open_court
        & ~psr_before_t0
        & ~branch_silent
        & ~branch_cooloff
    )

    report = {
        "enabled": True,
        "policy": "silent_dual_horizon_or_cooloff_after_last_psr",
        "t0_column": t0_column,
        "silent_horizon_months_pretension": pret_months,
        "silent_horizon_months_fu_court": fu_court_months,
        # эффективный горизонт тишины (keep) = ФУ/суд
        "silent_horizon_months": fu_court_months,
        "cooloff_months_after_psr": cooloff_months,
        "drop_psr_before_t0": drop_before,
        "horizon_months_no_court": fu_court_months,
        "snapshot_date": str(snapshot.date()),
        "snapshot_source": snapshot_source,
        "mature_until": str(pd.Timestamp(silent_until).date()),
        "silent_until": str(pd.Timestamp(silent_until).date()),
        "silent_pretension_until": str(pd.Timestamp(pret_until).date()),
        "cooloff_until": str(pd.Timestamp(cooloff_until).date()),
        "n_before": n_before,
        "n_after": int(len(out)),
        "n_dropped_t0_missing": int(missing_t0.sum()),
        "n_dropped_open_court": int((~missing_t0 & open_court).sum()),
        "n_dropped_psr_before_t0": int((~missing_t0 & ~open_court & psr_before_t0).sum()),
        "n_kept_silent": int(silent_kept.sum()),
        "n_kept_cooloff": int(cooloff_kept.sum()),
        "n_dropped_still_maturing": int(still_maturing.sum()),
        "n_with_psr_event": int((~missing_t0 & has_psr).sum()),
        "n_without_psr_event": int((~missing_t0 & ~has_psr).sum()),
        "n_kept_branch_a": int(cooloff_kept.sum()),
        "n_kept_branch_b_only": int(silent_kept.sum()),
        "n_dropped_horizon": int(still_maturing.sum()),
        "lag_days_target_freq_1": lags,
        "lag_definition": "T0 → CLAIMEDVALUEPERIOD.min (first claim instance)",
        "pretensions_loaded": pretensions is not None and len(pretensions) > 0,
    }
    logger.info(
        "maturity: S=%s (%s) silent pret=%sм fu/court=%sм cooloff=%sм "
        "rows %s → %s (open=%s before_t0=%s silent=%s cooloff=%s maturing=%s)",
        report["snapshot_date"],
        snapshot_source,
        pret_months,
        fu_court_months,
        cooloff_months,
        n_before,
        report["n_after"],
        report["n_dropped_open_court"],
        report["n_dropped_psr_before_t0"],
        report["n_kept_silent"],
        report["n_kept_cooloff"],
        report["n_dropped_still_maturing"],
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
    maturity_enabled: bool | None = None,
) -> pd.DataFrame:
    """Вызревание по ``target_3_claims`` (+ опционально pretensions)."""
    filters = with_maturity_enabled(filters, maturity_enabled)
    cfg = _maturity_config(filters)
    report_path = paths.processed_dir / MATURITY_REPORT_NAME if save_report else None
    if not cfg.get("enabled", True):
        _, _ = apply_target_maturity(
            df,
            pd.DataFrame(),
            pretensions=None,
            filters=filters,
            report_path=report_path,
        )
        return df
    claims = read_artifact(paths, paths.raw_dir, CLAIMS_ARTIFACT)
    pretensions = try_load_pretensions(paths)
    out, _ = apply_target_maturity(
        df,
        claims,
        pretensions=pretensions,
        filters=filters,
        report_path=report_path,
    )
    return out
