"""NPV-анализ: окупают ли инвестиции на pred_sev весь ПСР к дате суда."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from querulus.dataset.constants import RENAME_DICT
from querulus.dataset.steps.targets import (
    _CLAIM_PERIOD_COL,
    _TARGET_FREQ_CLAIMS_GROUP,
    _pick_last_claim_instances,
)

_T0_COL = "PAYMENT_ORDER_DATE_TIME"
_PSR_COL = "TARGET_FREQ_AMOUNT"
_COURT_DATE_COL = "COURTWORKOVERDATE"

DEFAULT_RATES: tuple[float, ...] = (0.08, 0.12, 0.16)


@dataclass(frozen=True)
class NpvReport:
    """Результат NPV-анализа."""

    date_coverage: pd.DataFrame
    rate_table: pd.DataFrame
    detail: pd.DataFrame


def _load_claims(project_root: Path) -> pd.DataFrame:
    """Загрузить target_3_claims.parquet и нормализовать колонки."""
    raw_dir = project_root / "data" / "raw"
    path = raw_dir / "target_3_claims.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"target_3_claims.parquet не найден: {path}. "
            "Запустите build_targets с USE_SQL=True."
        )
    df = pd.read_parquet(str(path))
    df = df.rename(columns=RENAME_DICT)
    df.columns = df.columns.str.upper()
    df = df.rename(columns=RENAME_DICT)
    return df


def _court_end_date_by_incident(claims: pd.DataFrame) -> pd.DataFrame:
    """Max CourtWorkOverDate последней принятой инстанции по инциденту."""
    court_col = None
    for candidate in (_COURT_DATE_COL, "COURTWORKOVERDATE"):
        uc = candidate.upper()
        for col in claims.columns:
            if col.upper() == uc:
                court_col = col
                break
        if court_col is not None:
            break
    if court_col is None:
        raise KeyError(
            f"В target_3_claims нет колонки {_COURT_DATE_COL}. "
            f"Доступные: {sorted(claims.columns[:20])}"
        )

    last = _pick_last_claim_instances(claims)
    last["_court_date"] = pd.to_datetime(last[court_col], errors="coerce")
    incident_col = "INCIDENT_NUMBER"
    if incident_col not in last.columns:
        for col in last.columns:
            if col.upper() in ("INCIDENT_NUMBER", "INCIDENTNUMBER"):
                incident_col = col
                break
    agg = (
        last.groupby(incident_col, as_index=False)["_court_date"]
        .max()
        .rename(columns={"_court_date": "t_end"})
    )
    return agg


def _compute_npv_detail(
    holdout: pd.DataFrame,
    pred_freq: pd.Series,
    pred_sev: pd.Series,
    t_end_map: pd.DataFrame,
    rates: tuple[float, ...],
) -> pd.DataFrame:
    """Поинцидентная таблица profit для каждого r."""
    work = holdout[["INCIDENT_NUMBER", _T0_COL, _PSR_COL, "TARGET_FREQ"]].copy()
    work["pred_freq"] = pred_freq.values
    work["pred_sev"] = pred_sev.values
    work["_inc_key"] = work["INCIDENT_NUMBER"].astype(str).str.strip()
    t_end_map = t_end_map.copy()
    t_end_map["_inc_key"] = t_end_map["INCIDENT_NUMBER"].astype(str).str.strip()
    work = work.merge(
        t_end_map[["_inc_key", "t_end"]], on="_inc_key", how="left"
    )

    t0 = pd.to_datetime(work[_T0_COL], errors="coerce")
    t_end = pd.to_datetime(work["t_end"], errors="coerce")
    dt_days = (t_end - t0).dt.days.fillna(0).clip(lower=0)
    dt_years = dt_days / 365.0

    work["dt_days"] = dt_days
    work["dt_years"] = dt_years

    p = np.where(work["pred_freq"] == 1, work["pred_sev"].fillna(0).values, 0.0)
    f = work[_PSR_COL].fillna(0).values

    for r in rates:
        col = f"profit_r{int(r * 100)}"
        fv = p * (1 + r) ** dt_years.values
        work[col] = fv - f

    return work


def _date_coverage_table(detail: pd.DataFrame) -> pd.DataFrame:
    """Статистика покрытия дат t_end."""
    has_date = detail["t_end"].notna()
    dt = detail.loc[has_date, "dt_days"]
    rows = [
        {"показатель": "строк holdout", "значение": len(detail)},
        {"показатель": "с валидной CourtWorkOverDate", "значение": int(has_date.sum())},
        {"показатель": "без даты суда", "значение": int((~has_date).sum())},
        {"показатель": "медиана лага T0→t_end (дни)", "значение": float(dt.median()) if len(dt) else None},
        {"показатель": "p90 лага (дни)", "значение": float(dt.quantile(0.9)) if len(dt) else None},
    ]
    return pd.DataFrame(rows)


def _rate_summary_table(detail: pd.DataFrame, rates: tuple[float, ...]) -> pd.DataFrame:
    """Агрегат по ставкам: сколько дел окупилось, суммарный профит."""
    # Только дела с TARGET_FREQ=1 и pred_freq=1
    sub = detail[(detail["TARGET_FREQ"] == 1) & (detail["pred_freq"] == 1)]
    rows = []
    for r in rates:
        col = f"profit_r{int(r * 100)}"
        profit = sub[col]
        ok = profit >= 0
        rows.append({
            "r (%)": int(r * 100),
            "n_дел": len(sub),
            "инвестиции_окупили": int(ok.sum()),
            "не_окупили": int((~ok).sum()),
            "доля_надо_платить": round(float((~ok).mean()), 4) if len(sub) else None,
            "суммарный_profit (₽)": round(float(profit.sum()), 2),
            "суммарный_profit_окупившихся (₽)": round(float(profit[ok].sum()), 2),
            "суммарный_убыток (₽)": round(float(profit[~ok].sum()), 2),
        })
    return pd.DataFrame(rows)


def run_npv_analysis(
    df: pd.DataFrame,
    training,
    index: pd.Index,
    project_root: Path,
    *,
    rates: tuple[float, ...] = DEFAULT_RATES,
) -> NpvReport:
    """Запуск NPV-анализа на holdout стека new.

    Parameters
    ----------
    df : основной датафрейм (с TARGET_FREQ, TARGET_FREQ_AMOUNT и т.д.)
    training : TrainingArtifacts стека new
    index : holdout-индекс (test)
    project_root : корень проекта querulus (для target_3_claims.parquet)
    rates : годовые доходности для сценариев
    """
    from querulus.training.stack_eval import _stack_predictions

    claims = _load_claims(project_root)
    t_end_map = _court_end_date_by_incident(claims)

    _, pred_freq, pred_sev = _stack_predictions(training, df, index)
    holdout = df.loc[index].copy()

    detail = _compute_npv_detail(holdout, pred_freq, pred_sev, t_end_map, rates)
    date_cov = _date_coverage_table(detail)
    rate_tbl = _rate_summary_table(detail, rates)

    return NpvReport(
        date_coverage=date_cov,
        rate_table=rate_tbl,
        detail=detail,
    )
