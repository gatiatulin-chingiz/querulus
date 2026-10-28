"""FE_* для second_ (только фичи из config 2.0.0). Без импорта querulus.src."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from features.cpi import (
        INFLATION_BASE_YEAR,
        add_real_monetary_columns,
        deflate_to_base_year,
        real_feature_name,
    )
except ImportError:
    from integration.features.cpi import (
        INFLATION_BASE_YEAR,
        add_real_monetary_columns,
        deflate_to_base_year,
        real_feature_name,
    )

SHARE_WORK_BINS: tuple[float, ...] = (0.3, 0.6)
SHARE_WORK_LABELS: tuple[str, ...] = ("low", "mid", "high")


#new
def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(pd.NA, index=df.index)


#new
def _to_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


#new
def _days_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    return (_to_dt(later) - _to_dt(earlier)).dt.days


#new
def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return num / den.where(den != 0)


#new
def _tier_from_bins(
    value: pd.Series,
    edges: tuple[float, ...] = SHARE_WORK_BINS,
    labels: tuple[str, ...] = SHARE_WORK_LABELS,
) -> pd.Series:
    values = pd.to_numeric(value, errors="coerce")
    low, high = edges
    cut = pd.cut(
        values,
        bins=[-np.inf, low, high, np.inf],
        labels=list(labels),
        right=False,
        ordered=True,
    )
    # object: иначе outboxml fillna падает на Categorical (new category).
    return cut.astype(object).where(cut.notna(), np.nan)


#new
def _ensure_share_work(df: pd.DataFrame) -> pd.DataFrame:
    """SHARE_WORK = Работы / VALUE_BEFORE_WITHOUT (не AMOUNT_REPAIR)."""
    out = df
    if "SHARE_WORK" in out.columns:
        return out
    works = None
    for name in ("WORK", "WORKS", "РАБОТЫ", "Raboty", "SHARE_WORK_NUM"):
        if name in out.columns:
            works = pd.to_numeric(out[name], errors="coerce")
            break
    if works is None:
        return out
    denom = pd.to_numeric(_col(out, "VALUE_BEFORE_WITHOUT"), errors="coerce")
    out = out.copy()
    out["SHARE_WORK"] = _safe_div(works, denom)
    return out


#new
def vehicle_age_months_from_year(
    object_year: pd.Series,
    base_date: pd.Series,
) -> pd.Series:
    base = _to_dt(base_date)
    year = pd.to_numeric(object_year, errors="coerce")
    age = (base.dt.year - year) * 12 + (base.dt.month - 1)
    age = pd.to_numeric(age, errors="coerce")
    age = age.where(age.isna(), age.clip(lower=0))
    return age.round().astype("Int64")


#new
def _cpi_asof_dates(df: pd.DataFrame) -> pd.Series:
    """Год CPI как в обучении: t0 = PAYMENT_ORDER_DATE_TIME, иначе EVENT_DATE."""
    pay = _to_dt(_col(df, "PAYMENT_ORDER_DATE_TIME"))
    event = _to_dt(_col(df, "EVENT_DATE"))
    return pay.where(pay.notna(), event)


#new
def add_second_model_features(
    df: pd.DataFrame,
    *,
    base_year: int = INFLATION_BASE_YEAR,
) -> pd.DataFrame:
    """Добавить FE_* / age months, нужные CF+RG 2.0.0.

    REAL-суммы дефлируются по дате выплаты (как ``features/derived.py`` на t0),
    не по дате ДТП. Дни event→loss считаются по ``LOSS_DATE_TIME``;
    если даты убытка нет, остаётся уже посчитанный ``FE_DAYS_EVENT_TO_LOSS``.
    """
    precomputed_days = (
        pd.to_numeric(df["FE_DAYS_EVENT_TO_LOSS"], errors="coerce")
        if "FE_DAYS_EVENT_TO_LOSS" in df.columns
        else None
    )
    out = df.copy()
    out["_CPI_ASOF"] = _cpi_asof_dates(out)
    out = add_real_monetary_columns(
        out, event_date_column="_CPI_ASOF", base_year=base_year
    )
    out = out.drop(columns=["_CPI_ASOF"])
    out = _ensure_share_work(out)

    t0 = "PAYMENT_ORDER_DATE_TIME"
    value_without_real = pd.to_numeric(
        _col(out, real_feature_name("VALUE_BEFORE_WITHOUT", base_year)), errors="coerce"
    )
    value_with_real = pd.to_numeric(
        _col(out, real_feature_name("VALUE_BEFORE_WITH", base_year)), errors="coerce"
    )

    both_ok = (value_without_real > 0) & (value_with_real >= 0)
    both_positive = (value_without_real > 0) & (value_with_real > 0)
    out["FE_VALUE_BEFORE_DIFF"] = (value_without_real - value_with_real).where(both_ok)
    out["FE_VALUE_BEFORE_RATIO"] = _safe_div(value_with_real, value_without_real).where(
        both_positive
    )

    loss_dt = _to_dt(_col(out, "LOSS_DATE_TIME"))
    event_dt = _to_dt(_col(out, "EVENT_DATE"))
    days = (loss_dt - event_dt).dt.days
    if precomputed_days is not None:
        days = days.where(loss_dt.notna(), precomputed_days)
    out["FE_DAYS_EVENT_TO_LOSS"] = days
    out["FE_DAYS_TO_PH_CONTRACT_END"] = _days_between(
        _col(out, "POLICYHOLDER_CONTRACT_END_DATE"),
        _col(out, "EVENT_DATE"),
    )

    out["FE_SHARE_WORK_TIER"] = _tier_from_bins(_col(out, "SHARE_WORK"))

    out["FE_VICTIM_POWER_PER_TON"] = (
        _safe_div(_col(out, "VICTIM_CAPACITY_ENGINE"), _col(out, "VICTIM_MAX_WEIGHT"))
        * 1e6
    )

    if "VICTIM_OBJECT_YEAR" in out.columns:
        base = _col(out, "EVENT_DATE")
        if base.isna().all():
            base = _col(out, t0)
        out["VICTIM_OBJECT_AGE_MONTHS"] = vehicle_age_months_from_year(
            out["VICTIM_OBJECT_YEAR"], base
        )

    # Клип отрицательного DIFF → 0 (как в train DQ).
    if "FE_VALUE_BEFORE_DIFF" in out.columns:
        diff = pd.to_numeric(out["FE_VALUE_BEFORE_DIFF"], errors="coerce")
        out["FE_VALUE_BEFORE_DIFF"] = diff.clip(lower=0)

    return out
