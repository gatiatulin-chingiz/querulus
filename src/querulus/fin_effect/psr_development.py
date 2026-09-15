"""Сроки развития претензий / ФУ / суда на ретро (лаги от T0)."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_PERCENTILES: tuple[int, ...] = (50, 70, 80, 90, 95)
_FU_CLAIM_ORIGIN = "Обращение к ФУ"


def _first_col(columns: Iterable[str], *names: str) -> str | None:
    lower = {str(c).casefold(): str(c) for c in columns}
    for name in names:
        hit = lower.get(name.casefold())
        if hit is not None:
            return hit
    return None


def _lag_percentiles(
    lag_days: pd.Series,
    *,
    stage: str,
    percentiles: tuple[int, ...],
) -> dict[str, float | int | None]:
    clean = pd.to_numeric(lag_days, errors="coerce")
    clean = clean[clean.notna() & (clean >= 0)]
    row: dict[str, float | int | None] = {
        "stage": stage,
        "n": int(clean.shape[0]),
        "definition": "",
    }
    for p in percentiles:
        row[f"p{p}"] = float(clean.quantile(p / 100.0)) if not clean.empty else None
    return row


def compute_psr_development_lags(
    *,
    pretensions: pd.DataFrame | None = None,
    claims: pd.DataFrame | None = None,
    incidents: pd.DataFrame | None = None,
    t0_column: str = "PAYMENT_ORDER_DATE_TIME",
    percentiles: tuple[int, ...] = DEFAULT_PERCENTILES,
) -> pd.DataFrame:
    """Перцентили лагов T0 → первая претензия / ФУ / суд.

    ``incidents`` — incident-level df с T0 (обычно train parquet).
    Без pretensions/claims соответствующие стадии пропускаются.
    """
    rows: list[dict[str, float | int | None]] = []
    if incidents is None or incidents.empty:
        return pd.DataFrame(rows)

    t0_col = t0_column if t0_column in incidents.columns else _first_col(
        incidents.columns,
        "PAYMENT_ORDER_DATE_TIME",
        "T0",
        "LOSS_DATE",
    )
    inc_col = _first_col(
        incidents.columns,
        "INCIDENT_NUMBER",
        "НомерИнцидент",
        "НомерИнцидента",
    )
    if t0_col is None or inc_col is None:
        return pd.DataFrame(rows)

    base = incidents[[inc_col, t0_col]].copy()
    base["_inc"] = base[inc_col].astype("string").str.strip()
    base["_t0"] = pd.to_datetime(base[t0_col], errors="coerce")
    base = base.dropna(subset=["_inc", "_t0"]).drop_duplicates("_inc")

    if pretensions is not None and not pretensions.empty:
        p_inc = _first_col(
            pretensions.columns,
            "INCIDENT_NUMBER",
            "НомерИнцидент",
            "НомерИнцидента",
        )
        p_date = _first_col(
            pretensions.columns,
            "PRETENSION_GET_DATE",
            "PRETENSIONGETDATE",
            "PRETENSION_DATE",
            "PRETENSIONDATE",
        )
        if p_inc is not None and p_date is not None:
            work = pretensions[[p_inc, p_date]].copy()
            work["_inc"] = work[p_inc].astype("string").str.strip()
            work["_event"] = pd.to_datetime(work[p_date], errors="coerce")
            first = work.dropna(subset=["_inc", "_event"]).groupby("_inc", as_index=False)[
                "_event"
            ].min()
            merged = base.merge(first, on="_inc", how="inner")
            lag = (merged["_event"] - merged["_t0"]).dt.days
            row = _lag_percentiles(lag, stage="pretension_first", percentiles=percentiles)
            row["definition"] = "T0 → min(PRETENSION_GET_DATE) по инциденту"
            rows.append(row)

    if claims is not None and not claims.empty:
        c_inc = _first_col(
            claims.columns,
            "INCIDENT_NUMBER",
            "НомерИнцидент",
            "НомерИнцидента",
        )
        c_period = _first_col(
            claims.columns,
            "CLAIMEDVALUEPERIOD",
            "INCOMING_CLAIM_GET_DATE",
            "CLAIM_PERIOD",
        )
        c_origin = _first_col(
            claims.columns,
            "CLAIMORIGIN",
            "CLAIM_ORIGIN",
            "IncomingClaimOrigin",
        )
        c_over = _first_col(
            claims.columns,
            "COURTWORKOVERDATE",
            "CourtWorkOverDate",
        )
        if c_inc is not None and c_period is not None:
            work = claims[[c_inc, c_period]].copy()
            work["_inc"] = work[c_inc].astype("string").str.strip()
            work["_event"] = pd.to_datetime(work[c_period], errors="coerce")
            if c_origin is not None:
                work["_origin"] = claims[c_origin].astype("string")
                fu = work.loc[
                    work["_origin"].fillna("").str.contains(_FU_CLAIM_ORIGIN, case=False)
                ]
                if not fu.empty:
                    first_fu = fu.dropna(subset=["_inc", "_event"]).groupby(
                        "_inc", as_index=False
                    )["_event"].min()
                    merged = base.merge(first_fu, on="_inc", how="inner")
                    lag = (merged["_event"] - merged["_t0"]).dt.days
                    row = _lag_percentiles(
                        lag, stage="fu_first", percentiles=percentiles
                    )
                    row["definition"] = (
                        f"T0 → min(CLAIMEDVALUEPERIOD) при origin «{_FU_CLAIM_ORIGIN}»"
                    )
                    rows.append(row)
                court = work.loc[
                    ~work["_origin"]
                    .fillna("")
                    .str.contains(_FU_CLAIM_ORIGIN, case=False)
                ]
            else:
                court = work
            first_court = court.dropna(subset=["_inc", "_event"]).groupby(
                "_inc", as_index=False
            )["_event"].min()
            merged = base.merge(first_court, on="_inc", how="inner")
            lag = (merged["_event"] - merged["_t0"]).dt.days
            row = _lag_percentiles(lag, stage="court_claim_first", percentiles=percentiles)
            row["definition"] = "T0 → min(CLAIMEDVALUEPERIOD) по искам"
            rows.append(row)

        if c_inc is not None and c_over is not None:
            work = claims[[c_inc, c_over]].copy()
            work["_inc"] = work[c_inc].astype("string").str.strip()
            work["_event"] = pd.to_datetime(work[c_over], errors="coerce")
            first = work.dropna(subset=["_inc", "_event"]).groupby("_inc", as_index=False)[
                "_event"
            ].min()
            merged = base.merge(first, on="_inc", how="inner")
            lag = (merged["_event"] - merged["_t0"]).dt.days
            row = _lag_percentiles(
                lag, stage="court_over_first", percentiles=percentiles
            )
            row["definition"] = "T0 → min(CourtWorkOverDate)"
            rows.append(row)

    return pd.DataFrame(rows)


def maturity_policy_reference() -> pd.DataFrame:
    """Ориентиры политики maturity (не эмпирические перцентили)."""
    return pd.DataFrame(
        [
            {
                "stage": "pretension_silent_policy",
                "days": 183,
                "note": "Тишина по претензиям: ~6 месяцев после T0",
            },
            {
                "stage": "fu_court_silent_policy",
                "days": 730,
                "note": "Тишина по ФУ/суду: ~24 месяца после T0",
            },
            {
                "stage": "cooloff_after_psr_policy",
                "days": 730,
                "note": "Охлаждение после последнего события ПСР: ~24 месяца",
            },
        ]
    )


__all__ = [
    "DEFAULT_PERCENTILES",
    "compute_psr_development_lags",
    "maturity_policy_reference",
]
