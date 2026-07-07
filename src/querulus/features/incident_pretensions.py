"""Incident-level pretension features as-of T0 (текущий инцидент, без утечки)."""
from __future__ import annotations

import pandas as pd

from querulus.dataset.pretension_utils import dedupe_pretension_rows
from querulus.features.config import FeatureConfig
from querulus.features.derived import _series

INCIDENT_COLUMN = "INCIDENT_NUMBER"
T0_COLUMN = "PAYMENT_ORDER_DATE_TIME"


def _incident_key(series: pd.Series) -> pd.Series:
    """Ключ инцидента для join."""
    return pd.to_numeric(series, errors="coerce")


def _pretension_incident_column(df: pd.DataFrame) -> str | None:
    for name in ("INCIDENT_NUMBER", "INCIDENTNUMBER"):
        if name in df.columns:
            return name
    return None


def _pretension_date_column(df: pd.DataFrame) -> str | None:
    for name in ("PRETENSION_GET_DATE", "PRETENSIONGETDATE"):
        if name in df.columns:
            return name
    return None


def _dedupe_pretensions(df: pd.DataFrame) -> pd.DataFrame:
    """Убрать дубликаты строк претензий после JOIN с IncidentToLoss."""
    return dedupe_pretension_rows(df)


def add_incident_pretension_features(
    df: pd.DataFrame,
    df_pretensions: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Агрегаты Declared_* и сумм по претензиям текущего инцидента до T0."""
    config = config or FeatureConfig()
    out = df.copy()
    if df_pretensions.empty:
        return out

    pret = _dedupe_pretensions(df_pretensions)
    incident_col = _pretension_incident_column(pret)
    date_col = _pretension_date_column(pret)
    if incident_col is None or date_col is None:
        return out

    pret = pret.copy()
    pret["_incident"] = _incident_key(pret[incident_col])
    pret["_pret_date"] = pd.to_datetime(pret[date_col], errors="coerce")

    t0 = pd.to_datetime(_series(out, config.t0_column), errors="coerce")
    out["_incident"] = _incident_key(_series(out, INCIDENT_COLUMN))
    out["_t0"] = t0

    merged = out[["_incident", "_t0"]].merge(
        pret,
        on="_incident",
        how="left",
        suffixes=("", "_pret"),
    )
    mask = merged["_pret_date"].notna() & (merged["_pret_date"] <= merged["_t0"])
    filtered = merged.loc[mask].copy()

    if filtered.empty:
        out = out.drop(columns=["_incident", "_t0"], errors="ignore")
        return out

    declared_cols = [
        column
        for column in filtered.columns
        if column.upper().startswith("DECLARED_")
    ]
    agg_map: dict[str, str] = {"FE_INCIDENT_PRET_COUNT": "_pret_date"}
    agg_funcs: dict[str, str] = {"FE_INCIDENT_PRET_COUNT": "count"}
    for column in declared_cols:
        safe = column.upper().replace("DECLARED_", "")
        name = f"FE_INCIDENT_DECLARED_{safe}_SUM"
        agg_map[name] = column
        agg_funcs[name] = "sum"

    for value_col in ("PRETENSION_VALUE", "PRETENSIONVALUE", "UTSVALUE"):
        if value_col in filtered.columns:
            name = f"FE_INCIDENT_{value_col}_SUM"
            agg_map[name] = value_col
            agg_funcs[name] = "sum"

    grouped = filtered.groupby("_incident").agg(
        **{name: (agg_map[name], agg_funcs[name]) for name in agg_map}
    ).reset_index()

    out = out.merge(grouped, on="_incident", how="left")
    for column in grouped.columns:
        if column != "_incident" and column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0)

    return out.drop(columns=["_incident", "_t0"], errors="ignore")
