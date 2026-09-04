"""Pandas-утилиты претензий (без SQL)."""
from __future__ import annotations

import pandas as pd

PRETENSION_PAID_ANSWER_TYPES: tuple[str, ...] = ("Выплата", "Частичная выплата")


def dedupe_pretension_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Убрать дубликаты строк претензий после JOIN с IncidentToLoss."""
    result = df.loc[:, ~df.columns.duplicated()].copy()
    for key in ("PRETENSION_ID", "PRETENSIONID"):
        if key in result.columns:
            result = result.drop_duplicates(subset=[key])
            break
    else:
        if "PRETENSION_NUMBER" in result.columns:
            subset = ["PRETENSION_NUMBER"]
            if "LOSS_ID" in result.columns:
                subset.append("LOSS_ID")
            elif "LOSSID" in result.columns:
                subset.append("LOSSID")
            result = result.drop_duplicates(subset=subset)

    incident_col = _first_present(result.columns, "INCIDENT_NUMBER", "INCIDENTNUMBER")
    pret_num_col = _first_present(result.columns, "PRETENSION_NUMBER", "PRETENSIONNUMBER")
    if incident_col is not None and pret_num_col is not None:
        result = result.drop_duplicates(subset=[incident_col, pret_num_col])
    return result


def _first_present(columns: pd.Index, *candidates: str) -> str | None:
    upper = {c.upper(): c for c in columns}
    for name in candidates:
        if name.upper() in upper:
            return upper[name.upper()]
    return None


def collapse_pretension_surcharge_to_incident(
    pret: pd.DataFrame,
    *,
    surcharge_out: str,
    uts_out: str,
    pretension_types: tuple[str, ...] | None = None,
    answer_types: tuple[str, ...] = PRETENSION_PAID_ANSWER_TYPES,
) -> pd.DataFrame:
    """Схлопнуть Surcharge/UTS на инцидент без двойного счёта одного номера претензии."""
    if pret.empty:
        return pd.DataFrame(columns=["INCIDENT_NUMBER", surcharge_out, uts_out])

    incident_col = _first_present(pret.columns, "INCIDENT_NUMBER", "INCIDENTNUMBER")
    pret_num_col = _first_present(pret.columns, "PRETENSION_NUMBER", "PRETENSIONNUMBER")
    surcharge_col = _first_present(pret.columns, "SURCHARGE_VALUE", "SURCHARGEVALUE")
    uts_col = _first_present(pret.columns, "UTS_SURCHARGE_VALUE", "UTSSURCHARGEVALUE")
    answer_col = _first_present(pret.columns, "ANSWER_TYPE", "ANSWERTYPE")
    type_col = _first_present(pret.columns, "PRETENSION_TYPE", "PRETENSIONTYPE")

    if incident_col is None or pret_num_col is None:
        raise KeyError("Для схлопывания нужны INCIDENT_NUMBER и PRETENSION_NUMBER")
    if surcharge_col is None or uts_col is None:
        raise KeyError("Для схлопывания нужны SurchargeValue и UTSSurchargeValue")

    work = pret.copy()
    if answer_col is not None and answer_types:
        work = work[work[answer_col].astype(str).isin(answer_types)]
    if pretension_types is not None and type_col is not None:
        work = work[work[type_col].astype(str).isin(pretension_types)]

    work[surcharge_col] = pd.to_numeric(work[surcharge_col], errors="coerce")
    work[uts_col] = pd.to_numeric(work[uts_col], errors="coerce")

    per_pret = (
        work.groupby([incident_col, pret_num_col], as_index=False)
        .agg({surcharge_col: "max", uts_col: "max"})
    )
    out = (
        per_pret.groupby(incident_col, as_index=False)
        .agg({surcharge_col: "sum", uts_col: "sum"})
        .rename(
            columns={
                incident_col: "INCIDENT_NUMBER",
                surcharge_col: surcharge_out,
                uts_col: uts_out,
            }
        )
    )
    return out
