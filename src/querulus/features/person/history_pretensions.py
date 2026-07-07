"""Historical pretension features per role as-of T0 (previous incidents only)."""

from __future__ import annotations

import pandas as pd

from querulus.features.person.config import INCIDENT_COLUMN, PERSON_PREFIX, ROLES, T0_COLUMN
from querulus.features.person.loaders import normalize_person_id_series


def _prep_pretensions(df_pret: pd.DataFrame) -> pd.DataFrame:
    df = df_pret.copy()
    # Standardize key columns.
    for col in ("INCIDENTNUMBER", "INCIDENT_NUMBER"):
        if col in df.columns and INCIDENT_COLUMN not in df.columns:
            df[INCIDENT_COLUMN] = df[col]
    if "PRETENSIONGETDATE" in df.columns and "PRETENSION_GET_DATE" not in df.columns:
        df["PRETENSION_GET_DATE"] = df["PRETENSIONGETDATE"]
    if "APPLICANTPERSONID" in df.columns and "APPLICANT_PERSON_ID" not in df.columns:
        df["APPLICANT_PERSON_ID"] = df["APPLICANTPERSONID"]
    if "PRETENSIONVALUE" in df.columns and "PRETENSION_VALUE" not in df.columns:
        df["PRETENSION_VALUE"] = df["PRETENSIONVALUE"]
    if "SURCHARGEVALUE" in df.columns and "SURCHARGE_VALUE" not in df.columns:
        df["SURCHARGE_VALUE"] = df["SURCHARGEVALUE"]
    if "UTSSURCHARGEVALUE" in df.columns and "UTS_SURCHARGE_VALUE" not in df.columns:
        df["UTS_SURCHARGE_VALUE"] = df["UTSSURCHARGEVALUE"]
    if "PRETENSIONNUMBER" in df.columns and "PRETENSION_NUMBER" not in df.columns:
        df["PRETENSION_NUMBER"] = df["PRETENSIONNUMBER"]
    if "PRETENSIONTYPES" in df.columns and "PRETENSION_TYPES" not in df.columns:
        df["PRETENSION_TYPES"] = df["PRETENSIONTYPES"]
    if "PRETENSIONGETMETHOD" in df.columns and "PRETENSION_GET_METHOD" not in df.columns:
        df["PRETENSION_GET_METHOD"] = df["PRETENSIONGETMETHOD"]
    if "ANSWERTYPE" in df.columns and "ANSWER_TYPE" not in df.columns:
        df["ANSWER_TYPE"] = df["ANSWERTYPE"]

    df["PRETENSION_GET_DATE"] = pd.to_datetime(df.get("PRETENSION_GET_DATE"), errors="coerce")
    df[INCIDENT_COLUMN] = pd.to_numeric(df.get(INCIDENT_COLUMN), errors="coerce")
    df["APPLICANT_PERSON_ID"] = normalize_person_id_series(df.get("APPLICANT_PERSON_ID"))
    return df


def _aggregate_pret_history(
    df_pret: pd.DataFrame,
    *,
    person_id: pd.Series,
    t0: pd.Series,
    current_incident: pd.Series,
) -> pd.DataFrame:
    """Return a frame indexed by original df index with aggregated features."""
    # Expand to row-level join: keep only pretensions for persons present in this batch.
    base = pd.DataFrame(
        {
            "_row": person_id.index,
            "_pid": person_id.values,
            "_t0": pd.to_datetime(t0, errors="coerce").values,
            "_inc": pd.to_numeric(current_incident, errors="coerce").values,
        }
    ).dropna(subset=["_pid"])

    pret = df_pret.dropna(subset=["APPLICANT_PERSON_ID", "PRETENSION_GET_DATE", INCIDENT_COLUMN]).copy()
    pret = pret.rename(columns={"APPLICANT_PERSON_ID": "_pid", "PRETENSION_GET_DATE": "_pret_date", INCIDENT_COLUMN: "_pret_inc"})

    merged = base.merge(pret, on="_pid", how="left")
    # history only: before T0 AND different incident
    mask = (merged["_pret_date"] < merged["_t0"]) & (merged["_pret_inc"] != merged["_inc"])
    merged = merged[mask]

    # Numeric columns (money): allowed only for previous incidents.
    money_cols = [c for c in ("PRETENSION_VALUE", "SURCHARGE_VALUE", "UTS_SURCHARGE_VALUE", "PRETENSION_VALUE_PENALTY", "SURCHARGE_VALUE_PENALTY") if c in merged.columns]
    for col in money_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # Aggregations per row.
    grouped = merged.groupby("_row", dropna=False)
    out = pd.DataFrame(index=person_id.index)
    out[f"{PERSON_PREFIX}PRET_COUNT"] = grouped.size().reindex(out.index).fillna(0).astype(int)

    if "PRETENSION_NUMBER" in merged.columns:
        out[f"{PERSON_PREFIX}PRET_PRETENSION_NUMBER_NUNIQUE"] = (
            grouped["PRETENSION_NUMBER"].nunique().reindex(out.index).fillna(0).astype(int)
        )
    if "PRETENSION_TYPES" in merged.columns:
        out[f"{PERSON_PREFIX}PRET_TYPES_NUNIQUE"] = (
            grouped["PRETENSION_TYPES"].nunique().reindex(out.index).fillna(0).astype(int)
        )
    if "PRETENSION_GET_METHOD" in merged.columns:
        out[f"{PERSON_PREFIX}PRET_GET_METHOD_MODE"] = (
            grouped["PRETENSION_GET_METHOD"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA).reindex(out.index)
        )
    if "ANSWER_TYPE" in merged.columns:
        out[f"{PERSON_PREFIX}PRET_ANSWER_TYPE_MODE"] = (
            grouped["ANSWER_TYPE"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA).reindex(out.index)
        )

    for col in money_cols:
        out[f"{PERSON_PREFIX}PRET_{col}_SUM"] = grouped[col].sum(min_count=1).reindex(out.index)

    return out


def add_person_pretension_history(df: pd.DataFrame, df_pretensions: pd.DataFrame) -> pd.DataFrame:
    """Добавить FE_PERSON_PRET_{ROLE}_* для всех ролей (история как applicant pretensions)."""
    out = df.copy()
    pret = _prep_pretensions(df_pretensions)

    t0 = out.get(T0_COLUMN)
    current_incident = out.get(INCIDENT_COLUMN)

    for role in ROLES:
        if role.person_id_column not in out.columns:
            continue
        pid = normalize_person_id_series(out[role.person_id_column])
        agg = _aggregate_pret_history(
            pret,
            person_id=pid,
            t0=t0,
            current_incident=current_incident,
        )
        # namespace per role
        agg = agg.add_prefix(f"{PERSON_PREFIX}PRET_{role.suffix}_")
        out = out.join(agg)

    return out

