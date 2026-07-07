"""Historical court (incoming claims) features per role as-of T0 (previous incidents only)."""

from __future__ import annotations

import pandas as pd

from querulus.features.person.config import INCIDENT_COLUMN, PERSON_PREFIX, ROLES, T0_COLUMN
from querulus.features.person.loaders import normalize_person_id_series, normalize_person_id_series as _norm_pid


def _prep_claims_persons(df_persons: pd.DataFrame) -> pd.DataFrame:
    df = df_persons.copy()
    # Map columns from Russian names to stable ones.
    if "НомерИск" in df.columns and "INCOMING_CLAIM_NUMBER" not in df.columns:
        df["INCOMING_CLAIM_NUMBER"] = df["НомерИск"]
    if "Лицо" in df.columns and "PERSON_ID" not in df.columns:
        df["PERSON_ID"] = df["Лицо"]
    df["PERSON_ID"] = _norm_pid(df["PERSON_ID"])
    return df


def _prep_claims_incoming(df_claims: pd.DataFrame) -> pd.DataFrame:
    df = df_claims.copy()
    # Ensure canonical columns.
    if "INCIDENTNUMBER" in df.columns and INCIDENT_COLUMN not in df.columns:
        df[INCIDENT_COLUMN] = df["INCIDENTNUMBER"]
    if "INCOMINGCLAIMNUMBER" in df.columns and "INCOMING_CLAIM_NUMBER" not in df.columns:
        df["INCOMING_CLAIM_NUMBER"] = df["INCOMINGCLAIMNUMBER"]
    if "INCOMINGCLAIMGETDATE" in df.columns and "INCOMING_CLAIM_GET_DATE" not in df.columns:
        df["INCOMING_CLAIM_GET_DATE"] = df["INCOMINGCLAIMGETDATE"]

    df[INCIDENT_COLUMN] = pd.to_numeric(df.get(INCIDENT_COLUMN), errors="coerce")
    df["INCOMING_CLAIM_GET_DATE"] = pd.to_datetime(df.get("INCOMING_CLAIM_GET_DATE"), errors="coerce")
    return df


def _aggregate_court_history(
    df_claims: pd.DataFrame,
    df_persons: pd.DataFrame,
    *,
    person_id: pd.Series,
    t0: pd.Series,
    current_incident: pd.Series,
) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "_row": person_id.index,
            "_pid": person_id.values,
            "_t0": pd.to_datetime(t0, errors="coerce").values,
            "_inc": pd.to_numeric(current_incident, errors="coerce").values,
        }
    ).dropna(subset=["_pid"])

    claims = df_claims.merge(
        df_persons[["INCOMING_CLAIM_NUMBER", "PERSON_ID"]],
        on="INCOMING_CLAIM_NUMBER",
        how="inner",
    )
    claims = claims.rename(
        columns={
            "PERSON_ID": "_pid",
            "INCOMING_CLAIM_GET_DATE": "_claim_date",
            INCIDENT_COLUMN: "_claim_inc",
        }
    )
    claims = claims.dropna(subset=["_pid", "_claim_date", "_claim_inc"])

    merged = base.merge(claims, on="_pid", how="left")
    mask = (merged["_claim_date"] < merged["_t0"]) & (merged["_claim_inc"] != merged["_inc"])
    merged = merged[mask]

    grouped = merged.groupby("_row", dropna=False)
    out = pd.DataFrame(index=person_id.index)

    out[f"{PERSON_PREFIX}COURT_CLAIM_COUNT_ROWS"] = grouped.size().reindex(out.index).fillna(0).astype(int)
    out[f"{PERSON_PREFIX}COURT_INCOMING_CLAIM_NUMBER_NUNIQUE"] = (
        grouped["INCOMING_CLAIM_NUMBER"].nunique().reindex(out.index).fillna(0).astype(int)
    )

    # Representative/cessionary flags if present.
    for flag in ("ПРЕДСТАВИТЕЛЬ", "ЦЕССИОНАРИЙ"):
        if flag in merged.columns:
            values = pd.to_numeric(merged[flag], errors="coerce")
            merged[flag] = values
            out[f"{PERSON_PREFIX}COURT_{flag}_MAX"] = grouped[flag].max().reindex(out.index)
            out[f"{PERSON_PREFIX}COURT_{flag}_SUM"] = grouped[flag].sum(min_count=1).reindex(out.index)

    # Money columns: CLAIMED* and RECOVERED* (as requested), sum and mean.
    money_cols = [c for c in merged.columns if c.startswith("CLAIMED") or c.startswith("RECOVERED")]
    for col in money_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
        out[f"{PERSON_PREFIX}COURT_{col}_SUM"] = grouped[col].sum(min_count=1).reindex(out.index)
        out[f"{PERSON_PREFIX}COURT_{col}_MEAN"] = grouped[col].mean().reindex(out.index)

    # Claim item/origin modes.
    for col in ("CLAIMITEM", "CLAIMORIGIN"):
        if col in merged.columns:
            out[f"{PERSON_PREFIX}COURT_{col}_MODE"] = (
                grouped[col].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA).reindex(out.index)
            )

    return out


def add_person_court_history(df: pd.DataFrame, df_claims_incoming: pd.DataFrame, df_claims_persons: pd.DataFrame) -> pd.DataFrame:
    """Добавить FE_PERSON_COURT_{ROLE}_* для всех ролей."""
    out = df.copy()
    claims = _prep_claims_incoming(df_claims_incoming)
    persons = _prep_claims_persons(df_claims_persons)

    t0 = out.get(T0_COLUMN)
    current_incident = out.get(INCIDENT_COLUMN)

    for role in ROLES:
        if role.person_id_column not in out.columns:
            continue
        pid = normalize_person_id_series(out[role.person_id_column])
        agg = _aggregate_court_history(
            claims,
            persons,
            person_id=pid,
            t0=t0,
            current_incident=current_incident,
        )
        agg = agg.add_prefix(f"{PERSON_PREFIX}COURT_{role.suffix}_")
        out = out.join(agg)

    return out

