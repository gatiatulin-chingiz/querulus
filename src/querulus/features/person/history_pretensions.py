"""Historical pretension features per role as-of T0 (previous incidents only)."""

from __future__ import annotations

import gc
import logging

import pandas as pd

from querulus.features.person.config import (
    INCIDENT_COLUMN,
    PERSON_PREFIX,
    PRETENSION_APPLICANT_COL,
    PRETENSION_RECIPIENT_COL,
    ROLES,
    T0_COLUMN,
)
from querulus.features.person.loaders import normalize_person_id_series
from querulus.features.person.utils import collect_person_ids

logger = logging.getLogger("querulus.features.person")

# Чанки base-строк при merge с историей претензий (снижение пика ОЗУ).
_AGG_CHUNK_SIZE = 8_000

_MONEY_COLS = (
    "PRETENSION_VALUE",
    "SURCHARGE_VALUE",
    "UTS_SURCHARGE_VALUE",
    "PRETENSION_VALUE_PENALTY",
    "SURCHARGE_VALUE_PENALTY",
)


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
    if "RECIPIENTPERSONID" in df.columns and "RECIPIENT_PERSON_ID" not in df.columns:
        df["RECIPIENT_PERSON_ID"] = df["RECIPIENTPERSONID"]
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

    if "PRETENSION_GET_DATE" in df.columns:
        df["PRETENSION_GET_DATE"] = pd.to_datetime(df["PRETENSION_GET_DATE"], errors="coerce")
    if INCIDENT_COLUMN in df.columns:
        df[INCIDENT_COLUMN] = pd.to_numeric(df[INCIDENT_COLUMN], errors="coerce")
    if "APPLICANT_PERSON_ID" in df.columns:
        df["APPLICANT_PERSON_ID"] = normalize_person_id_series(df["APPLICANT_PERSON_ID"])
    if "RECIPIENT_PERSON_ID" in df.columns:
        df["RECIPIENT_PERSON_ID"] = normalize_person_id_series(df["RECIPIENT_PERSON_ID"])
    return df


def _pretensions_for_join(
    df_pret: pd.DataFrame,
    pretension_person_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Long-формат претензий: одна строка на (person_id, претензия) для join."""
    if not pretension_person_columns:
        return pd.DataFrame()

    keep_cols = {
        INCIDENT_COLUMN,
        "PRETENSION_GET_DATE",
        "PRETENSION_NUMBER",
        "PRETENSION_TYPES",
        "PRETENSION_GET_METHOD",
        "ANSWER_TYPE",
        *_MONEY_COLS,
    }
    keep_cols.update(pretension_person_columns)

    parts: list[pd.DataFrame] = []
    for person_col in pretension_person_columns:
        if person_col not in df_pret.columns:
            continue
        cols = [c for c in keep_cols if c in df_pret.columns]
        part = df_pret.loc[df_pret[person_col].notna(), cols].copy()
        part = part.dropna(subset=[person_col, "PRETENSION_GET_DATE", INCIDENT_COLUMN])
        part["_pid"] = part[person_col]
        parts.append(part)

    if not parts:
        return pd.DataFrame()

    pret = pd.concat(parts, ignore_index=True)
    dedupe_keys = ["_pid", "PRETENSION_GET_DATE", INCIDENT_COLUMN]
    if "PRETENSION_NUMBER" in pret.columns:
        dedupe_keys = ["_pid", "PRETENSION_NUMBER", INCIDENT_COLUMN]
    pret = pret.drop_duplicates(subset=dedupe_keys)

    return pret.rename(
        columns={
            "PRETENSION_GET_DATE": "_pret_date",
            INCIDENT_COLUMN: "_pret_inc",
        }
    )


def _aggregate_grouped_pret_history(
    grouped: pd.core.groupby.DataFrameGroupBy,
    *,
    money_cols: list[str],
    index: pd.Index,
) -> pd.DataFrame:
    """Собрать агрегаты по сгруппированной истории претензий."""
    out = pd.DataFrame(index=index)
    out[f"{PERSON_PREFIX}PRET_COUNT"] = grouped.size().reindex(index).fillna(0).astype(int)

    if "PRETENSION_NUMBER" in grouped.obj.columns:
        out[f"{PERSON_PREFIX}PRET_PRETENSION_NUMBER_NUNIQUE"] = (
            grouped["PRETENSION_NUMBER"].nunique().reindex(index).fillna(0).astype(int)
        )
    if "PRETENSION_TYPES" in grouped.obj.columns:
        out[f"{PERSON_PREFIX}PRET_TYPES_NUNIQUE"] = (
            grouped["PRETENSION_TYPES"].nunique().reindex(index).fillna(0).astype(int)
        )
    if "PRETENSION_GET_METHOD" in grouped.obj.columns:
        out[f"{PERSON_PREFIX}PRET_GET_METHOD_MODE"] = (
            grouped["PRETENSION_GET_METHOD"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA)
            .reindex(index)
        )
    if "ANSWER_TYPE" in grouped.obj.columns:
        out[f"{PERSON_PREFIX}PRET_ANSWER_TYPE_MODE"] = (
            grouped["ANSWER_TYPE"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA)
            .reindex(index)
        )

    for col in money_cols:
        out[f"{PERSON_PREFIX}PRET_{col}_SUM"] = grouped[col].sum(min_count=1).reindex(index)

    return out


def _aggregate_pret_history(
    df_pret: pd.DataFrame,
    *,
    person_id: pd.Series,
    t0: pd.Series,
    current_incident: pd.Series,
    pretension_person_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Return a frame indexed by original df index with aggregated features."""
    base = pd.DataFrame(
        {
            "_row": person_id.index,
            "_pid": person_id.values,
            "_t0": pd.to_datetime(t0, errors="coerce").values,
            "_inc": pd.to_numeric(current_incident, errors="coerce").values,
        }
    ).dropna(subset=["_pid"])

    pret = _pretensions_for_join(df_pret, pretension_person_columns)
    if pret.empty:
        out = pd.DataFrame(index=person_id.index)
        out[f"{PERSON_PREFIX}PRET_COUNT"] = 0
        return out

    needed_pids = frozenset(base["_pid"].astype(str))
    pret = pret[pret["_pid"].astype(str).isin(needed_pids)]

    money_cols = [c for c in _MONEY_COLS if c in pret.columns]
    chunk_parts: list[pd.DataFrame] = []

    for start in range(0, len(base), _AGG_CHUNK_SIZE):
        base_chunk = base.iloc[start : start + _AGG_CHUNK_SIZE]
        chunk_pids = base_chunk["_pid"].unique()
        pret_chunk = pret[pret["_pid"].isin(chunk_pids)]
        if pret_chunk.empty:
            continue

        merged = base_chunk.merge(pret_chunk, on="_pid", how="left")
        mask = (merged["_pret_date"] < merged["_t0"]) & (merged["_pret_inc"] != merged["_inc"])
        merged = merged.loc[mask]
        if merged.empty:
            continue

        for col in money_cols:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

        grouped = merged.groupby("_row", dropna=False)
        chunk_index = pd.Index(base_chunk["_row"].unique())
        chunk_parts.append(
            _aggregate_grouped_pret_history(grouped, money_cols=money_cols, index=chunk_index)
        )
        del merged, pret_chunk, grouped
        gc.collect()

    out = pd.DataFrame(index=person_id.index)
    out[f"{PERSON_PREFIX}PRET_COUNT"] = 0
    if chunk_parts:
        combined = pd.concat(chunk_parts)
        out = combined.reindex(person_id.index)
        out[f"{PERSON_PREFIX}PRET_COUNT"] = out[f"{PERSON_PREFIX}PRET_COUNT"].fillna(0).astype(int)
        for col in out.columns:
            if col.endswith("_NUNIQUE"):
                out[col] = out[col].fillna(0).astype(int)

    return out


def _filter_pretensions_by_person_ids(
    pret: pd.DataFrame,
    person_ids: frozenset[str],
) -> pd.DataFrame:
    """Оставить претензии только по person_id из victim-датасета."""
    if not person_ids:
        return pret.iloc[0:0].copy()
    mask = pd.Series(False, index=pret.index)
    for column in (PRETENSION_APPLICANT_COL, PRETENSION_RECIPIENT_COL):
        if column in pret.columns:
            mask = mask | pret[column].astype(str).isin(person_ids)
    return pret.loc[mask]


def add_person_pretension_history(df: pd.DataFrame, df_pretensions: pd.DataFrame) -> pd.DataFrame:
    """Добавить FE_PERSON_PRET_{ROLE}_* для всех ролей (история по Applicant/Recipient)."""
    out = df
    pret = _prep_pretensions(df_pretensions)
    person_ids = collect_person_ids(out)
    pret = _filter_pretensions_by_person_ids(pret, person_ids)
    del df_pretensions
    logger.info(
        "Person pretension history: pret_rows=%s, person_ids=%s",
        len(pret),
        len(person_ids),
    )

    t0 = out.get(T0_COLUMN)
    current_incident = out.get(INCIDENT_COLUMN)

    for role in ROLES:
        if role.person_id_column not in out.columns:
            continue
        pid = normalize_person_id_series(out[role.person_id_column])
        logger.info("Person pretension history: role=%s", role.suffix)
        agg = _aggregate_pret_history(
            pret,
            person_id=pid,
            t0=t0,
            current_incident=current_incident,
            pretension_person_columns=role.pretension_person_columns,
        )
        agg = agg.add_prefix(f"{PERSON_PREFIX}PRET_{role.suffix}_")
        out = out.join(agg)
        del agg
        gc.collect()

    return out
