"""Утилиты дедупликации и схлопывания претензий на инцидент."""
from __future__ import annotations

import pandas as pd

# Доплаты учитываем только при выплате / частичной выплате.
PRETENSION_PAID_ANSWER_TYPES: tuple[str, ...] = ("Выплата", "Частичная выплата")


def dedupe_pretension_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Убрать дубликаты строк претензий после JOIN с IncidentToLoss.

    Порядок:
    1. PretensionID — артефакты дублирующего джойна LossID→IncidentToLoss.
    2. (INCIDENT_NUMBER, PRETENSION_NUMBER) — один номер претензии на инцидент
       (несколько убытков / повторяющиеся номера не размножают строку).
    """
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
    """Схлопнуть Surcharge/UTS на инцидент без двойного счёта одного номера претензии.

    Алгоритм:
    1. Связка pretension → incident уже должна быть в таблице (JOIN по LossID → IncidentToLoss).
    2. Оставить только AnswerType из ``answer_types`` (по умолчанию выплата / частичная).
    3. Опционально отфильтровать PretensionType.
    4. Один ряд на (INCIDENT_NUMBER, PRETENSION_NUMBER): доплаты не суммируются
       внутри одного номера претензии (max по дубликатам от нескольких убытков/джойна).
    5. Sum доплат по INCIDENT_NUMBER.
    """
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

    # Один номер претензии на инцидент — не суммировать дубли.
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


def pretension_surcharge_by_incident_sql(
    *,
    surcharge_alias: str,
    uts_alias: str,
    pretension_types: tuple[str, ...] | None = None,
    answer_types: tuple[str, ...] = PRETENSION_PAID_ANSWER_TYPES,
) -> str:
    """SQL-эквивалент ``collapse_pretension_surcharge_to_incident`` (JOIN по LossID)."""
    answer_list = ", ".join(f"'{v}'" for v in answer_types)
    type_filter = ""
    if pretension_types:
        types_list = ", ".join(f"'{v}'" for v in pretension_types)
        type_filter = f"\n      AND p.[PretensionType] IN ({types_list})"

    return f"""
    WITH pret_paid AS (
        SELECT
            itl.[IncidentNumber] AS IncidentNumber,
            p.[PretensionNumber] AS PretensionNumber,
            MAX(p.[SurchargeValue]) AS SurchargeValue,
            MAX(p.[UTSSurchargeValue]) AS UTSSurchargeValue
        FROM [OISUU_report].[dbo].[oisuu81_t_Pretensions] AS p
        LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_IncidentToLoss] AS itl
            ON p.[LossID] = itl.[LossID]
        WHERE p.[InsuranceTypeGroups] = 'ОСАГО'
          AND p.[AnswerType] IN ({answer_list}){type_filter}
        GROUP BY itl.[IncidentNumber], p.[PretensionNumber]
    )
    SELECT
        IncidentNumber,
        SUM(SurchargeValue) AS {surcharge_alias},
        SUM(UTSSurchargeValue) AS {uts_alias}
    FROM pret_paid
    GROUP BY IncidentNumber
    """
