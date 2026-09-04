"""Шаг пайплайна: targets."""
from __future__ import annotations

import numpy as np
import pandas as pd

from querulus.dataset.constants import RENAME_DICT
from querulus.dataset.load import targets as load_targets
from querulus.dataset.paths import DataPaths
from querulus.dataset.preprocess.filters import (
    ensure_victim_object_type_column,
    select_primary_loss_per_incident,
)
from querulus.dataset.preprocess.maturity import MATURITY_REPORT_NAME, apply_target_maturity

TARGET_FREQ_CLAIMS_GROUP = ("LOSS_NUMBER", "INCOMING_CLAIM_NUMBER")
_FU_CLAIM_ORIGIN = "Обращение к ФУ"
CLAIM_PERIOD_COL = "CLAIMEDVALUEPERIOD"
_VOID_DECISION = "не принято"
# Сигналы взыскания для отсева «пустой» / отменённой инстанции (Null во всех).
_RECOVERY_SIGNAL_COLS = (
    "RECOVEREDVALUEWITHSD",
    "RECOVEREDMAINDEBT",
    "RECOVEREDWEAROUT",
    "RECOVEREDLOSSCOMMODYVALUE",
)
_SURCHARGE_INCIDENT_COL = "SurchargeValue_cumsum_by_incident_all"
_UTS_SURCHARGE_INCIDENT_COL = "UTSSurchargeValue_cumsum_by_incident_all"
_TARGET_SEV_CLAIM_AMOUNT_COLS = (
    "RECOVEREDMAINDEBT",
    "RECOVEREDWEAROUT",
    "RECOVEREDLOSSCOMMODYVALUE",
)
_TARGET_3_SEV_SEVERITY_COLS: tuple[str, ...] = tuple(
    f"{col}_{instance}"
    for instance in range(1, 6)
    for col in _TARGET_SEV_CLAIM_AMOUNT_COLS
)
_TARGET_FREQ_PRET_COMPONENT_COLS = (
    _SURCHARGE_INCIDENT_COL,
    _UTS_SURCHARGE_INCIDENT_COL,
)
TARGET_FREQ_CLAIMS_COMPONENT_COLS = ("RECOVEREDVALUEWITHSD_LAST_INST_SUM",)
TARGET_FREQ_COMPONENT_COLS = TARGET_FREQ_CLAIMS_COMPONENT_COLS + _TARGET_FREQ_PRET_COMPONENT_COLS
TARGET_SEV_CLAIMS_COMPONENT_COLS = tuple(
    f"{col}_LAST_INST_SUM" for col in _TARGET_SEV_CLAIM_AMOUNT_COLS
)
TARGET_SEV_COMPONENT_COLS = TARGET_SEV_CLAIMS_COMPONENT_COLS + _TARGET_FREQ_PRET_COMPONENT_COLS
TARGET_3_SEV_COMPONENT_COLS = _TARGET_3_SEV_SEVERITY_COLS


def _is_fu_instance(inst: pd.Series, claim_origin: pd.Series | None) -> pd.Series:
    """ФУ: InstByOisuu=6 или InstByOisuu=1 при ClaimOrigin='Обращение к ФУ'."""
    inst_num = pd.to_numeric(inst, errors="coerce")
    is_fu = inst_num == 6
    if claim_origin is not None:
        origin = claim_origin.fillna("").astype(str).str.strip()
        is_fu = is_fu | ((inst_num == 1) & origin.eq(_FU_CLAIM_ORIGIN))
    return is_fu.fillna(False)


def is_void_claim_instance(df: pd.DataFrame) -> pd.Series:
    """Инстанция без принятого взыскания: Decision='Не принято' или все суммы Null.

    Не опирается на номер инстанции — только Decision и наличие сумм.
    """
    void = pd.Series(False, index=df.index)
    if "DECISION" in df.columns:
        decision = df["DECISION"].fillna("").astype(str).str.strip().str.casefold()
        void = void | decision.eq(_VOID_DECISION)
    signal_cols = [col for col in _RECOVERY_SIGNAL_COLS if col in df.columns]
    if signal_cols:
        amounts = df[signal_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
        void = void | amounts.isna().all(axis=1)
    return void


def _optional_claim_meta_cols(claims: pd.DataFrame) -> list[str]:
    """Опциональные колонки для выбора инстанции (origin, decision, сигналы сумм)."""
    cols: list[str] = []
    for col in ("CLAIMORIGIN", "DECISION", *_RECOVERY_SIGNAL_COLS):
        if col in claims.columns and col not in cols:
            cols.append(col)
    return cols


def pick_last_claim_instances(claims: pd.DataFrame) -> pd.DataFrame:
    """Последняя принятая инстанция каждого иска на убытке.

    Порядок: IncomingClaimNumber, ClaimedValuePeriod (хронология).
    Пропуск строк Decision='Не принято' и строк со всеми Null-взысканиями —
    берётся предыдущая принятая по хронологии (без привязки к InstByOisuu).
    ФУ — первоначальная стадия: если есть принятые судебные инстанции, ФУ не берём.
    """
    if CLAIM_PERIOD_COL not in claims.columns:
        raise KeyError(f"Для выбора инстанции не хватает колонки: {CLAIM_PERIOD_COL}")

    work = claims.copy()
    origin = work["CLAIMORIGIN"] if "CLAIMORIGIN" in work.columns else None
    inst = pd.to_numeric(work["INSTBYOISUU"], errors="coerce")
    is_fu = _is_fu_instance(inst, origin)
    is_court = (~is_fu) & inst.between(1, 5, inclusive="both")
    is_void = is_void_claim_instance(work)

    group_cols = list(TARGET_FREQ_CLAIMS_GROUP)
    sort_cols = [*group_cols, CLAIM_PERIOD_COL]

    # Только принятые судебные инстанции (хронология, не номер).
    court = work[is_court & ~is_void]
    from_court = (
        court.sort_values(sort_cols, ascending=[True, True, True], na_position="first")
        .drop_duplicates(group_cols, keep="last")
    )

    court_keys = court[group_cols].drop_duplicates()
    fu = work[is_fu & ~is_void].merge(court_keys, on=group_cols, how="left", indicator=True)
    fu_only = fu[fu["_merge"] == "left_only"].drop(columns="_merge")
    from_fu = (
        fu_only.sort_values(sort_cols, ascending=[True, True, True], na_position="first")
        .drop_duplicates(group_cols, keep="last")
    )

    return pd.concat([from_court, from_fu], ignore_index=True)


def _sum_last_claim_instances_by_incident(
    claims: pd.DataFrame,
    *,
    amount_cols: tuple[str, ...],
    output_col: str,
) -> pd.DataFrame:
    """Сумма amount_cols по последней принятой инстанции каждого иска, агрегат на инцидент."""
    required = {
        "INCIDENT_NUMBER",
        "LOSS_NUMBER",
        "INCOMING_CLAIM_NUMBER",
        "INSTBYOISUU",
        CLAIM_PERIOD_COL,
        *amount_cols,
    }
    missing = required - set(claims.columns)
    if missing:
        raise KeyError(f"Для {output_col} не хватает колонок: {sorted(missing)}")

    optional_cols = _optional_claim_meta_cols(claims)
    work = claims[list(required | set(optional_cols))].copy()
    work = work[work["INCIDENT_NUMBER"].notna() & work["LOSS_NUMBER"].notna()]
    # Не fillna до выбора инстанции — иначе Null неотличимы от нуля.
    for col in amount_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    for col in _RECOVERY_SIGNAL_COLS:
        if col in work.columns and col not in amount_cols:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    last_per_claim = pick_last_claim_instances(work)
    for col in amount_cols:
        last_per_claim[col] = last_per_claim[col].fillna(0)
    if len(amount_cols) == 1:
        last_per_claim[output_col] = last_per_claim[amount_cols[0]]
    else:
        last_per_claim[output_col] = last_per_claim[list(amount_cols)].sum(axis=1)

    per_loss = (
        last_per_claim.groupby(["INCIDENT_NUMBER", "LOSS_NUMBER"], as_index=False)[output_col]
        .sum()
    )
    return per_loss.groupby("INCIDENT_NUMBER", as_index=False)[output_col].sum()


def _sum_last_claim_components_by_incident(
    claims: pd.DataFrame,
    amount_cols: tuple[str, ...],
    *,
    suffix: str = "",
) -> pd.DataFrame:
    """Сумма amount_cols по последней принятой инстанции каждого иска, отдельно по каждой колонке."""
    required = {
        "INCIDENT_NUMBER",
        "LOSS_NUMBER",
        "INCOMING_CLAIM_NUMBER",
        "INSTBYOISUU",
        CLAIM_PERIOD_COL,
        *amount_cols,
    }
    missing = required - set(claims.columns)
    if missing:
        raise KeyError(f"Для компонентов иска не хватает колонок: {sorted(missing)}")

    optional_cols = _optional_claim_meta_cols(claims)
    work = claims[list(required | set(optional_cols))].copy()
    work = work[work["INCIDENT_NUMBER"].notna() & work["LOSS_NUMBER"].notna()]
    for col in amount_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    for col in _RECOVERY_SIGNAL_COLS:
        if col in work.columns and col not in amount_cols:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    last_per_claim = pick_last_claim_instances(work)
    for col in amount_cols:
        last_per_claim[col] = last_per_claim[col].fillna(0)
    output_cols = [f"{col}{suffix}" for col in amount_cols]
    per_loss = (
        last_per_claim.groupby(["INCIDENT_NUMBER", "LOSS_NUMBER"], as_index=False)[list(amount_cols)]
        .sum()
        .rename(columns=dict(zip(amount_cols, output_cols, strict=True)))
    )
    return per_loss.groupby("INCIDENT_NUMBER", as_index=False)[output_cols].sum()


def _build_target_freq_by_incident(
    claims: pd.DataFrame,
    pretensions: pd.DataFrame,
) -> pd.DataFrame:
    """Собрать TARGET_FREQ на уровне инцидента из исков (без ПСР).

    На каждый убыток: сумма RecoveredValueWithSD по последней принятой судебной инстанции
    каждого иска (пропуск Decision='Не принято' и Null-взысканий; ФУ не берём, если есть суды).
    На инцидент: сумма по всем убыткам + доплаты по претензиям (все типы ОСАГО, *_all).
    """
    claims_amount = _sum_last_claim_instances_by_incident(
        claims,
        amount_cols=("RECOVEREDVALUEWITHSD",),
        output_col="RECOVEREDVALUEWITHSD_LAST_INST_SUM",
    )
    claims_amount["TARGET_FREQ_CLAIMS_AMOUNT"] = claims_amount["RECOVEREDVALUEWITHSD_LAST_INST_SUM"]

    pret_cols = [
        "INCIDENT_NUMBER",
        _SURCHARGE_INCIDENT_COL,
        _UTS_SURCHARGE_INCIDENT_COL,
    ]
    pret = pretensions[pret_cols].copy()
    pret["TARGET_FREQ_PRET_AMOUNT"] = (
        pret[_SURCHARGE_INCIDENT_COL].fillna(0)
        + pret[_UTS_SURCHARGE_INCIDENT_COL].fillna(0)
    )

    out = claims_amount.merge(
        pret[["INCIDENT_NUMBER", "TARGET_FREQ_PRET_AMOUNT"]],
        on="INCIDENT_NUMBER",
        how="outer",
    )
    out["TARGET_FREQ_CLAIMS_AMOUNT"] = out["TARGET_FREQ_CLAIMS_AMOUNT"].fillna(0)
    out["TARGET_FREQ_PRET_AMOUNT"] = out["TARGET_FREQ_PRET_AMOUNT"].fillna(0)
    out["TARGET_FREQ_AMOUNT"] = out["TARGET_FREQ_CLAIMS_AMOUNT"] + out["TARGET_FREQ_PRET_AMOUNT"]
    out["TARGET_FREQ"] = (out["TARGET_FREQ_AMOUNT"] > 0).astype(int)
    return out


def _build_target_sev_claims_by_incident(claims: pd.DataFrame) -> pd.DataFrame:
    """Сумма взысканий (ОД + износ + УТС) по последней принятой инстанции каждого иска."""
    out = _sum_last_claim_components_by_incident(
        claims,
        _TARGET_SEV_CLAIM_AMOUNT_COLS,
        suffix="_LAST_INST_SUM",
    )
    out["TARGET_SEV_CLAIMS_AMOUNT"] = out[list(TARGET_SEV_CLAIMS_COMPONENT_COLS)].sum(axis=1)
    return out


def _last_nonzero_target_3_sev(row: pd.Series) -> float:
    """Последнее ненулевое среди RECOVEREDMAINDEBT/WEAROUT/LOSSCOMMODYVALUE_{1..5} (Litigant)."""
    for col in reversed(_TARGET_3_SEV_SEVERITY_COLS):
        val = row[col]
        if pd.notna(val) and val != 0:
            return float(val)
    return np.nan


def _build_target_3_sev_by_incident(claims: pd.DataFrame) -> pd.DataFrame:
    """TARGET_3_SEV: pivot по принятым инстанциям + последний ненулевой RECOVERED* (без претензий)."""
    required = {
        "INCIDENT_NUMBER",
        "INCOMING_CLAIM_NUMBER",
        CLAIM_PERIOD_COL,
        *_TARGET_SEV_CLAIM_AMOUNT_COLS,
    }
    missing = required - set(claims.columns)
    if missing:
        raise KeyError(f"Для TARGET_3_SEV не хватает колонок: {sorted(missing)}")

    optional_cols = _optional_claim_meta_cols(claims)
    work = claims[list(required | set(optional_cols))].copy()
    work = work[work["INCIDENT_NUMBER"].notna() & work["INCOMING_CLAIM_NUMBER"].notna()]
    for col in _TARGET_SEV_CLAIM_AMOUNT_COLS:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    for col in _RECOVERY_SIGNAL_COLS:
        if col in work.columns and col not in _TARGET_SEV_CLAIM_AMOUNT_COLS:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    # Отбрасываем «Не принято» / все-Null до pivot (порядок — по дате, не по InstByOisuu).
    work = work.loc[~is_void_claim_instance(work)].copy()
    for col in _TARGET_SEV_CLAIM_AMOUNT_COLS:
        work[col] = work[col].fillna(0)

    work = work.sort_values(
        ["INCOMING_CLAIM_NUMBER", CLAIM_PERIOD_COL],
        ascending=[True, True],
        na_position="first",
    )
    work["Instance"] = work.groupby("INCOMING_CLAIM_NUMBER").cumcount() + 1

    pivot_index = ["INCOMING_CLAIM_NUMBER", "INCIDENT_NUMBER"]
    wide = work.pivot_table(
        index=pivot_index,
        columns="Instance",
        values=list(_TARGET_SEV_CLAIM_AMOUNT_COLS),
        aggfunc="sum",
        fill_value=0,
    )
    wide.columns = [f"{col}_{int(instance)}" for col, instance in wide.columns]
    wide = wide.reset_index()

    pivot_cols = [col for col in wide.columns if col not in pivot_index]
    incident_pivot = wide.groupby("INCIDENT_NUMBER", as_index=False)[pivot_cols].sum()
    incident_pivot["TARGET_3_SEV"] = incident_pivot.apply(_last_nonzero_target_3_sev, axis=1)
    return incident_pivot


def ensure_claims_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить TARGET_FREQ_CLAIMS / TARGET_SEV_CLAIMS, если есть только *_AMOUNT."""
    out = df
    if "TARGET_FREQ_CLAIMS" not in out.columns and "TARGET_FREQ_CLAIMS_AMOUNT" in out.columns:
        out = out.copy()
        out["TARGET_FREQ_CLAIMS"] = (
            pd.to_numeric(out["TARGET_FREQ_CLAIMS_AMOUNT"], errors="coerce").fillna(0) > 0
        ).astype(int)
    if "TARGET_SEV_CLAIMS" not in out.columns and "TARGET_SEV_CLAIMS_AMOUNT" in out.columns:
        if out is df:
            out = out.copy()
        out["TARGET_SEV_CLAIMS"] = pd.to_numeric(
            out["TARGET_SEV_CLAIMS_AMOUNT"], errors="coerce"
        ).fillna(0)
    return out


def build_targets(
    paths: DataPaths,
    conn,
    df: pd.DataFrame,
    *,
    save_checkpoint: bool = True,
    use_sql: bool = False,
):
    """Добавить TARGET_2 (ПСР), TARGET_3_SEV, TARGET_FREQ и TARGET_SEV к victim-фрейму."""
    # Первичный убыток на инцидент: min LOSS_NUMBER
    df = select_primary_loss_per_incident(df)

    df_calc = load_targets.fetch_calc_agg(
        paths, conn, use_sql=use_sql, save_checkpoint=save_checkpoint
    )

    df_calc.loc[df_calc['ПроцентИзноса'] > 50, 'ПроцентИзноса'] = 50
    df_calc = df_calc.rename(columns={'Убыток': 'LOSS_NUMBER'})
    df_calc['SHARE_WORK'] = (df_calc['Работы'] / df_calc['СуммаРемонтаБезУчётаИзноса']).round(3)
    df_calc = df_calc.rename(
        columns={'СуммаРемонтаБезУчётаИзноса': 'AMOUNT_REPAIR', 'ПроцентИзноса': 'SHARE_WEAROUT'}
    )

    df = df.merge(
        df_calc[['INCIDENT_NUMBER', 'SHARE_WORK', 'AMOUNT_REPAIR', 'SHARE_WEAROUT']],
        how='left',
        on='INCIDENT_NUMBER',
    )

    df['FLAG_APPLICANT_SAME_VICTIM_PH'] = (
        df['APPLICANT_ID'] == df['VICTIM_POLICYHOLDER_PERSON_ID']
    ).astype(int)

    target_psr = load_targets.fetch_target_psr(
        paths, conn, use_sql=use_sql, save_checkpoint=save_checkpoint
    )

    # В ПСР на 1 инцидент может быть несколько убытков. Если часть сумм на уровне убытка NULL,
    # а часть заполнена, то merge к "первичному убытку" может потерять заполненные значения.
    # Поэтому схлопываем ПСР до уровня INCIDENT_NUMBER суммированием по всем убыткам инцидента.
    incident_key = "Номер_инциндента"
    if incident_key in target_psr.columns:
        value_cols = [col for col in target_psr.columns if col != incident_key]
        if value_cols:
            target_psr[value_cols] = target_psr[value_cols].apply(
                pd.to_numeric, errors="coerce"
            )
        target_psr = (
            target_psr.groupby(incident_key, as_index=False)[value_cols]
            .sum(min_count=1)
            .reset_index(drop=True)
        )

    df = df.merge(target_psr, how='left', left_on='INCIDENT_NUMBER', right_on='Номер_инциндента')

    # Доплаты претензий → инцидент: AnswerType=выплата/частичная,
    # дедуп по PretensionNumber, затем sum (см. pretension_utils).
    target_3_pretensions = load_targets.fetch_target_3_pretensions(
        paths, conn, use_sql=use_sql, save_checkpoint=save_checkpoint
    )
    target_3_pretensions = target_3_pretensions.rename(columns=RENAME_DICT)

    target_3_pretensions_all = load_targets.fetch_target_3_pretensions_all(
        paths, conn, use_sql=use_sql, save_checkpoint=save_checkpoint
    )
    target_3_pretensions_all = target_3_pretensions_all.rename(columns=RENAME_DICT)

    target_3_claims = load_targets.fetch_target_3_claims(
        paths, conn, use_sql=use_sql, save_checkpoint=save_checkpoint
    )
    target_3_claims = target_3_claims.rename(columns=RENAME_DICT)
    target_3_claims.columns = target_3_claims.columns.str.upper()
    target_3_claims = target_3_claims.rename(columns=RENAME_DICT)

    target_freq = _build_target_freq_by_incident(target_3_claims, target_3_pretensions_all)
    df = df.merge(
        target_freq[
            [
                "INCIDENT_NUMBER",
                "TARGET_FREQ",
                "TARGET_FREQ_AMOUNT",
                "RECOVEREDVALUEWITHSD_LAST_INST_SUM",
                "TARGET_FREQ_CLAIMS_AMOUNT",
                "TARGET_FREQ_PRET_AMOUNT",
            ]
        ],
        how="left",
        on="INCIDENT_NUMBER",
    )

    target_sev_claims = _build_target_sev_claims_by_incident(target_3_claims)
    target_3_sev = _build_target_3_sev_by_incident(target_3_claims)
    df = df.merge(target_sev_claims, how="left", on="INCIDENT_NUMBER")
    df = df.merge(target_3_sev, how="left", on="INCIDENT_NUMBER")
    df = df.merge(target_3_pretensions, how="left", on="INCIDENT_NUMBER")
    df = df.merge(target_3_pretensions_all, how="left", on="INCIDENT_NUMBER")

    df["TARGET_SEV"] = (
        df["TARGET_SEV_CLAIMS_AMOUNT"].fillna(0)
        + df[_SURCHARGE_INCIDENT_COL].fillna(0)
        + df[_UTS_SURCHARGE_INCIDENT_COL].fillna(0)
    )
    # Стек new_claims: только иски, без претензий.
    df["TARGET_FREQ_CLAIMS"] = (df["TARGET_FREQ_CLAIMS_AMOUNT"].fillna(0) > 0).astype(int)
    df["TARGET_SEV_CLAIMS"] = df["TARGET_SEV_CLAIMS_AMOUNT"].fillna(0)

    if 'VICTIM_VEHICLE_IS_JAPAN' in df.columns:
        df['VICTIM_VEHICLE_IS_JAPAN'] = df['VICTIM_VEHICLE_IS_JAPAN'].astype(str)

    df = df[df['VICTIM_POLICYHOLDER_TYPE'] == 'Физ. Лицо'].reset_index(drop=True)

    df["TARGET_2"] = (
        df["Сумма_выплат_по_претензиям"].fillna(0)
        + df["Сумма_взыскано_по_ФУ"].fillna(0)
        + df["Суммы_взыскано_по_иску"].fillna(0)
    )
    df["TARGET_2"] = df["TARGET_2"].apply(lambda x: 1 if x > 0 else 0).astype(int)
    df["TARGET_FREQ"] = df["TARGET_FREQ"].fillna(0).astype(int)
    df["TARGET_FREQ_CLAIMS"] = df["TARGET_FREQ_CLAIMS"].fillna(0).astype(int)
    for col in (
        "TARGET_FREQ_AMOUNT",
        "RECOVEREDVALUEWITHSD_LAST_INST_SUM",
        "TARGET_FREQ_CLAIMS_AMOUNT",
        "TARGET_FREQ_PRET_AMOUNT",
        *TARGET_SEV_CLAIMS_COMPONENT_COLS,
        "TARGET_SEV_CLAIMS_AMOUNT",
        "TARGET_SEV_CLAIMS",
        "TARGET_3_SEV",
        *TARGET_3_SEV_COMPONENT_COLS,
    ):
        if col in df.columns:
            df[col] = df[col].fillna(0)

    df = ensure_victim_object_type_column(df)

    report_path = paths.processed_dir / MATURITY_REPORT_NAME if save_checkpoint else None
    df, _ = apply_target_maturity(
        df, target_3_claims, report_path=report_path
    )
    return df