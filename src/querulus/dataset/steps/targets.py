"""Шаг пайплайна: targets."""
from __future__ import annotations

import numpy as np
import pandas as pd

from querulus.dataset.constants import RENAME_DICT
from querulus.dataset.filters import claims_sql_predicate, ensure_victim_object_type_column, select_primary_loss_per_incident
from querulus.dataset.io import load_sql_artifact
from querulus.dataset.paths import DataPaths

_TARGET_FREQ_CLAIMS_GROUP = ("LOSS_NUMBER", "INCOMING_CLAIM_NUMBER")
_FU_CLAIM_ORIGIN = "Обращение к ФУ"
_CLAIM_PERIOD_COL = "CLAIMEDVALUEPERIOD"
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


def _is_fu_instance(inst: pd.Series, claim_origin: pd.Series | None) -> pd.Series:
    """ФУ: InstByOisuu=6 или InstByOisuu=1 при ClaimOrigin='Обращение к ФУ'."""
    inst_num = pd.to_numeric(inst, errors="coerce")
    is_fu = inst_num == 6
    if claim_origin is not None:
        origin = claim_origin.fillna("").astype(str).str.strip()
        is_fu = is_fu | ((inst_num == 1) & origin.eq(_FU_CLAIM_ORIGIN))
    return is_fu.fillna(False)


def _pick_last_claim_instances(claims: pd.DataFrame) -> pd.DataFrame:
    """Последняя инстанция каждого иска на убытке.

    Порядок инстанций: IncomingClaimNumber, ClaimedValuePeriod (хронология).
    ФУ — первоначальная стадия: если есть судебные инстанции (1..5), ФУ не берём.
    """
    if _CLAIM_PERIOD_COL not in claims.columns:
        raise KeyError(f"Для выбора инстанции не хватает колонки: {_CLAIM_PERIOD_COL}")

    work = claims.copy()
    origin = work["CLAIMORIGIN"] if "CLAIMORIGIN" in work.columns else None
    inst = pd.to_numeric(work["INSTBYOISUU"], errors="coerce")
    is_fu = _is_fu_instance(inst, origin)
    is_court = (~is_fu) & inst.between(1, 5, inclusive="both")

    group_cols = list(_TARGET_FREQ_CLAIMS_GROUP)
    sort_cols = [*group_cols, _CLAIM_PERIOD_COL]

    court = work[is_court]
    from_court = (
        court.sort_values(sort_cols, ascending=[True, True, True], na_position="first")
        .drop_duplicates(group_cols, keep="last")
    )

    court_keys = court[group_cols].drop_duplicates()
    fu = work[is_fu].merge(court_keys, on=group_cols, how="left", indicator=True)
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
    """Сумма amount_cols по последней инстанции каждого иска, агрегат на инцидент."""
    required = {
        "INCIDENT_NUMBER",
        "LOSS_NUMBER",
        "INCOMING_CLAIM_NUMBER",
        "INSTBYOISUU",
        _CLAIM_PERIOD_COL,
        *amount_cols,
    }
    missing = required - set(claims.columns)
    if missing:
        raise KeyError(f"Для {output_col} не хватает колонок: {sorted(missing)}")

    optional_cols = ["CLAIMORIGIN"] if "CLAIMORIGIN" in claims.columns else []
    work = claims[list(required | set(optional_cols))].copy()
    work = work[work["INCIDENT_NUMBER"].notna() & work["LOSS_NUMBER"].notna()]
    for col in amount_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)

    last_per_claim = _pick_last_claim_instances(work)
    if len(amount_cols) == 1:
        last_per_claim[output_col] = last_per_claim[amount_cols[0]]
    else:
        last_per_claim[output_col] = last_per_claim[list(amount_cols)].sum(axis=1)

    per_loss = (
        last_per_claim.groupby(["INCIDENT_NUMBER", "LOSS_NUMBER"], as_index=False)[output_col]
        .sum()
    )
    return per_loss.groupby("INCIDENT_NUMBER", as_index=False)[output_col].sum()


def _build_target_freq_by_incident(
    claims: pd.DataFrame,
    pretensions: pd.DataFrame,
) -> pd.DataFrame:
    """Собрать TARGET_FREQ на уровне инцидента из исков (без ПСР).

    На каждый убыток: сумма RecoveredValueWithSD по последней судебной инстанции каждого иска
    (ФУ — первоначальная стадия и не считается «последней», если есть суды).
    На инцидент: сумма по всем убыткам + доплаты по претензиям (все типы ОСАГО, *_all).
    """
    claims_amount = _sum_last_claim_instances_by_incident(
        claims,
        amount_cols=("RECOVEREDVALUEWITHSD",),
        output_col="TARGET_FREQ_CLAIMS_AMOUNT",
    )

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
    """Сумма взысканий (ОД + износ + УТС) по последней инстанции каждого иска на инциденте."""
    return _sum_last_claim_instances_by_incident(
        claims,
        amount_cols=_TARGET_SEV_CLAIM_AMOUNT_COLS,
        output_col="TARGET_SEV_CLAIMS_AMOUNT",
    )


def _last_nonzero_target_3_sev(row: pd.Series) -> float:
    """Последнее ненулевое среди RECOVEREDMAINDEBT/WEAROUT/LOSSCOMMODYVALUE_{1..5} (Litigant)."""
    for col in reversed(_TARGET_3_SEV_SEVERITY_COLS):
        val = row[col]
        if pd.notna(val) and val != 0:
            return float(val)
    return np.nan


def _build_target_3_sev_by_incident(claims: pd.DataFrame) -> pd.DataFrame:
    """TARGET_3_SEV: pivot по инстанциям иска + последний ненулевой RECOVERED* (без претензий)."""
    required = {
        "INCIDENT_NUMBER",
        "INCOMING_CLAIM_NUMBER",
        _CLAIM_PERIOD_COL,
        *_TARGET_SEV_CLAIM_AMOUNT_COLS,
    }
    missing = required - set(claims.columns)
    if missing:
        raise KeyError(f"Для TARGET_3_SEV не хватает колонок: {sorted(missing)}")

    work = claims[list(required)].copy()
    work = work[work["INCIDENT_NUMBER"].notna() & work["INCOMING_CLAIM_NUMBER"].notna()]
    for col in _TARGET_SEV_CLAIM_AMOUNT_COLS:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)

    work = work.sort_values(
        ["INCOMING_CLAIM_NUMBER", _CLAIM_PERIOD_COL],
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
    return incident_pivot[["INCIDENT_NUMBER", "TARGET_3_SEV"]]


def build_targets(
    paths: DataPaths,
    conn,
    df: pd.DataFrame,
    *,
    save_checkpoint: bool = True,
    use_sql: bool = False,
):
    """Добавить TARGET (ПСР), TARGET_FREQ (иски) и TARGET_SEV к victim-фрейму."""
    # Первичный убыток на инцидент: max LOSS_NUMBER
    df = select_primary_loss_per_incident(df)

    query_calc_agg = \
    """
    with tmp as (
    	SELECT
    		_Period         Период	,
    		itl.IncidentNumber AS INCIDENT_NUMBER,
    		l.LossNumber 	AS LOSS_NUMBER	,
    		_Fld14748	СуммаРемонтаБезУчётаИзноса	,
    		_Fld14749	СтоимостьЗапчастей	,
    		_Fld14750	Работы	,
    		_Fld14751	Материалы	,
    		_Fld14752	ДеталиРазовогоМонтажа	,
    		_Fld14753	ПроцентИзноса	,
    		_Fld14754	СтоимостьСУчетомИзноса	,
    		_Fld16561	Пробег	,
    		_Fld16562	ДатаНачалаЭксплуатации	,
    		_Fld14787	НомерРасчета	,
    		cast(_Fld14788 as INT)	СканыКалькуляцииОбработаны	,
    		_Fld15038	ДатаРасчета,
    		-- Внутри одного убытка могут быть несколько расчётов AMOUNT_REPAIR.
    		-- Нам нужен "последний" расчёт по Период.
    		ROW_NUMBER() over (
                partition by itl.IncidentNumber
                order by l.LossNumber desc, _Period desc
            ) as rn
    	from oisuu81.dbo._InfoRg14746 i
    	left join oisuu81_t_losses l on l.LossID = _Fld14747RRef
    	LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_IncidentToLoss] AS itl on l.LossID=itl.LossID
    	where year(l.IssueDate) is not null
    	and l.LossProcess in ('Прямое ОСАГО (с 1 марта 2009)','Традиционное ОСАГО')
    	and Risk = 'Ущерб имуществу третьих лиц'
    )
    SELECT *
    FROM tmp
    WHERE rn = 1
    """
    df_calc = load_sql_artifact(
        paths,
        conn,
        paths.raw_dir,
        "df_calc_agg.parquet",
        query_calc_agg,
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
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

    target_psr_sql = \
    """
    SELECT 
    		 psr.[Номер_инциндента]
            ,sum(l.EMRValue) as EMRValue
    		,sum(psr.[Выплата_по_основному_убытку]) as Выплата_по_основному_убытку
    		,sum(psr.[Сумма_выплат_по_претензиям]) as Сумма_выплат_по_претензиям
    		,sum(psr.[Общая_сумма_заявленных_требований_ФУ]) as Общая_сумма_заявленных_требований_ФУ
    		,sum(psr.[Сумма_взыскано_по_ФУ]) as Сумма_взыскано_по_ФУ
    		,sum(psr.[Общая_сумма_заявленных_требований_ИСК]) as Общая_сумма_заявленных_требований_ИСК
    		,sum(psr.[Суммы_взыскано_по_иску]) as Суммы_взыскано_по_иску
    		,sum(psr.[Сумма_утс]) as Сумма_утс
    		,sum(psr.[Сумма_износа_по_калькуляции_инцидент]) as Сумма_износа_по_калькуляции_инцидент
    		,sum(psr.[Выплаченный_износ_инцидент]) as Выплаченный_износ_инцидент
    		,sum(psr.[Доп_расходы_инцидент]) as Доп_расходы_инцидент
    		,sum(psr.[Взысканный_износ_ФУ]) as Взысканный_износ_ФУ
    		,sum(psr.[Взысканный_износ_ИСК]) as Взысканный_износ_ИСК
      FROM [OISUU_report].[Datamart].[oisuu81_t_ПСР] as psr
      LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_Losses] as l on l.LossNumber=psr.Номер_убытка
      group by [Номер_инциндента]
    """
    target_psr = load_sql_artifact(
        paths,
        conn,
        paths.raw_dir,
        "target_2.parquet",
        target_psr_sql,
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
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

    target_3_pretensions_sql = \
    """
    SELECT 
           itl.IncidentNumber
          ,sum(p.[SurchargeValue]) as SurchargeValue_cumsum_by_incident
          ,sum(p.[UTSSurchargeValue]) as UTSSurchargeValue_cumsum_by_incident
      FROM [OISUU_report].[dbo].[oisuu81_t_Pretensions] AS p
      LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_IncidentToLoss] AS itl on itl.LossID=p.LossID
      WHERE InsuranceTypeGroups = 'ОСАГО'
      and PretensionType in ('Несогласие с суммой выплаты','Претензия на принятое решение')
      group by IncidentNumber
    """
    target_3_pretensions = load_sql_artifact(
        paths,
        conn,
        paths.raw_dir,
        "target_3_pretensions.parquet",
        target_3_pretensions_sql,
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
    )
    target_3_pretensions = target_3_pretensions.rename(columns=RENAME_DICT)

    target_3_pretensions_all_sql = \
    """
    SELECT 
           itl.IncidentNumber
          ,sum(p.[SurchargeValue]) as SurchargeValue_cumsum_by_incident_all
          ,sum(p.[UTSSurchargeValue]) as UTSSurchargeValue_cumsum_by_incident_all
      FROM [OISUU_report].[dbo].[oisuu81_t_Pretensions] AS p
      LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_IncidentToLoss] AS itl on itl.LossID=p.LossID
      WHERE InsuranceTypeGroups = 'ОСАГО'
      group by IncidentNumber
    """
    target_3_pretensions_all = load_sql_artifact(
        paths,
        conn,
        paths.raw_dir,
        "target_3_pretensions_all.parquet",
        target_3_pretensions_all_sql,
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
    )
    target_3_pretensions_all = target_3_pretensions_all.rename(columns=RENAME_DICT)

    claims_where = claims_sql_predicate(icnl_alias="icnl", loss_alias="l")
    target_3_claims_sql = f"""
    SELECT itl.[LossID]
          ,itl.[LossNumber]
          ,itl.[IncidentID]
          ,itl.[IncidentNumber]
          ,icnl.[IncomingClaimID]
          ,icnl.[InstByOisuu]
          ,icnl.[IncomingClaimNumber]
          ,icnl.[ClaimedValuePeriod]
          ,icnl.[ClaimedMainDebt]
          ,icnl.[ClaimedPlaintiffExamination]
          ,icnl.[ClaimedCourtExamination]
          ,icnl.[ClaimedRepresentativeExpenses]
          ,icnl.[ClaimedPercentForUses]
          ,icnl.[ClaimedPenaltyFee]
          ,icnl.[ClaimedFine]
          ,icnl.[ClaimedMoralDamage]
          ,icnl.[ClaimedOtherExpenses]
          ,icnl.[ClaimedStateDuty]
          ,icnl.[ClaimedLossCommodyValue]
          ,icnl.[ClaimedWearout]
          ,icnl.[ClaimedValueWithoutSD]
          ,icnl.[ClaimedValueWithSD]
          ,icnl.[ClaimedAmountLoss]
          ,icnl.[ClaimedCourtExaminationBeforeResolve]
          ,icnl.[RecoveredValuePeriod]
          ,icnl.[RecoveredMainDebt]
          ,icnl.[RecoveredPlaintiffExamination]
          ,icnl.[RecoveredCourtExamination]
          ,icnl.[RecoveredRepresentativeExpenses]
          ,icnl.[RecoveredPercentForUses]
          ,icnl.[RecoveredPenaltyFee]
          ,icnl.[RecoveredFine]
          ,icnl.[RecoveredMoralDamage]
          ,icnl.[RecoveredOtherExpenses]
          ,icnl.[RecoveredStateDuty]
          ,icnl.[RecoveredLossCommodyValue]
          ,icnl.[RecoveredWearout]
          ,icnl.[RecoveredValueWithoutSD]
          ,icnl.[RecoveredValueWithSD]
          ,icnl.[RecoveredAmountLoss]
          ,icnl.[RecoveredCourtExaminationBeforeResolve]
          ,icnl.[Instance]
          ,icnl.[instID]
          ,icnl.[Court]
          ,icnl.[Judge]
          ,icnl.[Applicant]
          ,icnl.[ExpertOrg]
          ,icnl.[Decision]
          ,icnl.[IncomingClaimGetDate]
          ,icnl.[CourtWorkOverDate]
          ,icnl.[ClaimItem]
          ,icnl.[ClaimOrigin]
          ,icnl.[CourtWorkUnit]
          ,icnl.[EmployeeName]
          ,icnl.[FLSimpleOrder]
          ,icnl.[LinkLossNumber]
    FROM [OISUU_report].[Datamart].[oisuu81_t_IncomingClaimNewLogicByInst] as icnl
    LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_IncidentToLoss] as itl on itl.LossNumber=icnl.LinkLossNumber
    LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_Losses] as l on l.LossNumber=itl.LossNumber
    WHERE {claims_where}
    """
    target_3_claims = load_sql_artifact(
        paths,
        conn,
        paths.raw_dir,
        "target_3_claims.parquet",
        target_3_claims_sql,
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
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

    if 'VICTIM_VEHICLE_IS_JAPAN' in df.columns:
        df['VICTIM_VEHICLE_IS_JAPAN'] = df['VICTIM_VEHICLE_IS_JAPAN'].astype(str)

    df = df[df['VICTIM_POLICYHOLDER_TYPE'] == 'Физ. Лицо'].reset_index(drop=True)

    df['TARGET'] = (
        df['Сумма_выплат_по_претензиям'].fillna(0)
        + df['Сумма_взыскано_по_ФУ'].fillna(0)
        + df['Суммы_взыскано_по_иску'].fillna(0)
    )
    df['TARGET'] = df['TARGET'].apply(lambda x: 1 if x > 0 else 0).astype(int)
    df["TARGET_FREQ"] = df["TARGET_FREQ"].fillna(0).astype(int)
    for col in (
        "TARGET_FREQ_AMOUNT",
        "TARGET_FREQ_CLAIMS_AMOUNT",
        "TARGET_FREQ_PRET_AMOUNT",
        "TARGET_SEV_CLAIMS_AMOUNT",
        "TARGET_3_SEV",
    ):
        df[col] = df[col].fillna(0)

    df = ensure_victim_object_type_column(df)

    return df