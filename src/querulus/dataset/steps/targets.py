"""Шаг пайплайна: targets."""
from __future__ import annotations

import numpy as np
import pandas as pd

from querulus.dataset.constants import RENAME_DICT
from querulus.dataset.filters import claims_sql_predicate, select_primary_loss_per_incident
from querulus.dataset.io import checkpoint, load_sql_artifact
from querulus.dataset.paths import DataPaths


def build_targets(
    paths: DataPaths,
    conn,
    df: pd.DataFrame,
    *,
    save_checkpoint: bool = True,
    use_sql: bool = False,
):
    """Добавить TARGET (ПСР) и TARGET_SEV (сумма взыскания) к victim-фрейму."""
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
    		ROW_NUMBER() over (partition by itl.IncidentNumber order by l.LossNumber desc) as rn
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

    for col in target_psr.columns:
        target_psr[col] = target_psr[col].fillna(0)

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

    df_base = target_3_claims[
        ['INCIDENT_NUMBER', 'LOSS_ID', 'LOSS_NUMBER', 'INCOMINGCLAIMID', 'INCOMING_CLAIM_NUMBER', 'LINK_LOSS_NUMBER']
    ].drop_duplicates(['INCOMING_CLAIM_NUMBER'])

    df_sorted = target_3_claims.loc[:, ['LOSS_NUMBER'] + list(target_3_claims.iloc[:, 6:].columns)]
    df_sorted = df_sorted.drop(['INSTID', 'LINK_LOSS_NUMBER'], axis=1)
    df_sorted = df_sorted.sort_values(by=['INCOMING_CLAIM_NUMBER', 'CLAIMEDVALUEPERIOD']).reset_index(drop=True)
    df_sorted['Instance'] = df_sorted.groupby(['INCOMING_CLAIM_NUMBER']).cumcount() + 1
    df_pivot = df_sorted.pivot(index=['INCOMING_CLAIM_NUMBER'], columns='Instance')
    df_pivot.columns = [f'{col[0]}_{int(col[1])}' for col in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    target_3_claims = df_base.merge(df_pivot, how='left', on=['INCOMING_CLAIM_NUMBER'])
    target_3_claims = target_3_claims[
        ['INCIDENT_NUMBER',
         'RECOVEREDMAINDEBT_1', 'RECOVEREDWEAROUT_1', 'RECOVEREDLOSSCOMMODYVALUE_1',
         'RECOVEREDMAINDEBT_2', 'RECOVEREDWEAROUT_2', 'RECOVEREDLOSSCOMMODYVALUE_2',
         'RECOVEREDMAINDEBT_3', 'RECOVEREDWEAROUT_3', 'RECOVEREDLOSSCOMMODYVALUE_3',
         'RECOVEREDMAINDEBT_4', 'RECOVEREDWEAROUT_4', 'RECOVEREDLOSSCOMMODYVALUE_4',
         'RECOVEREDMAINDEBT_5', 'RECOVEREDWEAROUT_5', 'RECOVEREDLOSSCOMMODYVALUE_5',
         'RECOVEREDVALUEWITHSD_1', 'RECOVEREDVALUEWITHSD_2', 'RECOVEREDVALUEWITHSD_3',
         'RECOVEREDVALUEWITHSD_4', 'RECOVEREDVALUEWITHSD_5']
    ].fillna(0)
    target_3_claims = target_3_claims.groupby('INCIDENT_NUMBER').sum().reset_index(drop=False)

    df = df.merge(target_3_claims, how='left', on='INCIDENT_NUMBER')
    df = df.merge(target_3_pretensions, how='left', on='INCIDENT_NUMBER')
    df = df.merge(target_3_pretensions_all, how='left', on='INCIDENT_NUMBER')

    severity_cols = [
        'RECOVEREDMAINDEBT_1', 'RECOVEREDWEAROUT_1', 'RECOVEREDLOSSCOMMODYVALUE_1',
        'RECOVEREDMAINDEBT_2', 'RECOVEREDWEAROUT_2', 'RECOVEREDLOSSCOMMODYVALUE_2',
        'RECOVEREDMAINDEBT_3', 'RECOVEREDWEAROUT_3', 'RECOVEREDLOSSCOMMODYVALUE_3',
        'RECOVEREDMAINDEBT_4', 'RECOVEREDWEAROUT_4', 'RECOVEREDLOSSCOMMODYVALUE_4',
        'RECOVEREDMAINDEBT_5', 'RECOVEREDWEAROUT_5', 'RECOVEREDLOSSCOMMODYVALUE_5',
    ]

    def get_last_nonzero_or_valid(row):
        for col in reversed(severity_cols):
            val = row[col]
            if pd.notna(val) and val != 0:
                return val
        return np.nan

    df['TARGET_SEV'] = df.apply(get_last_nonzero_or_valid, axis=1)
    df[['TARGET_SEV', 'SurchargeValue_cumsum_by_incident', 'UTSSurchargeValue_cumsum_by_incident']] = df[
        ['TARGET_SEV', 'SurchargeValue_cumsum_by_incident', 'UTSSurchargeValue_cumsum_by_incident']
    ].fillna(0)
    df['TARGET_SEV'] = (
        df['TARGET_SEV']
        + df['SurchargeValue_cumsum_by_incident']
        + df['UTSSurchargeValue_cumsum_by_incident']
    )

    if 'VICTIM_VEHICLE_IS_JAPAN' in df.columns:
        df['VICTIM_VEHICLE_IS_JAPAN'] = df['VICTIM_VEHICLE_IS_JAPAN'].astype(str)

    df = df[df['VICTIM_POLICYHOLDER_TYPE'] == 'Физ. Лицо'].reset_index(drop=True)

    df['TARGET'] = (
        df['Сумма_выплат_по_претензиям'] + df['Сумма_взыскано_по_ФУ'] + df['Суммы_взыскано_по_иску']
    ).fillna(0)
    df['TARGET'] = df['TARGET'].apply(lambda x: 1 if x > 0 else 0).astype(int)

    df = checkpoint(
        df,
        paths,
        paths.processed_dir,
        "df_final_3.parquet",
        save=save_checkpoint,
    )
    return df
