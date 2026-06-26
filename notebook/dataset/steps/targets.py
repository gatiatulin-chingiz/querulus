"""Шаг пайплайна: targets."""
from __future__ import annotations

import numpy as np
import pandas as pd

from dataset.constants import RENAME_DICT
from dataset.io import checkpoint
from dataset.paths import DataPaths


def build_targets(paths: DataPaths, conn, df: pd.DataFrame, df_claims_: pd.DataFrame, *, save_checkpoint: bool = True):
    df_claims_inc_agg = df_claims_.groupby('INCIDENT_NUMBER')['INCOMING_CLAIM_NUMBER'].count().reset_index()
    df_claims_inc_agg[:2]

    df = df.merge(df_claims_inc_agg,how='left',on='INCIDENT_NUMBER')

    df['TARGET'] = df['INCOMING_CLAIM_NUMBER'].fillna(0).apply(lambda x: 1 if x > 0 else 0)


    df = df.drop(columns=['INCOMING_CLAIM_NUMBER'])

    # оставляем только первичный убыток
    df_ = df.copy()
    df = df.sort_values(['INCIDENT_NUMBER','PAYMENT_ORDER_DATE_TIME']).drop_duplicates(subset=['INCIDENT_NUMBER'],keep='first')
    len(df)


    query_calc_agg = \
    """
    with tmp as (
    	SELECT
    		_Period         Период	,
    		itl.IncidentNumber AS INCIDENT_NUMBER,
    		l.LossNumber 	AS LOSS_NUMBER	,
    --		l.RefundFormDetailed,
    --		l.LossProcess,
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
    	--IncidentNumber, count(*)
    	from oisuu81.dbo._InfoRg14746 i
    	left join oisuu81_t_losses l on l.LossID = _Fld14747RRef
    	LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_IncidentToLoss] AS itl on l.LossID=itl.LossID
    	where year(l.IssueDate) is not null
    --	and l.RefundFormDetailed  in ('Ремонт','Денежная','Денежная. Отказ от ремонта','Ремонт. Смена СТОА')
    	and l.LossProcess in ('Прямое ОСАГО (с 1 марта 2009)','Традиционное ОСАГО')
    	and Risk = 'Ущерб имуществу третьих лиц'
    --	and IncidentNumber = 10825255
    )
    SELECT *
    FROM tmp
    WHERE rn = 1
    """
    df_calc = pd.read_sql(query_calc_agg,conn)
    df_calc


    df_calc.loc[df_calc['ПроцентИзноса'] > 50, 'ПроцентИзноса'] = 50


    df_calc = df_calc.rename(columns={'Убыток':'LOSS_NUMBER'})


    df_calc['SHARE_WORK'] = (df_calc['Работы']/df_calc['СуммаРемонтаБезУчётаИзноса']).round(3)
    df_calc = df_calc.rename(columns={'СуммаРемонтаБезУчётаИзноса':'AMOUNT_REPAIR','ПроцентИзноса':'SHARE_WEAROUT'})


    df = df.merge(df_calc[['INCIDENT_NUMBER','SHARE_WORK','AMOUNT_REPAIR','SHARE_WEAROUT']],how='left',on='INCIDENT_NUMBER')
    df['AMOUNT_REPAIR'].isna().mean()


    df['ISSUE_YEAR'] = df['ISSUE_DATE'].dt.year


    df = df.drop(columns=['ISSUE_YEAR'])


    df['FLAG_APPLICANT_SAME_VICTIM_PH'] = df.apply(lambda row: 1 if row['APPLICANT_ID']==row['VICTIM_POLICYHOLDER_PERSON_ID'] else 0, axis=1)
    df['FLAG_APPLICANT_SAME_VICTIM_PH'].mean()


    df['TOTAL_COUNT_PRETENSION'] = df['APPLICANT_FTRS_PRET_PRETENSION_NUMBER_nunique'] + df['VICTIM_PH_FTRS_PRET_PRETENSION_NUMBER_nunique']
    df['TOTAL_COUNT_COURT'] = df['APPLICANT_FTRS_COURT_INCOMING_CLAIM_NUMBER_NUNIQUE'] + df['VICTIM_PH_FTRS_COURT_INCOMING_CLAIM_NUMBER_NUNIQUE']


    target_2_ = \
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
    target_2 = pd.read_sql(target_2_, conn)
    target_2.shape




    target_2 = checkpoint(
        target_2,
        paths,
        paths.raw_dir,
        "target_2.parquet",
        save=save_checkpoint,
    )


    for col in target_2.columns:
        target_2[col] = target_2[col].fillna(0)


    df = df.merge(target_2, how='left', left_on='INCIDENT_NUMBER', right_on='Номер_инциндента')
    df.shape




    target_3_pretensions = \
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
    --  and LossNumber = 1304948
    --  order by PretensionGetDate
    """
    target_3_pretensions = pd.read_sql(target_3_pretensions, conn)
    target_3_pretensions.shape



    target_3_pretensions = checkpoint(
        target_3_pretensions,
        paths,
        paths.raw_dir,
        "target_3_pretensions.parquet",
        save=save_checkpoint,
    )
    target_3_pretensions = target_3_pretensions.rename(columns=RENAME_DICT)
    target_3_pretensions

    target_3_pretensions_all = \
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
    target_3_pretensions_all = pd.read_sql(target_3_pretensions_all, conn)
    target_3_pretensions_all.shape



    target_3_pretensions_all = checkpoint(
        target_3_pretensions_all,
        paths,
        paths.raw_dir,
        "target_3_pretensions_all.parquet",
        save=save_checkpoint,
    )
    target_3_pretensions_all = target_3_pretensions_all.rename(columns=RENAME_DICT)
    target_3_pretensions_all


    target_3_claims = \
    """
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
    WHERE (ClaimOrigin in ('ВСК', 'Обращение к ФУ') or ClaimOrigin is null)
      AND (ClaimItem != 'ВСК - 3 лицо' or ClaimItem is null)
      AND l.LossProcess IN ('Прямое ОСАГО (с 1 марта 2009)', 'Традиционное ОСАГО')
    """
    target_3_claims = pd.read_sql(target_3_claims, conn)
    target_3_claims.shape


    target_3_claims = checkpoint(
        target_3_claims,
        paths,
        paths.raw_dir,
        "target_3_claims.parquet",
        save=save_checkpoint,
    )
    target_3_claims = target_3_claims.rename(columns=RENAME_DICT)
    target_3_claims


    target_3_claims.columns = target_3_claims.columns.str.upper()
    target_3_claims = target_3_claims.rename(columns=RENAME_DICT)


    df_base = target_3_claims[['INCIDENT_NUMBER', 'LOSS_ID', 'LOSS_NUMBER',
                               'INCOMINGCLAIMID', 'INCOMING_CLAIM_NUMBER',
                               'LINK_LOSS_NUMBER']].drop_duplicates(['INCOMING_CLAIM_NUMBER'])
    df_base = df_base.sort_values(by=['INCOMING_CLAIM_NUMBER']).reset_index(drop=True)


    # Отбираем только необходимые колонки (относящиеся к судам)
    df_sorted = target_3_claims.loc[:, ['LOSS_NUMBER'] + list(target_3_claims.iloc[:, 6:].columns)]
    # Удаляем ненужные колонки
    df_sorted = df_sorted.drop(['INSTID', 'LINK_LOSS_NUMBER'], axis=1)
    # Сортируем датафрейм по номеру иска и дате выставления, чтобы инстанции шли по временному порядку
    df_sorted = df_sorted.sort_values(by=['INCOMING_CLAIM_NUMBER', 'CLAIMEDVALUEPERIOD']).reset_index(drop=True)
    # Первое вхождение в какую-либо инстанцию (будь то 6, или 1 инстанции) в отдельно взятом номере иска == перва инстанция
    # для данного иска (cumcount)
    df_sorted['Instance'] = df_sorted.groupby(['INCOMING_CLAIM_NUMBER']).cumcount() + 1
    # Сводная таблица по искам и инстанциям в строку
    df_pivot = df_sorted.pivot(index=['INCOMING_CLAIM_NUMBER'], columns='Instance')
    # Колонки переименовываем в соответствии с номером инстанции (ClaimedValueWithSD_1, ClaimedValueWithSD_2 И т.д.)
    df_pivot.columns = [f'{col[0]}_{int(col[1])}' for col in df_pivot.columns]
    df_pivot = df_pivot.reset_index()
    df_pivot


    target_3_claims = df_base.merge(df_pivot, how='left', on=['INCOMING_CLAIM_NUMBER'])
    target_3_claims


    target_3_claims = target_3_claims[['INCIDENT_NUMBER',
                                       'RECOVEREDMAINDEBT_1', 'RECOVEREDWEAROUT_1', 'RECOVEREDLOSSCOMMODYVALUE_1',
                                       'RECOVEREDMAINDEBT_2', 'RECOVEREDWEAROUT_2', 'RECOVEREDLOSSCOMMODYVALUE_2',
                                       'RECOVEREDMAINDEBT_3', 'RECOVEREDWEAROUT_3', 'RECOVEREDLOSSCOMMODYVALUE_3',
                                       'RECOVEREDMAINDEBT_4', 'RECOVEREDWEAROUT_4', 'RECOVEREDLOSSCOMMODYVALUE_4',
                                       'RECOVEREDMAINDEBT_5', 'RECOVEREDWEAROUT_5', 'RECOVEREDLOSSCOMMODYVALUE_5',
                                  
                                       'RECOVEREDVALUEWITHSD_1','RECOVEREDVALUEWITHSD_2','RECOVEREDVALUEWITHSD_3',
                                       'RECOVEREDVALUEWITHSD_4','RECOVEREDVALUEWITHSD_5']].fillna(0)


    target_3_claims = target_3_claims.groupby('INCIDENT_NUMBER').sum().reset_index(drop=False)
    target_3_claims


    df = df.merge(target_3_claims, how='left', left_on='INCIDENT_NUMBER', right_on='INCIDENT_NUMBER')
    df.shape


    df = df.merge(target_3_pretensions, how='left', left_on='INCIDENT_NUMBER', right_on='INCIDENT_NUMBER')
    df.shape


    df = df.merge(target_3_pretensions_all, how='left', left_on='INCIDENT_NUMBER', right_on='INCIDENT_NUMBER')
    df.shape


    # Список колонок в порядке от высшей инстанции к низшей
    cols = ['RECOVEREDMAINDEBT_1', 'RECOVEREDWEAROUT_1', 'RECOVEREDLOSSCOMMODYVALUE_1',
            'RECOVEREDMAINDEBT_2', 'RECOVEREDWEAROUT_2', 'RECOVEREDLOSSCOMMODYVALUE_2',
            'RECOVEREDMAINDEBT_3', 'RECOVEREDWEAROUT_3', 'RECOVEREDLOSSCOMMODYVALUE_3',
            'RECOVEREDMAINDEBT_4', 'RECOVEREDWEAROUT_4', 'RECOVEREDLOSSCOMMODYVALUE_4',
            'RECOVEREDMAINDEBT_5', 'RECOVEREDWEAROUT_5', 'RECOVEREDLOSSCOMMODYVALUE_5']

    # Функция для выбора первого ненулевого и не-NaN значения
    def get_last_nonzero_or_valid(row):
        for col in reversed(cols):
            val = row[col]
            # Если значение не NaN и не 0 (если 0 считается "пустым")
            if pd.notna(val) and val != 0:
                return val
        # Если все пустые или нули — вернуть NaN или 0, как вам нужно
        return np.nan  # или 0, если предпочитаете

    df['TARGET_3_SEV'] = df.apply(get_last_nonzero_or_valid, axis=1)


    df[['TARGET_3_SEV', 'SurchargeValue_cumsum_by_incident', 'UTSSurchargeValue_cumsum_by_incident']] = df[['TARGET_3_SEV', 'SurchargeValue_cumsum_by_incident', 'UTSSurchargeValue_cumsum_by_incident']].fillna(0)


    df['TARGET_3_SEV'] = df['TARGET_3_SEV'] + df['SurchargeValue_cumsum_by_incident'] + df['UTSSurchargeValue_cumsum_by_incident']
    df['TARGET_3_SEV'].value_counts(dropna=False)


    df['TARGET_3_FREQ'] = df['TARGET_3_SEV'].apply(lambda x: 1 if x > 0 else 0)
    df['TARGET_3_FREQ'].value_counts(dropna=False)


    df['VICTIM_VEHICLE_IS_JAPAN'] = df['VICTIM_VEHICLE_IS_JAPAN'].astype(str)


    df = df[df['VICTIM_POLICYHOLDER_TYPE'] == 'Физ. Лицо'].reset_index(drop=True)

    df['TARGET_2'] = (
        df['Сумма_выплат_по_претензиям'] + df['Сумма_взыскано_по_ФУ'] + df['Суммы_взыскано_по_иску']
    ).fillna(0)
    df['TARGET_2'] = df['TARGET_2'].apply(lambda x: 1 if x > 0 else 0).astype(int)

    df = checkpoint(
        df,
        paths,
        paths.processed_dir,
        "df_final_3.parquet",
        save=save_checkpoint,
    )
    return df
