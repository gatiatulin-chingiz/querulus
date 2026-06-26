"""Шаг пайплайна: claims."""
from __future__ import annotations

import gc
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from dataset.constants import RENAME_DICT
from dataset.io import checkpoint, read_artifact
from dataset.paths import DataPaths
from dataset.utils import convert_to_binary

logger = logging.getLogger("querulus.dataset")


def load_claims(paths: DataPaths, conn, *, use_sql: bool = False, save_checkpoint: bool = True):
    query = \
    """
    SELECT 
        Лицо,
        CAST ( Представитель as int ) Представитель,
        CAST ( Цессионарий as int ) Цессионарий,
        ПолноеФИОЛица,
        ТипЛица,
        ПолЛица,
        НомерИск,
        ОбращениеКФУОтЗаявителяПоступилоПосредством
    FROM [OISUU_report].[Datamart].[oisuu81_t_Истцы]
    WHERE ПометкаУдаленияИск=0x00
    """
    #df_claims_persons = pd.read_sql_query(query, conn)
    if use_sql:
        logger.info("LOAD sql: df_claims_persons")
        df_claims_persons = pd.read_sql_query(query, conn)
    else:
        df_claims_persons = read_artifact(
            paths, paths.processed_dir, "df_claims_persons.parquet"
        )
    df_claims_persons



    query_pay = \
    """
    WITH doc as (
        SELECT [_Document5472_IDRRef]
              ,[_KeyField]
              ,[_LineNo14756]
              ,[_Fld14757]
              ,[_Fld14758]
              ,[_Fld14759]
              ,[_Fld14760]
              ,[_Fld14761]
              ,[_Fld14762RRef]
              ,[_Fld14763]
              ,ROW_NUMBER() OVER (PARTITION BY [_Document5472_IDRRef] ORDER BY [_LineNo14756] DESC) as rn
          FROM [oisuu81].[dbo].[_Document5472_VT14755]
    )
    SELECT
           itl.[IncidentNumber]
          ,icnl.[IncomingClaimNumber]
          ,RecoveredValueWithSD
          ,docs.[_Fld14757] as [Payment_fee_fu]
          ,ROW_NUMBER() OVER (PARTITION BY IncomingClaimNumber ORDER BY RecoveredValuePeriod desc) rn_inst
    FROM [OISUU_report].[Datamart].[oisuu81_t_IncomingClaimNewLogicByInst] as icnl
    LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_IncidentToLoss] as itl on itl.LossNumber=icnl.LinkLossNumber
    LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_Losses] as l on l.LossNumber=itl.LossNumber
    LEFT JOIN doc as docs on docs.[_Document5472_IDRRef]=icnl.[IncomingClaimID]
    WHERE (ClaimOrigin not in ('Исходящий иск') or ClaimOrigin is null )
        AND (docs.rn = 1 or docs.rn is null)
        and icnl.Instance != 'Суд кассационной инстанции'
        and icnl.RecoveredValuePeriod is not null
    """

    df_claims_pay = pd.read_sql_query(query_pay, conn)
    df_claims_pay


    # Группируем все взыскания по инциденты 
    df_claims_pay['RecoveredValueWithSD'] = df_claims_pay['RecoveredValueWithSD'].fillna(0)
    df_claims_pay['Payment_fee_fu'] = df_claims_pay['Payment_fee_fu'].fillna(0)
    df_claims_pay = df_claims_pay[df_claims_pay['rn_inst']==1].groupby('IncidentNumber')[['RecoveredValueWithSD','Payment_fee_fu']].agg('sum').reset_index()
    df_claims_pay[:2]




    query = \
    """
    WITH doc as (
        SELECT [_Document5472_IDRRef]
              ,[_KeyField]
              ,[_LineNo14756]
              ,[_Fld14757]
              ,[_Fld14758]
              ,[_Fld14759]
              ,[_Fld14760]
              ,[_Fld14761]
              ,[_Fld14762RRef]
              ,[_Fld14763]
              ,ROW_NUMBER() OVER (PARTITION BY [_Document5472_IDRRef] ORDER BY [_LineNo14756] DESC) as rn
          FROM [oisuu81].[dbo].[_Document5472_VT14755]
    ),
    cessia as (
        SELECT 
                НомерИск
                ,max([Представитель]) as [Представитель]
                ,max([Цессионарий]) as [Цессионарий]
          FROM [OISUU_report].[Datamart].[oisuu81_t_Истцы]
          group by НомерИск
    )
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
          ,docs.[_Fld14757] as [Payment_fee_fu]
          ,cs.[Представитель]
          ,cs.[Цессионарий]
    FROM [OISUU_report].[Datamart].[oisuu81_t_IncomingClaimNewLogicByInst] as icnl
    LEFT JOIN cessia as cs on cs.НомерИск=icnl.IncomingClaimNumber
    LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_IncidentToLoss] as itl on itl.LossNumber=icnl.LinkLossNumber
    LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_Losses] as l on l.LossNumber=itl.LossNumber
    LEFT JOIN doc as docs on docs.[_Document5472_IDRRef]=icnl.[IncomingClaimID]
    WHERE (ClaimOrigin not in ('Исходящий иск') or ClaimOrigin is null )
        AND (docs.rn = 1 or docs.rn is null)
    """


    df_claims = pd.read_sql_query(query, conn)
    df_claims = checkpoint(
        df_claims,
        paths,
        paths.processed_dir,
        "df_claims.parquet",
        save=save_checkpoint,
    )

    df_claims.columns = df_claims.columns.str.upper()
    df_claims = df_claims.rename(columns=RENAME_DICT)

    df_claims['ПРЕДСТАВИТЕЛЬ'] = df_claims['ПРЕДСТАВИТЕЛЬ'].apply(convert_to_binary)
    df_claims['ЦЕССИОНАРИЙ'] = df_claims['ЦЕССИОНАРИЙ'].apply(convert_to_binary)


    df_base = df_claims[['INCIDENT_NUMBER', 'LOSS_ID', 'LOSS_NUMBER',
                         'INCOMINGCLAIMID', 'INCOMING_CLAIM_NUMBER',
                         'LINK_LOSS_NUMBER']].drop_duplicates(['INCOMING_CLAIM_NUMBER'])
    df_base = df_base.sort_values(by=['INCOMING_CLAIM_NUMBER']).reset_index(drop=True)



    # Отбираем только необходимые колонки (относящиеся к судам)
    df_sorted = df_claims.loc[:, ['LOSS_NUMBER'] + list(df_claims.iloc[:, 6:].columns)]
    # Удаляем ненужные колонки
    df_sorted = df_sorted.drop(['INSTID', 'LINK_LOSS_NUMBER'], axis=1)
    #df_sorted = df_sorted.drop(['INSTID'], axis=1)
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
    df_pivot[df_pivot['INCOMING_CLAIM_NUMBER']==434485]


    df_claims = df_base.merge(df_pivot, how='left', on=['INCOMING_CLAIM_NUMBER'])



    group_map_item = {
 
     'ВРЕД ЗДОРОВЬЮ':'ЖиЗ',
     'ВРЕД ЖИЗНИ':'ЖиЗ',
     'ВРЕД ЖИЗНИ И/ИЛИ ЗДОРОВЬЮ':'ЖиЗ',
     'ВОЗМЕЩЕНИЕ ВРЕДА, ПРИЧИНЕННОГО ЖИЗНИ И ЗДОРОВЬЮ':'ЖиЗ',
     'ВОЗМЕЩЕНИЕ ПО ПОТЕРЕ КОРМИЛЬЦА':'ЖиЗ',
 
     'МОРАЛЬНЫЙ ВРЕД':'МОРАЛЬНЫЙ ВРЕД',
     'О ВЗЫСКАНИИ МОРАЛЬНОГО ВРЕДА':'МОРАЛЬНЫЙ ВРЕД',
 
     'НЕДОСТАТКИ РЕМОНТА':'НЕДОСТАТКИ РЕМОНТА',
     'УСТРАНЕНИЕ НЕДОСТАТКОВ РЕМОНТА':'НЕДОСТАТКИ РЕМОНТА',
     'КОМПЕНСАЦИЯ НЕДОСТАТКОВ РЕМОНТА':'НЕДОСТАТКИ РЕМОНТА',
 
     'НЕДОСТОВЕРНЫЕ СВЕДЕНИЯ Е-ОСАГО':'НЕДОСТОВЕРНЫЕ СВЕДЕНИЯ',
     'НИЧТОЖНЫЙ ИЛИ ПОДДЕЛЬНЫЙ ПОЛИС ОСАГО':'НЕДОСТОВЕРНЫЕ СВЕДЕНИЯ',
     'СОБЩЕНИЕ ЗАВЕДОМО ЛОЖНЫХ СВЕДЕНИЙ':'НЕДОСТОВЕРНЫЕ СВЕДЕНИЯ',
     'СООБЩЕНИЕ ЗАВЕДОМО ЛОЖНЫХ СВЕДЕНИЙ':'НЕДОСТОВЕРНЫЕ СВЕДЕНИЯ',
 
     'НЕ ПРЕДОСТАВЛЕНИЕ ПОВРЕЖДЕННОГО ОБЪЕКТА СТРАХОВАНИЯ НА ОСМОТР':'НЕ ПРЕДОСТАВЛЕНИЕ НА ОСМОТР',
     'НЕ ПРЕДОСТАВЛЕНИЕ ПОВРЕЖДЕННОГО ИМУЩЕСТВА НА ОСМОТР':'НЕ ПРЕДОСТАВЛЕНИЕ НА ОСМОТР',
 
     'ДОПЛАТА':'СПОР ПО СУММЕ',
     'СПОР ПО СУММЕ':'СПОР ПО СУММЕ',
     'ДОП. РАСХОДЫ':'СПОР ПО СУММЕ',

 
     'НЕПРИЗНАНИЕ СОБЫТИЯ СТРАХОВЫМ СЛУЧАЕМ ПО ЗАКОНУ': 'НЕПРИЗНАНИЕ СТРАХОВЫМ СЛУЧАЕМ',
     'НЕПРИЗНАНИЕ СОБЫТИЯ СТРАХОВЫМ СЛУЧАЕМ ПО ПРАВИЛАМ': 'НЕПРИЗНАНИЕ СТРАХОВЫМ СЛУЧАЕМ',
     'НЕПРИЗНАНИЕ\xa0 СОБЫТИЯ СТРАХОВЫМ СЛУЧАЕМ ПО ЗАКОНУ': 'НЕПРИЗНАНИЕ СТРАХОВЫМ СЛУЧАЕМ',
     'СОБЫТИЕ НЕ В СТРАХОВОЙ\xa0 ПЕРИОД': 'НЕПРИЗНАНИЕ СТРАХОВЫМ СЛУЧАЕМ',
     'НЕПРИЗНАНИЕ СОБЫТИЯ СТРАХОВЫМ СЛУЧАЕМ': 'НЕПРИЗНАНИЕ СТРАХОВЫМ СЛУЧАЕМ',
     'СОБЫТИЕ ПРОИЗОШЛО НЕ В ПЕРИОД ДЕЙСТВИЯ ДОГОВОРА СТРАХОВАНИЯ': 'НЕПРИЗНАНИЕ СТРАХОВЫМ СЛУЧАЕМ',
 
     'НЕУСТОЙКА/ШТРАФ ПО ЗОЗПП ОТДЕЛЬНЫМ ИСКОМ': 'НЕУСТОЙКА',
     'НЕУСТОЙКА ПО ОСАГО ОТДЕЛЬНЫМ ИСКОМ': 'НЕУСТОЙКА',
     'ВЗЫСКАНИЕ НЕУСТОЙКИ': 'НЕУСТОЙКА',
     'ПРОЦЕНТЫ ЗА ПОЛЬЗОВАНИЕ ЧУЖИМИ ДЕНЕЖНЫМИ СРЕДСТВАМИ':'НЕУСТОЙКА',
 

     'ИНДЕКСАЦИЯ ПРИСУЖДЕННЫХ СУММ':'НЕУСТОЙКА',
 
     'ИЗМЕНЕНИЕ УСЛОВИЙ СТРАХОВАНИЯ': 'УСЛОВИЯ СТРАХОВАНИЯ',
     'ПО УСЛОВИЯМ ДОГОВОРА': 'УСЛОВИЯ СТРАХОВАНИЯ',
 
     'ВОЗВРАТ СТОИМОСТИ ТУРИСТИЧЕСКОГО ПРОДУКТА': 'ВОЗВРАТ СТОИМОСТИ ТУРИСТИЧЕСКОГО ПРОДУКТА',
     'ГО ПО ДОГОВОРУ О РЕАЛИЗАЦИИ ТУРИСТИЧЕСКОГО ПРОДУКТА': 'ВОЗВРАТ СТОИМОСТИ ТУРИСТИЧЕСКОГО ПРОДУКТА',
 
     'ПРАВО НА ПВУ': 'ПВУ',
     'ПО ПВУ': 'ПВУ', 
     'СПОР О ПВУ': 'ПВУ', 
 
     'СПОР О ФОРМЕ СТРАХОВОГО ВОЗМЕЩЕНИЯ (РЕМОНТ/ВЫПЛАТА)': 'ФОРМА ВОЗМЕЩЕНИЯ',
     'СПОР О ФОРМЕ ВЫПЛАТЫ (РЕМОНТ/ДЕНЬГИ)': 'ФОРМА ВОЗМЕЩЕНИЯ', 
     'УСЛОВИЕ "РЕМОНТ"': 'ФОРМА ВОЗМЕЩЕНИЯ',
     'ВЫДАТЬ НАПРАВЛЕНИЕ НА РЕМОНТ': 'ФОРМА ВОЗМЕЩЕНИЯ',
     'ОРГАНИЗАЦИЯ РЕМОНТА НА СТОА': 'ФОРМА ВОЗМЕЩЕНИЯ',
     'ОБЯЗАТЕЛЬНЫЙ РЕМОНТ ОСАГО.ЦЕССИЯ ЮЛ': 'ФОРМА ВОЗМЕЩЕНИЯ',
     'ЦЕССИЯ ОСАГО.ОБЯЗАТЕЛЬНЫЙ РЕМОНТ': 'ФОРМА ВОЗМЕЩЕНИЯ',
 
     'ВОЗВРАТ СТРАХОВОЙ ПРЕМИИ': 'ВОЗВРАТ СТРАХОВОЙ ПРЕМИИ',
     'ВОЗВРАТ ПРЕМИИ (КБМ)': 'ВОЗВРАТ СТРАХОВОЙ ПРЕМИИ', 
     'ОСПАРИВАНИЕ КБМ (ОСАГО)': 'ВОЗВРАТ СТРАХОВОЙ ПРЕМИИ',
 
     'ДОСРОЧНОЕ ПОГАШЕНИЕ КРЕДИТА': 'ДОСРОЧНОЕ ПОГАШЕНИЕ КРЕДИТА',
     'ДОСРОЧНОЕ ПОГАШЕНИЕ КРЕДИТА (КАСКО)': 'ДОСРОЧНОЕ ПОГАШЕНИЕ КРЕДИТА', 
     'ДОСРОЧНОЕ ПОГАШЕНИЕ КРЕДИТА (НС)': 'ДОСРОЧНОЕ ПОГАШЕНИЕ КРЕДИТА',

 
 
     'ПРЕКРАЩЕНИЕ ДОГОВОРА ОСАГО': 'ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ',
     'ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ': 'ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ', 
     'О ПРИЗНАНИИ НЕДЕЙСТВИТЕЛЬНЫМ ДОГОВОРА СТРАХОВАНИЯ': 'ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ',
     'ПЕРИОД ОХЛАЖДЕНИЯ (КАСКО)': 'ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ', 
     'ПЕРИОД ОХЛАЖДЕНИЯ (НС)': 'ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ',
     'ПЕРИОД ОХЛАЖДЕНИЯ (ИФЛ)': 'ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ',
     'ДОСРОЧНОЕ ПРЕКРАЩЕНИЕ ДОГОВОРА КАСКО': 'ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ',
     'ПЕРИОД ОХЛАЖДЕНИЯ': 'ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ',
     'РАСТОРЖЕНИЕ, ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ': 'ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ',
     'ПРИЗНАНИЕ УСЛОВИЙ КРЕДИТНОГО ДОГОВОРА НЕДЕЙСТВИТЕЛЬНЫМИ': 'ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ',
     'ПРЕКРАЩЕНИЕ ДОГОВОРА ОСАГО': 'ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ',
     'ПОНУЖДЕНИЕ К ЗАКЛЮЧЕНИЮ ДОГОВОРА': 'ПРЕКРАЩЕНИЕ ДОГОВОРА СТРАХОВАНИЯ',
 
     'НЕВРУЧЕНИЕ КИД (НС)': 'КИД', 
     'НЕВРУЧЕНИЕ КИД ': 'КИД',
     'НЕКОРРЕКТНЫЙ КИД': 'КИД',

     'ПРЕВЫШЕНИЕ СТРАХОВОЙ СУММЫ НАД СТОИМОСТЬЮ': 'ПРЕВЫШЕНИЕ СТРАХОВОЙ СУММЫ НАД СТОИМОСТЬЮ', 
     'ПРЕВЫШЕНИЕ СТРАХОВОЙ СУММЫ НАД СТРАХОВОЙ СТОИМОСТЬЮ': 'ПРЕВЫШЕНИЕ СТРАХОВОЙ СУММЫ НАД СТОИМОСТЬЮ',


     'НЕДОКАЗАННОСТЬ ОБСТОЯТЕЛЬСТВ НАСТУПЛЕНИЯ СТРАХОВОГО СЛУЧАЯ (ТРАСОЛОГИЯ)': 'ТРАСОЛОГИЯ', 
     'НЕДОКАЗАННОСТЬ РАЗМЕРА УЩЕРБА (ТРАСОЛОГИЯ)': 'ТРАСОЛОГИЯ',
     'ЧАСТИЧНЫЙ ОТКАЗ ПО ТРАСОЛОГИИ': 'ТРАСОЛОГИЯ',
     'ПОЛНЫЙ ОТКАЗ ПО ТРЭ': 'ТРАСОЛОГИЯ',
     'ЧАСТИЧНЫЙ ОТКАЗ ПО ТРЭ': 'ТРАСОЛОГИЯ',
 
     'ИНОЕ – РСА 3-Е ЛИЦО':'3_Е_ЛИЦО',
     'ВСК - 3 ЛИЦО':'3_Е_ЛИЦО',
     'ИНОЕ – ВСК 3-Е ЛИЦО':'3_Е_ЛИЦО',
 
 
     'ОСПАРИВАНИЕ СОГЛАШЕНИЯ ОБ УРЕГУЛИРОВАНИИ ОСАГО':'ОСПАРИВАНИЕ СОГЛАШЕНИЯ ОБ УРЕГУЛИРОВАНИЕ' ,
 
     'ЖИЗ СОЛИДАРНАЯ ОТВЕТСТВЕННОСТЬ':'ОБОЮДНАЯ ВИНА',
 
     'СУДЕБНЫЕ РАСХОДЫ':'СУДЕБНЫЕ РАСХОДЫ И САНКЦИИ ЗА НЕИСПОЛНЕНИЕ РЕШЕНИЯ СУДА', 
     'САНКЦИИ ЗА НЕИСПОЛНЕНИЕ РЕШЕНИЯ СУДА':'СУДЕБНЫЕ РАСХОДЫ И САНКЦИИ ЗА НЕИСПОЛНЕНИЕ РЕШЕНИЯ СУДА',
     'САНКЦИИ ЗА НЕИСПОЛНЕНИЕ РЕШЕНИЯ':'СУДЕБНЫЕ РАСХОДЫ И САНКЦИИ ЗА НЕИСПОЛНЕНИЕ РЕШЕНИЯ СУДА',
     'ТРЕБОВАНИЕ ПО ВЫПЛАТЕ ШТРАФНЫХ САНКЦИЙ':'СУДЕБНЫЕ РАСХОДЫ И САНКЦИИ ЗА НЕИСПОЛНЕНИЕ РЕШЕНИЯ СУДА'
    }
    df_claims['CLAIM_ITEM_1'] = df_claims['CLAIM_ITEM_1'].str.upper()
    df_claims['CLAIM_ITEM'] = df_claims['CLAIM_ITEM_1'].str.upper().replace(group_map_item).fillna(df_claims['CLAIM_ITEM_1'])


    cat_claim_item = list(df_claims['CLAIM_ITEM'].value_counts().reset_index()[0:9]['CLAIM_ITEM'].unique())
    df_claims['CLAIM_ITEM'] = df_claims['CLAIM_ITEM'].apply(lambda x : x if x in cat_claim_item else 'ПРОЧЕЕ')
    df_claims['CLAIM_ITEM'].value_counts()


    #ohe_enc = OneHotEncoder(drop='first', handle_unknown='ignore')
    ohe_enc = OneHotEncoder( handle_unknown='ignore')
    cat_features = ['CLAIM_ITEM']
    ohe_enc.fit(df_claims[cat_features])
    enc = ohe_enc.transform(df_claims[cat_features]).toarray()
    enc = pd.DataFrame(enc, columns=ohe_enc.get_feature_names_out())
    df_claims_ = pd.concat([df_claims.reset_index(drop=True), enc.reset_index(drop=True)], axis=1)
    df_claims_[:2]


    df_claims_ = checkpoint(
        df_claims_,
        paths,
        paths.processed_dir,
        "df_claims_.parquet",
        save=save_checkpoint,
    )
    df_claims = checkpoint(
        df_claims,
        paths,
        paths.processed_dir,
        "df_claims.parquet",
        save=save_checkpoint,
    )
    return df_claims_persons, df_claims, df_claims_
