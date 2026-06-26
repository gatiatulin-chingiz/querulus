"""Шаг пайплайна: pretensions."""
from __future__ import annotations

import logging

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from dataset.constants import RENAME_DICT
from dataset.io import checkpoint, read_artifact
from dataset.paths import DataPaths
from dataset.utils import convert_to_binary, hex_upper

logger = logging.getLogger("querulus.dataset")


def load_pretensions(paths: DataPaths, conn, *, use_sql: bool = False, save_checkpoint: bool = True):
    df_pretensions_ = ("""
    SELECT *
      FROM [OISUU_report].[dbo].[oisuu81_t_Pretensions] AS P
      LEFT JOIN [OISUU_report].[dbo].[oisuu81_t_IncidentToLoss] AS ITL ON ITL.LossID=P.LossID
    """)
    if use_sql:
        logger.info("LOAD sql: df_pretensions")
        df_pretensions = pd.read_sql(df_pretensions_, conn)
    else:
        df_pretensions = read_artifact(paths, paths.raw_dir, "df_pretensions.parquet")

    df_pretensions = df_pretensions.loc[:, ~df_pretensions.columns.duplicated()].copy()

    # хадуп не работает, поэтому берем нужные id фио из sql
    sql_pret_id = ( """
    select 
        pr.PretensionNumber PRETENSION_NUMBER
        ,L.LossNumber LOSS_NUMBER
        ,(L.PolicyholderPersonID) as POLICYHOLDER_PERSON_ID
        ,(L.VictimPersonID) as VICTIM_PERSON_ID
        ,(L.VIctimPolicyholderPersonID) as VICTIM_POLICYHOLDER_PERSON_ID
        , L.VictimObjectOwnerPersonID as VICTIM_OBJECT_OWNER_PERSON_ID
     from OISUU_REPORT.DBO.oisuu81_t_Pretensions pr
     left join OISUU_REPORT.DBO.oisuu81_t_Losses L on l.LossID=pr.LossID

    """)

    pretension_fio_id = pd.read_sql(sql_pret_id,conn)
    pretension_fio_id.shape

    pretension_fio_id['POLICYHOLDER_PERSON_ID'] = pretension_fio_id['POLICYHOLDER_PERSON_ID'].apply(hex_upper)
    pretension_fio_id['VICTIM_PERSON_ID'] = pretension_fio_id['VICTIM_PERSON_ID'].apply(hex_upper)
    pretension_fio_id['VICTIM_POLICYHOLDER_PERSON_ID'] = pretension_fio_id[
        'VICTIM_POLICYHOLDER_PERSON_ID'
    ].apply(hex_upper)
    pretension_fio_id['VICTIM_OBJECT_OWNER_PERSON_ID'] = pretension_fio_id[
        'VICTIM_OBJECT_OWNER_PERSON_ID'
    ].apply(hex_upper)

    df_pretensions.columns = df_pretensions.columns.str.upper()
    df_pretensions = df_pretensions.rename(columns=RENAME_DICT)

    #Добавим id в основной датасет по претензиям
    print(df_pretensions.shape)
    df_pretensions = df_pretensions.merge(pretension_fio_id[['PRETENSION_NUMBER','VICTIM_POLICYHOLDER_PERSON_ID','VICTIM_OBJECT_OWNER_PERSON_ID']],how='left',on='PRETENSION_NUMBER')
    print(df_pretensions.shape)

    query = \
    """
    -- ОТСЮДА БЕРУТСЯ ПРЕТЕНЗИИ С СУММОЙ ПО НЕУСТОЙКЕ
    with tmp as (
    	SELECT 
    			itl.IncidentID
    		 ,itl.IncidentNumber
    		 ,itl.LossID
    		 ,itl.LossNumber
    		 ,p.[PretensionID]
    		 ,p.[IsMarked]
    		 ,p.PretensionDate
    		 ,p.PretensionNumber
    		 ,p.[IsOver]
    		 ,p.[PretensionGetDate]
    		 ,p.[PretensionType]
    		 ,p.[ApplicantPersonID]
    		 ,p.[PretensionStage]
    		 ,p.[PretensionGetMethod]
    		 ,p.[PretensionValue]
    		 ,p.[PretensionCurrency]
    		 ,p.[UTSValue]
    		 ,p.[UTSCurrency]
    		 ,p.[LossUnit]
    		 ,p.[LossUnitZone]
    		 ,p.[AnswerType]
    		 ,p.[AnswerDate]
    		 ,p.[SurchargeValue]
    		 ,p.[SurchargeCurrency]
    		 ,p.[UTSSurchargeValue]
    		 ,p.[UTSSurchargeCurrency]
    		 ,p.[Comment]
    		 ,p.[HaveRequisitesOfApplicant]
    		 ,p.[RequiredReviewSNEO]
    		 ,p.[DateCancellationReviews]
    		 ,p.[ExternalOrderSNEO]
    		 ,p.[IsFullPretensionAmountsWithBreakdown]
    		 ,p.[SendInRSADate]
    		 ,p.[DateSentScannedCopies]
    		 ,p.[PretensionTypeID]
    		 ,p.[PretensionTypes]
    		 ,p.[PretensionKinds]
    		 ,p.[InsuranceTypes]
    		 ,p.[InsuranceTypeGroups]
    		 ,p.[Cession]
    		 ,p.[LinkedLossID]
    		 ,_Fld9244RRef
    		 ,_Fld9243 as pretension_value_
    		 ,SUM(_Fld9243) OVER(PARTITION BY IncidentNumber, PretensionNumber) as PRETENSION_VALUE_PENALTY
    		 ,ROW_NUMBER() OVER(PARTITION BY IncidentNumber, PretensionNumber ORDER BY PretensionNumber) as rn
    		 ,0 as SURCHARGE_VALUE_PENALTY
    	 FROM [OISUU_report].[dbo].[oisuu81_t_IncidentToLoss] as itl
    	 left join [OISUU_report].[dbo].[oisuu81_t_Pretensions] as p on p.LossID=itl.LossID
    	 left join oisuu81.dbo._Document6169_VT9241 vt on vt._Document6169_IDRRef=p.PretensionID
    	-- left join [oisuu81].[dbo].[_Document6169_VT9237] dcv on dcv._Document6169_IDRRef = p.PretensionID
    	 WHERE _Fld9244RRef in (0xB6B5441EA172DD2611E8AC27427E4644, 0xB6B5441EA172DD2611E8AC282FFD5C5A)
     ),
     -- ОТСЮДА БЕРУТСЯ ТЕ ЖЕ ПРЕТЕНЗИИ, КОТОРЫЕ В ЗАПРОСЕ ВЫШЕ, НО ВЫТАСКИВАЮТСЯ СУММЫ ДОПЛАТ
    penalty as (
    	SELECT 
    			itl.IncidentID
    		 ,itl.IncidentNumber
    		 ,itl.LossID
    		 ,itl.LossNumber
    		 ,p.[PretensionID]
    		 ,p.[IsMarked]
    		 ,p.PretensionDate
    		 ,p.PretensionNumber
    		 ,p.[IsOver]
    		 ,p.[PretensionGetDate]
    		 ,p.[PretensionType]
    		 ,p.[ApplicantPersonID]
    		 ,p.[PretensionStage]
    		 ,p.[PretensionGetMethod]
    		 ,p.[PretensionValue]
    		 ,p.[PretensionCurrency]
    		 ,p.[UTSValue]
    		 ,p.[UTSCurrency]
    		 ,p.[LossUnit]
    		 ,p.[LossUnitZone]
    		 ,p.[AnswerType]
    		 ,p.[AnswerDate]
    		 ,p.[SurchargeValue]
    		 ,p.[SurchargeCurrency]
    		 ,p.[UTSSurchargeValue]
    		 ,p.[UTSSurchargeCurrency]
    		 ,p.[Comment]
    		 ,p.[HaveRequisitesOfApplicant]
    		 ,p.[RequiredReviewSNEO]
    		 ,p.[DateCancellationReviews]
    		 ,p.[ExternalOrderSNEO]
    		 ,p.[IsFullPretensionAmountsWithBreakdown]
    		 ,p.[SendInRSADate]
    		 ,p.[DateSentScannedCopies]
    		 ,p.[PretensionTypeID]
    		 ,p.[PretensionTypes]
    		 ,p.[PretensionKinds]
    		 ,p.[InsuranceTypes]
    		 ,p.[InsuranceTypeGroups]
    		 ,p.[Cession]
    		 ,p.[LinkedLossID]
    		 ,_Fld9244RRef
    		 ,_Fld9243 as pretension_value_
    		 ,SUM(_Fld9243) OVER(PARTITION BY IncidentNumber, PretensionNumber) as PRETENSION_VALUE_PENALTY
    		 ,ROW_NUMBER() OVER(PARTITION BY IncidentNumber, PretensionNumber ORDER BY PretensionNumber) as rn
    		 ,pai.Amount AS SURCHARGE_VALUE_PENALTY
    	 FROM [OISUU_report].[dbo].[oisuu81_t_IncidentToLoss] as itl
    	 left join [OISUU_report].[dbo].[oisuu81_t_Pretensions] as p on p.LossID=itl.LossID
    	 left join oisuu81.dbo._Document6169_VT9241 vt on vt._Document6169_IDRRef=p.PretensionID
    	 left join [OISUU_report].[dbo].[oisuu81_vLossPaimentsTable] as pai on pai.LossID=p.LinkedLossID
    	 WHERE _Fld9244RRef in (0xB6B5441EA172DD2611E8AC27427E4644, 0xB6B5441EA172DD2611E8AC282FFD5C5A)
    	 and BudgetLine = 'Неустойка/пеня'
    )
     SELECT *
     FROM tmp as tmp
     --left join tmp_1 as tmp_1 on tmp.PretensionNumber=tmp_1.PretensionNumber
     WHERE rn = 1
     union select * from penalty where rn = 1
    """
    df_pretensions_3 = pd.read_sql_query(query, conn)

    df_pretensions_3 = checkpoint(
        df_pretensions_3,
        paths,
        paths.raw_dir,
        "df_pretensions_3.parquet",
        save=save_checkpoint,
    )

    df_pretensions_3.columns = df_pretensions_3.columns.str.upper()
    df_pretensions_3 = df_pretensions_3.rename(columns=RENAME_DICT)
    df_pretensions_3 = df_pretensions_3.sort_values(['PRETENSION_NUMBER', 'SURCHARGE_VALUE_PENALTY'])

    df_pretensions_3 = df_pretensions_3.drop_duplicates(['PRETENSION_NUMBER'], keep='last')
    df_pretensions_3.duplicated('PRETENSION_NUMBER').sum()

    df_pretensions = df_pretensions.merge(df_pretensions_3[['PRETENSION_NUMBER',
                                                            'PRETENSION_VALUE_PENALTY',
                                                            'SURCHARGE_VALUE_PENALTY']], how='left', on='PRETENSION_NUMBER')
    df_pretensions['PRETENSION_VALUE_PENALTY'] = df_pretensions['PRETENSION_VALUE_PENALTY'].fillna(0)
    df_pretensions['SURCHARGE_VALUE_PENALTY'] = df_pretensions['SURCHARGE_VALUE_PENALTY'].fillna(0)


    # Применение функции к колонке DataFrame
    df_pretensions['IS_MARKED'] = df_pretensions['IS_MARKED'].apply(convert_to_binary)
    df_pretensions['IS_OVER'] = df_pretensions['IS_OVER'].apply(convert_to_binary)
    df_pretensions['CESSION'] = df_pretensions['CESSION'].apply(convert_to_binary)
    df_pretensions['HAVE_REQUISITES_OF_APPLICANT'] = df_pretensions['HAVE_REQUISITES_OF_APPLICANT'].apply(convert_to_binary)
    df_pretensions['REQUIRED_REVIEWS_NEO'] = df_pretensions['REQUIRED_REVIEWS_NEO'].apply(convert_to_binary)
    df_pretensions['IS_FULL_PRETENSION_AMOUNTS_WITH_BREAK_DOWN'] = df_pretensions['IS_FULL_PRETENSION_AMOUNTS_WITH_BREAK_DOWN'].apply(convert_to_binary)

    df_pretensions = df_pretensions[df_pretensions['IS_MARKED'] == 0]

    df_pretensions = df_pretensions.sort_values(by=['INCIDENT_NUMBER', 'PRETENSION_GET_DATE'])

    list_PRETENSION_TYPES = list(df_pretensions['PRETENSION_TYPES'].value_counts().reset_index()[:9]['PRETENSION_TYPES'].unique())
    df_pretensions['PRETENSION_TYPES_'] = df_pretensions['PRETENSION_TYPES'].apply(lambda x: x if x in list_PRETENSION_TYPES else 'ПРОЧЕЕ')
    df_pretensions['INSURANCE_TYPE_GROUPS_'] = df_pretensions['INSURANCE_TYPE_GROUPS'].apply(lambda x: x if x in ['ОСАГО','КАСКО+ГО'] else 'ПРОЧЕЕ')

    map_ANSWER_TYPE = {
    'Уведомление о доплате':'Частичная выплата',
    'Согласован дополнительный объем ремонта' :'Выплата',
    'Уведомление о проделанной работе': 'Направлен ответ',
    'Приглашение в офис'  : 'Направлен ответ',
    'Отказ повторный': 'Отказ в удовлетворении претензии',
    'Отказ, нет реквизитов': 'Отказ в удовлетворении претензии', 
    'Направлен на ремонт' :'Выплата',
    'Приглашение на ремонт' :'Выплата'
    
    }
    df_pretensions['PRETENSION_GET_METHOD_'] = df_pretensions['PRETENSION_GET_METHOD'].apply(lambda x: 'ПРОЧЕЕ' if x in ['Иное','Партнер'] else x)
    df_pretensions['ANSWER_TYPE_'] = df_pretensions['ANSWER_TYPE'].replace(map_ANSWER_TYPE)

    #ohe_enc = OneHotEncoder(drop='first', handle_unknown='ignore')
    ohe_enc = OneHotEncoder( handle_unknown='ignore')
    cat_features = ['INSURANCE_TYPE_GROUPS_','PRETENSION_TYPES_', 'PRETENSION_GET_METHOD_', 'ANSWER_TYPE_']
    ohe_enc.fit(df_pretensions[cat_features])
    enc = ohe_enc.transform(df_pretensions[cat_features]).toarray()
    enc = pd.DataFrame(enc, columns=ohe_enc.get_feature_names_out())

    df_pretensions['PRETENSION_CUMCOUNT'] = df_pretensions.groupby(['INCIDENT_NUMBER'])['PRETENSION_NUMBER'].cumcount() + 1

    df_pretensions['PRETENSION_VALUE_CUMSUM'] = df_pretensions.groupby(['INCIDENT_NUMBER'])['PRETENSION_VALUE'].cumsum()

    df_pretensions['SURCHARGE_VALUE_CUMSUM'] = df_pretensions.groupby(['INCIDENT_NUMBER'])['SURCHARGE_VALUE'].cumsum()

    df_pretensions['PRETENSION_VALUE_PENALTY_CUMSUM'] = df_pretensions.groupby(['INCIDENT_NUMBER'])['PRETENSION_VALUE_PENALTY'].cumsum()

    df_pretensions['SURCHARGE_VALUE_PENALTY_CUMSUM'] = df_pretensions.groupby(['INCIDENT_NUMBER'])['SURCHARGE_VALUE_PENALTY'].cumsum()

    df_pretensions['UTS_VALUE_CUMSUM'] = df_pretensions.groupby(['INCIDENT_NUMBER'])['UTS_VALUE'].cumsum()

    df_pretensions['UTS_SURCHARGE_VALUE_CUMSUM'] = df_pretensions.groupby(['INCIDENT_NUMBER'])['UTS_SURCHARGE_VALUE'].cumsum()

    df_pretensions['HAVE_REQUISITES_OF_APPLICANT_MAX'] = df_pretensions.groupby(['INCIDENT_NUMBER'])['HAVE_REQUISITES_OF_APPLICANT'].transform(max)
    df_pretensions['REQUIRED_REVIEWS_NEO_MAX'] = df_pretensions.groupby(['INCIDENT_NUMBER'])['REQUIRED_REVIEWS_NEO'].transform(max)
    df_pretensions['IS_FULL_PRETENSION_AMOUNTS_WITH_BREAK_DOWN_MAX'] = df_pretensions.groupby(['INCIDENT_NUMBER'])['IS_FULL_PRETENSION_AMOUNTS_WITH_BREAK_DOWN'].transform(max)
    df_pretensions['CESSION_MAX'] = df_pretensions.groupby(['INCIDENT_NUMBER'])['CESSION'].transform(max)

    df_pretensions = pd.concat([df_pretensions.reset_index(drop=True), enc.reset_index(drop=True)], axis=1)

    df_pretensions.loc[(df_pretensions['PRETENSION_TYPES'] == 'Требование по выплате только неустойки') &
                       (df_pretensions['PRETENSION_VALUE_PENALTY'] == 0) &
                       (df_pretensions['PRETENSION_VALUE_PENALTY'] != df_pretensions['PRETENSION_VALUE']), \
                       'PRETENSION_VALUE_PENALTY'] = df_pretensions.loc[:, 'PRETENSION_VALUE']

    df_pretensions = checkpoint(
        df_pretensions,
        paths,
        paths.processed_dir,
        "df_pretensions_enriched.parquet",
        save=save_checkpoint,
    )
    return df_pretensions, pretension_fio_id
