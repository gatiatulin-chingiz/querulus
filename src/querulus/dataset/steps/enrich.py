"""Шаг пайплайна: enrich.

LEGACY (Litigant): не вызывается при include_enrich=False.
Колонки *_FTRS_* содержат агрегаты претензий/судов и дают утечку таргета ПСР.
Модуль сохранён как справочник SQL/агрегаций для будущего feature engineering (as-of T0).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from querulus.dataset.io import checkpoint
from querulus.dataset.paths import DataPaths
from querulus.dataset.utils import my_mode


def enrich_dataset(paths: DataPaths, df_victim, df_claims, df_claims_, df_claims_payments, df_pretensions, df_claims_persons, pretension_fio_id, *, save_checkpoint: bool = True):
    df = df_victim
    df = checkpoint(
        df,
        paths,
        paths.local_data_dir,
        "df_pre_final.parquet",
        save=save_checkpoint,
    )
    df_claims_payments = checkpoint(
        df_claims_payments,
        paths,
        paths.local_data_dir,
        "df_claims_pre_final.parquet",
        save=save_checkpoint,
    )
    df_pretensions = checkpoint(
        df_pretensions,
        paths,
        paths.local_data_dir,
        "df_pretensions_pre_final.parquet",
        save=save_checkpoint,
    )
    # df_claims_pre_final.parquet хранит payments (legacy-имя из тетрадки)
    df_claims = df_claims_payments.copy()

    #агрегаты по заявителю убытка 
    #присоединяем претензии где ФИО Заявитель убытка  выступал в качестве заявителя претензии (Applicant)
    df_applicant_agg = df.sort_values(['INCIDENT_NUMBER','LOSS_DATE_TIME'])[['APPLICANT_ID','INCIDENT_NUMBER','LOSS_DATE_TIME']]\
                         .drop_duplicates(subset=['INCIDENT_NUMBER'],keep='first')\
                         .merge(df_pretensions[df_pretensions['APPLICANT_PERSON_ID']!='00000000000000000000000000000000']\
                         .drop(columns=['INCIDENT_NUMBER'],axis=1),how='left',left_on='APPLICANT_ID',right_on='APPLICANT_PERSON_ID')

    df_applicant_agg = df_applicant_agg[df_applicant_agg['LOSS_DATE_TIME']>=df_applicant_agg['PRETENSION_GET_DATE']]


    sum_cols = ['PRETENSION_VALUE','UTS_VALUE','SURCHARGE_VALUE','UTS_SURCHARGE_VALUE','CESSION','PRETENSION_VALUE_PENALTY','SURCHARGE_VALUE_PENALTY',
                "INSURANCE_TYPE_GROUPS__КАСКО+ГО",
                "INSURANCE_TYPE_GROUPS__ОСАГО",
                "INSURANCE_TYPE_GROUPS__ПРОЧЕЕ",

                 'PRETENSION_TYPES__Жалоба по ОСАГО 5+',
                 'PRETENSION_TYPES__Запрос документов по делу',
                 'PRETENSION_TYPES__Несогласие с суммой выплаты',
                 'PRETENSION_TYPES__Отказ от ремонта (смена формы возмещения)',
                 'PRETENSION_TYPES__ПРОЧЕЕ',
                 'PRETENSION_TYPES__Претензия на принятое решение',
                 'PRETENSION_TYPES__Претензия на сроки ремонта и согласования',
                 'PRETENSION_TYPES__Претензия по качеству ремонта',
                 'PRETENSION_TYPES__Претензия по смене СТОА',
                 'PRETENSION_TYPES__Требование по выплате только неустойки',
            
                 'PRETENSION_GET_METHOD__Интернет',
                 'PRETENSION_GET_METHOD__Лично',
                 'PRETENSION_GET_METHOD__ПРОЧЕЕ',
                 'PRETENSION_GET_METHOD__Почта',
                 'PRETENSION_GET_METHOD__Электронное',
                 'PRETENSION_GET_METHOD__None',
            
                 'ANSWER_TYPE__Выплата',
                 'ANSWER_TYPE__Направлен ответ',
                 'ANSWER_TYPE__Направлены документы',
                 'ANSWER_TYPE__Отказ в удовлетворении претензии',
                 'ANSWER_TYPE__Частичная выплата',
                 'ANSWER_TYPE__None'
               ]
    mode_cols = ['PRETENSIONTYPE','PRETENSION_GET_METHOD','LOSS_UNIT','LOSS_UNIT_ZONE','ANSWER_TYPE','PRETENSION_TYPES','PRETENSION_KINDS','INSURANCE_TYPE_GROUPS']
    agg_dict = {}
    nunique_col = ['PRETENSION_NUMBER','PRETENSION_TYPES','PRETENSION_KINDS','INSURANCE_TYPE_GROUPS']

    agg_dict = {}

    def my_mode(series):
        m = series.mode()
        return m.iloc[0] if not m.empty else np.nan  # или np.nan, если хотите

    for col in sum_cols:
        agg_dict[f"{col}_sum"] = (col, 'sum')

    for col in nunique_col:
        agg_dict[f"{col}_nunique"] = (col, 'nunique')
    for col in mode_cols:
        agg_dict[f"{col}_unique_list"] = (col, lambda x: list(set(x.dropna())))

    df_applicant_agg = df_applicant_agg.groupby(['APPLICANT_ID','INCIDENT_NUMBER']).agg(**agg_dict).reset_index()

    df_pretensions = df_pretensions.merge(pretension_fio_id[['PRETENSION_NUMBER','VICTIM_PERSON_ID']],how='left',on='PRETENSION_NUMBER')


    import numpy as np

    df_pretensions['VICTIM_POLICYHOLDER_PERSON_ID'] = np.where(
        df_pretensions['VICTIM_POLICYHOLDER_PERSON_ID'] == '00000000000000000000000000000000',
        df_pretensions['VICTIM_OBJECT_OWNER_PERSON_ID'],
        df_pretensions['VICTIM_POLICYHOLDER_PERSON_ID']
    )
    df_pretensions['VICTIM_POLICYHOLDER_PERSON_ID'] = np.where(
        df_pretensions['VICTIM_POLICYHOLDER_PERSON_ID'] == '00000000000000000000000000000000',
        df_pretensions['VICTIM_PERSON_ID'],
        df_pretensions['VICTIM_POLICYHOLDER_PERSON_ID']
    )

    #агрегируем для заявителя инфо по претензиям где он был страхователем, исключая те случаи где в претензии страховтель = заявителю, чтобы не учесть дважды
    df_applicant_agg_2 = df.sort_values(['INCIDENT_NUMBER','LOSS_DATE_TIME'])[['APPLICANT_ID','INCIDENT_NUMBER','LOSS_DATE_TIME']]\
                        .drop_duplicates(subset=['INCIDENT_NUMBER'],keep='first')\
                        .merge(df_pretensions[df_pretensions['VICTIM_POLICYHOLDER_PERSON_ID']!='00000000000000000000000000000000']\
                               .drop(columns=['INCIDENT_NUMBER'],axis=1),how='left',left_on='APPLICANT_ID',right_on='VICTIM_POLICYHOLDER_PERSON_ID')

    df_applicant_agg_2 = df_applicant_agg_2[df_applicant_agg_2['VICTIM_POLICYHOLDER_PERSON_ID']!=df_applicant_agg_2['APPLICANT_PERSON_ID']]
    df_applicant_agg_2 = df_applicant_agg_2[df_applicant_agg_2['LOSS_DATE_TIME']>=df_applicant_agg_2['PRETENSION_GET_DATE']]
    df_applicant_agg_2 = df_applicant_agg_2.groupby(['APPLICANT_ID','INCIDENT_NUMBER']).agg(**agg_dict).reset_index()

    # соединяем все агрегированные датасеты с инфо по заявителю
    df_applicant_agg_ = df_applicant_agg.merge(df_applicant_agg_2,on = ['APPLICANT_ID','INCIDENT_NUMBER'],how='outer')


    col_for_agg = nunique_col + sum_cols
    for col in col_for_agg:
        if col in sum_cols:
            df_applicant_agg_[f'APPLICANT_FTRS_PRET_{col}_SUM'] = df_applicant_agg_[f'{col}_sum_x'].fillna(0) + df_applicant_agg_[f'{col}_sum_y'].fillna(0)
            df_applicant_agg_ = df_applicant_agg_.drop(columns=[f'{col}_sum_x',f'{col}_sum_y'],axis=1)
        if col in nunique_col:
            df_applicant_agg_[f'APPLICANT_FTRS_PRET_{col}_nunique'] =  df_applicant_agg_[f'{col}_nunique_x'].fillna(0) + df_applicant_agg_[f'{col}_nunique_y'].fillna(0)
            df_applicant_agg_ = df_applicant_agg_.drop(columns=[f'{col}_nunique_x',f'{col}_nunique_y'],axis=1)
    # агрегируем моду
    mode_cols_result = {}

    for col in mode_cols:
        # получаем три столбца из слипания -- без суффикса, _x и _y
        colnames = [ f"{col}_unique_list_x", f"{col}_unique_list_y"]
        def get_mode(row):
            # row - pandas.Series из трех элементов, каждый - список (или nan)
            # собираем в один список все значения
            values = []
            for val in row:
                if isinstance(val, list):
                    values.extend(val)
            if values:
                return Counter(values).most_common(1)[0][0]
            else:
                return None

        # применяем по строкам
        df_applicant_agg_[f'APPLICANT_FTRS_PRET_{col}_mode'] = df_applicant_agg_[colnames].apply(get_mode, axis=1)
        df_applicant_agg_ = df_applicant_agg_.drop(columns=colnames, axis=1)

    df_applicant_agg_[df_applicant_agg_['INCIDENT_NUMBER']==7996344.0]


    # добавляем инфо по претензиям заявителя первичного убытка в основной датасет
    df = df.merge(df_applicant_agg_,how='left',on=['INCIDENT_NUMBER','APPLICANT_ID'])
    df.shape

    list_for_fillna = list(df_applicant_agg_.drop(columns=['APPLICANT_ID','INCIDENT_NUMBER']).columns)
    list_for_fillna
    df[list_for_fillna] = df[list_for_fillna].fillna(0)


    #агрегаты по страхователю убытка 
    #присоединяем претензии где Страхователь  выступал в качестве заявителя претензии (Applicant)
    df_applicant_agg = df.sort_values(['INCIDENT_NUMBER','LOSS_DATE_TIME'])[['VICTIM_POLICYHOLDER_PERSON_ID','INCIDENT_NUMBER','LOSS_DATE_TIME']]\
                        .drop_duplicates(subset=['INCIDENT_NUMBER'],keep='first')\
                        .merge(df_pretensions[df_pretensions['APPLICANT_PERSON_ID']!='00000000000000000000000000000000']\
                               .drop(columns=['INCIDENT_NUMBER', 'VICTIM_POLICYHOLDER_PERSON_ID'],axis=1),how='left',left_on='VICTIM_POLICYHOLDER_PERSON_ID',right_on='APPLICANT_PERSON_ID')

    df_applicant_agg = df_applicant_agg[df_applicant_agg['LOSS_DATE_TIME']>=df_applicant_agg['PRETENSION_GET_DATE']]



    sum_cols = ['PRETENSION_VALUE','UTS_VALUE','SURCHARGE_VALUE','UTS_SURCHARGE_VALUE','CESSION','PRETENSION_VALUE_PENALTY','SURCHARGE_VALUE_PENALTY',
               'INSURANCE_TYPE_GROUPS__КАСКО+ГО',
                'INSURANCE_TYPE_GROUPS__ОСАГО',
                'INSURANCE_TYPE_GROUPS__ПРОЧЕЕ',
                'PRETENSION_TYPES__Жалоба по ОСАГО 5+',
                'PRETENSION_TYPES__Запрос документов по делу',
                'PRETENSION_TYPES__Несогласие с суммой выплаты',
                'PRETENSION_TYPES__Отказ от ремонта (смена формы возмещения)',
                'PRETENSION_TYPES__ПРОЧЕЕ',
                'PRETENSION_TYPES__Претензия на принятое решение',
                'PRETENSION_TYPES__Претензия на сроки ремонта и согласования',
                'PRETENSION_TYPES__Претензия по качеству ремонта',
                'PRETENSION_TYPES__Претензия по смене СТОА',
                'PRETENSION_TYPES__Требование по выплате только неустойки',
                'PRETENSION_GET_METHOD__Интернет',
                'PRETENSION_GET_METHOD__Лично',
                'PRETENSION_GET_METHOD__ПРОЧЕЕ',
                'PRETENSION_GET_METHOD__Почта',
                'PRETENSION_GET_METHOD__Электронное',
                'PRETENSION_GET_METHOD__None',
                'ANSWER_TYPE__Выплата',
                'ANSWER_TYPE__Направлен ответ',
                'ANSWER_TYPE__Направлены документы',
                'ANSWER_TYPE__Отказ в удовлетворении претензии',
                'ANSWER_TYPE__Частичная выплата',
                'ANSWER_TYPE__None'
               ]
    mode_cols = ['PRETENSIONTYPE','PRETENSION_GET_METHOD','LOSS_UNIT','LOSS_UNIT_ZONE','ANSWER_TYPE','PRETENSION_TYPES','PRETENSION_KINDS','INSURANCE_TYPE_GROUPS']
    agg_dict = {}
    nunique_col = ['PRETENSION_NUMBER','PRETENSION_TYPES','PRETENSION_KINDS','INSURANCE_TYPE_GROUPS']

    agg_dict = {}

    def my_mode(series):
        m = series.mode()
        return m.iloc[0] if not m.empty else np.nan  # или np.nan, если хотите

    for col in sum_cols:
        agg_dict[f"{col}_sum"] = (col, 'sum')
    for col in nunique_col:
        agg_dict[f"{col}_nunique"] = (col, 'nunique')
    for col in mode_cols:
        agg_dict[f"{col}_unique_list"] = (col, lambda x: list(set(x.dropna())))

    df_applicant_agg = df_applicant_agg.groupby(['VICTIM_POLICYHOLDER_PERSON_ID','INCIDENT_NUMBER']).agg(**agg_dict).reset_index()


    #агрегируем для заявителя инфо по претензиям где он был страхователем, исключая те случаи где в претензии страховтель = заявителю, чтобы не учесть дважды
    df_applicant_agg_2 = df.sort_values(['INCIDENT_NUMBER','LOSS_DATE_TIME'])[['VICTIM_POLICYHOLDER_PERSON_ID','INCIDENT_NUMBER','LOSS_DATE_TIME']]\
                        .drop_duplicates(subset=['INCIDENT_NUMBER'],keep='first')\
                        .merge(df_pretensions[df_pretensions['VICTIM_POLICYHOLDER_PERSON_ID']!='00000000000000000000000000000000']\
                               .drop(columns=['INCIDENT_NUMBER'],axis=1),how='left',on='VICTIM_POLICYHOLDER_PERSON_ID')

    df_applicant_agg_2 = df_applicant_agg_2[df_applicant_agg_2['VICTIM_POLICYHOLDER_PERSON_ID']!=df_applicant_agg_2['APPLICANT_PERSON_ID']]
    df_applicant_agg_2 = df_applicant_agg_2[df_applicant_agg_2['LOSS_DATE_TIME']>=df_applicant_agg_2['PRETENSION_GET_DATE']]
    df_applicant_agg_2 = df_applicant_agg_2.groupby(['VICTIM_POLICYHOLDER_PERSON_ID','INCIDENT_NUMBER']).agg(**agg_dict).reset_index()


    # соединяем все агрегированные датасеты с инфо по заявителю
    df_applicant_agg_ = df_applicant_agg.merge(df_applicant_agg_2,on = ['VICTIM_POLICYHOLDER_PERSON_ID','INCIDENT_NUMBER'],how='outer')


    col_for_agg = nunique_col + sum_cols
    for col in col_for_agg:
        if col in sum_cols:
            df_applicant_agg_[f'VICTIM_PH_FTRS_PRET_{col}_SUM'] =  df_applicant_agg_[f'{col}_sum_x'].fillna(0) + df_applicant_agg_[f'{col}_sum_y'].fillna(0)
            df_applicant_agg_ = df_applicant_agg_.drop(columns=[f'{col}_sum_x',f'{col}_sum_y'],axis=1)
        if col in nunique_col:
            df_applicant_agg_[f'VICTIM_PH_FTRS_PRET_{col}_nunique'] =  df_applicant_agg_[f'{col}_nunique_x'].fillna(0) + df_applicant_agg_[f'{col}_nunique_y'].fillna(0)
            df_applicant_agg_ = df_applicant_agg_.drop(columns=[f'{col}_nunique_x',f'{col}_nunique_y'],axis=1)
    # агрегируем моду
    mode_cols_result = {}

    for col in mode_cols:
        # получаем три столбца из слипания  _x и _y
        colnames = [ f"{col}_unique_list_x", f"{col}_unique_list_y"]
        def get_mode(row):
            # row - pandas.Series из трех элементов, каждый - список (или nan)
            # собираем в один список все значения
            values = []
            for val in row:
                if isinstance(val, list):
                    values.extend(val)
            if values:
                return Counter(values).most_common(1)[0][0]
            else:
                return None

        # применяем по строкам
        df_applicant_agg_[f'VICTIM_PH_FTRS_PRET_{col}_mode'] = df_applicant_agg_[colnames].apply(get_mode, axis=1)
        df_applicant_agg_ = df_applicant_agg_.drop(columns=colnames, axis=1)

    df_applicant_agg_[:2]

    # добавляем инфо по претензиям заявителя первичного убытка в основной датасет
    df = df.merge(df_applicant_agg_,how='left',on=['INCIDENT_NUMBER','VICTIM_POLICYHOLDER_PERSON_ID'])
    df.shape

    list_for_fillna = list(df_applicant_agg_.drop(columns=['VICTIM_POLICYHOLDER_PERSON_ID','INCIDENT_NUMBER']).columns)
    df[list_for_fillna] = df[list_for_fillna].fillna(0)

    df_claims_persons['Лицо'] = df_claims_persons['Лицо'].apply(lambda x: x.hex().upper() if x is not None  else np.nan)
    df_claims_persons['ПолноеФИОЛица'] = df_claims_persons['ПолноеФИОЛица'].str.upper()
    df_claims_persons[:2]

    #удаляем дубликаты
    df_claims_persons = df_claims_persons.drop_duplicates()
    df_claims_persons = df_claims_persons.sort_values(by=['Представитель','Цессионарий']).drop_duplicates(subset=['ПолноеФИОЛица','НомерИск'],keep='first')
    len(df_claims_persons)

    # добавляем инфо , что по судам
    df_claims = df_claims_persons.rename(columns={'НомерИск':'INCOMING_CLAIM_NUMBER'})\
                .merge(df_claims_.drop(columns=['INCIDENT_NUMBER','LOSS_ID','LOSS_NUMBER','INCOMINGCLAIMID']),how='left',on='INCOMING_CLAIM_NUMBER')


    #агрегаты по заявителю убытка , suffixes=('', '_pret')
    df_applicant_agg = df.sort_values(['INCIDENT_NUMBER','LOSS_DATE_TIME'])[['APPLICANT_ID','INCIDENT_NUMBER','LOSS_DATE_TIME']]\
                        .drop_duplicates(subset=['INCIDENT_NUMBER'],keep='first')\
                        .merge(df_claims[df_claims['Лицо']!='00000000000000000000000000000000'],how='left',left_on='APPLICANT_ID',right_on='Лицо')

    df_applicant_agg = df_applicant_agg[df_applicant_agg['LOSS_DATE_TIME']>=df_applicant_agg['INCOMING_CLAIM_GET_DATE_1']]

    df_applicant_agg.columns = df_applicant_agg.columns.str.replace(r'[\s\-]+', '_', regex=True)

    sum_cols = [
        'Представитель',
        'Цессионарий',
        'CLAIMEDMAINDEBT_1',
        'CLAIMEDPLAINTIFFEXAMINATION_1',
        'CLAIMEDCOURTEXAMINATION_1',
        'CLAIMEDREPRESENTATIVEEXPENSES_1',
        'CLAIMEDPERCENTFORUSES_1',
        'CLAIMEDPENALTYFEE_1',
        'CLAIMEDFINE_1',
        'CLAIMEDMORALDAMAGE_1',
        'CLAIMEDOTHEREXPENSES_1',
        'CLAIMEDSTATEDUTY_1',
        'CLAIMEDLOSSCOMMODYVALUE_1',
        'CLAIMEDWEAROUT_1',
        'CLAIMEDVALUEWITHOUTSD_1',
        'CLAIMEDVALUEWITHSD_1',
        'CLAIMEDAMOUNTLOSS_1',
        'CLAIMEDCOURTEXAMINATIONBEFORERESOLVE_1',
        'CLAIM_ITEM_3_Е_ЛИЦО',
        'CLAIM_ITEM_НЕ_ПРЕДОСТАВЛЕНИЕ_НА_ОСМОТР',
        'CLAIM_ITEM_НЕКОМПЛЕКТНОСТЬ_ДОКУМЕНТОВ',
        'CLAIM_ITEM_НЕПРИЗНАНИЕ_СТРАХОВЫМ_СЛУЧАЕМ',
        'CLAIM_ITEM_НЕУСТОЙКА',
        'CLAIM_ITEM_ПРОЧЕЕ',
        'CLAIM_ITEM_СПОР_ПО_СУММЕ',
        'CLAIM_ITEM_СУДЕБНЫЕ_РАСХОДЫ_И_САНКЦИИ_ЗА_НЕИСПОЛНЕНИЕ_РЕШЕНИЯ_СУДА',
        'CLAIM_ITEM_ТРАСОЛОГИЯ',
        'CLAIM_ITEM_ФОРМА_ВОЗМЕЩЕНИЯ']
    mode_cols = ['ОбращениеКФУОтЗаявителяПоступилоПосредством', 'CLAIM_ITEM','CLAIMORIGIN_1' ]

    nunique_col = ['INCOMING_CLAIM_NUMBER','LINK_LOSS_NUMBER']
    mean_col = ['CLAIMEDMAINDEBT_1',
            'CLAIMEDPLAINTIFFEXAMINATION_1',
            'CLAIMEDCOURTEXAMINATION_1',
            'CLAIMEDREPRESENTATIVEEXPENSES_1',
            'CLAIMEDPERCENTFORUSES_1',
            'CLAIMEDPENALTYFEE_1',
            'CLAIMEDFINE_1',
            'CLAIMEDMORALDAMAGE_1',
            'CLAIMEDOTHEREXPENSES_1', 
            'CLAIMEDSTATEDUTY_1',
            'CLAIMEDLOSSCOMMODYVALUE_1',
            'CLAIMEDWEAROUT_1',
            'CLAIMEDVALUEWITHOUTSD_1',
            'CLAIMEDVALUEWITHSD_1',
            'CLAIMEDAMOUNTLOSS_1', 
            'CLAIMEDCOURTEXAMINATIONBEFORERESOLVE_1']
    agg_dict = {}

    def my_mode(series):
        m = series.mode()
        return m.iloc[0] if not m.empty else np.nan  # или np.nan, если хотите

    for col in sum_cols:
        agg_dict[f'APPLICANT_FTRS_COURT_{col}_SUM'] = (col, 'sum')
    for col in mean_col:
        agg_dict[f'APPLICANT_FTRS_COURT_{col}_MEAN'] = (col, 'mean')
    # for col in count_cols:
    #     agg_dict[f"{col}_count"] = (col, 'count')
    for col in nunique_col:
        agg_dict[f'APPLICANT_FTRS_COURT_{col}_NUNIQUE'] = (col, 'nunique')
    for col in mode_cols:
        agg_dict[f'APPLICANT_FTRS_COURT_{col}_MODE'] = (col, my_mode)


    df_applicant_agg = df_applicant_agg.groupby(['APPLICANT_ID','INCIDENT_NUMBER']).agg(**agg_dict).reset_index()

    # добавляем инфо по судам заявителя первичного убытка в основной датасет
    df = df.merge(df_applicant_agg,how='left',on=['INCIDENT_NUMBER','APPLICANT_ID'])
    df.shape

    list_for_fillna = list(df_applicant_agg.drop(columns=['APPLICANT_ID','INCIDENT_NUMBER']).columns)
    list_for_fillna
    df[list_for_fillna] = df[list_for_fillna].fillna(0)


    #агрегаты по страхователю жертве убытка , suffixes=('', '_pret')
    df_applicant_agg = df.sort_values(['INCIDENT_NUMBER','LOSS_DATE_TIME'])[['VICTIM_POLICYHOLDER_PERSON_ID','INCIDENT_NUMBER','LOSS_DATE_TIME','APPLICANT_ID']]\
                        .drop_duplicates(subset=['INCIDENT_NUMBER'],keep='first')\
                        .merge(df_claims[df_claims['Лицо']!='00000000000000000000000000000000'],how='left',left_on='VICTIM_POLICYHOLDER_PERSON_ID',right_on='Лицо')

    df_applicant_agg = df_applicant_agg[df_applicant_agg['LOSS_DATE_TIME']>=df_applicant_agg['INCOMING_CLAIM_GET_DATE_1']]


    df_applicant_agg.columns = df_applicant_agg.columns.str.replace(r'[\s\-]+', '_', regex=True)

    sum_cols = [
        'Представитель',
        'Цессионарий',
        'CLAIMEDMAINDEBT_1',
        'CLAIMEDPLAINTIFFEXAMINATION_1',
        'CLAIMEDCOURTEXAMINATION_1',
        'CLAIMEDREPRESENTATIVEEXPENSES_1',
        'CLAIMEDPERCENTFORUSES_1',
        'CLAIMEDPENALTYFEE_1',
        'CLAIMEDFINE_1',
        'CLAIMEDMORALDAMAGE_1',
        'CLAIMEDOTHEREXPENSES_1',
        'CLAIMEDSTATEDUTY_1',
        'CLAIMEDLOSSCOMMODYVALUE_1',
        'CLAIMEDWEAROUT_1',
        'CLAIMEDVALUEWITHOUTSD_1',
        'CLAIMEDVALUEWITHSD_1',
        'CLAIMEDAMOUNTLOSS_1',
        'CLAIMEDCOURTEXAMINATIONBEFORERESOLVE_1',
        'CLAIM_ITEM_3_Е_ЛИЦО',
        'CLAIM_ITEM_НЕ_ПРЕДОСТАВЛЕНИЕ_НА_ОСМОТР',
        'CLAIM_ITEM_НЕКОМПЛЕКТНОСТЬ_ДОКУМЕНТОВ',
        'CLAIM_ITEM_НЕПРИЗНАНИЕ_СТРАХОВЫМ_СЛУЧАЕМ',
        'CLAIM_ITEM_НЕУСТОЙКА',
        'CLAIM_ITEM_ПРОЧЕЕ',
        'CLAIM_ITEM_СПОР_ПО_СУММЕ',
        'CLAIM_ITEM_СУДЕБНЫЕ_РАСХОДЫ_И_САНКЦИИ_ЗА_НЕИСПОЛНЕНИЕ_РЕШЕНИЯ_СУДА',
        'CLAIM_ITEM_ТРАСОЛОГИЯ',
        'CLAIM_ITEM_ФОРМА_ВОЗМЕЩЕНИЯ']
    mode_cols = ['ОбращениеКФУОтЗаявителяПоступилоПосредством', 'CLAIM_ITEM','CLAIMORIGIN_1' ]
    nunique_col = ['INCOMING_CLAIM_NUMBER','LINK_LOSS_NUMBER','CLAIM_ITEM']
    mean_col = ['CLAIMEDMAINDEBT_1',
            'CLAIMEDPLAINTIFFEXAMINATION_1',
            'CLAIMEDCOURTEXAMINATION_1',
            'CLAIMEDREPRESENTATIVEEXPENSES_1',
            'CLAIMEDPERCENTFORUSES_1',
            'CLAIMEDPENALTYFEE_1',
            'CLAIMEDFINE_1',
            'CLAIMEDMORALDAMAGE_1',
            'CLAIMEDOTHEREXPENSES_1', 
            'CLAIMEDSTATEDUTY_1',
            'CLAIMEDLOSSCOMMODYVALUE_1',
            'CLAIMEDWEAROUT_1',
            'CLAIMEDVALUEWITHOUTSD_1',
            'CLAIMEDVALUEWITHSD_1',
            'CLAIMEDAMOUNTLOSS_1', 
            'CLAIMEDCOURTEXAMINATIONBEFORERESOLVE_1']

    agg_dict = {}

    def my_mode(series):
        m = series.mode()
        return m.iloc[0] if not m.empty else np.nan  # или np.nan, если хотите

    for col in sum_cols:
        agg_dict[f'VICTIM_PH_FTRS_COURT_{col}_SUM'] = (col, 'sum')
    for col in mean_col:
        agg_dict[f'VICTIM_PH_FTRS_COURT_{col}_MEAN'] = (col, 'mean')
    # for col in count_cols:
    #     agg_dict[f"{col}_count"] = (col, 'count')
    for col in nunique_col:
        agg_dict[f'VICTIM_PH_FTRS_COURT_{col}_NUNIQUE'] = (col, 'nunique')
    for col in mode_cols:
        agg_dict[f'VICTIM_PH_FTRS_COURT_{col}_MODE'] = (col, my_mode)

    df_applicant_agg = df_applicant_agg.groupby(['VICTIM_POLICYHOLDER_PERSON_ID','INCIDENT_NUMBER']).agg(**agg_dict).reset_index()


    # добавляем инфо по судам заявителя первичного убытка в основной датасет
    df = df.merge(df_applicant_agg,how='left',on=['INCIDENT_NUMBER','VICTIM_POLICYHOLDER_PERSON_ID'])
    df.shape


    list_for_fillna = list(df_applicant_agg.drop(columns=['VICTIM_POLICYHOLDER_PERSON_ID','INCIDENT_NUMBER']).columns)
    list_for_fillna
    df[list_for_fillna] = df[list_for_fillna].fillna(0)


    col_mode = []
    for i in df.columns:
        if '_mode' in i or '_MODE' in i:
            col_mode.append(i)
    for c in col_mode:
        df[c] = df[c].replace(0,'nan').astype(str)


    df = checkpoint(
        df,
        paths,
        paths.raw_dir,
        "pre_final.parquet",
        save=save_checkpoint,
    )
    return df
