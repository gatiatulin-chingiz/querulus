"""Шаг пайплайна: pretensions.

LEGACY (Litigant): не вызывается при include_enrich=False.
SQL и обогащение претензий — справочник для будущего feature engineering (as-of T0).
"""
from __future__ import annotations

import logging

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from querulus.dataset.constants import RENAME_DICT
from querulus.dataset.load.io import checkpoint
from querulus.dataset.load.pretensions import (
    fetch_pretension_fio_ids,
    fetch_pretensions_base,
    fetch_pretensions_penalty,
)
from querulus.dataset.paths import DataPaths
from querulus.dataset.preprocess.pretension import dedupe_pretension_rows
from querulus.dataset.utils import convert_to_binary, hex_upper

logger = logging.getLogger("querulus.dataset")


def load_pretensions(paths: DataPaths, conn, *, use_sql: bool = False, save_checkpoint: bool = True):
    df_pretensions = fetch_pretensions_base(
        paths, conn, use_sql=use_sql, save_checkpoint=save_checkpoint
    )

    df_pretensions = df_pretensions.loc[:, ~df_pretensions.columns.duplicated()].copy()
    df_pretensions = dedupe_pretension_rows(df_pretensions)

    pretension_fio_id = fetch_pretension_fio_ids(
        paths, conn, use_sql=use_sql, save_checkpoint=save_checkpoint
    )

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

    # Добавим id в основной датасет по претензиям.
    df_pretensions = df_pretensions.merge(pretension_fio_id[['PRETENSION_NUMBER','VICTIM_POLICYHOLDER_PERSON_ID','VICTIM_OBJECT_OWNER_PERSON_ID']],how='left',on='PRETENSION_NUMBER')

    df_pretensions_3 = fetch_pretensions_penalty(
        paths, conn, use_sql=use_sql, save_checkpoint=save_checkpoint
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

    df_pretensions['HAVE_REQUISITES_OF_APPLICANT_MAX'] = df_pretensions.groupby(['INCIDENT_NUMBER'])['HAVE_REQUISITES_OF_APPLICANT'].transform('max')
    df_pretensions['REQUIRED_REVIEWS_NEO_MAX'] = df_pretensions.groupby(['INCIDENT_NUMBER'])['REQUIRED_REVIEWS_NEO'].transform('max')
    df_pretensions['IS_FULL_PRETENSION_AMOUNTS_WITH_BREAK_DOWN_MAX'] = df_pretensions.groupby(['INCIDENT_NUMBER'])['IS_FULL_PRETENSION_AMOUNTS_WITH_BREAK_DOWN'].transform('max')
    df_pretensions['CESSION_MAX'] = df_pretensions.groupby(['INCIDENT_NUMBER'])['CESSION'].transform('max')

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
