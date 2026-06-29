"""Шаг пайплайна: payments."""
from __future__ import annotations

import gc
import logging

import pandas as pd

from querulus.dataset.constants import RENAME_DICT
from querulus.dataset.io import checkpoint, load_sql_artifact
from querulus.dataset.paths import DataPaths

logger = logging.getLogger("querulus.dataset")


def load_claims_payments(paths: DataPaths, conn, df_claims: pd.DataFrame, *, use_sql: bool = False, save_checkpoint: bool = True):
    df_payments_q = ("""
        SELECT *
        FROM [OISUU_report].[dbo].[oisuu81_t_IncidentToLoss] as ITL
        LEFT JOIN [OISUU_report].[dbo].oisuu81_t_payments AS p on p.LOSSID = ITL.LOSSID
    """)

    df_payments = load_sql_artifact(
        paths,
        conn,
        paths.raw_dir,
        "df_payments.parquet",
        df_payments_q,
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
    )

    df_payments = df_payments.loc[:, ~df_payments.columns.duplicated()].copy()


    column_mapping = {
        'LossID': 'LOSSID',
        'LossNumber': 'LOSSNUMBER',
        'IncidentID': 'INCIDENTID',
        'IncidentNumber': 'INCIDENTNUMBER',
        'PaymentDateTime': 'PAYMENTDATETIME',
        'PaymentNumber': 'PAYMENTNUMBER',
        'PaymentID': 'PAYMENTID',
        'PaymentValue': 'PAYMENTVALUE',
        'DocumentNumber': 'DOCUMENTNUMBER',
        'ValueRub': 'VALUERUB',
        'Comment': 'COL_COMMENT',
        'DocumentOperation': 'DOCUMENTOPERATION',
        'CurrencyCode': 'CURRENCYCODE',
        'Currency': 'CURRENCY',
        'PaymentRecipientPersonName': 'PAYMENTRECIPIENTPERSONNAME',
        'PaymentRecipientPersonCode': 'PAYMENTRECIPIENTPERSONCODE',
        'PaymentRecipientPersonID': 'PAYMENTRECIPIENTPERSONID',
        'IsReasonForPaymentExternalOrder': 'ISREASONFORPAYMENTEXTERNALORDER',
        'IsReasonForPaymentVPRS': 'ISREASONFORPAYMENTVPRS',
        'IsReasonForPaymentInsuranceAct': 'ISREASONFORPAYMENTINSURANCEACT',
        'IsReasonForPaymentCompletedWorkGetAct': 'ISREASONFORPAYMENTCOMPLETEDWORKGETACT',
        'ReasonForPayment': 'REASONFORPAYMENT'
    }

    df_payments.rename(columns=column_mapping, inplace=True)


    df_payments = df_payments.rename(columns=RENAME_DICT)
    df_payments = df_payments.sort_values('PAYMENT_DATETIME')


    df_payments['PAYMENT_VALUE_CUMSUM'] = df_payments.groupby(['INCIDENT_NUMBER'])['PAYMENT_VALUE'].cumsum()


    df_claims_payments = df_claims.merge(df_payments[['INCIDENT_NUMBER',
                                                      'PAYMENT_DATETIME',
                                                      'PAYMENT_VALUE_CUMSUM']],
                                         left_on='INCIDENT_NUMBER',
                                         right_on='INCIDENT_NUMBER',
                                         how='left')
    df_claims_payments

    df_claims_payments = checkpoint(
        df_claims_payments,
        paths,
        paths.processed_dir,
        "df_claims_payments.parquet",
        save=save_checkpoint,
    )

    # --- фильтрация и дедупликация ---
    mask = (df_claims_payments['PAYMENT_DATETIME'] <= df_claims_payments['RECOVEREDVALUEPERIOD_1']) | \
           df_claims_payments['PAYMENT_DATETIME'].isnull()

    # Применяем маску и сразу освобождаем исходный df, если он больше не нужен
    df_claims_payments = df_claims_payments[mask]

    # Шаг 2: Сортировка — делаем только по необходимым столбцам
    # Убедитесь, что столбцы datetime имеют правильный тип (datetime64)
    df_claims_payments['RECOVEREDVALUEPERIOD_1'] = pd.to_datetime(df_claims_payments['RECOVEREDVALUEPERIOD_1'])
    df_claims_payments['PAYMENT_DATETIME'] = pd.to_datetime(df_claims_payments['PAYMENT_DATETIME'])

    df_claims_payments = df_claims_payments.sort_values(
        ['RECOVEREDVALUEPERIOD_1', 'PAYMENT_DATETIME'],
        na_position='first',
    #    kind='mergesort'  # стабильная сортировка, иногда эффективнее по памяти
    )

    # Шаг 3: Удаление дубликатов — используем keep='last'
    df_claims_payments = df_claims_payments.drop_duplicates('INCOMING_CLAIM_NUMBER', keep='last')

    df_claims_payments = checkpoint(
        df_claims_payments,
        paths,
        paths.processed_dir,
        "df_claims_payments.parquet",
        save=save_checkpoint,
    )
    gc.collect()
    return df_claims_payments
