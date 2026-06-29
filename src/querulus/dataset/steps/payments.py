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
        SELECT
            ITL.IncidentNumber,
            p.PaymentDateTime,
            p.PaymentValue
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
        columns=["IncidentNumber", "PaymentDateTime", "PaymentValue"],
    )

    df_payments = df_payments.loc[:, ~df_payments.columns.duplicated()].copy()
    df_payments = df_payments.rename(
        columns={
            "IncidentNumber": "INCIDENTNUMBER",
            "PaymentDateTime": "PAYMENTDATETIME",
            "PaymentValue": "PAYMENTVALUE",
        }
    )
    df_payments = df_payments.rename(columns=RENAME_DICT)
    df_payments = df_payments[
        ["INCIDENT_NUMBER", "PAYMENT_DATETIME", "PAYMENT_VALUE"]
    ].copy()
    df_payments["PAYMENT_DATETIME"] = pd.to_datetime(df_payments["PAYMENT_DATETIME"])
    df_payments = df_payments.sort_values(["INCIDENT_NUMBER", "PAYMENT_DATETIME"])
    df_payments["PAYMENT_VALUE_CUMSUM"] = df_payments.groupby(
        ["INCIDENT_NUMBER"], sort=False
    )["PAYMENT_VALUE"].cumsum()
    payment_features = df_payments[
        ["INCIDENT_NUMBER", "PAYMENT_DATETIME", "PAYMENT_VALUE_CUMSUM"]
    ].copy()
    del df_payments
    gc.collect()

    claim_keys = df_claims[
        ["INCOMING_CLAIM_NUMBER", "INCIDENT_NUMBER", "RECOVEREDVALUEPERIOD_1"]
    ].copy()
    claim_keys["RECOVEREDVALUEPERIOD_1"] = pd.to_datetime(
        claim_keys["RECOVEREDVALUEPERIOD_1"]
    )
    # Мержим только ключи, чтобы широкий df_claims не размножался на все платежи.
    df_claims_payment_keys = claim_keys.merge(
        payment_features,
        on="INCIDENT_NUMBER",
        how="left",
    )
    del claim_keys, payment_features
    gc.collect()

    # --- фильтрация и дедупликация ---
    mask = (
        df_claims_payment_keys["PAYMENT_DATETIME"]
        <= df_claims_payment_keys["RECOVEREDVALUEPERIOD_1"]
    ) | df_claims_payment_keys["PAYMENT_DATETIME"].isnull()

    df_claims_payment_keys = df_claims_payment_keys.loc[mask]
    df_claims_payment_keys = df_claims_payment_keys.sort_values(
        ["RECOVEREDVALUEPERIOD_1", "PAYMENT_DATETIME"],
        na_position='first',
    #    kind='mergesort'  # стабильная сортировка, иногда эффективнее по памяти
    )

    # Шаг 3: Удаление дубликатов — используем keep='last'
    df_claims_payment_keys = df_claims_payment_keys.drop_duplicates(
        "INCOMING_CLAIM_NUMBER", keep="last"
    )
    df_claims_payments = df_claims.merge(
        df_claims_payment_keys[
            ["INCOMING_CLAIM_NUMBER", "PAYMENT_DATETIME", "PAYMENT_VALUE_CUMSUM"]
        ],
        on="INCOMING_CLAIM_NUMBER",
        how="left",
    )
    del df_claims_payment_keys
    gc.collect()

    df_claims_payments = checkpoint(
        df_claims_payments,
        paths,
        paths.processed_dir,
        "df_claims_payments.parquet",
        save=save_checkpoint,
    )
    gc.collect()
    return df_claims_payments
