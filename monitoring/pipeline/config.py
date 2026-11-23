"""Константы пайплайна (модели, окна, таблицы Motor).

Сборка runtime-настроек из env — ``pipeline.settings.load_settings``.

Имена env (стиль MONITORING_BUU)::

* OISUU: ``database_host``, ``database_name``, ``database_user_name``, ``database_user_pass``
* Motor: ``motor_host``, ``motor_name``, ``motor_user_name``, ``motor_user_pass``
* Oracle: ``SERVER_PROD8``, ``PORT_PROD8``, ``USER_PROD8``, ``PASSWORD_PROD8``, ``SERVICE_NAME_PROD8``
"""
from __future__ import annotations


# --- целевые таблицы Motor.agrafana ------------------------------------------
SCHEMA = "agrafana"
TABLE_ERRORS = "Querulus_errors"
TABLE_DRIFT = "querulus_DataDrift"
TABLE_FIN_EFFECT = "Querulus_FinEffect"

# =============================================================================
# РЕДАКТИРОВАТЬ ПРИ СМЕНЕ МОДЕЛИ / ДАТАСЕТА / ФИЧ — ТОЛЬКО ЗДЕСЬ
# =============================================================================
# DataDrift: 1 Hive на slot (полный train), 1 расчёт на slot.
# features = union(cf, rg); без среза по таргету; track в datadrift.py = "all".

DEFAULT_DRIFT_MODELS: tuple[dict, ...] = (
    {
        "slot": "main",
        "stem": "querulus_ansamble_2026_04_18_v1",
        "hive": "models.querulus_ansamble_2026_04_18_v1",
        # cf_3 ∪ rg_3
        "features": (
            "AMOUNT_REPAIR",
            "APPLICANT_AGE",
            "APPLICANT_FORM",
            "APPLY_DELAY",
            "EVENT_CREATED_BY_GIBDD_FLAG",
            "EVENT_YEAR",
            "FILIAL",
            "GUILTY_CAPACITY_ENGINE",
            "LOSS_UNIT_ZONE",
            "RECIEVE_METHOD",
            "VICTIM_MAX_WEIGHT",
            "VICTIM_VEHICLE_AGE",
            "VICTIM_VEHICLE_CATEGORY",
            "VICTIM_VEHICLE_COUNTRY",
        ),
    },
    {
        "slot": "second",
        "stem": "UU_querulus_2026_11_05_12_44_24",
        "hive": "models.UU_querulus_2026_11_05_12_44_24",
        # cf ∪ rg second (сырые + FE_*)
        "features": (
            "APPLICANT_AGE",
            "APPLICANT_FORM",
            "EVENT_CREATED_BY_GIBDD_FLAG",
            "FE_DAYS_EVENT_TO_LOSS",
            "FE_DAYS_TO_PH_CONTRACT_END",
            "FE_SHARE_WORK_TIER",
            "FE_VALUE_BEFORE_DIFF",
            "FE_VALUE_BEFORE_RATIO",
            "FE_VALUE_BEFORE_WITH_BIN",
            "FE_VALUE_BEFORE_WITH_REAL_2022",
            "FE_VALUE_BEFORE_WITHOUT_REAL_2022",
            "FE_VICTIM_POWER_PER_TON",
            "FILIAL",
            "FL_PHOTO_VIDEO",
            "IS_TOTAL",
            "LOSS_UNIT_TYPE",
            "NOT_NOTIFICATION",
            "PAYMENT_RECIPIENT_TYPE",
            "RECIEVE_METHOD",
            "REGION",
            "VICTIM_AGE",
            "VICTIM_OBJECT_AGE_MONTHS",
            "VICTIM_VEHICLE_AGE",
            "VICTIM_VEHICLE_CATEGORY",
        ),
    },
)

# Опциональный срез train по PAYMENT_ORDER_DATE_TIME; пусто = без среза
DEFAULT_REFERENCE_TRAIN_START = ""
DEFAULT_REFERENCE_TRAIN_END = ""

DEFAULT_PSI_THRESHOLD = 0.3
DEFAULT_DRIFT_LOOKBACK_DAYS = 14
DEFAULT_ERRORS_LOOKBACK_DAYS = 14
# =============================================================================

DEFAULT_EMAIL_RECEIVERS = (
    "Gatyatulin@vsk.ru",
    "AVBondarev@vsk.ru",
    "Matsera@VSK.RU",
)
