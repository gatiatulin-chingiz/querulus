"""Константы пайплайна (модели, окна, таблицы Motor).

Сборка runtime-настроек из env — ``pipeline.settings.load_settings``.
"""
from __future__ import annotations


SCHEMA = "agrafana"
TABLE_ERRORS = "Querulus_errors"
TABLE_DRIFT = "Querulus_DataDrift"
TABLE_FIN_EFFECT = "Querulus_FinEffect"

# =============================================================================
# РЕДАКТИРОВАТЬ ПРИ СМЕНЕ МОДЕЛИ / ДАТАСЕТА — ТОЛЬКО ЗДЕСЬ
# =============================================================================
#
# DataDrift:
#   hive, train_filter, pickle/{stem}.pickle (фичи + decode),
#   DEFAULT_DRIFT_LOOKBACK_DAYS, DEFAULT_PSI_THRESHOLD
# FinEffect:
#   hive у slot=main (без train_filter); витрина/PILOT_FILIALS — не здесь
# Errors / alert:
#   DEFAULT_ERRORS_*, DEFAULT_EMAIL_RECEIVERS
#
# main: окно как config_cf_3 / выгрузка X_train ансамбля
# second: окно как config_prod.json (UU 2.0.0) separation.train_period

_FILIALS = (
    "Архангельский",
    "Владимирский",
    "Кемеровский",
    "Курский",
    "Магнитогорский",
    "Марийский",
    "Мурманский",
    "Омский",
    "Пермский",
    "Петропавловск-Камчатский",
    "Уфимский",
    "Ярославский",
)

_TRAIN_FILTER_MAIN = {
    "date_col": "PAYMENT_ORDER_DATE_TIME",
    "date_start": "2022-01-01",
    "date_end": "2024-05-31",
    "filial_col": "FILIAL",
    "filials": _FILIALS,
}

_TRAIN_FILTER_SECOND = {
    "date_col": "PAYMENT_ORDER_DATE_TIME",
    "date_start": "2022-01-01",
    "date_end": "2025-02-09",
    "filial_col": "FILIAL",
    "filials": _FILIALS,
}

DEFAULT_DRIFT_MODELS: tuple[dict, ...] = (
    {
        "slot": "main",
        "stem": "querulus_ansamble_2026_04_18_v1",
        "hive": "models.querulus_ansamble_2026_04_18_v1",
        "train_filter": dict(_TRAIN_FILTER_MAIN),
    },
    {
        "slot": "second",
        "stem": "UU_querulus_2026_11_05_12_44_24",
        "hive": "models.UU_querulus_2026_11_05_12_44_24",
        "train_filter": dict(_TRAIN_FILTER_SECOND),
    },
)

DEFAULT_PSI_THRESHOLD = 0.3
DEFAULT_DRIFT_LOOKBACK_DAYS = 14
DEFAULT_ERRORS_LOOKBACK_DAYS = 14
DEFAULT_ERRORS_MODEL_KINDS: tuple[str, ...] = ("Сутяжность",)

DEFAULT_EMAIL_RECEIVERS = (
    "Gatyatulin@vsk.ru",
)
