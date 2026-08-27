"""Признаки прод-моделей (config_cf_3 / config_rg_3) и baseline для selected.

``PROD_*`` — фактический прод (для HTML «старая модель»).
``DEFAULT_*`` — пул ``features_source=selected`` (без FS); severity уже
на ``FE_VALUE_BEFORE_WITHOUT_REAL_2022`` вместо номинала / AMOUNT_REPAIR.
"""

from __future__ import annotations

# config_cf_3.json → querulus_cf_3.features
PROD_FREQUENCY_FEATURES: tuple[str, ...] = (
    "EVENT_CREATED_BY_GIBDD_FLAG",
    "FILIAL",
    "VICTIM_VEHICLE_CATEGORY",
    "APPLICANT_FORM",
    "RECIEVE_METHOD",
    "VICTIM_VEHICLE_AGE",
    "VICTIM_MAX_WEIGHT",
    "GUILTY_CAPACITY_ENGINE",
    "APPLICANT_AGE",
    "EVENT_YEAR",
)

# config_rg_3.json → querulus_rg_3.features
PROD_SEVERITY_FEATURES: tuple[str, ...] = (
    "LOSS_UNIT_ZONE",
    "VICTIM_VEHICLE_COUNTRY",
    "APPLY_DELAY",
    "AMOUNT_REPAIR",
)

DEFAULT_FREQUENCY_FEATURES: tuple[str, ...] = PROD_FREQUENCY_FEATURES

DEFAULT_SEVERITY_FEATURES: tuple[str, ...] = (
    "LOSS_UNIT_ZONE",
    "VICTIM_VEHICLE_COUNTRY",
    "APPLY_DELAY",
    "FE_VALUE_BEFORE_WITHOUT_REAL_2022",
)
