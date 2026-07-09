"""Признаки финальных моделей из configs/config_cf_3.json и config_rg_3.json."""

from __future__ import annotations

DEFAULT_FREQUENCY_FEATURES: tuple[str, ...] = (
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

DEFAULT_SEVERITY_FEATURES: tuple[str, ...] = (
    "LOSS_UNIT_ZONE",
    "VICTIM_VEHICLE_COUNTRY",
    "APPLY_DELAY",
    "VALUE_BEFORE_WITHOUT",
)
