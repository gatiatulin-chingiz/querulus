"""Конфигурация feature engineering v1."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from querulus import PROJECT_ROOT

_CONFIG_PATH = PROJECT_ROOT / "configs" / "features_v1.json"


@dataclass(frozen=True)
class FeatureThresholds:
    """Пороги бакетов и флагов."""

    apply_delay_high: int = 30
    apply_delay_notify: int = 7
    vehicle_weight_heavy: float = 3500.0
    amount_repair_high: float = 300_000.0
    amount_repair_bins: tuple[float, ...] = (100_000.0, 300_000.0)
    insurance_amount_bins: tuple[float, ...] = (400_000.0, 1_000_000.0)
    victim_loss_sum_bins: tuple[float, ...] = (50_000.0, 200_000.0)
    kbm_low: float = 1.0
    kbm_mid: float = 1.17
    share_wearout_bins: tuple[float, ...] = (20.0, 50.0)
    share_work_bins: tuple[float, ...] = (0.3, 0.6)


@dataclass(frozen=True)
class FeatureConfig:
    """Параметры этапов 0–1 FE."""

    t0_column: str = "PAYMENT_ORDER_DATE_TIME"
    thresholds: FeatureThresholds = field(default_factory=FeatureThresholds)
    vehicle_age_bins: tuple[float, ...] = (3.0, 7.0, 15.0)
    dedup_target: str = "VICTIM_VEHICLE_TYPE_BY_CLASSIFICATOR"
    dedup_sources: tuple[str, ...] = (
        "VICTIM_VEHICLE_TYPE_BY_CLASSIFICATOR_x",
        "VICTIM_VEHICLE_TYPE_BY_CLASSIFICATOR_y",
    )
    fe_columns: tuple[str, ...] = ()


FE_CATEGORICAL_SUFFIXES: tuple[str, ...] = ("_BIN", "_BUCKET", "_TIER")
FE_CATEGORICAL_EXPLICIT: frozenset[str] = frozenset(
    {
        "FE_SEASON_EVENT",
        "FE_HOUR_BUCKET_EVENT",
        "FE_PARTICIPANTS_BIN",
        "FE_DTPOSAGO_TYPE",
        "FE_EVENT_SCHEME",
    }
)


def is_fe_categorical(column: str) -> bool:
    """Строковые FE-колонки (бакеты), которые CatBoost должен получать как cat."""
    return column.startswith("FE_") and (
        column in FE_CATEGORICAL_EXPLICIT or column.endswith(FE_CATEGORICAL_SUFFIXES)
    )


def fe_categorical_columns(columns: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Отфильтровать категориальные FE из списка имён колонок."""
    return tuple(column for column in columns if is_fe_categorical(column))


def load_feature_config(path: Path | None = None) -> FeatureConfig:
    """Загрузить configs/features_v1.json."""
    config_path = path or _CONFIG_PATH
    data = json.loads(config_path.read_text(encoding="utf-8"))

    raw_thresholds = data.get("thresholds", {})
    thresholds = FeatureThresholds(
        apply_delay_high=int(raw_thresholds.get("apply_delay_high", 30)),
        apply_delay_notify=int(raw_thresholds.get("apply_delay_notify", 7)),
        vehicle_weight_heavy=float(raw_thresholds.get("vehicle_weight_heavy", 3500)),
        amount_repair_high=float(raw_thresholds.get("amount_repair_high", 300_000)),
        amount_repair_bins=tuple(raw_thresholds.get("amount_repair_bins", [100_000, 300_000])),
        insurance_amount_bins=tuple(raw_thresholds.get("insurance_amount_bins", [400_000, 1_000_000])),
        victim_loss_sum_bins=tuple(raw_thresholds.get("victim_loss_sum_bins", [50_000, 200_000])),
        kbm_low=float(raw_thresholds.get("kbm_low", 1.0)),
        kbm_mid=float(raw_thresholds.get("kbm_mid", 1.17)),
        share_wearout_bins=tuple(raw_thresholds.get("share_wearout_bins", [20, 50])),
        share_work_bins=tuple(raw_thresholds.get("share_work_bins", [0.3, 0.6])),
    )

    dedup = data.get("dedup_columns", {})
    return FeatureConfig(
        t0_column=str(data.get("t0_column", "PAYMENT_ORDER_DATE_TIME")),
        thresholds=thresholds,
        vehicle_age_bins=tuple(data.get("vehicle_age_bins", [3, 7, 15])),
        dedup_target=str(dedup.get("target", "VICTIM_VEHICLE_TYPE_BY_CLASSIFICATOR")),
        dedup_sources=tuple(dedup.get("sources", FeatureConfig.dedup_sources)),
        fe_columns=tuple(data.get("fe_columns", [])),
    )
