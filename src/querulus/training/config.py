"""Конфигурация обучения моделей querulus."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from querulus.training.mvp_types import DEFAULT_MVP_INPUT_TYPES, DEFAULT_OTHER_COLS
from querulus.training.selected_features import (
    DEFAULT_FREQUENCY_FEATURES,
    DEFAULT_SEVERITY_FEATURES,
)


def _default_mvp_input_types() -> dict[str, tuple[str, ...]]:
    """Словарь типов признаков MVP из Litigant/model_learn.py."""
    return {key: tuple(value) for key, value in DEFAULT_MVP_INPUT_TYPES.items()}


FeaturesSource = Literal["selected", "mvp"]


@dataclass(frozen=True)
class TrainingConfig:
    """Параметры разбиения, целей и обучения CatBoost.

    ``features_source``:
    - ``selected`` — фичи из ``selected_features.py`` (без feature select);
    - ``mvp`` — полный MVP-пул (``frequency_select_features`` / ``severity_select_features``).

    Гиперпараметры (как финальные модели в model_learn.py / configs):
    - ``frequency_iterations`` / ``severity_iterations`` — число итераций CatBoost;
    - ``frequency_random_state`` / ``severity_random_state`` — seed для каждой модели;
    - ``frequency_classifier_params`` / ``severity_regressor_params`` — прочие kwargs CatBoost.

    Категориальные признаки: AutoMVP ``CATEGORIAL + BINARY`` после ``correct_types``,
    пересечение с признаками конкретной модели — как в Litigant.

    ``mvp_cutoff_nan``: доля пропусков, выше которой AutoMVP убирает колонку из пула.

    ``severity_range``:
    - ``None`` — только ``target > 0`` (без верхней границы);
    - ``(low, high)`` — ``target.between(low, high)``.
    """

    date_column: str = "LOSS_DATE_TIME"
    train_period: tuple[str, str] = ("2022-01-01", "2024-05-31")
    test_period: tuple[str, str] = ("2024-06-01", "2025-06-01")
    # Внутренние периоды (если None — режутся из train_period хвостом val/cal).
    val_period: tuple[str, str] | None = None
    cal_period: tuple[str, str] | None = None
    frequency_target: str = "TARGET_FREQ"
    severity_target: str = "TARGET_SEV"
    features_source: FeaturesSource = "selected"
    severity_range: tuple[float, float] | None = None
    frequency_iterations: int = 375
    severity_iterations: int = 100
    frequency_random_state: int = 0
    severity_random_state: int = 0
    mvp_cutoff_nan: float = 0.95
    modeldiagnostics_root: Path | str | None = "/home/jovyan/old_home"
    frequency_features: tuple[str, ...] | None = None
    severity_features: tuple[str, ...] | None = None
    frequency_select_features: bool = False
    frequency_num_features_to_select: int = 30
    frequency_select_iterations: int = 100
    frequency_select_early_stopping_rounds: int = 50
    severity_select_features: bool = False
    severity_num_features_to_select: int = 30
    severity_select_iterations: int = 100
    severity_select_early_stopping_rounds: int = 50
    frequency_calibration_enabled: bool = False
    frequency_calibration_method: Literal["isotonic", "sigmoid"] = "isotonic"
    frequency_leak_importance_level: float = 50.0
    mvp_input_types: dict[str, tuple[str, ...]] = field(default_factory=_default_mvp_input_types)
    base_drop_columns: tuple[str, ...] = DEFAULT_OTHER_COLS
    extra_drop_columns: tuple[str, ...] = field(default_factory=tuple)
    frequency_classifier_params: dict[str, object] = field(
        default_factory=lambda: {
            "auto_class_weights": "Balanced",
            "verbose": 250,
        }
    )
    severity_regressor_params: dict[str, object] = field(
        default_factory=lambda: {
            "verbose": 250,
        }
    )
    severity_target_transform: Literal["raw", "log1p"] = "raw"
    severity_sample_weight: Literal["none", "sqrt", "linear"] = "none"
    # Train-loop (блок B): FE в пуле обучения
    use_fe_features: bool = True
    corr_filter_threshold: float = 0.95

    @property
    def drop_columns(self) -> tuple[str, ...]:
        """Колонки other_cols из Litigant (таргеты и ID), исключаемые из признаков."""
        return self.base_drop_columns + self.extra_drop_columns


def resolve_features_config(config: TrainingConfig) -> TrainingConfig:
    """Применить ``features_source``: selected_features или полный MVP.

    Для ``mvp`` явные ``frequency_features`` / ``severity_features`` не затираются
    (нужно, чтобы после select на new переиспользовать те же фичи на других стеках).
    """
    if config.features_source == "mvp":
        return config
    return replace(
        config,
        frequency_features=(
            config.frequency_features
            if config.frequency_features is not None
            else DEFAULT_FREQUENCY_FEATURES
        ),
        severity_features=(
            config.severity_features
            if config.severity_features is not None
            else DEFAULT_SEVERITY_FEATURES
        ),
        frequency_select_features=False,
        severity_select_features=False,
    )
