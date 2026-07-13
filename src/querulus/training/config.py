"""Конфигурация обучения моделей querulus."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from querulus.training.mvp_types import DEFAULT_MVP_INPUT_TYPES, DEFAULT_OTHER_COLS
from querulus.training.selected_features import DEFAULT_SEVERITY_FEATURES


def _default_mvp_input_types() -> dict[str, tuple[str, ...]]:
    """Словарь типов признаков MVP из Litigant/model_learn.py."""
    return {key: tuple(value) for key, value in DEFAULT_MVP_INPUT_TYPES.items()}


@dataclass(frozen=True)
class TrainingConfig:
    """Параметры разбиения, целей и обучения CatBoost.

    Гиперпараметры (как финальные модели в model_learn.py / configs):
    - ``frequency_iterations`` / ``severity_iterations`` — число итераций CatBoost;
    - ``frequency_random_state`` / ``severity_random_state`` — seed для каждой модели;
    - ``frequency_classifier_params`` / ``severity_regressor_params`` — прочие kwargs CatBoost.

    Категориальные признаки: AutoMVP ``CATEGORIAL + BINARY`` после ``correct_types``,
    пересечение с признаками конкретной модели — как в Litigant.

    ``mvp_cutoff_nan``: доля пропусков, выше которой AutoMVP убирает колонку из пула.
  """

    date_column: str = "LOSS_DATE_TIME"
    train_period: tuple[str, str] = ("2022-01-01", "2024-05-31")
    test_period: tuple[str, str] = ("2024-06-01", "2025-06-01")
    frequency_target: str = "TARGET_FREQ"
    severity_target: str = "TARGET_SEV"
    severity_range: tuple[float, float] = (1.0, 1_500_000.0)
    frequency_iterations: int = 375
    severity_iterations: int = 100
    frequency_random_state: int = 0
    severity_random_state: int = 0
    mvp_cutoff_nan: float = 0.95
    modeldiagnostics_root: Path | str | None = "/home/jovyan/old_home"
    frequency_features: tuple[str, ...] | None = None
    severity_features: tuple[str, ...] | None = DEFAULT_SEVERITY_FEATURES
    frequency_select_features: bool = True
    frequency_num_features_to_select: int = 20
    frequency_select_iterations: int = 100
    frequency_select_early_stopping_rounds: int = 50
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

    @property
    def drop_columns(self) -> tuple[str, ...]:
        """Колонки other_cols из Litigant (таргеты и ID), исключаемые из признаков."""
        return self.base_drop_columns + self.extra_drop_columns
