"""Конфигурация обучения моделей querulus."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from querulus.training.mvp_types import DEFAULT_MVP_INPUT_TYPES, DEFAULT_OTHER_COLS


def _default_mvp_input_types() -> dict[str, tuple[str, ...]]:
    """Словарь типов признаков MVP из Litigant/model_learn.py."""
    return {key: tuple(value) for key, value in DEFAULT_MVP_INPUT_TYPES.items()}


@dataclass(frozen=True)
class TrainingConfig:
    """Параметры разбиения, целей и обучения CatBoost.

    Гиперпараметры CatBoost:
    - ``iterations``, ``random_state`` — общие для обеих моделей;
    - ``frequency_classifier_params`` — доп. аргументы ``CatBoostClassifier``
      (как в model_learn.py: ``auto_class_weights='Balanced'``, ``verbose=250``);
    - ``severity_regressor_params`` — доп. аргументы ``CatBoostRegressor``.

    Категориальные признаки определяются автоматически через AutoMVP:
    ``types_dict['CATEGORIAL'] + types_dict['BINARY']`` после ``correct_types``.
    Явный список в конфиге не задаётся — как в Litigant.
    """

    date_column: str = "LOSS_DATE_TIME"
    train_period: tuple[str, str] = ("2022-01-01", "2024-05-31")
    test_period: tuple[str, str] = ("2024-06-01", "2025-06-01")
    frequency_target: str = "TARGET_2"
    severity_target: str = "TARGET_3_SEV"
    severity_range: tuple[float, float] = (1.0, 1_500_000.0)
    random_state: int = 2026
    iterations: int = 100
    modeldiagnostics_root: Path | str | None = "/home/jovyan/old_home"
    best_threshold_metric: str = "f1_score"
    frequency_features: tuple[str, ...] | None = None
    severity_features: tuple[str, ...] | None = None
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