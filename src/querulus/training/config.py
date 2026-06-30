"""Конфигурация обучения моделей querulus."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """Параметры разбиения, целей и обучения CatBoost."""

    date_column: str = "LOSS_DATE_TIME"
    train_period: tuple[str, str] = ("2022-01-01", "2024-05-31")
    test_period: tuple[str, str] = ("2024-06-01", "2025-06-01")
    frequency_target: str = "TARGET_2"
    severity_target: str = "TARGET_3_SEV"
    severity_range: tuple[float, float] = (1.0, 1_500_000.0)
    random_state: int = 2026
    iterations: int = 100
    modeldiagnostics_root: Path | str | None = "/home/jovyan/old_home"
    mvp_input_types: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "NUMERIC": ("LONGITUDE", "LATITUDE"),
            "CATEGORIAL": (),
            "BINARY": ("VICTIM_VEHICLE_IS_JAPAN",),
        }
    )
    base_drop_columns: tuple[str, ...] = (
        "TARGET",
        "TARGET_2",
        "TARGET_3",
        "TARGET_3_FREQ",
        "TARGET_3_SEV",
        "LOSS_NUMBER",
        "INCIDENT_NUMBER",
        "Номер_инциндента",
        "PAYMENT_VALUE",
        "PAYMENTS_SUM_RUR",
        "LOSS_AMOUNT",
        "FIX",
        "FIN",
        "REFUND_FORM_INCIDENT",
        "VICTIM_LOSS_SUM_FUTURE",
        "VICTIM_LOSS_COUNT_FUTURE",
        "GUILTY_LOSS_SUM_FUTURE",
        "GUILTY_LOSS_COUNT_FUTURE",
        "PREMIUM_SUM_FUTURE_ALL",
        "PREMIUM_COUNT_FUTURE_ALL",
        "Сумма_выплат_по_претензиям",
        "Сумма_взыскано_по_ФУ",
        "Суммы_взыскано_по_иску",
        "Общая_сумма_заявленных_требований_ФУ",
        "Общая_сумма_заявленных_требований_ИСК",
        "Доп_расходы_инцидент",
    )
    extra_drop_columns: tuple[str, ...] = field(default_factory=tuple)

    @property
    def drop_columns(self) -> tuple[str, ...]:
        """Колонки, исключаемые из признаков."""
        return self.base_drop_columns + self.extra_drop_columns
