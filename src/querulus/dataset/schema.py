"""Контракт колонок датасета Querulus (единый allow-list ролей).

Имена таргетов, даты, ID, денежных и known-categorical читаются отсюда —
не копируются строковыми литералами по модулям. Препроцессинг collect и
OutBoxML по-прежнему **разный**; schema задаёт только *что* за колонки, не *как*
их готовить.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# SemVer модели, для которой зафиксирован контракт (в т.ч. max_year).
SCHEMA_MODEL_VERSION: str = "2.0.0"


@dataclass(frozen=True)
class DatasetSchema:
    """Декларативный контракт входного датасета.

    ``validate`` — лёгкая проверка мусора на входе (обязательные колонки,
    размер, доля позитивов, согласованность SEV⇒FREQ). Не заменяет DQ/winsorize.
    """

    model_version: str = SCHEMA_MODEL_VERSION
    date_column: str = "PAYMENT_ORDER_DATE_TIME"
    incident_column: str = "INCIDENT_NUMBER"
    loss_number_column: str = "LOSS_NUMBER"

    frequency_target: str = "TARGET_FREQ"
    severity_target: str = "TARGET_SEV"
    frequency_amount: str = "TARGET_FREQ_AMOUNT"
    frequency_claims_amount: str = "TARGET_FREQ_CLAIMS_AMOUNT"
    frequency_pret_amount: str = "TARGET_FREQ_PRET_AMOUNT"

    # Потолок фич-годов = контракт версии модели (2.0.0 не видела 2027).
    max_year_feature_value: int = 2026
    inflation_base_year: int = 2022

    # Календарные окна обучения (совпадают с TrainingConfig / FinEffectConfig).
    train_period: tuple[str, str] = ("2022-01-01", "2024-05-31")
    test_period: tuple[str, str] = ("2024-06-01", "2025-06-01")

    id_columns: tuple[str, ...] = (
        "INCIDENT_NUMBER",
        "LOSS_NUMBER",
    )

    # Таргеты / ID — не winsorize и не в feature pool без явного include.
    other_cols: tuple[str, ...] = (
        "INCIDENT_NUMBER",
        "LOSS_NUMBER",
        "TARGET_2",
        "TARGET_FREQ",
        "TARGET_FREQ_CLAIMS",
        "TARGET_SEV",
        "TARGET_SEV_CLAIMS",
        "TARGET_3_SEV",
        "TARGET_FREQ_AMOUNT",
        "TARGET_FREQ_CLAIMS_AMOUNT",
        "TARGET_FREQ_PRET_AMOUNT",
        "TARGET_SEV_CLAIMS_AMOUNT",
    )

    monetary_columns: tuple[str, ...] = (
        "VALUE_BEFORE_WITH",
        "VALUE_BEFORE_WITHOUT",
        "PREMIUM_SUM_ALL",
        "FE_PERSON_PRET_PAYMENT_RECIPIENT_FE_PERSON_PRET_SURCHARGE_VALUE_SUM",
        "FE_PERSON_PRET_APPLICANT_FE_PERSON_PRET_PRETENSION_VALUE_SUM",
        "FE_PERSON_PRET_APPLICANT_FE_PERSON_PRET_SURCHARGE_VALUE_SUM",
    )

    known_categorical: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "FILIAL",
                "REGION",
                "REGION_EVENT",
                "VICTIM_TS_REGION",
                "GUILTY_TS_REGION",
                "VIC_TS_COUNTRY",
                "GUIL_TS_COUNTRY",
                "VICTIM_VEHICLE_COUNTRY",
                "GUILTY_VEHICLE_COUNTRY",
                "PAYMENT_RECIPIENT_TYPE",
                "RECIEVE_METHOD",
                "APPLICANT_FORM",
                "VICTIM_VEHICLE_CATEGORY",
                "GUILTY_VEHICLE_CATEGORY",
                "LOSS_UNIT_TYPE",
                "LOSS_UNIT_ZONE",
                "LOSS_UNIT",
                "ACCEPTED_UNIT",
            }
        )
    )
    known_categorical_max_nunique: int = 200

    # Доля TARGET_FREQ=1: вне диапазона — warning в validate (не hard-fail по умолчанию).
    positive_rate_warn_low: float = 0.001
    positive_rate_warn_high: float = 0.5

    def required_columns(self) -> tuple[str, ...]:
        """Минимум колонок для обучения / финэффекта / периодов."""
        return (
            self.date_column,
            self.frequency_target,
            self.severity_target,
        )

    def winsorize_exclude_columns(self) -> tuple[str, ...]:
        """Колонки вне winsorize: other_cols + дата + id."""
        return tuple(
            dict.fromkeys(
                (
                    *self.other_cols,
                    self.date_column,
                    *self.id_columns,
                )
            )
        )

    def validate(
        self,
        df: pd.DataFrame,
        *,
        min_rows: int = 1,
        raise_on_error: bool = True,
    ) -> list[str]:
        """Проверить контракт входа.

        Returns:
            Список проблем (пустой = ок). При ``raise_on_error`` и непустом
            списке критичных ошибок — ``DatasetSchemaError``.
        """
        problems: list[str] = []
        critical: list[str] = []

        if df is None:
            critical.append("df is None")
            if raise_on_error:
                raise DatasetSchemaError(critical)
            return critical

        n = len(df)
        if n < int(min_rows):
            critical.append(f"len(df)={n} < min_rows={min_rows}")

        missing = [c for c in self.required_columns() if c not in df.columns]
        if missing:
            critical.append(f"нет обязательных колонок: {missing}")

        if self.date_column in df.columns:
            dates = pd.to_datetime(df[self.date_column], errors="coerce")
            if int(dates.notna().sum()) == 0:
                critical.append(f"колонка даты {self.date_column!r}: все значения NaT")

        if self.frequency_target in df.columns and n > 0:
            y = pd.to_numeric(df[self.frequency_target], errors="coerce")
            pos = float((y.fillna(0).astype(int) == 1).mean())
            if pos < self.positive_rate_warn_low or pos > self.positive_rate_warn_high:
                problems.append(
                    f"доля {self.frequency_target}=1 равна {pos:.4f} "
                    f"(ожидали примерно [{self.positive_rate_warn_low}, "
                    f"{self.positive_rate_warn_high}])"
                )

        if (
            self.frequency_target in df.columns
            and self.severity_target in df.columns
            and n > 0
        ):
            freq = pd.to_numeric(df[self.frequency_target], errors="coerce").fillna(0)
            sev = pd.to_numeric(df[self.severity_target], errors="coerce").fillna(0)
            bad = int(((sev > 0) & (freq.astype(int) != 1)).sum())
            if bad > 0:
                problems.append(
                    f"{self.severity_target}>0 при {self.frequency_target}≠1: {bad} строк"
                )

        all_issues = critical + problems
        if critical and raise_on_error:
            raise DatasetSchemaError(critical + problems)
        return all_issues

    def to_manifest(self) -> dict[str, Any]:
        """Сериализация для period/run манифеста."""
        return {
            "model_version": self.model_version,
            "date_column": self.date_column,
            "frequency_target": self.frequency_target,
            "severity_target": self.severity_target,
            "max_year_feature_value": self.max_year_feature_value,
            "inflation_base_year": self.inflation_base_year,
            "train_period": list(self.train_period),
            "test_period": list(self.test_period),
            "id_columns": list(self.id_columns),
            "monetary_columns": list(self.monetary_columns),
            "known_categorical": sorted(self.known_categorical),
        }


class DatasetSchemaError(ValueError):
    """Нарушение контракта DatasetSchema."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        ValueError.__init__(self, "; ".join(self.problems))


DEFAULT_DATASET_SCHEMA: DatasetSchema = DatasetSchema()
