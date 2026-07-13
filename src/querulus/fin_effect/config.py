"""Конфигурация расчёта финансового эффекта (Litigant fin_effect.py)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

FactMode = Literal["icnl", "legacy_psr"]


@dataclass(frozen=True)
class FinEffectConfig:
    """Имена колонок и параметры расчёта."""

    fact_mode: FactMode = "icnl"
    incident_column: str = "INCIDENT_NUMBER"
    filial_column: str = "FILIAL"
    date_column: str = "LOSS_DATE_TIME"
    frequency_target_column: str = "TARGET_FREQ"
    severity_target_column: str = "TARGET_SEV"
    fact_amount_column: str = "TARGET_FREQ_AMOUNT"
    freq_claims_amount_column: str = "TARGET_FREQ_CLAIMS_AMOUNT"
    freq_pret_amount_column: str = "TARGET_FREQ_PRET_AMOUNT"
    base_payment_column: str = "Выплата_по_основному_убытку"
    # Только boolean-триггер взноса ФУ (сумма в fact не входит).
    fu_fee_trigger_column: str = "Сумма_взыскано_по_ФУ"
    pretension_payments_column: str = "Сумма_выплат_по_претензиям"
    fu_recovery_column: str = "Сумма_взыскано_по_ФУ"
    court_recovery_column: str = "Суммы_взыскано_по_иску"
    premiums_column: str = "Взносы"
    surcharge_column: str = "SurchargeValue_cumsum_by_incident_all"
    uts_surcharge_column: str = "UTSSurchargeValue_cumsum_by_incident_all"
    fu_fee_amount: float = 100_000.0
    court_fee_amount: float = 15_000.0
    apply_court_fee: bool = False
    include_surcharge_in_fact: bool = False
    negate_fact_for_report: bool = True
    threshold_start: float = 0.0
    threshold_stop: float = 1.1
    threshold_step: float = 0.01
    train_period: tuple[str, str] = ("2022-01-01", "2024-05-31")
    test_period: tuple[str, str] = ("2024-06-01", "2025-06-01")
    export_columns: tuple[str, ...] = field(
        default_factory=lambda: (
            "INCIDENT_NUMBER",
            "FILIAL",
            "Выплата_по_основному_убытку",
            "TARGET_FREQ_AMOUNT",
            "TARGET_FREQ_CLAIMS_AMOUNT",
            "TARGET_FREQ_PRET_AMOUNT",
            "Взносы",
            "fin_effect_fact",
            "TARGET_SEV",
            "TARGET_FREQ",
            "TARGET_2",
            "TARGET_3_SEV",
            "pred_freq",
            "pred_sev",
            "fin_effect_model",
        )
    )

    @property
    def uses_legacy_psr_fact(self) -> bool:
        return self.fact_mode == "legacy_psr"

    @property
    def fill_zero_columns(self) -> tuple[str, ...]:
        """Колонки, которые заполняются нулями перед расчётом."""
        if self.uses_legacy_psr_fact:
            return (
                self.pretension_payments_column,
                self.fu_recovery_column,
                self.court_recovery_column,
                self.fu_fee_trigger_column,
                self.base_payment_column,
            )
        return (
            self.fact_amount_column,
            self.freq_claims_amount_column,
            self.freq_pret_amount_column,
            self.fu_fee_trigger_column,
            self.base_payment_column,
        )


ANALYTICS_RENAME_DICT: dict[str, str] = {
    "INCIDENT_NUMBER": "НОМЕР ИНЦИДЕНТА",
    "FILIAL": "ФИЛИАЛ",
    "Выплата_по_основному_убытку": "ВЫПЛАТА ПО ОСНОВНОМУ УБЫТКУ",
    "TARGET_FREQ_AMOUNT": "ИСКОВАЯ СУММА (TARGET_FREQ_AMOUNT)",
    "TARGET_FREQ_CLAIMS_AMOUNT": "ИСКИ (TARGET_FREQ_CLAIMS)",
    "TARGET_FREQ_PRET_AMOUNT": "ПРЕТЕНЗИИ (TARGET_FREQ_PRET)",
    "Взносы": "ВЗНОСЫ",
    "fin_effect_fact": "ФАКТ ФИН. ЭФФЕКТ ",
    "TARGET_SEV": "ФАКТ СУММА ВЗЫСКАНИЯ ОСНОВНОГО ДОЛГА/УТС/ИЗНОСА",
    "TARGET_FREQ": "БЫЛ ИСК (TARGET_FREQ)",
    "TARGET_2": "БЫЛ ПСР (TARGET_2)",
    "TARGET_3_SEV": "ФАКТ СУММА (TARGET_3_SEV)",
    "pred_freq": "МОДЕЛЬ БУДЕТ ЛИ ВЗЫСКАНИЕ ОСНОВНОГО ДОЛГА/УТС/ИЗНОСА",
    "pred_sev": "МОДЕЛЬ СУММА ВЗЫСКАНИЯ ОСНОВНОГО ДОЛГА/УТС/ИЗНОСА",
    "fin_effect_model": "МОДЕЛЬ ФИН. ЭФФЕКТ",
}
