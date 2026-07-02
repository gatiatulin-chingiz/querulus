"""Конфигурация расчёта финансового эффекта (Litigant fin_effect.py)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FinEffectConfig:
    """Имена колонок и параметры расчёта."""

    incident_column: str = "INCIDENT_NUMBER"
    filial_column: str = "FILIAL"
    date_column: str = "LOSS_DATE_TIME"
    frequency_target_column: str = "TARGET"
    severity_target_column: str = "TARGET_SEV"
    base_payment_column: str = "Выплата_по_основному_убытку"
    pretension_payments_column: str = "Сумма_выплат_по_претензиям"
    fu_recovery_column: str = "Сумма_взыскано_по_ФУ"
    court_recovery_column: str = "Суммы_взыскано_по_иску"
    premiums_column: str = "Взносы"
    surcharge_column: str = "SurchargeValue_cumsum_by_incident"
    uts_surcharge_column: str = "UTSSurchargeValue_cumsum_by_incident"
    fu_fee_amount: float = 100_000.0
    court_fee_amount: float = 15_000.0
    apply_court_fee: bool = False
    include_surcharge_in_fact: bool = False
    negate_fact_for_report: bool = True
    threshold_start: float = 0.0
    threshold_stop: float = 1.1
    threshold_step: float = 0.1
    train_period: tuple[str, str] = ("2022-01-01", "2024-05-31")
    test_period: tuple[str, str] = ("2024-06-01", "2025-06-01")
    export_columns: tuple[str, ...] = field(
        default_factory=lambda: (
            "INCIDENT_NUMBER",
            "FILIAL",
            "Выплата_по_основному_убытку",
            "Сумма_выплат_по_претензиям",
            "Сумма_взыскано_по_ФУ",
            "Суммы_взыскано_по_иску",
            "Взносы",
            "fin_effect_fact",
            "TARGET_SEV",
            "TARGET",
            "pred_freq",
            "pred_sev",
            "fin_effect_model",
        )
    )

    @property
    def fill_zero_columns(self) -> tuple[str, ...]:
        """Колонки, которые заполняются нулями перед расчётом."""
        columns = [
            self.pretension_payments_column,
            self.fu_recovery_column,
            self.court_recovery_column,
            self.surcharge_column,
            self.uts_surcharge_column,
        ]
        if self.include_surcharge_in_fact:
            return tuple(columns)
        return tuple(columns[:3] + columns[3:])


ANALYTICS_RENAME_DICT: dict[str, str] = {
    "INCIDENT_NUMBER": "НОМЕР ИНЦИДЕНТА",
    "FILIAL": "ФИЛИАЛ",
    "Выплата_по_основному_убытку": "ВЫПЛАТА ПО ОСНОВНОМУ УБЫТКУ",
    "Сумма_выплат_по_претензиям": "СУММА ВЫПЛАТ ПО ПРЕТЕНЗИЯМ",
    "Сумма_взыскано_по_ФУ": "СУММА ВЗЫСКАННАЯ У ФУ",
    "Суммы_взыскано_по_иску": "СУММА ВЗЫСКАННАЯ В СУДЕ",
    "Взносы": "ВЗНОСЫ",
    "fin_effect_fact": "ФАКТ ФИН. ЭФФЕКТ ",
    "TARGET_SEV": "ФАКТ СУММА ВЗЫСКАНИЯ ОСНОВНОГО ДОЛГА/УТС/ИЗНОСА",
    "TARGET": "БЫЛ ПСР",
    "pred_freq": "МОДЕЛЬ БУДЕТ ЛИ ВЗЫСКАНИЕ ОСНОВНОГО ДОЛГА/УТС/ИЗНОСА",
    "pred_sev": "МОДЕЛЬ СУММА ВЗЫСКАНИЯ ОСНОВНОГО ДОЛГА/УТС/ИЗНОСА",
    "fin_effect_model": "МОДЕЛЬ ФИН. ЭФФЕКТ",
}
