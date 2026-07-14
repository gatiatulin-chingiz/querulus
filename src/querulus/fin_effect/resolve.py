"""Подбор FinEffectConfig под legacy/new/new_claims датасет и таргеты."""
from __future__ import annotations

from typing import Literal

from dataclasses import replace

import pandas as pd

from querulus.fin_effect.config import FinEffectConfig

FactMode = Literal["icnl", "legacy_psr"]

LEGACY_FREQUENCY_TARGETS: frozenset[str] = frozenset({"TARGET_2"})
LEGACY_SEVERITY_TARGETS: frozenset[str] = frozenset({"TARGET_3_SEV"})
ICNL_FREQUENCY_TARGETS: frozenset[str] = frozenset({"TARGET_FREQ", "TARGET_FREQ_CLAIMS"})
ICNL_SEVERITY_TARGETS: frozenset[str] = frozenset({"TARGET_SEV", "TARGET_SEV_CLAIMS"})
CLAIMS_FREQUENCY_TARGETS: frozenset[str] = frozenset({"TARGET_FREQ_CLAIMS"})
CLAIMS_SEVERITY_TARGETS: frozenset[str] = frozenset({"TARGET_SEV_CLAIMS"})

_PSR_FACT_COLUMNS: tuple[str, ...] = (
    "Сумма_выплат_по_претензиям",
    "Сумма_взыскано_по_ФУ",
    "Суммы_взыскано_по_иску",
)


def _icnl_amount_signal(df: pd.DataFrame) -> float:
    """Доля строк с ненулевым TARGET_FREQ_AMOUNT."""
    if "TARGET_FREQ_AMOUNT" not in df.columns:
        return 0.0
    values = pd.to_numeric(df["TARGET_FREQ_AMOUNT"], errors="coerce").fillna(0)
    return float((values > 0).mean())


def _has_psr_fact_columns(df: pd.DataFrame) -> bool:
    return all(column in df.columns for column in _PSR_FACT_COLUMNS)


def _legacy_targets_selected(frequency_target: str, severity_target: str) -> bool:
    return frequency_target in LEGACY_FREQUENCY_TARGETS or severity_target in LEGACY_SEVERITY_TARGETS


def _claims_targets_selected(frequency_target: str, severity_target: str) -> bool:
    return (
        frequency_target in CLAIMS_FREQUENCY_TARGETS
        or severity_target in CLAIMS_SEVERITY_TARGETS
    )


def _icnl_targets_selected(frequency_target: str, severity_target: str) -> bool:
    return (
        frequency_target in ICNL_FREQUENCY_TARGETS
        or severity_target in ICNL_SEVERITY_TARGETS
    )


def infer_legacy_dataset(df: pd.DataFrame) -> bool:
    """Эвристика: старый parquet с ПСР без содержательного icnl-факта.

    Новый querulus-датасет тоже содержит колонки ПСР (merge target_psr),
    поэтому наличие ПСР + ненулевой TARGET_FREQ_AMOUNT → не legacy.
    """
    has_psr = _has_psr_fact_columns(df)
    icnl_signal = _icnl_amount_signal(df)
    if icnl_signal >= 0.01:
        return False
    if has_psr and icnl_signal < 0.01:
        return True
    return has_psr


def resolve_fact_mode(
    df: pd.DataFrame,
    *,
    frequency_target: str,
    severity_target: str,
    loaded_from_checkpoint: bool = True,
    legacy_dataset: bool | None = None,
    fact_mode: FactMode | Literal["auto"] = "auto",
) -> FactMode:
    """Выбрать режим расчёта fin_effect_fact.

    - Явный fact_mode → как передали.
    - legacy_dataset=True → legacy_psr; False → по таргетам.
    - TARGET_2 / TARGET_3_SEV → legacy_psr.
    - TARGET_FREQ* / TARGET_SEV* → icnl.
    """
    if fact_mode in ("icnl", "legacy_psr"):
        return fact_mode

    if legacy_dataset is True:
        return "legacy_psr"
    if legacy_dataset is False:
        return "legacy_psr" if _legacy_targets_selected(frequency_target, severity_target) else "icnl"

    if _legacy_targets_selected(frequency_target, severity_target):
        return "legacy_psr"

    if _icnl_targets_selected(frequency_target, severity_target):
        return "icnl"

    if not loaded_from_checkpoint:
        return "icnl"
    return "legacy_psr" if infer_legacy_dataset(df) else "icnl"


def _legacy_config(
    frequency_target: str,
    severity_target: str,
) -> FinEffectConfig:
    return FinEffectConfig(
        frequency_target_column=frequency_target,
        severity_target_column=severity_target,
        fact_mode="legacy_psr",
        threshold_step=0.1,
        threshold_stop=1.0,
        export_columns=(
            "INCIDENT_NUMBER",
            "FILIAL",
            "Выплата_по_основному_убытку",
            "Сумма_выплат_по_претензиям",
            "Сумма_взыскано_по_ФУ",
            "Суммы_взыскано_по_иску",
            "Взносы",
            "fin_effect_fact",
            severity_target,
            frequency_target,
            "TARGET_2",
            "pred_freq",
            "pred_sev",
            "fin_effect_model",
        ),
    )


def _icnl_config(
    frequency_target: str,
    severity_target: str,
) -> FinEffectConfig:
    """icnl: full amount или только claims (без претензий)."""
    fact_amount = (
        "TARGET_FREQ_CLAIMS_AMOUNT"
        if _claims_targets_selected(frequency_target, severity_target)
        else "TARGET_FREQ_AMOUNT"
    )
    return FinEffectConfig(
        frequency_target_column=frequency_target,
        severity_target_column=severity_target,
        fact_mode="icnl",
        fact_amount_column=fact_amount,
        export_columns=(
            "INCIDENT_NUMBER",
            "FILIAL",
            "Выплата_по_основному_убытку",
            "TARGET_FREQ_AMOUNT",
            "TARGET_FREQ_CLAIMS_AMOUNT",
            "TARGET_FREQ_PRET_AMOUNT",
            "Взносы",
            "fin_effect_fact",
            severity_target,
            frequency_target,
            "TARGET_FREQ",
            "TARGET_FREQ_CLAIMS",
            "TARGET_2",
            "TARGET_3_SEV",
            "pred_freq",
            "pred_sev",
            "fin_effect_model",
        ),
    )


def resolve_fin_effect_config(
    df: pd.DataFrame,
    *,
    frequency_target: str = "TARGET_FREQ",
    severity_target: str = "TARGET_SEV",
    loaded_from_checkpoint: bool = True,
    legacy_dataset: bool | None = None,
    fact_mode: FactMode | Literal["auto"] = "auto",
    **overrides: object,
) -> FinEffectConfig:
    """Собрать FinEffectConfig под датасет и выбранные таргеты."""
    mode = resolve_fact_mode(
        df,
        frequency_target=frequency_target,
        severity_target=severity_target,
        loaded_from_checkpoint=loaded_from_checkpoint,
        legacy_dataset=legacy_dataset,
        fact_mode=fact_mode,
    )
    base = _legacy_config(frequency_target, severity_target) if mode == "legacy_psr" else _icnl_config(
        frequency_target, severity_target
    )
    if overrides:
        return replace(base, **overrides)
    return base
