"""Политика порога frequency: подбор только на Val (max net_effect)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from querulus.fin_effect.calculator import (
    FinEffectResult,
    run_fin_effect_from_training,
    run_fin_effect_pipeline,
)
from querulus.fin_effect.config import FinEffectConfig


@dataclass(frozen=True)
class ValThresholdResult:
    """Порог, подобранный на Val по фин. эффекту."""

    threshold: float
    fe_val: FinEffectResult
    n_val: int


def pick_threshold_on_val(
    df: pd.DataFrame,
    val_index: pd.Index,
    frequency_proba: pd.Series | pd.Index,
    severity_prediction: pd.Series | pd.Index,
    y_true_freq: pd.Series | pd.Index,
    *,
    config: FinEffectConfig | None = None,
) -> ValThresholdResult:
    """Подбор порога на Val; поиск max net_effect внутри ``run_fin_effect_pipeline``."""
    index = pd.Index(val_index).intersection(df.index)
    if len(index) == 0:
        raise ValueError("val_index пуст или не пересекается с df.index")

    proba = pd.Series(frequency_proba, dtype=float).reindex(index)
    sev = pd.Series(severity_prediction, dtype=float).reindex(index)
    y_true = pd.Series(y_true_freq).reindex(index)
    common = proba.dropna().index.intersection(sev.dropna().index).intersection(y_true.index)
    if len(common) == 0:
        raise ValueError("Нет пересечения proba/sev/y_true на Val")

    fe_val = run_fin_effect_pipeline(
        df.loc[common],
        proba.reindex(common),
        sev.reindex(common),
        y_true.reindex(common),
        threshold=None,
        config=config,
    )
    return ValThresholdResult(
        threshold=float(fe_val.best_threshold),
        fe_val=fe_val,
        n_val=int(len(common)),
    )


def pick_threshold_on_val_from_training(
    df: pd.DataFrame,
    training: object,
    val_index: pd.Index,
    *,
    config: FinEffectConfig | None = None,
    frequency_target_column: str | None = None,
) -> ValThresholdResult:
    """Подбор порога на Val для ``TrainingArtifacts`` (freq + severity из training)."""
    fe_val = run_fin_effect_from_training(
        df,
        training,
        effect_index=val_index,
        frequency_target_column=frequency_target_column,
        threshold=None,
        config=config,
    )
    return ValThresholdResult(
        threshold=float(fe_val.best_threshold),
        fe_val=fe_val,
        n_val=int(len(val_index)),
    )


def resolve_val_threshold(
    training: object | None,
    *,
    explicit: float | None = None,
) -> float:
    """Взять ``val_threshold`` из training или явное значение."""
    if explicit is not None:
        return float(explicit)
    if training is None:
        raise ValueError("Нужен training.val_threshold или explicit threshold")
    thr = getattr(training, "val_threshold", None)
    if thr is None:
        raise ValueError(
            "training.val_threshold не задан; сначала pick_threshold_on_val_from_training"
        )
    return float(thr)
