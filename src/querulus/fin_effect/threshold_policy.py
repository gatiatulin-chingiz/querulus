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


def val_index_from_training(training: object | None) -> pd.Index | None:
    """Index Val из ``TrainingArtifacts.frequency_split``, если есть."""
    split = getattr(training, "frequency_split", None)
    if split is None or not getattr(split, "has_val", False):
        return None
    x_val = getattr(split, "x_val", None)
    if x_val is None or len(x_val) == 0:
        return None
    return pd.Index(x_val.index)


def val_index_from_trainings(
    trainings: dict[str, object],
    *,
    prefer: tuple[str, ...] = ("new", "new_claims", "legacy"),
) -> pd.Index | None:
    """Общий Val-календарь для стеков без собственного Val (legacy train/test)."""
    for name in prefer:
        idx = val_index_from_training(trainings.get(name))
        if idx is not None and len(idx) > 0:
            return idx
    for training in trainings.values():
        idx = val_index_from_training(training)
        if idx is not None and len(idx) > 0:
            return idx
    return None


def resolve_or_pick_val_threshold(
    df: pd.DataFrame,
    training: object,
    *,
    val_index: pd.Index | None = None,
    frequency_target_column: str | None = None,
    config: FinEffectConfig | None = None,
    explicit: float | None = None,
) -> float:
    """``training.val_threshold`` или подбор на Val (свой split или ``val_index``)."""
    if explicit is not None:
        return float(explicit)
    cached = getattr(training, "val_threshold", None)
    if cached is not None:
        return float(cached)

    own_val = val_index_from_training(training)
    pick_index = own_val if own_val is not None and len(own_val) > 0 else val_index
    if pick_index is None or len(pick_index) == 0:
        raise ValueError(
            "Нужен training.val_threshold или val_index для подбора порога на Val"
        )
    return pick_threshold_on_val_from_training(
        df,
        training,
        pick_index,
        config=config,
        frequency_target_column=frequency_target_column,
    ).threshold
