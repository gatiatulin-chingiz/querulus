"""Грубый correlation-filter числовых признаков (до HPO).

Категориальные колонки не трогаем (Pearson к ним не применяется).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CorrFilterResult:
    """Результат filter по одному таску (freq или sev)."""

    kept_features: tuple[str, ...]
    eliminated_features: tuple[str, ...]
    threshold: float
    target_column: str


def _is_numeric_series(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def correlation_filter_features(
    df: pd.DataFrame,
    features: list[str] | tuple[str, ...],
    target_column: str,
    *,
    threshold: float = 0.95,
) -> CorrFilterResult:
    """Убрать один из пары числовых признаков с |corr| > threshold.

    Правило: оставляем признак с большей |corr| к ``target_column``;
    при равенстве — с меньшим % пропусков. Категории остаются в пуле без изменений.
    """
    present = [f for f in features if f in df.columns]
    if target_column not in df.columns:
        raise ValueError(f"Нет колонки таргета: {target_column}")

    y = pd.to_numeric(df[target_column], errors="coerce")
    numeric: list[str] = []
    categorical: list[str] = []
    for name in present:
        if _is_numeric_series(df[name]):
            numeric.append(name)
        else:
            categorical.append(name)

    eliminated: list[str] = []
    kept_numeric = list(numeric)

    if len(kept_numeric) >= 2:
        frame = df[kept_numeric].apply(pd.to_numeric, errors="coerce")
        corr = frame.corr().abs()
        # Целевые корреляции
        target_corr = {
            col: float(frame[col].corr(y)) if frame[col].notna().sum() > 1 else 0.0
            for col in kept_numeric
        }
        for col in target_corr:
            if target_corr[col] != target_corr[col]:  # NaN
                target_corr[col] = 0.0

        nan_share = {col: float(frame[col].isna().mean()) for col in kept_numeric}
        drop: set[str] = set()

        # Обходим верхний треугольник
        cols = list(kept_numeric)
        for i, a in enumerate(cols):
            if a in drop:
                continue
            for b in cols[i + 1 :]:
                if b in drop:
                    continue
                pair = corr.loc[a, b]
                if pair != pair or pair <= threshold:
                    continue
                # Кто хуже к таргету
                score_a = abs(target_corr[a])
                score_b = abs(target_corr[b])
                if score_a > score_b:
                    loser = b
                elif score_b > score_a:
                    loser = a
                else:
                    loser = a if nan_share[a] >= nan_share[b] else b
                drop.add(loser)

        eliminated = sorted(drop)
        kept_numeric = [c for c in kept_numeric if c not in drop]

    kept = tuple(kept_numeric + categorical)
    # Сохранить исходный порядок features насколько возможно
    order = {name: idx for idx, name in enumerate(present)}
    kept = tuple(sorted(kept, key=lambda name: order.get(name, 10_000)))
    return CorrFilterResult(
        kept_features=kept,
        eliminated_features=tuple(eliminated),
        threshold=threshold,
        target_column=target_column,
    )


def slice_mvp_types(
    mvp_types: dict[str, tuple[str, ...]],
    features: list[str] | tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    """Оставить в types_dict только колонки из ``features``."""
    feature_set = set(features)
    return {
        key: tuple(col for col in cols if col in feature_set)
        for key, cols in mvp_types.items()
    }
