"""Инклюзивные календарные периоды для колонок datetime.

Date-only граница ``end="YYYY-MM-DD"`` означает весь календарный день:
``[start, end + 1 day)``, а не ``<= midnight``.
"""
from __future__ import annotations

import pandas as pd


def period_end_exclusive(end: str | pd.Timestamp) -> pd.Timestamp:
    """Правый конец полуинтервала ``[start, end_exclusive)``."""
    ts = pd.Timestamp(end)
    if ts == ts.normalize():
        return ts + pd.Timedelta(days=1)
    return ts + pd.Timedelta(nanoseconds=1)


def mask_date_period(dates: pd.Series, start: str, end: str) -> pd.Series:
    """Маска строк с датой в ``[start, end]`` включительно (с учётом времени)."""
    start_ts = pd.Timestamp(start)
    end_excl = period_end_exclusive(end)
    return (dates >= start_ts) & (dates < end_excl)
