"""Временные сплиты Train / Val / Cal / Test для train-loop."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DateSplitParts:
    """Индексы строк по ролям сплита."""

    train: pd.Index
    val: pd.Index
    cal: pd.Index
    test: pd.Index
    fit: pd.Index  # train ∪ val — финальный fit до калибровки


def _mask_period(dates: pd.Series, start: str, end: str) -> pd.Series:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return (dates >= start_ts) & (dates <= end_ts)


def split_by_date_periods(
    df: pd.DataFrame,
    *,
    date_column: str,
    train_period: tuple[str, str],
    val_period: tuple[str, str],
    cal_period: tuple[str, str],
    test_period: tuple[str, str],
) -> DateSplitParts:
    """Разбить df по календарным периодам (границы включительно)."""
    if date_column not in df.columns:
        raise ValueError(f"Нет колонки даты: {date_column}")
    dates = pd.to_datetime(df[date_column], errors="coerce")
    train = df.index[_mask_period(dates, *train_period).fillna(False)]
    val = df.index[_mask_period(dates, *val_period).fillna(False)]
    cal = df.index[_mask_period(dates, *cal_period).fillna(False)]
    test = df.index[_mask_period(dates, *test_period).fillna(False)]
    fit = train.union(val)
    return DateSplitParts(train=train, val=val, cal=cal, test=test, fit=fit)


def default_inner_periods_from_train(
    train_period: tuple[str, str],
    *,
    val_days: int = 90,
    cal_days: int = 60,
) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str]]:
    """Из одного train-окна вырезать хвост: … | train_core | val | cal.

    Возвращает (train_core, val_period, cal_period).
    """
    start = pd.Timestamp(train_period[0])
    end = pd.Timestamp(train_period[1])
    cal_start = end - pd.Timedelta(days=cal_days - 1)
    val_end = cal_start - pd.Timedelta(days=1)
    val_start = val_end - pd.Timedelta(days=val_days - 1)
    train_end = val_start - pd.Timedelta(days=1)
    if train_end <= start:
        raise ValueError(
            "train_period слишком короткий для val/cal; задайте периоды явно в TrainingConfig"
        )
    return (
        (start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")),
        (val_start.strftime("%Y-%m-%d"), val_end.strftime("%Y-%m-%d")),
        (cal_start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")),
    )
