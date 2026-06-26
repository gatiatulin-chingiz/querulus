"""Вспомогательные функции пайплайна."""
from __future__ import annotations

import numpy as np
import pandas as pd


def convert_to_binary(value) -> int:
    """Преобразование бинарных полей OISUU в 0/1."""
    if value in ("00", b"\x00"):
        return 0
    if value in ("01", b"\x01"):
        return 1
    return 0


def my_mode(series: pd.Series):
    """Мода серии; NaN если пусто."""
    m = series.mode()
    return m.iloc[0] if not m.empty else np.nan


def hex_upper(value):
    """bytes -> HEX или NaN."""
    return value.hex().upper() if value is not None else np.nan
