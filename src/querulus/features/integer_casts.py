"""Приведение age/year и бинарных float-флагов к Int64 (без полного copy)."""
from __future__ import annotations

import re

import pandas as pd

# Колонки возраста / года / счётчиков — целые.
_INT_NAME_RE = re.compile(
    r"(?:^|_)("
    r"AGE|YEAR|MONTH|HOUR|DAY|DOORS|SEATS|PLACE|COUNT|PARTICIPANTS"
    r")(?:$|_)",
    re.IGNORECASE,
)
_INT_EXACT = {
    "EVENT_YEAR",
    "EVENT_MONTH",
    "EVENT_HOUR",
    "EVENT_DAY",
    "APPLY_DELAY",
    "PARTICIPANTS_COUNT",
    "GUILTY_OBJECT_YEAR",
    "VICTIM_OBJECT_YEAR",
    "POLICYHOLDER_OBJECT_YEAR",
}

# Бинарные флаги (0/1/NaN), часто приходят float.
_BINARY_NAME_RE = re.compile(
    r"(?:^|_)("
    r"FLAG|IS_|MADE_IN|USED_AS|NOT_NOTIFICATION|HAS_|HEAVY|REPEAT|"
    r"MISMATCH|WEEKEND|JOINT|REGRESS|EV_|JAPAN|COMMERCIAL|"
    r"HIGH_APPLY|HIGH_VALUE|SAME_|CORRECTED"
    r")",
    re.IGNORECASE,
)


def _is_binary_values(series: pd.Series) -> bool:
    """True, если ненулевые значения ⊆ {0, 1} (без построения set всех unique)."""
    values = pd.to_numeric(series, errors="coerce")
    finite = values.dropna()
    if finite.empty:
        return False
    if float(finite.min()) < 0.0 or float(finite.max()) > 1.0:
        return False
    return bool(((finite == 0) | (finite == 1)).all())


def _should_try_binary(upper: str) -> bool:
    return bool(_BINARY_NAME_RE.search(upper)) or (
        "MADE_IN" in upper or upper.endswith("_FLAG") or "_IS_" in upper
    )


def cast_integer_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Возраст/год/счётчики и бинарные флаги → Int64.

    Мутирует ``df`` на месте (без ``df.copy()``), чтобы не удваивать ОЗУ.
    """
    for col in list(df.columns):
        if pd.api.types.is_integer_dtype(df[col]):
            continue
        upper = str(col).upper()
        as_int = upper in _INT_EXACT or bool(_INT_NAME_RE.search(upper))
        as_bin = False
        if _should_try_binary(upper):
            # Бинарная проверка только для кандидатов по имени
            try:
                as_bin = _is_binary_values(df[col])
            except (TypeError, ValueError):
                as_bin = False
        if not as_int and not as_bin:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        df[col] = numeric.round().astype("Int64")
    return df
