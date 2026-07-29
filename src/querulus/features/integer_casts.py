"""Приведение age/year и бинарных float-флагов к Int64."""
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
    """True, если ненулевые значения ⊆ {0, 1}."""
    values = pd.to_numeric(series, errors="coerce")
    uniq = set(values.dropna().unique().tolist())
    if not uniq:
        return False
    return uniq.issubset({0, 1, 0.0, 1.0})


def cast_integer_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Возраст/год/счётчики и бинарные флаги → Int64 (NaN сохраняются)."""
    out = df.copy()
    for col in out.columns:
        name = str(col)
        upper = name.upper()
        as_int = upper in _INT_EXACT or bool(_INT_NAME_RE.search(upper))
        as_bin = bool(_BINARY_NAME_RE.search(upper)) and _is_binary_values(out[col])
        if not as_int and not as_bin:
            # Бинарный float без «флагового» имени, но значения только 0/1
            # (например VICTIM_VEHICLE_MADE_IN_RF) — тоже Int64.
            if _is_binary_values(out[col]) and (
                "MADE_IN" in upper or upper.endswith("_FLAG") or "_IS_" in upper
            ):
                as_bin = True
        if not as_int and not as_bin:
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        out[col] = numeric.round().astype("Int64")
    return out
