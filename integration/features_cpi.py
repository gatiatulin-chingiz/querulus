"""CPI / дефляция для shadow-модели 2.0.0 (без импорта querulus.src).

Снимок таблицы из обучения. Год инцидента без коэффициента в таблице →
последний доступный по дате (max year); для лет раньше таблицы — min year.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from loguru import logger

INFLATION_BASE_YEAR: int = 2022

# Источник: Росстат ipc_mes (как в обучении Querulus). Норма к дек.2020=1.0;
# при дефляции дополнительно нормируется к INFLATION_BASE_YEAR.
RU_CPI_LEVEL_VS_BASE: dict[int, float] = {
    2017: 0.887278,
    2018: 0.925076,
    2019: 0.953198,
    2020: 1.0,
    2021: 1.0839,
    2022: 1.213318,
    2023: 1.303346,
    2024: 1.427424,
    2025: 1.507217,
    2026: 1.5704,
}

MONETARY_COLUMNS_FOR_REAL: tuple[str, ...] = (
    "VALUE_BEFORE_WITH",
    "VALUE_BEFORE_WITHOUT",
)


def real_feature_name(column: str, base_year: int = INFLATION_BASE_YEAR) -> str:
    stem = column if column.startswith("FE_") else f"FE_{column}"
    suffix = f"_REAL_{base_year}"
    if stem.endswith(suffix):
        return stem
    return f"{stem}{suffix}"


def _normalized_table(
    levels: Mapping[int, float] | None = None,
    *,
    base_year: int = INFLATION_BASE_YEAR,
) -> dict[int, float]:
    table = dict(levels or RU_CPI_LEVEL_VS_BASE)
    if base_year not in table:
        raise ValueError(f"Нет CPI для base_year={base_year}")
    base = float(table[base_year])
    return {y: float(v) / base for y, v in table.items()}


def resolve_cpi_year(
    year: float | int | None,
    *,
    levels: Mapping[int, float] | None = None,
) -> int | None:
    """Год для lookup: exact → иначе clamp к [min, max] таблицы (новые убытки → max)."""
    table = dict(levels or RU_CPI_LEVEL_VS_BASE)
    if not table:
        return None
    known = sorted(table)
    min_y, max_y = known[0], known[-1]
    if year is None or (isinstance(year, float) and year != year):
        return max_y
    y = int(year)
    if y in table:
        return y
    if y > max_y:
        logger.warning(
            "CPI: год инцидента {} позже таблицы (max={}); берём последний коэффициент",
            y,
            max_y,
        )
        return max_y
    if y < min_y:
        logger.warning(
            "CPI: год инцидента {} раньше таблицы (min={}); берём самый ранний коэффициент",
            y,
            min_y,
        )
        return min_y
    # дыра внутри диапазона — ближайший известный
    nearest = min(known, key=lambda k: abs(k - y))
    logger.warning(
        "CPI: год {} отсутствует в таблице; ближайший {}",
        y,
        nearest,
    )
    return nearest


def cpi_level_for_years(
    years: pd.Series | np.ndarray,
    *,
    base_year: int = INFLATION_BASE_YEAR,
    levels: Mapping[int, float] | None = None,
) -> pd.Series:
    """Уровень цен относительно base_year (1.0 = базис); fallback max/min year."""
    normalized = _normalized_table(levels, base_year=base_year)
    year = pd.to_numeric(pd.Series(years), errors="coerce")
    resolved: list[float] = []
    for y in year.tolist():
        key = resolve_cpi_year(y, levels=levels or RU_CPI_LEVEL_VS_BASE)
        if key is None:
            resolved.append(float("nan"))
        else:
            resolved.append(float(normalized[key]))
    return pd.Series(resolved, index=year.index, dtype=float)


def deflate_to_base_year(
    amounts: pd.Series,
    event_dates: pd.Series,
    *,
    base_year: int = INFLATION_BASE_YEAR,
    levels: Mapping[int, float] | None = None,
) -> pd.Series:
    """Номинал → рубли base_year: amount / cpi_level."""
    nominal = pd.to_numeric(amounts, errors="coerce")
    dates = pd.to_datetime(event_dates, errors="coerce")
    years = dates.dt.year if hasattr(dates, "dt") else pd.Series(dates).year
    level = cpi_level_for_years(years, base_year=base_year, levels=levels)
    return nominal / level.where(level > 0)


def add_real_monetary_columns(
    df: pd.DataFrame,
    *,
    event_date_column: str = "EVENT_DATE",
    columns: tuple[str, ...] | list[str] | None = None,
    base_year: int = INFLATION_BASE_YEAR,
    levels: Mapping[int, float] | None = None,
    bridge_amount_repair: bool = True,
) -> pd.DataFrame:
    """Добавить ``FE_*_REAL_{base}``; при отсутствии VALUE_BEFORE_WITHOUT — bridge из AMOUNT_REPAIR."""
    out = df.copy()
    if (
        bridge_amount_repair
        and "VALUE_BEFORE_WITHOUT" not in out.columns
        and "AMOUNT_REPAIR" in out.columns
    ):
        logger.warning(
            "CPI: VALUE_BEFORE_WITHOUT отсутствует; bridge AMOUNT_REPAIR → VALUE_BEFORE_WITHOUT"
        )
        out["VALUE_BEFORE_WITHOUT"] = pd.to_numeric(out["AMOUNT_REPAIR"], errors="coerce")

    if event_date_column not in out.columns:
        logger.warning("CPI: нет колонки {} — REAL-колонки не считаем", event_date_column)
        return out

    event_dates = out[event_date_column]
    for col in columns or MONETARY_COLUMNS_FOR_REAL:
        if col not in out.columns:
            continue
        out[real_feature_name(col, base_year)] = deflate_to_base_year(
            out[col],
            event_dates,
            base_year=base_year,
            levels=levels,
        )
    return out
