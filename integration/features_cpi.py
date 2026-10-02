"""CPI / дефляция для shadow-модели 2.0.0 (без импорта querulus.src).

Таблица — ``configs/cpi_levels.json`` (тот же файл, что у обучения).
Год инцидента вне таблицы → LOCF с warning «обновите JSON».
"""
from __future__ import annotations

import json
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from loguru import logger

INFLATION_BASE_YEAR: int = 2022

_INTEGRATION_ROOT = Path(__file__).resolve().parent
DEFAULT_CPI_JSON_PATH: Path = _INTEGRATION_ROOT.parent / "configs" / "cpi_levels.json"

MONETARY_COLUMNS_FOR_REAL: tuple[str, ...] = (
    "VALUE_BEFORE_WITH",
    "VALUE_BEFORE_WITHOUT",
)

_last_cpi_year_usage: dict[int, int] = {}


@lru_cache(maxsize=4)
def load_cpi_levels(path: str | None = None) -> dict[int, float]:
    json_path = Path(path) if path is not None else DEFAULT_CPI_JSON_PATH
    if not json_path.is_file():
        raise FileNotFoundError(f"Нет CPI-таблицы: {json_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    raw = payload.get("levels", payload)
    return {int(k): float(v) for k, v in raw.items()}


try:
    RU_CPI_LEVEL_VS_BASE: dict[int, float] = load_cpi_levels()
except FileNotFoundError:  # pragma: no cover
    RU_CPI_LEVEL_VS_BASE = {}


def cpi_year_usage() -> dict[int, int]:
    return dict(_last_cpi_year_usage)


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
    table = dict(levels) if levels is not None else load_cpi_levels()
    if base_year not in table:
        raise ValueError(f"Нет CPI для base_year={base_year}")
    base = float(table[base_year])
    return {y: float(v) / base for y, v in table.items()}


def resolve_cpi_year(
    year: float | int | None,
    *,
    levels: Mapping[int, float] | None = None,
) -> int | None:
    """Год для lookup: exact → иначе LOCF/clamp к [min, max] таблицы."""
    table = dict(levels) if levels is not None else load_cpi_levels()
    if not table:
        return None
    known = sorted(table)
    min_y, max_y = known[0], known[-1]
    if year is None or (isinstance(year, float) and year != year):
        _last_cpi_year_usage[-1] = max_y
        return max_y
    y = int(year)
    if y in table:
        _last_cpi_year_usage[y] = y
        return y
    if y > max_y:
        msg = (
            f"CPI-таблица устарела: год инцидента {y} > max={max_y}. "
            f"LOCF year={max_y}. Обновите configs/cpi_levels.json."
        )
        logger.warning(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
        _last_cpi_year_usage[y] = max_y
        return max_y
    if y < min_y:
        logger.warning(
            "CPI: год инцидента {} раньше таблицы (min={}); берём earliest",
            y,
            min_y,
        )
        _last_cpi_year_usage[y] = min_y
        return min_y
    nearest = min(known, key=lambda k: abs(k - y))
    logger.warning("CPI: год {} отсутствует в таблице; ближайший {}", y, nearest)
    _last_cpi_year_usage[y] = nearest
    return nearest


def cpi_level_for_years(
    years: pd.Series | np.ndarray,
    *,
    base_year: int = INFLATION_BASE_YEAR,
    levels: Mapping[int, float] | None = None,
) -> pd.Series:
    """Уровень цен относительно base_year (1.0 = базис); fallback LOCF."""
    table = dict(levels) if levels is not None else load_cpi_levels()
    normalized = _normalized_table(table, base_year=base_year)
    year = pd.to_numeric(pd.Series(years), errors="coerce")
    resolved: list[float] = []
    for y in year.tolist():
        key = resolve_cpi_year(y, levels=table)
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
