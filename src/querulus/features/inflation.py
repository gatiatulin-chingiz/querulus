"""Дефляция денежных сумм к базисному году (инфляция / CPI).

Номинальные VALUE_BEFORE_* дрейфуют во времени из‑за роста цен — для модели и PSI
используем суммы в рублях базисного года: ``real = nominal / cpi_level[year]``,
где ``cpi_level[base_year] = 1.0``.

Таблица уровней — ``configs/cpi_levels.json``. Год вне таблицы → LOCF
(last observation carried forward / backward к ближайшему доступному году)
с warning и пометкой использованного года.
"""
from __future__ import annotations

import json
import logging
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from querulus.dataset.schema import DEFAULT_DATASET_SCHEMA

logger = logging.getLogger("querulus.features.inflation")

# Базис для «реальных» рублей (совпадает с началом train_period).
INFLATION_BASE_YEAR: int = DEFAULT_DATASET_SCHEMA.inflation_base_year

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CPI_JSON_PATH: Path = _PROJECT_ROOT / "configs" / "cpi_levels.json"

# Денежные колонки → *_REAL_{year} (номинал остаётся в df; в обучение — TO_DROP).
MONETARY_COLUMNS_FOR_REAL: tuple[str, ...] = DEFAULT_DATASET_SCHEMA.monetary_columns

# Старый базис в FS-артефактах / OutBoxML JSON до смены CPI.
LEGACY_INFLATION_ALIAS_YEAR: int = 2020

# Кэш последних LOCF-резолвов для отчётов (requested_year → used_year).
_last_cpi_year_usage: dict[int, int] = {}


@lru_cache(maxsize=4)
def load_cpi_levels(path: str | None = None) -> dict[int, float]:
    """Загрузить уровни CPI из JSON (ключи — годы)."""
    json_path = Path(path) if path is not None else DEFAULT_CPI_JSON_PATH
    if not json_path.is_file():
        raise FileNotFoundError(
            f"Нет CPI-таблицы: {json_path}. Создайте configs/cpi_levels.json."
        )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    raw = payload.get("levels", payload)
    return {int(k): float(v) for k, v in raw.items()}


# Обратная совместимость: модульный dict = снимок из JSON при импорте.
try:
    RU_CPI_LEVEL_VS_BASE: dict[int, float] = load_cpi_levels()
except FileNotFoundError:  # pragma: no cover - dev без configs/
    RU_CPI_LEVEL_VS_BASE = {}


def cpi_year_usage() -> dict[int, int]:
    """Словарь ``requested_year → used_year`` после последнего вызова resolve/cpi_level."""
    return dict(_last_cpi_year_usage)


def clear_cpi_year_usage() -> None:
    """Сбросить кэш пометок LOCF (для тестов / нового прогона)."""
    _last_cpi_year_usage.clear()


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
            f"Используем LOCF year={max_y}. Обновите configs/cpi_levels.json."
        )
        logger.warning(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
        _last_cpi_year_usage[y] = max_y
        return max_y
    if y < min_y:
        msg = (
            f"CPI: год инцидента {y} < min={min_y}; берём earliest={min_y}."
        )
        logger.warning(msg)
        _last_cpi_year_usage[y] = min_y
        return min_y
    # дыра внутри диапазона — ближайший известный (LOCF по |Δ|)
    nearest = min(known, key=lambda k: abs(k - y))
    logger.warning("CPI: год %s отсутствует; ближайший %s", y, nearest)
    _last_cpi_year_usage[y] = nearest
    return nearest


def cpi_level_for_years(
    years: pd.Series | np.ndarray,
    *,
    base_year: int = INFLATION_BASE_YEAR,
    levels: Mapping[int, float] | None = None,
) -> pd.Series:
    """Уровень цен относительно ``base_year`` (1.0 = базис); вне таблицы — LOCF."""
    table = dict(levels) if levels is not None else load_cpi_levels()
    if base_year not in table:
        raise ValueError(f"Нет CPI для base_year={base_year}")
    base = float(table[base_year])
    normalized = {y: float(v) / base for y, v in table.items()}
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
    """Перевести номинал в рубли ``base_year``: amount / cpi_level."""
    nominal = pd.to_numeric(amounts, errors="coerce")
    dates = pd.to_datetime(event_dates, errors="coerce")
    years = dates.year if isinstance(dates, pd.DatetimeIndex) else dates.dt.year
    level = cpi_level_for_years(years, base_year=base_year, levels=levels)
    return nominal / level.where(level > 0)


def real_feature_name(column: str, base_year: int = INFLATION_BASE_YEAR) -> str:
    """Имя FE-колонки в рублях базисного года."""
    stem = column if column.startswith("FE_") else f"FE_{column}"
    suffix = f"_REAL_{base_year}"
    if stem.endswith(suffix):
        return stem
    return f"{stem}{suffix}"


def ensure_legacy_real_column_aliases(
    df: pd.DataFrame,
    *,
    current_year: int = INFLATION_BASE_YEAR,
    legacy_year: int = LEGACY_INFLATION_ALIAS_YEAR,
) -> pd.DataFrame:
    """Скопировать ``*_REAL_{current}`` → ``*_REAL_{legacy}``, если алиаса ещё нет.

    Collect пишет итоговый parquet с алиасами; DSM/example не досоздают колонки.
    """
    if current_year == legacy_year:
        return df
    suffix_cur = f"_REAL_{current_year}"
    suffix_leg = f"_REAL_{legacy_year}"
    out: pd.DataFrame | None = None
    for col in df.columns:
        name = str(col)
        if not name.endswith(suffix_cur):
            continue
        legacy = f"{name[: -len(suffix_cur)]}{suffix_leg}"
        if legacy in df.columns:
            continue
        if out is None:
            out = df.copy()
        out[legacy] = out[name]
    return df if out is None else out


def add_real_monetary_columns(
    df: pd.DataFrame,
    event_dates: pd.Series,
    columns: tuple[str, ...] | list[str] | None = None,
    *,
    base_year: int = INFLATION_BASE_YEAR,
    levels: Mapping[int, float] | None = None,
) -> pd.DataFrame:
    """Добавить ``*_REAL_{base_year}`` для денежных колонок, если они есть в df."""
    out = df
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
