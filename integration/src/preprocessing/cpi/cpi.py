"""CPI / дефляция денежных полей для second_ (модель 2.0.0).

Таблица уровней — ``src/preprocessing/cpi/cpi_levels.json`` (копия обучения;
копируется в Docker-образ вместе с ``src/``). Базовый год —
``INFLATION_BASE_YEAR`` (2022): колонки ``FE_*_REAL_2022`` = номинал / CPI(год as-of).

Если год инцидента вне таблицы — LOCF/clamp к ближайшему краю с warning
«обновите JSON». Модуль не импортирует ``querulus.src``.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
from loguru import logger

INFLATION_BASE_YEAR: int = 2022

DEFAULT_CPI_JSON_PATH: Path = Path(__file__).resolve().parent / "cpi_levels.json"

MONETARY_COLUMNS_FOR_REAL: tuple[str, ...] = (
    "VALUE_BEFORE_WITH",
    "VALUE_BEFORE_WITHOUT",
)

# reviewed
# Таблица CPI и нормировка к base_year — один раз при импорте модуля.
if DEFAULT_CPI_JSON_PATH.is_file():
    _cpi_payload = json.loads(DEFAULT_CPI_JSON_PATH.read_text(encoding="utf-8"))
    _cpi_raw = _cpi_payload.get("levels", _cpi_payload)
    RU_CPI_LEVEL_VS_BASE: dict[int, float] = {
        int(k): float(v) for k, v in _cpi_raw.items()
    }
    del _cpi_payload, _cpi_raw
else:  # pragma: no cover
    RU_CPI_LEVEL_VS_BASE = {}

if RU_CPI_LEVEL_VS_BASE:
    if INFLATION_BASE_YEAR not in RU_CPI_LEVEL_VS_BASE:
        raise ValueError(f"Нет CPI для base_year={INFLATION_BASE_YEAR}")
    _base_level = float(RU_CPI_LEVEL_VS_BASE[INFLATION_BASE_YEAR])
    _NORMALIZED_CPI: dict[int, float] = {
        y: float(v) / _base_level for y, v in RU_CPI_LEVEL_VS_BASE.items()
    }
    del _base_level
else:  # pragma: no cover
    _NORMALIZED_CPI = {}


# reviewed
def resolve_cpi_year(year: float | int | None) -> int | None:
    """Подобрать год для lookup в CPI-таблице.

    Порядок: exact match → clamp выше max (LOCF max) → clamp ниже min →
    ближайший известный год. NaN/None → max таблицы.

    Args:
        year: календарный год as-of (дата выплаты).

    Returns:
        Год-ключ для lookup или ``None``, если таблица пуста.
    """
    table = RU_CPI_LEVEL_VS_BASE
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
        msg = (
            f"CPI-таблица устарела: год инцидента {y} > max={max_y}. "
            f"LOCF year={max_y}. Обновите integration/src/preprocessing/cpi/cpi_levels.json."
        )
        logger.warning(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
        return max_y
    if y < min_y:
        logger.warning(
            "CPI: год инцидента {} раньше таблицы (min={}); берём earliest",
            y,
            min_y,
        )
        return min_y
    nearest = min(known, key=lambda k: abs(k - y))
    logger.warning("CPI: год {} отсутствует в таблице; ближайший {}", y, nearest)
    return nearest


# reviewed
def real_feature_name(column: str) -> str:
    """Имя колонки дефлированного признака: ``FE_{stem}_REAL_{INFLATION_BASE_YEAR}``.

    Args:
        column: сырое имя (с ``FE_`` или без).

    Returns:
        Имя REAL-колонки; если суффикс уже есть — без изменений.
    """
    stem = column if column.startswith("FE_") else f"FE_{column}"
    suffix = f"_REAL_{INFLATION_BASE_YEAR}"
    if stem.endswith(suffix):
        return stem
    return f"{stem}{suffix}"


# reviewed
def deflate_to_base_year(amounts: pd.Series, years: pd.Series) -> pd.Series:
    """Перевести номинальные суммы в рубли ``INFLATION_BASE_YEAR``: amount / cpi_level.

    Год as-of уже посчитан в ``coerce_and_enrich`` (``PAYMENT_ORDER_YEAR``).
    Здесь только lookup CPI и деление.

    Args:
        amounts: номинал (numeric).
        years: календарный год as-of (numeric).

    Returns:
        Series дефлированных сумм (NaN там, где level ≤ 0 или нет данных).
    """
    levels: list[float] = []
    for y in years.tolist():
        key = resolve_cpi_year(y)
        if key is None:
            levels.append(float("nan"))
        else:
            levels.append(float(_NORMALIZED_CPI[key]))
    level = pd.Series(levels, index=years.index, dtype=float)
    return amounts / level.where(level > 0)


# reviewed
def add_real_monetary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить ``FE_*_REAL_2022`` для ``VALUE_BEFORE_*`` по ``PAYMENT_ORDER_YEAR``.

    Колонки ``MONETARY_COLUMNS_FOR_REAL`` и ``PAYMENT_ORDER_YEAR``
    должны уже пройти ``coerce_and_enrich`` / validate — здесь без soft-пропусков.

    Args:
        df: кадр после enrich и validate.

    Returns:
        Копия df с REAL-колонками.
    """
    out = df.copy()
    years = out["PAYMENT_ORDER_YEAR"]
    for col in MONETARY_COLUMNS_FOR_REAL:
        out[real_feature_name(col)] = deflate_to_base_year(out[col], years)
    return out
