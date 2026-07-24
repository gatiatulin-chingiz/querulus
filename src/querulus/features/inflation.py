"""Дефляция денежных сумм к базисному году (инфляция / CPI).

Номинальные VALUE_BEFORE_* дрейфуют во времени из‑за роста цен — для модели и PSI
используем суммы в рублях базисного года: ``real = nominal / cpi_level[year]``,
где ``cpi_level[base_year] = 1.0``.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

# Базис для «реальных» рублей (можно сменить в FeatureConfig / ноутбуке).
INFLATION_BASE_YEAR: int = 2020

# Уровень цен к декабрю 2020 (=1.0): цепочка ИПЦ «к декабрю предыдущего года»
# из Росстата, файл ipc_mes_06-2026.xlsx, лист «01»
# (https://rosstat.gov.ru/storage/mediabank/ipc_mes_06-2026.xlsx).
# Для 2026 в файле — накопленный ИПЦ янв–июнь к дек. 2025 (сноска).
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

# Денежные колонки → FE_*_REAL (номинал остаётся в df для сегментов/финэффекта).
MONETARY_COLUMNS_FOR_REAL: tuple[str, ...] = (
    "VALUE_BEFORE_WITH",
    "VALUE_BEFORE_WITHOUT",
)


def cpi_level_for_years(
    years: pd.Series | np.ndarray,
    *,
    base_year: int = INFLATION_BASE_YEAR,
    levels: Mapping[int, float] | None = None,
) -> pd.Series:
    """Уровень цен относительно ``base_year`` (1.0 = базис)."""
    table = dict(levels or RU_CPI_LEVEL_VS_BASE)
    if base_year not in table:
        raise ValueError(f"Нет CPI для base_year={base_year}")
    base = float(table[base_year])
    year = pd.to_numeric(pd.Series(years), errors="coerce")
    # Нормируем таблицу так, чтобы base_year давал ровно 1.0
    normalized = {y: float(v) / base for y, v in table.items()}
    level = year.map(normalized)
    # Вне таблицы — ближайший известный год
    known = sorted(normalized)
    if level.isna().any():
        filled = level.copy()
        for idx in level[level.isna()].index:
            y = year.loc[idx]
            if y != y:
                continue
            nearest = min(known, key=lambda k: abs(k - int(y)))
            filled.loc[idx] = normalized[nearest]
        level = filled
    return level.astype(float)


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
    level = cpi_level_for_years(dates.dt.year, base_year=base_year, levels=levels)
    return nominal / level.where(level > 0)


def real_feature_name(column: str, base_year: int = INFLATION_BASE_YEAR) -> str:
    """Имя FE-колонки в рублях базисного года."""
    return f"FE_{column}_REAL_{base_year}"
