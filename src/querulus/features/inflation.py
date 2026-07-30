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

# Денежные колонки → *_REAL_{year} (номинал остаётся в df; в обучение — TO_DROP).
# Person-претензии: сумма уже агрегирована; дефлируем по T0 строки (приближение).
MONETARY_COLUMNS_FOR_REAL: tuple[str, ...] = (
    "VALUE_BEFORE_WITH",
    "VALUE_BEFORE_WITHOUT",
    "PREMIUM_SUM_ALL",
    "FE_PERSON_PRET_PAYMENT_RECIPIENT_FE_PERSON_PRET_SURCHARGE_VALUE_SUM",
    "FE_PERSON_PRET_APPLICANT_FE_PERSON_PRET_PRETENSION_VALUE_SUM",
    "FE_PERSON_PRET_APPLICANT_FE_PERSON_PRET_SURCHARGE_VALUE_SUM",
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
