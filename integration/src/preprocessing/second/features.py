"""Инженерия FE_* для second_ (модель 2.0.0) из сырого вектора.

Контракт: во входе только сырые поля; FE_* считаются здесь.
Типы дат/чисел уже приведены в ``coerce_and_enrich``.

Серые признаки со списка моделей, которые собирает сервис:
    * FE_*_REAL_2022 ← VALUE_BEFORE_* + PAYMENT_ORDER_YEAR (CPI);
    * FE_VALUE_BEFORE_WITH_BIN ← REAL_WITH;
    * FE_VALUE_BEFORE_DIFF / RATIO ← REAL_WITH / REAL_WITHOUT;
    * FE_DAYS_* ← даты;
    * FE_SHARE_WORK_TIER ← WORK / VALUE_BEFORE_WITHOUT;
    * FE_VICTIM_POWER_PER_TON ← VICTIM_CAPACITY_ENGINE / VICTIM_MAX_WEIGHT;
    * EVENT_YEAR ← EVENT_DATE (в ``coerce_and_enrich``);
    * VICTIM_OBJECT_AGE_MONTHS ← VICTIM_OBJECT_YEAR + EVENT_DATE.

Не собираются здесь (ожидаются во входе уже готовыми):
    * VICTIM_TS_REGION, VICTIM_VEHICLE_COUNTRY
      (на схеме рядом REG_NUM — upstream ETL, не сервис).

VIC_TS_COUNTRY считается в ``coerce_and_enrich`` из ``VICTIM_VEHICLE_BRAND``
(``results/Other_dicts.json`` → ``dict_ts_country``).

Clip числовых FE делает ``prepare_dataset`` по ``model_config``, не здесь.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from integration.src.preprocessing.cpi import (
        add_real_monetary_columns,
        real_feature_name,
    )
except ImportError:
    from src.preprocessing.cpi import (
        add_real_monetary_columns,
        real_feature_name,
    )

SHARE_WORK_BINS: tuple[float, ...] = (0.3, 0.6)
SHARE_WORK_LABELS: tuple[str, ...] = ("low", "mid", "high")
# Как в querulus.features.config: бакеты REAL-суммы with.
AMOUNT_REPAIR_BINS: tuple[float, ...] = (120_000.0, 365_000.0)


# reviewed
def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Поэлементное деление; при нулевом знаменателе — NaN."""
    return numerator / denominator.where(denominator != 0)


# reviewed
def _amount_bins(amount: pd.Series, edges: tuple[float, ...]) -> pd.Series:
    """Трёхуровневые бакеты суммы как object (``<low``, ``low-high``, ``>high``)."""
    low, high = edges
    labels = [f"<{int(low)}", f"{int(low)}-{int(high)}", f">{int(high)}"]
    cut = pd.cut(
        amount,
        bins=[-np.inf, low, high, np.inf],
        labels=labels,
        right=False,
        ordered=True,
    )
    return cut.astype(object).where(cut.notna(), np.nan)


# reviewed
def add_second_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Собрать FE_* / age months из уже препроцессированного вектора."""
    out = df.copy()
    out = add_real_monetary_columns(out)

    if "SHARE_WORK" not in out.columns:
        # WORK / VALUE_BEFORE_WITHOUT уже обязательны во входе.
        out["SHARE_WORK"] = _safe_div(out["WORK"], out["VALUE_BEFORE_WITHOUT"])

    value_without_real = out[real_feature_name("VALUE_BEFORE_WITHOUT")]
    value_with_real = out[real_feature_name("VALUE_BEFORE_WITH")]

    both_ok = (value_without_real > 0) & (value_with_real >= 0)
    both_positive = (value_without_real > 0) & (value_with_real > 0)
    out["FE_VALUE_BEFORE_DIFF"] = (value_without_real - value_with_real).where(both_ok)
    out["FE_VALUE_BEFORE_RATIO"] = _safe_div(value_with_real, value_without_real).where(
        both_positive
    )
    out["FE_VALUE_BEFORE_WITH_BIN"] = _amount_bins(value_with_real, AMOUNT_REPAIR_BINS)

    out["FE_DAYS_EVENT_TO_LOSS"] = (
        out["LOSS_DATE_TIME"] - out["EVENT_DATE"]
    ).dt.days
    out["FE_DAYS_TO_PH_CONTRACT_END"] = (
        out["POLICYHOLDER_CONTRACT_END_DATE"] - out["EVENT_DATE"]
    ).dt.days

    low, high = SHARE_WORK_BINS
    cut = pd.cut(
        out["SHARE_WORK"],
        bins=[-np.inf, low, high, np.inf],
        labels=list(SHARE_WORK_LABELS),
        right=False,
        ordered=True,
    )
    out["FE_SHARE_WORK_TIER"] = cut.astype(object).where(cut.notna(), np.nan)

    out["FE_VICTIM_POWER_PER_TON"] = (
        _safe_div(out["VICTIM_CAPACITY_ENGINE"], out["VICTIM_MAX_WEIGHT"]) * 1e6
    )

    age = (
        (out["EVENT_DATE"].dt.year - out["VICTIM_OBJECT_YEAR"]) * 12
        + (out["EVENT_DATE"].dt.month - 1)
    )
    age = age.where(age.isna(), age.clip(lower=0))
    out["VICTIM_OBJECT_AGE_MONTHS"] = age.round().astype("Int64")

    out["FE_VALUE_BEFORE_DIFF"] = out["FE_VALUE_BEFORE_DIFF"].clip(lower=0)
    return out
