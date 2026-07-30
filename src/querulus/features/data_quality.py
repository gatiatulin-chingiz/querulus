"""Очистка значений фич: hard clip ≥0 + winsorize IQR на log1p (границы с train).

Без NaN/drop строк: отрицательные денежные → 0; выбросы зажимаются в Tukey-fence
на шкале ``log1p``, затем ``expm1`` обратно.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from querulus.features.inflation import (
    INFLATION_BASE_YEAR,
    MONETARY_COLUMNS_FOR_REAL,
    real_feature_name,
)

IQR_K: float = 1.5


def _default_monetary_columns(base_year: int = INFLATION_BASE_YEAR) -> tuple[str, ...]:
    """Номинал + REAL + DIFF износа."""
    reals = tuple(real_feature_name(col, base_year) for col in MONETARY_COLUMNS_FOR_REAL)
    return tuple(
        dict.fromkeys(
            [
                *MONETARY_COLUMNS_FOR_REAL,
                *reals,
                "FE_VALUE_BEFORE_DIFF",
            ]
        )
    )


@dataclass
class DataQualityReport:
    """Отчёт по манипуляциям с данными на этапе DQ."""

    iqr_k: float = IQR_K
    train_rows: int = 0
    rows_in: int = 0
    rows_out: int = 0
    policy: dict[str, str] = field(
        default_factory=lambda: {
            "negatives": "clip_to_zero",
            "outliers": "winsorize_iqr_log1p_expm1",
            "fit_on": "train",
            "no_row_drop": "true",
            "no_nan_impute": "true",
        }
    )
    hard_clip_nonnegative: list[dict[str, Any]] = field(default_factory=list)
    winsorize_log1p_iqr: list[dict[str, Any]] = field(default_factory=list)
    skipped_columns: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-сериализуемый словарь."""
        return {
            "policy": self.policy,
            "iqr_k": self.iqr_k,
            "train_rows": self.train_rows,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "summary": {
                "n_hard_clip_columns": len(self.hard_clip_nonnegative),
                "n_hard_clip_cells": int(
                    sum(int(r.get("n_clipped", 0)) for r in self.hard_clip_nonnegative)
                ),
                "n_winsorize_columns": len(self.winsorize_log1p_iqr),
                "n_winsorize_cells": int(
                    sum(int(r.get("n_clipped_total", 0)) for r in self.winsorize_log1p_iqr)
                ),
                "n_skipped_columns": len(self.skipped_columns),
            },
            "hard_clip_nonnegative": self.hard_clip_nonnegative,
            "winsorize_log1p_iqr": self.winsorize_log1p_iqr,
            "skipped_columns": self.skipped_columns,
        }


def clip_nonnegative_columns(
    df: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Зажать отрицательные значения в 0 (без drop/NaN)."""
    result = df
    details: list[dict[str, Any]] = []
    for column in columns:
        if column not in result.columns:
            continue
        values = pd.to_numeric(result[column], errors="coerce")
        neg = values < 0
        n_clipped = int(neg.sum())
        if n_clipped == 0:
            continue
        if result is df:
            result = df.copy()
            values = pd.to_numeric(result[column], errors="coerce")
            neg = values < 0
        min_before = float(values.loc[neg].min())
        values = values.where(~neg, 0.0)
        result[column] = values
        details.append(
            {
                "column": column,
                "n_clipped": n_clipped,
                "min_before": min_before,
                "action": "clip_to_0",
            }
        )
    return result, details


def _winsorize_log1p_iqr_column(
    series: pd.Series,
    train_mask: np.ndarray,
    *,
    iqr_k: float,
) -> tuple[pd.Series, dict[str, Any] | None, str | None]:
    """Winsorize одной колонки: log1p → IQR fence → expm1.

    Возвращает ``(series, detail|None, skip_reason|None)``.
    """
    values = pd.to_numeric(series, errors="coerce")
    train_vals = values.to_numpy(dtype=float)[train_mask]
    train_finite = train_vals[np.isfinite(train_vals)]
    if train_finite.size < 20:
        return series, None, "too_few_train_finite"
    if (train_finite < 0).any():
        return series, None, "negatives_remain_skip_log1p"

    train_log = np.log1p(train_finite)
    q1, q3 = np.quantile(train_log, [0.25, 0.75])
    iqr = float(q3 - q1)
    if not np.isfinite(iqr) or iqr <= 0:
        return series, None, "zero_iqr_log"

    low_log = float(q1 - iqr_k * iqr)
    high_log = float(q3 + iqr_k * iqr)
    low_raw = float(np.expm1(low_log))
    high_raw = float(np.expm1(high_log))
    # после hard clip нижняя граница не ниже 0
    low_raw = max(0.0, low_raw)

    arr = values.to_numpy(dtype=float, copy=True)
    finite = np.isfinite(arr)
    n_low = int(((arr < low_raw) & finite).sum())
    n_high = int(((arr > high_raw) & finite).sum())
    if n_low == 0 and n_high == 0:
        detail = {
            "q1_log": float(q1),
            "q3_log": float(q3),
            "iqr_log": iqr,
            "low_log": low_log,
            "high_log": high_log,
            "low_raw": low_raw,
            "high_raw": high_raw,
            "n_clipped_low": 0,
            "n_clipped_high": 0,
            "n_clipped_total": 0,
            "n_train_finite": int(train_finite.size),
        }
        return series, detail, None

    arr = np.where(finite, np.clip(arr, low_raw, high_raw), arr)
    out = pd.Series(arr, index=series.index, dtype=float)
    detail = {
        "q1_log": float(q1),
        "q3_log": float(q3),
        "iqr_log": iqr,
        "low_log": low_log,
        "high_log": high_log,
        "low_raw": low_raw,
        "high_raw": high_raw,
        "n_clipped_low": n_low,
        "n_clipped_high": n_high,
        "n_clipped_total": n_low + n_high,
        "n_train_finite": int(train_finite.size),
    }
    return out, detail, None


def apply_data_quality(
    df: pd.DataFrame,
    *,
    train_index: pd.Index,
    numeric_columns: list[str] | tuple[str, ...],
    monetary_columns: list[str] | tuple[str, ...] | None = None,
    iqr_k: float = IQR_K,
    base_year: int = INFLATION_BASE_YEAR,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Hard clip ≥0 для monetary/DIFF → winsorize log1p-IQR на numeric (fit=train)."""
    report = DataQualityReport(
        iqr_k=float(iqr_k),
        train_rows=int(len(train_index)),
        rows_in=int(len(df)),
    )
    money = list(
        monetary_columns
        if monetary_columns is not None
        else _default_monetary_columns(base_year)
    )
    hard_cols = [c for c in money if c in df.columns]
    result = df.copy()
    result, hard_details = clip_nonnegative_columns(result, hard_cols)
    report.hard_clip_nonnegative = hard_details

    train_mask = np.asarray(result.index.isin(train_index), dtype=bool)
    candidates = list(
        dict.fromkeys(
            [c for c in numeric_columns if c in result.columns] + hard_cols
        )
    )
    for column in candidates:
        series = result[column]
        if not pd.api.types.is_numeric_dtype(series):
            coerced = pd.to_numeric(series, errors="coerce")
            if int(coerced.notna().sum()) < 20:
                report.skipped_columns.append(
                    {"column": column, "reason": "not_numeric"}
                )
                continue
            series = coerced

        new_series, detail, skip = _winsorize_log1p_iqr_column(
            series, train_mask, iqr_k=iqr_k
        )
        if skip is not None:
            report.skipped_columns.append({"column": column, "reason": skip})
            continue
        if detail is None:
            continue
        result[column] = new_series
        report.winsorize_log1p_iqr.append({"column": column, **detail})

    report.rows_out = int(len(result))
    return result, report


def clip_negative_value_before_diff(df: pd.DataFrame) -> pd.DataFrame:
    """``FE_VALUE_BEFORE_DIFF < 0`` → 0 (без удаления строк)."""
    out, _ = clip_nonnegative_columns(df, ["FE_VALUE_BEFORE_DIFF"])
    return out
