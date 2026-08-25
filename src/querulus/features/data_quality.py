"""Очистка значений фич: hard clip ≥0 + winsorize IQR на log1p (границы с train).

Без NaN/drop строк: отрицательные денежные → 0; выбросы зажимаются в Tukey-fence
на шкале ``log1p``, затем ``expm1`` обратно.

Вызов на этапе сборки датасета: ``apply_dataset_data_quality`` (см. ``features.pipeline``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from querulus.features.inflation import (
    INFLATION_BASE_YEAR,
    MONETARY_COLUMNS_FOR_REAL,
    real_feature_name,
)

IQR_K: float = 1.5
# Совпадает с TrainingConfig.train_period / date_column (без импорта training).
DEFAULT_DQ_DATE_COLUMN: str = "PAYMENT_ORDER_DATE_TIME"
DEFAULT_DQ_TRAIN_PERIOD: tuple[str, str] = ("2022-01-01", "2024-05-31")


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


def train_index_by_period(
    df: pd.DataFrame,
    *,
    date_column: str = DEFAULT_DQ_DATE_COLUMN,
    train_period: tuple[str, str] = DEFAULT_DQ_TRAIN_PERIOD,
) -> pd.Index:
    """Индекс строк с датой в ``train_period`` (включительно)."""
    if date_column not in df.columns:
        raise ValueError(f"Нет колонки даты для DQ: {date_column}")
    dates = pd.to_datetime(df[date_column], errors="coerce")
    start = pd.Timestamp(train_period[0])
    end = pd.Timestamp(train_period[1])
    mask = (dates >= start) & (dates <= end)
    return df.index[mask]


def infer_numeric_feature_columns(
    df: pd.DataFrame,
    *,
    exclude_columns: Iterable[str],
) -> list[str]:
    """Числовые фичи для winsorize: не ID/таргеты/дата, не бинарные 0/1."""
    exclude = set(exclude_columns)
    out: list[str] = []
    for column in df.columns:
        if column in exclude:
            continue
        series = df[column]
        if pd.api.types.is_bool_dtype(series):
            continue
        if not pd.api.types.is_numeric_dtype(series):
            continue
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            continue
        uniq = set(np.round(values.to_numpy(dtype=float), 12))
        if len(uniq) <= 2 and uniq.issubset({0.0, 1.0}):
            continue
        out.append(column)
    return out


def apply_dataset_data_quality(
    df: pd.DataFrame,
    *,
    date_column: str = DEFAULT_DQ_DATE_COLUMN,
    train_period: tuple[str, str] = DEFAULT_DQ_TRAIN_PERIOD,
    exclude_columns: Iterable[str] | None = None,
    report_path: Path | str | None = None,
    iqr_k: float = IQR_K,
    base_year: int = INFLATION_BASE_YEAR,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """DQ на сборке датасета: clip≥0 + winsorize; IQR fit на train_period.

    ``exclude_columns`` по умолчанию — ``DEFAULT_OTHER_COLS`` + дата.
    """
    if exclude_columns is None:
        from querulus.training.mvp_types import DEFAULT_OTHER_COLS

        exclude = list(DEFAULT_OTHER_COLS)
    else:
        exclude = list(exclude_columns)
    exclude = list(dict.fromkeys([*exclude, date_column]))

    numeric_columns = infer_numeric_feature_columns(df, exclude_columns=exclude)
    train_index = train_index_by_period(
        df, date_column=date_column, train_period=train_period
    )
    result, report = apply_data_quality(
        df,
        train_index=train_index,
        numeric_columns=numeric_columns,
        iqr_k=iqr_k,
        base_year=base_year,
    )

    if report_path is not None:
        payload = report.to_dict()
        payload["pipeline_context"] = {
            "when": "dataset assembly (features.pipeline / load synthetic|final)",
            "fit_split": "train_period",
            "date_column": date_column,
            "train_period": [str(train_period[0]), str(train_period[1])],
            "train_rows_matched": int(len(train_index)),
            "numeric_columns_considered": numeric_columns,
            "n_numeric_considered": len(numeric_columns),
            "not_used": ["row_drop", "nan_impute", "percentile_winsor"],
        }
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return result, report


def build_service_dq_bounds(
    report: dict[str, Any] | DataQualityReport,
    *,
    model_version: str,
    source_report: str | Path | None = None,
) -> dict[str, Any]:
    """Замороженные границы DQ для сервиса (не пересчитывать IQR на заявке).

    Берём low_raw/high_raw из отчёта сборки parquet — тем же забором резали
    ``df_final_3``, на котором учится модель 2.
    """
    payload = report.to_dict() if isinstance(report, DataQualityReport) else dict(report)
    winsor_bounds: dict[str, dict[str, float]] = {}
    for row in payload.get("winsorize_log1p_iqr") or []:
        column = row.get("column")
        if not column:
            continue
        winsor_bounds[str(column)] = {
            "low_raw": float(row["low_raw"]),
            "high_raw": float(row["high_raw"]),
        }
    money_cols = [
        str(row.get("column"))
        for row in (payload.get("hard_clip_nonnegative") or [])
        if row.get("column")
    ]
    if not money_cols:
        money_cols = list(_default_monetary_columns())
    return {
        "version": model_version,
        "source_report": str(source_report) if source_report else None,
        "policy": payload.get("policy"),
        "pipeline_context": payload.get("pipeline_context"),
        "iqr_k": payload.get("iqr_k", IQR_K),
        "hard_clip_nonnegative_columns": money_cols,
        "winsorize_bounds": winsor_bounds,
        "service_contract": (
            "Before prepare_dataset / DSM: (1) clip listed monetary cols to >=0; "
            "(2) clip each winsorize_bounds col to [low_raw, high_raw]. "
            "Do not recompute IQR per request. Same fences as df_final_3 build."
        ),
    }


def write_service_dq_bounds(
    out_path: Path | str,
    *,
    model_version: str,
    report_path: Path | str | None = None,
    report: dict[str, Any] | DataQualityReport | None = None,
) -> Path:
    """Пишет ``querulus_dq_bounds_{version}.json`` рядом с прод-артефактами."""
    path = Path(out_path)
    if report is None:
        if report_path is None:
            raise ValueError("Нужен report или report_path")
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    payload = build_service_dq_bounds(
        report,
        model_version=model_version,
        source_report=report_path,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def apply_frozen_dq_bounds(
    df: pd.DataFrame,
    bounds: dict[str, Any],
) -> pd.DataFrame:
    """Применить замороженный DQ к сырому кадру (контракт сервиса).

    1) monetary: x < 0 → 0;
    2) winsorize_bounds: clip в [low_raw, high_raw] на сырой шкале
       (границы уже из log1p-IQR train при сборке датасета).
    """
    result = df.copy()
    money = list(bounds.get("hard_clip_nonnegative_columns") or [])
    if money:
        result, _ = clip_nonnegative_columns(result, money)
    for column, fence in (bounds.get("winsorize_bounds") or {}).items():
        if column not in result.columns:
            continue
        low = float(fence["low_raw"])
        high = float(fence["high_raw"])
        values = pd.to_numeric(result[column], errors="coerce")
        result[column] = values.clip(lower=low, upper=high)
    return result
