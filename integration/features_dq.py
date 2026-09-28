"""Frozen DQ (hard clip ≥0 + winsorize bounds) для shadow 2.0.0.

Границы из collect (`data_quality_report` → service bounds JSON).
IQR на заявке не пересчитываем.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


def load_dq_bounds(path: Path | str) -> dict[str, Any]:
    """Загрузить JSON замороженных границ."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Нет DQ bounds: {p}")
    return json.loads(p.read_text(encoding="utf-8-sig"))


def clip_nonnegative_columns(
    df: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """x < 0 → 0 для перечисленных колонок."""
    result = df
    for column in columns:
        if column not in result.columns:
            continue
        values = pd.to_numeric(result[column], errors="coerce")
        neg = values < 0
        if not bool(neg.any()):
            continue
        if result is df:
            result = df.copy()
            values = pd.to_numeric(result[column], errors="coerce")
            neg = values < 0
        result[column] = values.where(~neg, 0.0)
    return result


def apply_frozen_dq_bounds(
    df: pd.DataFrame,
    bounds: dict[str, Any],
) -> pd.DataFrame:
    """1) monetary ≥0; 2) clip в [low_raw, high_raw] на сырой шкале."""
    result = df.copy()
    money = list(bounds.get("hard_clip_nonnegative_columns") or [])
    if money:
        result = clip_nonnegative_columns(result, money)
    for column, fence in (bounds.get("winsorize_bounds") or {}).items():
        if column not in result.columns:
            continue
        low = float(fence["low_raw"])
        high = float(fence["high_raw"])
        values = pd.to_numeric(result[column], errors="coerce")
        result[column] = values.clip(lower=low, upper=high)
    return result


def apply_frozen_dq_from_path(
    df: pd.DataFrame,
    path: Path | str | None,
) -> pd.DataFrame:
    """Применить bounds с диска; если path None/нет файла — df без изменений + warning."""
    if path is None:
        logger.warning("DQ: путь bounds не задан — пропускаем frozen clip")
        return df
    p = Path(path)
    if not p.is_file():
        logger.warning("DQ: файл bounds не найден {} — пропускаем frozen clip", p)
        return df
    bounds = load_dq_bounds(p)
    return apply_frozen_dq_bounds(df, bounds)


def bounds_from_data_quality_report(report: dict[str, Any]) -> dict[str, Any]:
    """Собрать service-bounds из data_quality_report.json collect."""
    winsor_bounds: dict[str, dict[str, float]] = {}
    for row in report.get("winsorize_log1p_iqr") or []:
        column = row.get("column")
        if not column:
            continue
        winsor_bounds[str(column)] = {
            "low_raw": float(row["low_raw"]),
            "high_raw": float(row["high_raw"]),
        }
    money_cols = [
        str(row.get("column"))
        for row in (report.get("hard_clip_nonnegative") or [])
        if row.get("column")
    ]
    return {
        "hard_clip_nonnegative_columns": money_cols,
        "winsorize_bounds": winsor_bounds,
        "iqr_k": report.get("iqr_k"),
        "policy": report.get("policy"),
    }
