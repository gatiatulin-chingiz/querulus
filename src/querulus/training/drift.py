"""Календарный дрейф признаков train vs test (и по месяцам теста)."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from querulus.training.config import TrainingConfig


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index для числового признака."""
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 20 or len(actual) < 20:
        return float("nan")
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, quantiles))
    if len(cuts) < 3:
        return float("nan")
    exp_counts = np.histogram(expected, bins=cuts)[0].astype(float)
    act_counts = np.histogram(actual, bins=cuts)[0].astype(float)
    exp_share = (exp_counts + 1e-6) / (exp_counts.sum() + 1e-6 * len(exp_counts))
    act_share = (act_counts + 1e-6) / (act_counts.sum() + 1e-6 * len(act_counts))
    return float(np.sum((act_share - exp_share) * np.log(act_share / exp_share)))


def _categorical_share_l1(train: pd.Series, test: pd.Series, top_n: int = 20) -> float:
    """L1 между долями топ-категорий train/test."""
    train_vc = train.astype("string").fillna("<NA>").value_counts(normalize=True)
    test_vc = test.astype("string").fillna("<NA>").value_counts(normalize=True)
    cats = list(train_vc.head(top_n).index.union(test_vc.head(top_n).index))
    if not cats:
        return float("nan")
    t = train_vc.reindex(cats, fill_value=0.0)
    s = test_vc.reindex(cats, fill_value=0.0)
    return float((t - s).abs().sum())


def feature_drift_report(
    df: pd.DataFrame,
    features: Iterable[str],
    config: TrainingConfig,
    *,
    categorical_features: Iterable[str] | None = None,
    importance: pd.DataFrame | None = None,
    top_importance: int = 30,
) -> pd.DataFrame:
    """Сравнить распределения фич train vs test по ``date_column``.

    Числа: PSI + сдвиг mean/median/доля NaN.
    Категории: L1 долей + nunique + доля NaN.
    """
    data = df.copy()
    data[config.date_column] = pd.to_datetime(data[config.date_column], errors="coerce")
    train = data[data[config.date_column].between(*config.train_period)]
    test = data[data[config.date_column].between(*config.test_period)]
    cat_set = set(categorical_features or ())

    feature_list = [f for f in features if f in data.columns]
    if importance is not None and not importance.empty and "feature" in importance.columns:
        ranked = [
            name
            for name in importance["feature"].tolist()
            if name in feature_list
        ][:top_importance]
        # Топ по importance сначала, остальные следом.
        rest = [f for f in feature_list if f not in ranked]
        feature_list = ranked + rest

    rows: list[dict[str, object]] = []
    for column in feature_list:
        train_col = train[column]
        test_col = test[column]
        is_cat = column in cat_set or (
            not pd.api.types.is_numeric_dtype(train_col)
            and train_col.dtype == object
        )
        row: dict[str, object] = {
            "feature": column,
            "kind": "categorical" if is_cat else "numeric",
            "train_null_share": float(train_col.isna().mean()),
            "test_null_share": float(test_col.isna().mean()),
            "null_share_delta": float(test_col.isna().mean() - train_col.isna().mean()),
        }
        if is_cat:
            row["drift_score"] = _categorical_share_l1(train_col, test_col)
            row["train_nunique"] = int(train_col.nunique(dropna=True))
            row["test_nunique"] = int(test_col.nunique(dropna=True))
        else:
            train_num = pd.to_numeric(train_col, errors="coerce")
            test_num = pd.to_numeric(test_col, errors="coerce")
            row["drift_score"] = _psi(train_num.to_numpy(dtype=float), test_num.to_numpy(dtype=float))
            row["train_mean"] = float(train_num.mean()) if train_num.notna().any() else float("nan")
            row["test_mean"] = float(test_num.mean()) if test_num.notna().any() else float("nan")
            row["mean_delta"] = (
                float(test_num.mean() - train_num.mean())
                if train_num.notna().any() and test_num.notna().any()
                else float("nan")
            )
        rows.append(row)

    report = pd.DataFrame(rows)
    if report.empty:
        return report
    return report.sort_values("drift_score", ascending=False, na_position="last").reset_index(
        drop=True
    )


def monthly_target_drift(
    df: pd.DataFrame,
    target: str,
    config: TrainingConfig,
) -> pd.DataFrame:
    """Среднее таргета по месяцам train/test для визуального контроля дрейфа."""
    data = df[[config.date_column, target]].copy()
    data[config.date_column] = pd.to_datetime(data[config.date_column], errors="coerce")
    data = data.dropna(subset=[config.date_column])
    data["_month"] = data[config.date_column].dt.to_period("M").astype(str)

    def _split_label(ts: pd.Timestamp) -> str:
        if config.train_period[0] <= str(ts.date()) <= config.train_period[1]:
            return "train"
        if config.test_period[0] <= str(ts.date()) <= config.test_period[1]:
            return "test"
        return "other"

    data["_split"] = data[config.date_column].map(_split_label)
    data = data[data["_split"].isin(["train", "test"])]
    grouped = (
        data.groupby(["_split", "_month"], as_index=False)
        .agg(n=(target, "size"), target_mean=(target, "mean"))
        .rename(columns={"_split": "split", "_month": "month"})
    )
    return grouped.sort_values(["split", "month"]).reset_index(drop=True)
