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


def _categorical_set_diff(
    train: pd.Series,
    test: pd.Series,
    *,
    max_list: int = 15,
) -> tuple[str, str]:
    """Категории, появившиеся в test / пропавшие из train (строки через ``|``)."""
    train_set = set(train.astype("string").fillna("<NA>").unique())
    test_set = set(test.astype("string").fillna("<NA>").unique())
    appeared = sorted(test_set - train_set)
    disappeared = sorted(train_set - test_set)

    def _fmt(values: list[str]) -> str:
        if not values:
            return ""
        head = values[:max_list]
        text = " | ".join(head)
        if len(values) > max_list:
            text += f" | …(+{len(values) - max_list})"
        return text

    return _fmt(appeared), _fmt(disappeared)


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
            row["l1"] = _categorical_share_l1(train_col, test_col)
            row["psi"] = float("nan")
            row["train_nunique"] = int(train_col.nunique(dropna=True))
            row["test_nunique"] = int(test_col.nunique(dropna=True))
            appeared, disappeared = _categorical_set_diff(train_col, test_col)
            row["cats_appeared"] = appeared
            row["cats_disappeared"] = disappeared
        else:
            train_num = pd.to_numeric(train_col, errors="coerce")
            test_num = pd.to_numeric(test_col, errors="coerce")
            row["psi"] = _psi(train_num.to_numpy(dtype=float), test_num.to_numpy(dtype=float))
            row["l1"] = float("nan")
            row["train_mean"] = float(train_num.mean()) if train_num.notna().any() else float("nan")
            row["test_mean"] = float(test_num.mean()) if test_num.notna().any() else float("nan")
            row["mean_delta"] = (
                float(test_num.mean() - train_num.mean())
                if train_num.notna().any() and test_num.notna().any()
                else float("nan")
            )
            row["cats_appeared"] = ""
            row["cats_disappeared"] = ""
        rows.append(row)

    report = pd.DataFrame(rows)
    if report.empty:
        return report
    report["_sort"] = report["psi"].fillna(report["l1"])
    return (
        report.sort_values("_sort", ascending=False, na_position="last")
        .drop(columns=["_sort"])
        .reset_index(drop=True)
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


def filter_features_by_drift(
    df: pd.DataFrame,
    features: Iterable[str],
    *,
    date_column: str,
    reference_period: tuple[str, str],
    compare_period: tuple[str, str],
    threshold: float = 0.5,
    psi_threshold: float | None = None,
    l1_threshold: float | None = None,
    categorical_features: Iterable[str] | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Убрать признаки с сильным дрейфом: PSI (числа) / L1 (каты).

    ``threshold`` — общий fallback, если ``psi_threshold`` / ``l1_threshold`` не заданы.
    NaN score → не дропаем; в ``note`` причина (мало уникальных значений и т.п.).

    Returns:
        kept_features, report с колонками metric / score / dropped / note.
    """
    psi_th = float(threshold if psi_threshold is None else psi_threshold)
    l1_th = float(threshold if l1_threshold is None else l1_threshold)

    data = df.copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    ref = data[data[date_column].between(*reference_period)]
    cmp = data[data[date_column].between(*compare_period)]
    cat_set = set(categorical_features or ())
    feature_list = [f for f in features if f in data.columns]

    rows: list[dict[str, object]] = []
    for column in feature_list:
        ref_col = ref[column]
        cmp_col = cmp[column]
        is_cat = column in cat_set or not pd.api.types.is_numeric_dtype(ref_col)
        note = ""
        if is_cat:
            score = _categorical_share_l1(ref_col, cmp_col)
            kind = "categorical"
            metric = "L1"
            thr = l1_th
            appeared, disappeared = _categorical_set_diff(ref_col, cmp_col)
            if score != score:
                note = "L1 недоступен (пустые доли)"
        else:
            ref_num = pd.to_numeric(ref_col, errors="coerce")
            cmp_num = pd.to_numeric(cmp_col, errors="coerce")
            score = _psi(ref_num.to_numpy(dtype=float), cmp_num.to_numpy(dtype=float))
            kind = "numeric"
            metric = "PSI"
            thr = psi_th
            appeared, disappeared = "", ""
            if score != score:
                nuniq = int(ref_num.nunique(dropna=True))
                note = f"PSI недоступен (уник.≈{nuniq} / мало строк — бины не строятся)"
        drop = bool(score == score and score > thr)
        rows.append(
            {
                "feature": column,
                "kind": kind,
                "metric": metric,
                "score": score,
                "threshold": thr,
                "dropped": drop,
                "note": note,
                "cats_appeared": appeared,
                "cats_disappeared": disappeared,
            }
        )

    report = pd.DataFrame(rows)
    if report.empty:
        return [], report
    report = report.sort_values(
        "score", ascending=False, na_position="last"
    ).reset_index(drop=True)
    dropped = set(report.loc[report["dropped"], "feature"].tolist())
    kept = [f for f in feature_list if f not in dropped]
    return kept, report


def format_psi_filter_report(report: pd.DataFrame) -> str:
    """Читаемая таблица PSI/L1 без лишних NaN-колонок."""
    if report is None or report.empty:
        return "(empty)"
    cols = [
        c
        for c in (
            "feature",
            "metric",
            "score",
            "threshold",
            "dropped",
            "note",
            "cats_appeared",
            "cats_disappeared",
        )
        if c in report.columns
    ]
    view = report[cols].copy()
    if "score" in view.columns:
        view["score"] = view["score"].map(
            lambda x: f"{x:.3f}" if isinstance(x, (int, float)) and x == x else "—"
        )
    return view.to_string(index=False)
