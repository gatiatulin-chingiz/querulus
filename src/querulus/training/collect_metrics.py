"""Collect-style метрики без зависимости от modeldiagnostics.

Используется в ``example.ipynb`` (OutBoxML DSM) и anywhere, где нужен тот же
набор ключей, что в collect/HPO, но без внешнего пакета ModelDiagnostics.
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

TaskKind = Literal["classification", "regression"]


def gini_index(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    policy: str = "accurate",
) -> float:
    """Gini по кривой Лоренца (modeldiagnostics.Gini, без внешней зависимости).

    Строки сортируются по ``y_pred`` (score); накапливается доля ``y_true``.
    Для бинарной классификации ``y_pred`` — proba, порог τ не используется.
    """

    def special_cumsum(v, w):
        return sum(np.cumsum([0] + list(v * w)[:-1]) * w + v * w * (w + 1) / 2)

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true и y_pred должны иметь одинаковую длину")
    n_samples = y_true.shape[0]
    if n_samples == 0:
        return float("nan")

    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[np.lexsort(arr[:, [1, 0]].T, axis=-1)][::-1, 0]
    if policy == "accurate":
        pred_order = (
            pd.DataFrame(arr)
            .groupby([1], observed=True)[0]
            .transform("mean")[arr[:, 1].argsort()]
            .values[::-1]
        )
    elif policy == "fast":
        pred_order = arr[np.lexsort(arr[:, [0, 1]].T, axis=-1)][::-1, 0]
    else:
        raise ValueError(f"Неизвестная policy={policy!r}")

    l_true = np.cumsum(true_order) / np.sum(true_order)
    l_pred = np.cumsum(pred_order) / np.sum(pred_order)
    g_true = float(np.sum(l_true) - (n_samples + 1) / 2)
    g_pred = float(np.sum(l_pred) - (n_samples + 1) / 2)
    if g_true == 0:
        return float("nan")
    return float(g_pred / g_true)


def _to_finite_arrays(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(y_score, dtype=float)
    if len(y) != len(s):
        raise ValueError(f"Длины y_true={len(y)} и y_score={len(s)} не совпадают")
    mask = np.isfinite(y) & np.isfinite(s)
    return y[mask], s[mask]


def classification_gini(
    y_true: pd.Series | np.ndarray,
    proba: np.ndarray,
    *,
    policy: str = "accurate",
) -> float:
    """Gini frequency-модели: Lorenz по вероятностям класса 1."""
    y, p = _to_finite_arrays(y_true, proba)
    if len(y) == 0:
        return float("nan")
    try:
        return gini_index(y.astype(int), p, policy=policy)
    except Exception:  # noqa: BLE001
        return float("nan")


def _finalize_metrics(raw: dict[str, float | int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, (bool, str)):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(number) or math.isinf(number):
            continue
        out[str(key)] = number
    return out


def regression_metrics_bundle(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Регрессия: те же ключи, что ``ModelDiagnostics.compute_regression_metrics``."""
    y, pred = _to_finite_arrays(y_true, y_pred)
    if len(y) == 0:
        return {}

    mse = float(mean_squared_error(y, pred))
    raw: dict[str, float | int] = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        "max_error": float(max_error(y, pred)),
        "bias": float(np.mean(y - pred)),
        "median_error": float(np.median(y - pred)),
    }
    try:
        raw["mape"] = float(mean_absolute_percentage_error(y, pred))
    except Exception:  # noqa: BLE001
        raw["mape"] = float("nan")

    mask = y != 0
    if np.any(mask):
        raw["mpe"] = float(np.mean((y[mask] - pred[mask]) / y[mask]))
    else:
        raw["mpe"] = float("nan")

    try:
        raw["gini"] = gini_index(y, pred)
    except Exception:  # noqa: BLE001
        raw["gini"] = float("nan")

    y_mean = float(np.mean(y))
    raw["shift"] = float(np.mean(pred) / y_mean) if y_mean != 0 else float("nan")
    return _finalize_metrics(raw)


def classification_metrics_at_threshold(
    y_true: pd.Series | np.ndarray,
    proba: np.ndarray,
    *,
    threshold: float,
    include_ece: bool = False,
) -> dict[str, float]:
    """Классификация на одном пороге (как MD с ``thresholds=[0.5]``)."""
    y, p = _to_finite_arrays(y_true, proba)
    if len(y) == 0:
        return {}

    y_int = y.astype(int)
    pred_labels = (p >= float(threshold)).astype(int)

    raw: dict[str, float | int] = {
        "best_threshold": float(threshold),
    }
    try:
        raw["roc_auc"] = float(roc_auc_score(y_int, p))
    except Exception:  # noqa: BLE001
        raw["roc_auc"] = float("nan")
    try:
        raw["pr_auc"] = float(average_precision_score(y_int, p))
    except Exception:  # noqa: BLE001
        raw["pr_auc"] = float("nan")
    raw["gini"] = classification_gini(y_int, p)

    if include_ece:
        from querulus.training.calibration import expected_calibration_error

        try:
            raw["ece"] = float(
                expected_calibration_error(y_int, p, strategy="equal_mass")
            )
        except Exception:  # noqa: BLE001
            raw["ece"] = float("nan")

    tp = int(((y_int == 1) & (pred_labels == 1)).sum())
    tn = int(((y_int == 0) & (pred_labels == 0)).sum())
    fp = int(((y_int == 0) & (pred_labels == 1)).sum())
    fn = int(((y_int == 1) & (pred_labels == 0)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    raw.update(
        {
            "f1_score": float(f1_score(y_int, pred_labels, zero_division=0)),
            "precision_score": float(
                precision_score(y_int, pred_labels, zero_division=0)
            ),
            "recall_score": float(recall_score(y_int, pred_labels, zero_division=0)),
            "sensitivity": float(sensitivity)
            if sensitivity == sensitivity
            else float("nan"),
            "specificity": float(specificity)
            if specificity == specificity
            else float("nan"),
            "youden_index": (
                float(sensitivity + specificity - 1)
                if sensitivity == sensitivity and specificity == specificity
                else float("nan")
            ),
            "mcc": float(matthews_corrcoef(y_int, pred_labels)),
            "shift": (
                float(pred_labels.sum() / y_int.sum())
                if float(y_int.sum()) > 0
                else float("nan")
            ),
        }
    )
    return _finalize_metrics(raw)


def classification_metrics_threshold_free(
    y_true: pd.Series | np.ndarray,
    proba: np.ndarray,
) -> dict[str, float]:
    """ROC/PR/Gini без порога (для HPO до расчёта thr_val)."""
    y, p = _to_finite_arrays(y_true, proba)
    if len(y) == 0:
        return {}
    y_int = y.astype(int)
    raw: dict[str, float | int] = {}
    try:
        raw["roc_auc"] = float(roc_auc_score(y_int, p))
    except Exception:  # noqa: BLE001
        raw["roc_auc"] = float("nan")
    try:
        raw["pr_auc"] = float(average_precision_score(y_int, p))
    except Exception:  # noqa: BLE001
        raw["pr_auc"] = float("nan")
    raw["gini"] = classification_gini(y_int, p)
    return _finalize_metrics(raw)


def metrics_bundle(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    *,
    task_type: TaskKind,
    threshold: float | None = None,
) -> dict[str, float]:
    """Единая точка входа для collect-style bundle."""
    if task_type == "classification":
        if threshold is None:
            return classification_metrics_threshold_free(y_true, y_score)
        return classification_metrics_at_threshold(
            y_true,
            y_score,
            threshold=threshold,
            include_ece=False,
        )
    return regression_metrics_bundle(y_true, y_score)
