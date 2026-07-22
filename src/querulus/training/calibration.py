"""Калибровка вероятностей и ECE."""
from __future__ import annotations

import numpy as np
import pandas as pd


def expected_calibration_error(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (меньше — лучше)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if not np.any(in_bin):
            continue
        ece += (in_bin.mean()) * abs(y[in_bin].mean() - p[in_bin].mean())
    return float(ece)


def fit_probability_calibrator(
    model: object,
    x_cal: pd.DataFrame,
    y_cal: pd.Series,
    *,
    method: str = "isotonic",
):
    """Калибратор на отдельном Cal-set (cv='prefit')."""
    from sklearn.calibration import CalibratedClassifierCV

    calibrator = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrator.fit(x_cal, y_cal.astype(int))
    return calibrator
