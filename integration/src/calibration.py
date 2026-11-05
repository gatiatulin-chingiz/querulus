"""Калибраторы трека second_ (модель 2.0.0).

Контракт артефактов (фиксированный):
* **CF** — ``cf_calibrator.pickle``: 1D ``IsotonicRegression`` по raw score;
* **RG** — ``rg_calibrator.pickle``: 1D isotonic по raw регрессии;
  отрицательные значения после калибровки обрезаются до 0.

Публичный API: ``apply_cf_calibrator``, ``apply_rg_calibrator``.
"""
from __future__ import annotations

from typing import Any, List

import numpy as np


# reviewed
def apply_cf_calibrator(
    calibrator: Any,
    raw_scores: List[float],
) -> List[float]:
    """Откалибровать вероятности классификации (CF): isotonic ``predict(raw)``.

    Args:
        calibrator: загруженный CF-калибратор (isotonic; грузит preload).
        raw_scores: сырые P(класс 1) / скоры модели до калибровки.

    Returns:
        Список откалиброванных вероятностей той же длины, что ``raw_scores``.
    """
    arr = np.asarray(
        calibrator.predict(np.asarray(raw_scores, dtype=float)), dtype=float
    )
    return [float(x) for x in arr.ravel()]


# reviewed
def apply_rg_calibrator(
    calibrator: Any,
    raw_reg: List[float],
) -> List[float]:
    """Откалибровать регрессию (RG): isotonic ``predict(raw)``, затем clip ≥ 0.

    Args:
        calibrator: загруженный RG-калибратор (isotonic; грузит preload).
        raw_reg: сырые предсказания регрессии.

    Returns:
        Список неотрицательных откалиброванных значений.
    """
    arr = np.asarray(
        calibrator.predict(np.asarray(raw_reg, dtype=float)), dtype=float
    )
    return [float(max(0.0, x)) for x in arr.ravel()]
