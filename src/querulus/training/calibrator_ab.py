"""A/B сравнение raw vs калиброванных вероятностей на шкале финэффекта."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from querulus.fin_effect.calculator import run_fin_effect_pipeline
from querulus.fin_effect.config import FinEffectConfig
from querulus.training.calibration import expected_calibration_error


@dataclass(frozen=True)
class CalibratorAbResult:
    """A/B raw vs cal: таблица + финэффект на каждой шкале proba."""

    table: pd.DataFrame
    fe_raw: Any
    fe_cal: Any


def compare_calibrator_ab(
    df: pd.DataFrame,
    effect_index: pd.Index,
    proba_raw: pd.Series | np.ndarray,
    proba_cal: pd.Series | np.ndarray,
    severity_pred: pd.Series | np.ndarray,
    y_true_freq: pd.Series | np.ndarray,
    *,
    config: Any | None = None,
    val_threshold: float | None = None,
    title: str | None = None,
    print_summary: bool = True,
) -> CalibratorAbResult:
    """Сравнить raw vs cal: ECE + порог + финэффект на той же шкале proba.

    Порог — один с Val (``val_threshold`` или подбор на ``effect_index`` по raw proba).
    """

    cfg = config or FinEffectConfig()
    aligned = df.loc[effect_index]
    index = aligned.index

    def _as_series(values: pd.Series | np.ndarray) -> pd.Series:
        if isinstance(values, pd.Series):
            return values.reindex(index)
        return pd.Series(np.asarray(values, dtype=float), index=index)

    proba_raw_s = _as_series(proba_raw)
    proba_cal_s = _as_series(proba_cal)
    sev_s = _as_series(severity_pred)
    if isinstance(y_true_freq, pd.Series):
        y_s = y_true_freq.reindex(index)
    else:
        y_s = pd.Series(np.asarray(y_true_freq), index=index)

    fe_raw = run_fin_effect_pipeline(
        aligned,
        proba_raw_s,
        sev_s,
        y_s,
        threshold=None,
        config=cfg,
    )
    if val_threshold is None:
        val_threshold = float(fe_raw.best_threshold)
    thr = float(val_threshold)
    fe_raw = run_fin_effect_pipeline(
        aligned,
        proba_raw_s,
        sev_s,
        y_s,
        threshold=thr,
        config=cfg,
    )
    fe_cal = run_fin_effect_pipeline(
        aligned,
        proba_cal_s,
        sev_s,
        y_s,
        threshold=thr,
        config=cfg,
    )

    rows: list[dict[str, float | str]] = []
    for name, proba, fe in (
        ("raw", proba_raw_s, fe_raw),
        ("cal", proba_cal_s, fe_cal),
    ):
        pred = (proba.to_numpy(dtype=float) >= thr).astype(int)
        rows.append(
            {
                "branch": name,
                "ece": expected_calibration_error(y_s, proba, strategy="equal_mass"),
                "best_threshold": thr,
                "net_effect": float(fe.net_effect),
                "model_effect": float(fe.model_effect_total),
                "fact_effect": float(fe.fact_effect_total),
                "pred_rate": float(pred.mean()) if len(pred) else float("nan"),
                "n": float(len(index)),
            }
        )
    table = pd.DataFrame(rows).set_index("branch")

    if print_summary:
        label = title or "calibrator A/B"
        print(f"\n=== {label} ===")
        print(
            table.to_string(
                float_format=lambda x: f"{x:,.2f}" if abs(x) < 1e6 else f"{x:,.0f}"
            )
        )
        delta_net = float(fe_cal.net_effect) - float(fe_raw.net_effect)
        delta_ece = float(table.loc["cal", "ece"]) - float(table.loc["raw", "ece"])
        print(
            f"Δ net_effect (cal−raw)={delta_net:,.0f} ₽; "
            f"Δ ECE (cal−raw)={delta_ece:+.2f}"
        )

    return CalibratorAbResult(table=table, fe_raw=fe_raw, fe_cal=fe_cal)
