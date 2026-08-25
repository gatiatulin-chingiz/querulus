"""Калибровка вероятностей и ECE (positive-class, бинарная частота)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

BinStrategy = Literal["equal_mass", "equal_width"]


@dataclass(frozen=True)
class CalibratorAbResult:
    """A/B raw vs cal: таблица + финэффект на каждой шкале proba."""

    table: pd.DataFrame
    fe_raw: Any
    fe_cal: Any


def expected_calibration_error(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    *,
    n_bins: int = 10,
    strategy: BinStrategy = "equal_mass",
) -> float:
    """Positive-class ECE: сравнивает freq(y=1) и mean p в бинах.

    Не top-label accuracy: ``y_prob`` — P(y=1). При дисбалансе это честнее
    argmax-ECE. ``equal_mass`` (квантили) — дефолт; ``equal_width`` — классика.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) == 0:
        return float("nan")
    n_bins = max(2, int(n_bins))
    if strategy == "equal_width":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "equal_mass":
        edges = _equal_mass_edges(p, n_bins)
    else:
        raise ValueError(f"Неизвестная strategy={strategy!r}")
    return _ece_from_edges(y, p, edges)


def _equal_mass_edges(p: np.ndarray, n_bins: int) -> np.ndarray:
    """Границы бинов по квантилям p; дубликаты схлопываются."""
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(p, quantiles))
    if len(edges) < 2:
        return np.array([0.0, 1.0], dtype=float)
    # Гарантируем покрытие [min, max] предсказаний.
    edges[0] = min(float(edges[0]), float(p.min()))
    edges[-1] = max(float(edges[-1]), float(p.max()))
    return edges.astype(float)


def _ece_from_edges(y: np.ndarray, p: np.ndarray, edges: np.ndarray) -> float:
    ece = 0.0
    n_intervals = len(edges) - 1
    for i in range(n_intervals):
        lo, hi = edges[i], edges[i + 1]
        if i < n_intervals - 1:
            in_bin = (p >= lo) & (p < hi)
        else:
            in_bin = (p >= lo) & (p <= hi)
        if not np.any(in_bin):
            continue
        # weight = |Bm|/N; freq/conf — positive-class (не argmax-acc).
        weight = float(in_bin.mean())
        freq = float(y[in_bin].mean())
        conf = float(p[in_bin].mean())
        ece += weight * abs(freq - conf)
    return float(ece)


def balance_binary_cal_frame(
    x_cal: pd.DataFrame,
    y_cal: pd.Series,
    *,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Downsample majority до размера minority (для обучения калибратора).

    ECE всегда считайте на полном Cal; этот кадр — только fit Platt/Isotonic.
    """
    y = y_cal.astype(int)
    pos_idx = y.index[y == 1]
    neg_idx = y.index[y == 0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return x_cal, y_cal
    n = int(min(len(pos_idx), len(neg_idx)))
    rng = np.random.default_rng(random_state)
    if len(pos_idx) > n:
        pos_idx = pos_idx[rng.choice(len(pos_idx), size=n, replace=False)]
    if len(neg_idx) > n:
        neg_idx = neg_idx[rng.choice(len(neg_idx), size=n, replace=False)]
    keep = pd.Index(np.concatenate([pos_idx.to_numpy(), neg_idx.to_numpy()]))
    keep = pd.Series(keep).sample(frac=1.0, random_state=random_state).to_numpy()
    return x_cal.loc[keep].copy(), y_cal.loc[keep].copy()


def fit_probability_calibrator(
    model: object,
    x_cal: pd.DataFrame,
    y_cal: pd.Series,
    *,
    method: str = "isotonic",
    balance: bool = True,
    random_state: int = 42,
):
    """Калибратор на отдельном Cal-set (cv='prefit').

    ``balance=True``: учим на downsampled majority (честнее при дисбалансе
    для P(y=1)); оценку ECE делайте на полном Cal.
    """
    from sklearn.calibration import CalibratedClassifierCV

    x_fit, y_fit = x_cal, y_cal
    if balance:
        x_fit, y_fit = balance_binary_cal_frame(
            x_cal, y_cal, random_state=random_state
        )
    calibrator = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrator.fit(x_fit, y_fit.astype(int))
    return calibrator


def compare_calibrator_ab(
    df: pd.DataFrame,
    effect_index: pd.Index,
    proba_raw: pd.Series | np.ndarray,
    proba_cal: pd.Series | np.ndarray,
    severity_pred: pd.Series | np.ndarray,
    y_true_freq: pd.Series | np.ndarray,
    *,
    config: Any | None = None,
    title: str | None = None,
    print_summary: bool = True,
) -> CalibratorAbResult:
    """Сравнить raw vs cal: ECE + порог + финэффект на той же шкале proba.

    Порог для каждой ветки ищется заново на её вероятностях (``threshold=None``).
    """
    from querulus.fin_effect.calculator import run_fin_effect_pipeline
    from querulus.fin_effect.config import FinEffectConfig

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
    fe_cal = run_fin_effect_pipeline(
        aligned,
        proba_cal_s,
        sev_s,
        y_s,
        threshold=None,
        config=cfg,
    )

    rows: list[dict[str, float | str]] = []
    for name, proba, fe in (
        ("raw", proba_raw_s, fe_raw),
        ("cal", proba_cal_s, fe_cal),
    ):
        pred = (proba.to_numpy(dtype=float) >= float(fe.best_threshold)).astype(int)
        rows.append(
            {
                "branch": name,
                "ece": expected_calibration_error(y_s, proba, strategy="equal_mass"),
                "best_threshold": float(fe.best_threshold),
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
                float_format=lambda x: f"{x:,.4f}" if abs(x) < 1e6 else f"{x:,.0f}"
            )
        )
        delta_net = float(fe_cal.net_effect) - float(fe_raw.net_effect)
        delta_ece = float(table.loc["cal", "ece"]) - float(table.loc["raw", "ece"])
        print(
            f"Δ net_effect (cal−raw)={delta_net:,.0f} ₽; "
            f"Δ ECE (cal−raw)={delta_ece:+.4f}"
        )

    return CalibratorAbResult(table=table, fe_raw=fe_raw, fe_cal=fe_cal)
