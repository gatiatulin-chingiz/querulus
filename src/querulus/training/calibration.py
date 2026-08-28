"""Калибровка вероятностей (частота) и уровней сумм (severity)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

BinStrategy = Literal["equal_mass", "equal_width"]
SeverityCalMethod = Literal["isotonic", "affine", "scale"]


@dataclass
class SeverityCalibrator:
    """Постобработка raw severity → откалиброванный уровень суммы.

    Учится на парах (pred, fact) с ``fact > 0``; на инференсе применяется ко всем pred.
    """

    method: SeverityCalMethod
    n_fit: int
    bias_before: float
    bias_after: float
    _iso: Any = None
    _intercept: float = 0.0
    _coef: float = 1.0
    _scale: float = 1.0

    def predict(self, y_pred: pd.Series | np.ndarray) -> np.ndarray:
        """Сырой pred → откалиброванный; результат ``>= 0``."""
        raw = np.asarray(y_pred, dtype=float).reshape(-1)
        out = np.full(raw.shape, np.nan, dtype=float)
        ok = np.isfinite(raw)
        if not np.any(ok):
            return out
        x = raw[ok]
        if self.method == "isotonic":
            cal = np.asarray(self._iso.predict(x), dtype=float)
        elif self.method == "affine":
            cal = self._intercept + self._coef * x
        elif self.method == "scale":
            cal = self._scale * x
        else:
            raise ValueError(f"Неизвестный method={self.method!r}")
        out[ok] = np.maximum(cal, 0.0)
        return out

    def predict_series(self, y_pred: pd.Series) -> pd.Series:
        """То же, что ``predict``, с сохранением index."""
        return pd.Series(self.predict(y_pred), index=y_pred.index, dtype=float)


def severity_mean_bias(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> float:
    """Средний bias = mean(pred − fact) на конечных парах."""
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y) != len(p):
        raise ValueError(f"len(y_true)={len(y)} != len(y_pred)={len(p)}")
    mask = np.isfinite(y) & np.isfinite(p)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(p[mask] - y[mask]))


def fit_severity_calibrator(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    method: SeverityCalMethod = "isotonic",
    min_samples: int = 50,
) -> SeverityCalibrator:
    """Калибратор уровня суммы на Cal-set (только строки с fact > 0).

    Methods:
        isotonic — монотонное отображение pred→fact (sklearn IsotonicRegression);
        affine — fact ≈ a + b·pred (OLS);
        scale — fact ≈ s·pred, ``s = mean(fact)/mean(pred)``.
    """
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y) != len(p):
        raise ValueError(f"len(y_true)={len(y)} != len(y_pred)={len(p)}")
    mask = np.isfinite(y) & np.isfinite(p) & (y > 0)
    n = int(mask.sum())
    if n < int(min_samples):
        raise ValueError(
            f"Мало точек для severity-калибровки: n={n} < min_samples={min_samples}"
        )
    y_fit, p_fit = y[mask], p[mask]
    bias_before = float(np.mean(p_fit - y_fit))

    iso = None
    intercept, coef, scale = 0.0, 1.0, 1.0
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(y_min=0.0, out_of_bounds="clip")
        iso.fit(p_fit, y_fit)
        cal_fit = np.asarray(iso.predict(p_fit), dtype=float)
    elif method == "affine":
        # polyfit: y ≈ coef * p + intercept
        coef, intercept = [float(v) for v in np.polyfit(p_fit, y_fit, deg=1)]
        cal_fit = intercept + coef * p_fit
    elif method == "scale":
        mean_p = float(np.mean(p_fit))
        if abs(mean_p) < 1e-12:
            raise ValueError("mean(pred)≈0 — scale-калибровка невозможна")
        scale = float(np.mean(y_fit) / mean_p)
        cal_fit = scale * p_fit
    else:
        raise ValueError(f"Неизвестный method={method!r}")

    cal_fit = np.maximum(cal_fit, 0.0)
    bias_after = float(np.mean(cal_fit - y_fit))
    return SeverityCalibrator(
        method=method,
        n_fit=n,
        bias_before=bias_before,
        bias_after=bias_after,
        _iso=iso,
        _intercept=intercept,
        _coef=coef,
        _scale=scale,
    )


def apply_severity_calibrator(
    calibrator: SeverityCalibrator | None,
    y_pred: pd.Series | np.ndarray,
) -> pd.Series | np.ndarray:
    """Применить калибратор; ``None`` → вернуть ``y_pred`` без изменений."""
    if calibrator is None:
        return y_pred
    if isinstance(y_pred, pd.Series):
        return calibrator.predict_series(y_pred)
    return calibrator.predict(y_pred)


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
    val_threshold: float | None = None,
    title: str | None = None,
    print_summary: bool = True,
) -> CalibratorAbResult:
    """Сравнить raw vs cal: ECE + порог + финэффект на той же шкале proba.

    Порог — один с Val (``val_threshold`` или подбор на ``effect_index`` по raw proba).
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
