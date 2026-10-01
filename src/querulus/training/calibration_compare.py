"""Сравнение двух калибровок CF: ClassificationCalibration-стиль (train) vs Querulus (cal).

1) ``ClassificationCalibration`` / эквивалент: isotonic ``cv='prefit'`` на **DSM train**.
2) Querulus ``fit_probability_calibrator``: isotonic на **отдельном cal-индексе**.

Reliability / Brier / ECE — на train и test DSM (оценка; учить калибратор на train
для схемы 1 — методически плохо, здесь для явного сравнения со старым примером).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

from querulus.training.build_outboxml_configs import (
    ensure_predictable_model,
    unwrap_estimator,
)
from querulus.training.calibration import (
    expected_calibration_error,
    fit_probability_calibrator,
)
from querulus.training.outboxml_metrics import prepare_dsm_features

SplitName = Literal["train", "test"]


@dataclass
class ClassificationCalibration:
    """Эквивалент ``calibration/mldw_pipelines.ClassificationCalibration`` для OutBoxML DSM.

    Учит isotonic на **train** DSM (``cv='prefit'``). Не тащит mldataworker.
    """

    manager: Any
    model_name: str
    method: str = "isotonic"
    _calibrated_model: Any = None
    _feature_columns: list[str] = field(default_factory=list)

    def fit_transform(self) -> Any:
        result = self.manager.get_result()[self.model_name]
        subset = result.data_subset
        num = list(subset.features_numerical or [])
        cat = list(subset.features_categorical or [])
        cols = [*num, *cat]
        self._feature_columns = cols
        x_train = subset.X_train.loc[:, cols]
        y_train = subset.y_train
        estimator = unwrap_estimator(ensure_predictable_model(result.model))
        calibrator = CalibratedClassifierCV(
            estimator, method=self.method, cv="prefit"
        )
        sample_weight = None
        if getattr(subset, "exposure_train", None) is not None:
            sample_weight = subset.exposure_train
        try:
            calibrator.fit(x_train, y_train.astype(int), sample_weight=sample_weight)
        except TypeError:
            calibrator.fit(x_train, y_train.astype(int))
        self._calibrated_model = calibrator

        x_test = subset.X_test.loc[:, cols]
        pred_train = pd.Series(
            calibrator.predict_proba(x_train)[:, 1], index=x_train.index, dtype=float
        )
        pred_test = pd.Series(
            calibrator.predict_proba(x_test)[:, 1], index=x_test.index, dtype=float
        )
        # Как в mldw: подменяем predictions в result (если API есть).
        load = getattr(result, "load_predictions", None)
        if callable(load):
            load(pred_train, "train")
            load(pred_test, "test")
        return calibrator

    def predict_proba_positive(self, x: pd.DataFrame) -> pd.Series:
        if self._calibrated_model is None:
            raise RuntimeError("Сначала вызовите fit_transform()")
        cols = self._feature_columns or list(x.columns)
        matrix = x.loc[:, cols]
        return pd.Series(
            self._calibrated_model.predict_proba(matrix)[:, 1],
            index=matrix.index,
            dtype=float,
        )


@dataclass
class CalibrationSliceScores:
    y_true: pd.Series
    raw: pd.Series
    mldw: pd.Series
    ours: pd.Series


@dataclass
class CalibrationCompareResult:
    """Результаты двух калибровок + метрики + данные для графиков."""

    model_name: str
    cal_index: pd.Index
    cal_n: int
    mldw_calibrator: Any
    ours_calibrator: Any
    train: CalibrationSliceScores
    test: CalibrationSliceScores
    metrics: pd.DataFrame

    def plot_reliability(self, *, n_bins: int = 10, figsize: tuple[int, int] = (12, 8)):
        """Reliability (calibration curve) train/test × raw / mldw-train / ours-cal."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=True, sharey=True)
        specs = [
            ("raw", "Сырой CF"),
            ("mldw", "ClassificationCalibration\n(isotonic на train)"),
            ("ours", "Querulus\n(isotonic на cal)"),
        ]
        for row, (split_name, slice_scores) in enumerate(
            (("train", self.train), ("test", self.test))
        ):
            for col, (attr, title) in enumerate(specs):
                ax = axes[row, col]
                y = slice_scores.y_true
                p = getattr(slice_scores, attr)
                _plot_one_reliability(ax, y, p, n_bins=n_bins)
                ax.set_title(f"{split_name.upper()} · {title}", fontsize=10)
                if row == 1:
                    ax.set_xlabel("Mean predicted P(y=1)")
                if col == 0:
                    ax.set_ylabel("Fraction of positives")
        fig.suptitle(
            f"Calibration curves · {self.model_name} · cal n={self.cal_n}",
            fontsize=12,
        )
        fig.tight_layout()
        return fig


def _positive_proba(estimator: Any, x: pd.DataFrame) -> pd.Series:
    if hasattr(estimator, "predict_proba"):
        scores = np.asarray(estimator.predict_proba(x)[:, 1], dtype=float)
    else:
        scores = np.asarray(estimator.predict(x), dtype=float)
    return pd.Series(scores, index=x.index, dtype=float)


def _dsm_xy(dsm: Any, model_name: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    result = dsm.get_result()[model_name]
    subset = result.data_subset
    num = list(subset.features_numerical or [])
    cat = list(subset.features_categorical or [])
    cols = [*num, *cat]
    return (
        subset.X_train.loc[:, cols],
        subset.y_train.astype(int),
        subset.X_test.loc[:, cols],
        subset.y_test.astype(int),
        cols,
    )


def resolve_cal_index(periods: dict[str, Any], *, prefer_prod_tau: bool) -> pd.Index:
    """Индекс cal: для prod — ``prod_tau_cal_idx``, иначе ``splits.cal``."""
    if prefer_prod_tau:
        idx = periods.get("prod_tau_cal_idx")
        if idx is not None and len(idx) > 0:
            return pd.Index(idx)
    splits = periods.get("splits")
    if splits is not None and getattr(splits, "cal", None) is not None:
        cal = splits.cal
        if len(cal) > 0:
            return pd.Index(cal)
    # Фоллбек: хвост train / пересечение
    train = getattr(splits, "train", None) if splits is not None else None
    if train is not None and len(train) > 0:
        n = max(1, int(round(0.15 * len(train))))
        return pd.Index(train[-n:])
    raise ValueError("Не найден cal-индекс в periods (prod_tau_cal_idx / splits.cal)")


def _slice_metrics(y: pd.Series, p: pd.Series, *, split: str, kind: str) -> dict[str, Any]:
    y_a = y.astype(float)
    p_a = p.astype(float)
    mask = y_a.notna() & p_a.notna()
    y_m, p_m = y_a[mask], p_a[mask]
    return {
        "split": split,
        "kind": kind,
        "n": int(mask.sum()),
        "brier": float(brier_score_loss(y_m, p_m)) if len(y_m) else float("nan"),
        "ece": float(
            expected_calibration_error(y_m, p_m, n_bins=10, strategy="equal_mass")
        )
        if len(y_m)
        else float("nan"),
        "mean_pred": float(p_m.mean()) if len(p_m) else float("nan"),
        "base_rate": float(y_m.mean()) if len(y_m) else float("nan"),
    }


def _plot_one_reliability(ax: Any, y: pd.Series, p: pd.Series, *, n_bins: int) -> None:
    y_a = np.asarray(y, dtype=float)
    p_a = np.asarray(p, dtype=float)
    mask = np.isfinite(y_a) & np.isfinite(p_a)
    y_a, p_a = y_a[mask], p_a[mask]
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="ideal")
    if len(y_a) < 20:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        return
    frac_pos, mean_pred = calibration_curve(
        y_a, p_a, n_bins=n_bins, strategy="quantile"
    )
    ax.plot(mean_pred, frac_pos, "o-", color="#1f4e79", label="reliability")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)


def compare_cf_calibrations(
    dsm: Any,
    *,
    model_name: str,
    df: pd.DataFrame,
    periods: dict[str, Any],
    prefer_prod_tau_cal: bool = False,
    method: str = "isotonic",
    balance_ours: bool = True,
) -> CalibrationCompareResult:
    """Две калибровки CF + метрики на DSM train/test.

    Parameters
    ----------
    prefer_prod_tau_cal:
        True после prod-fit (окно τ-cal); False для parity (``splits.cal``).
    """
    x_train, y_train, x_test, y_test, cols = _dsm_xy(dsm, model_name)
    result = dsm.get_result()[model_name]
    raw_estimator = unwrap_estimator(ensure_predictable_model(result.model))

    # --- 1) ClassificationCalibration (train) ---
    mldw = ClassificationCalibration(
        manager=dsm, model_name=model_name, method=method
    )
    mldw_calibrator = mldw.fit_transform()

    # --- 2) Querulus на cal-индексе ---
    cal_index = resolve_cal_index(periods, prefer_prod_tau=prefer_prod_tau_cal)
    cal_index = cal_index.intersection(df.index)
    if len(cal_index) < 30:
        raise ValueError(
            f"Слишком мало строк cal для калибровки: n={len(cal_index)} (нужно ≥30)"
        )
    x_cal = prepare_dsm_features(dsm, model_name, df.loc[cal_index])
    # выровнять колонки под DSM matrix
    missing = [c for c in cols if c not in x_cal.columns]
    if missing:
        raise ValueError(f"В prepared cal нет колонок DSM: {missing[:8]}")
    x_cal = x_cal.loc[:, cols]
    y_cal = df.loc[x_cal.index, result.model_config.column_target].astype(int)
    ours_calibrator = fit_probability_calibrator(
        raw_estimator,
        x_cal,
        y_cal,
        method=method,
        balance=balance_ours,
    )

    def _pack(x: pd.DataFrame, y: pd.Series) -> CalibrationSliceScores:
        raw = _positive_proba(raw_estimator, x)
        mldw_p = _positive_proba(mldw_calibrator, x)
        ours_p = _positive_proba(ours_calibrator, x)
        return CalibrationSliceScores(
            y_true=y.astype(int),
            raw=raw,
            mldw=mldw_p,
            ours=ours_p,
        )

    train_scores = _pack(x_train, y_train)
    test_scores = _pack(x_test, y_test)

    rows = []
    for split, scores in (("train", train_scores), ("test", test_scores)):
        for kind, series in (
            ("raw", scores.raw),
            ("mldw_train_cal", scores.mldw),
            ("querulus_cal", scores.ours),
        ):
            rows.append(_slice_metrics(scores.y_true, series, split=split, kind=kind))
    metrics = pd.DataFrame(rows)

    return CalibrationCompareResult(
        model_name=model_name,
        cal_index=cal_index,
        cal_n=len(cal_index),
        mldw_calibrator=mldw_calibrator,
        ours_calibrator=ours_calibrator,
        train=train_scores,
        test=test_scores,
        metrics=metrics,
    )
