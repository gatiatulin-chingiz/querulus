"""Диагностика severity: вклад хвоста в ошибку и сравнение с log1p-таргетом."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from querulus.training.config import TrainingConfig
from querulus.training.pipeline import TrainingArtifacts, _require_catboost


def severity_error_by_quantile(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    quantiles: tuple[float, ...] = (0.0, 0.5, 0.9, 0.99, 1.0),
) -> pd.DataFrame:
    """Разрез ошибки severity по квантилям факта (true).

    Гипотеза «бьёт хвост»: в верхних бинах большая доля abs_err_sum / true_sum.
    """
    yt = pd.Series(np.asarray(y_true, dtype=float)).reset_index(drop=True)
    yp = pd.Series(np.asarray(y_pred, dtype=float)).reset_index(drop=True)
    mask = yt.notna() & yp.notna()
    yt = yt[mask]
    yp = yp[mask]
    if yt.empty:
        return pd.DataFrame()

    q_vals = sorted({float(q) for q in quantiles})
    if q_vals[0] > 0:
        q_vals = [0.0, *q_vals]
    if q_vals[-1] < 1:
        q_vals = [*q_vals, 1.0]
    edges = yt.quantile(q_vals).to_numpy(dtype=float)
    # Уникальные границы (при большом числе нулей квантили слипаются).
    uniq_edges = np.unique(edges)
    if len(uniq_edges) < 2:
        bins = pd.Series(["all"] * len(yt))
    else:
        bins = pd.cut(yt, bins=uniq_edges, include_lowest=True, duplicates="drop")

    abs_err = (yp - yt).abs()
    frame = pd.DataFrame({"y_true": yt, "y_pred": yp, "abs_err": abs_err, "bin": bins})
    total_abs = float(abs_err.sum())
    total_true = float(yt.sum())
    rows: list[dict[str, object]] = []
    for bin_label, group in frame.groupby("bin", observed=True):
        abs_sum = float(group["abs_err"].sum())
        true_sum = float(group["y_true"].sum())
        rows.append(
            {
                "bin": str(bin_label),
                "n": int(len(group)),
                "true_sum": true_sum,
                "true_share": true_sum / total_true if total_true else float("nan"),
                "abs_err_sum": abs_sum,
                "abs_err_share": abs_sum / total_abs if total_abs else float("nan"),
                "mae": float(group["abs_err"].mean()),
                "bias": float((group["y_pred"] - group["y_true"]).mean()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out.attrs["tail_abs_err_share_p90"] = float(
            out.loc[out["true_share"].cumsum() >= 0.9, "abs_err_share"].sum()
        ) if "true_share" in out else float("nan")
    return out


@dataclass(frozen=True)
class SeverityLog1pCompare:
    """Сравнение raw vs log1p severity на том же сплите/фичах."""

    quantile_raw: pd.DataFrame
    quantile_log1p: pd.DataFrame
    summary: pd.DataFrame
    y_pred_log1p: np.ndarray


def compare_severity_log1p(
    training: TrainingArtifacts,
    config: TrainingConfig | None = None,
) -> SeverityLog1pCompare:
    """Обучить severity на log1p(y) и сравнить ошибку по квантилям с raw-моделью."""
    if training.severity_split is None:
        raise ValueError("severity_split отсутствует в TrainingArtifacts")
    config = config or TrainingConfig()
    CatBoostClassifier, CatBoostRegressor, Pool, *_ = _require_catboost()
    del CatBoostClassifier  # не используется

    split = training.severity_split
    features = training.severity_features
    cat_features = training.severity_categorical_features
    y_train = np.asarray(split.y_train, dtype=float)
    y_test = np.asarray(split.y_test, dtype=float)
    # log1p только для y>=0; отрицательные (если появятся) клипуем в 0.
    y_train_log = np.log1p(np.clip(y_train, a_min=0.0, a_max=None))
    y_test_log = np.log1p(np.clip(y_test, a_min=0.0, a_max=None))

    train_pool = Pool(
        split.x_train[features],
        y_train_log,
        cat_features=cat_features,
        feature_names=features,
    )
    test_pool = Pool(
        split.x_test[features],
        y_test_log,
        cat_features=cat_features,
        feature_names=features,
    )
    model = CatBoostRegressor(
        iterations=config.severity_iterations,
        random_state=config.severity_random_state,
        **config.severity_regressor_params,
    )
    model.fit(train_pool, eval_set=test_pool, plot=False)
    y_pred_log = np.expm1(np.asarray(model.predict(split.x_test[features]), dtype=float))
    y_pred_log = np.clip(y_pred_log, a_min=0.0, a_max=None)

    y_pred_raw = np.asarray(
        training.severity_model.predict(split.x_test[features]),
        dtype=float,
    )
    q_raw = severity_error_by_quantile(y_test, y_pred_raw)
    q_log = severity_error_by_quantile(y_test, y_pred_log)

    def _global(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, object]]:
        err = y_pred - y_true
        return [
            {"model": name, "metric": "mae", "value": float(np.nanmean(np.abs(err)))},
            {"model": name, "metric": "bias", "value": float(np.nanmean(err))},
            {"model": name, "metric": "pred_sum", "value": float(np.nansum(y_pred))},
            {"model": name, "metric": "true_sum", "value": float(np.nansum(y_true))},
            {
                "model": name,
                "metric": "sum_bias",
                "value": float(np.nansum(y_pred) - np.nansum(y_true)),
            },
        ]

    summary = pd.DataFrame(_global("raw", y_test, y_pred_raw) + _global("log1p", y_test, y_pred_log))
    return SeverityLog1pCompare(
        quantile_raw=q_raw,
        quantile_log1p=q_log,
        summary=summary,
        y_pred_log1p=y_pred_log,
    )
