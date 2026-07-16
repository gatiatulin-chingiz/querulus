"""Единое сравнение severity-вариантов: метрики, квантили, fin-effect."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from querulus.fin_effect.calculator import FinEffectResult
from querulus.fin_effect.compare_report import plain_floats
from querulus.fin_effect.config import FinEffectConfig
from querulus.fin_effect.segment_eval import (
    _effect_index_and_freq,
    _fin_on_index,
    _value_mask,
    fin_effect_penalty_table,
    run_fin_effect_with_severity_predictions,
)
from querulus.training.config import TrainingConfig
from querulus.training.pipeline import TrainingArtifacts
from querulus.training.severity_diagnostics import severity_error_by_quantile
from querulus.training.severity_training import (
    SeveritySampleWeight,
    SeverityTargetTransform,
    fit_severity_model,
    severity_predict,
)

SegmentSide = Literal["all", "le", "gt"]


@dataclass(frozen=True)
class SeverityZooModel:
    """Одна severity-модель зоопарка."""

    name: str
    transform: SeverityTargetTransform
    sample_weight: SeveritySampleWeight
    segment: SegmentSide
    model: object
    eval_index: pd.Index
    y_pred: np.ndarray
    y_true: np.ndarray
    fin_effect: FinEffectResult | None


@dataclass(frozen=True)
class SeverityZooCompare:
    """Сводные таблицы по всем severity-вариантам."""

    metrics_summary: pd.DataFrame
    quantile_summary: pd.DataFrame
    fin_effect_summary: pd.DataFrame
    penalty_summary: pd.DataFrame
    bin_axis: str
    value_column: str
    value_threshold: float
    models: dict[str, SeverityZooModel]


def _shared_quantile_edges(
    y_true: pd.Series | np.ndarray,
    quantiles: tuple[float, ...] = (0.0, 0.5, 0.9, 0.99, 1.0),
) -> np.ndarray | None:
    yt = pd.Series(np.asarray(y_true, dtype=float)).dropna()
    if yt.empty:
        return None
    q_vals = sorted({float(q) for q in quantiles})
    if q_vals[0] > 0:
        q_vals = [0.0, *q_vals]
    if q_vals[-1] < 1:
        q_vals = [*q_vals, 1.0]
    edges = np.unique(yt.quantile(q_vals).to_numpy(dtype=float))
    return edges if len(edges) >= 2 else None


def _quantile_table_for_preds(
    *,
    variant: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bin_axis: str,
    bin_edges: np.ndarray | None,
) -> pd.DataFrame:
    """Квантили по фиксированным границам ``bin_axis`` (true severity)."""
    yt = pd.Series(np.asarray(y_true, dtype=float))
    yp = pd.Series(np.asarray(y_pred, dtype=float))
    mask = yt.notna() & yp.notna()
    yt = yt[mask].reset_index(drop=True)
    yp = yp[mask].reset_index(drop=True)
    if yt.empty:
        return pd.DataFrame()

    if bin_edges is None or len(bin_edges) < 2:
        table = severity_error_by_quantile(yt, yp)
    else:
        bins = pd.cut(yt, bins=bin_edges, include_lowest=True, duplicates="drop")
        abs_err = (yp - yt).abs()
        frame = pd.DataFrame({"y_true": yt, "y_pred": yp, "abs_err": abs_err, "bin": bins})
        total_abs = float(abs_err.sum()) or float("nan")
        total_true = float(yt.clip(lower=0).sum()) or float("nan")
        rows: list[dict[str, object]] = []
        for bin_label, group in frame.groupby("bin", observed=True):
            abs_sum = float(group["abs_err"].sum())
            true_sum = float(group["y_true"].clip(lower=0).sum())
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
        table = pd.DataFrame(rows)

    if table.empty:
        return table
    table = table.copy()
    table.insert(0, "variant", variant)
    table.insert(1, "bin_axis", bin_axis)
    return table


def _segment_indices(
    df: pd.DataFrame,
    index: pd.Index,
    value_column: str,
    value_threshold: float,
    side: SegmentSide,
) -> pd.Index:
    if side == "all":
        return index
    mask = _value_mask(df, index, value_column, value_threshold)
    if side == "gt":
        mask = ~mask.fillna(False)
    return index[mask.to_numpy()]


def run_severity_zoo_compare(
    df: pd.DataFrame,
    training: TrainingArtifacts,
    fin_config: FinEffectConfig,
    training_config: TrainingConfig | None = None,
    *,
    split: str = "test",
    value_column: str = "VALUE_BEFORE_WITH",
    value_threshold: float = 50_000.0,
    threshold: float | None = None,
) -> SeverityZooCompare:
    """Обучить все severity-варианты и собрать две сводные таблицы + fin-effect.

    Варианты: raw, log1p, weighted_*, raw/log1p × (≤threshold, >threshold).
    Квантили бьются по ``fin_config.severity_target_column`` (true severity).
    """
    training_config = training_config or TrainingConfig()
    if training.severity_split is None:
        raise ValueError("severity_split отсутствует")

    from querulus.fin_effect.calculator import _feature_rows_for_predict

    sev_target = fin_config.severity_target_column
    cat = training.severity_categorical_features
    features = training.severity_features

    sev_train_idx = training.severity_split.x_train.index
    sev_test_idx = training.severity_split.x_test.index
    effect_index, _ = _effect_index_and_freq(training, split)

    # Общий порог freq — от raw на полном test.
    predict_all = _feature_rows_for_predict(training, effect_index, df.loc[effect_index])
    raw_full_pred = severity_predict(
        training.severity_model,
        predict_all[features],
        cat,
        transform=getattr(training, "severity_target_transform", "raw"),
    )
    raw_full_fe = run_fin_effect_with_severity_predictions(
        df, training, raw_full_pred, split=split, threshold=threshold, config=fin_config
    )
    use_threshold = threshold if threshold is not None else raw_full_fe.best_threshold

    # Общие границы квантилей по true severity на severity-test.
    y_true_sev_test = pd.to_numeric(
        df.loc[sev_test_idx, sev_target], errors="coerce"
    ).fillna(0.0)
    bin_edges = _shared_quantile_edges(y_true_sev_test)

    specs: list[tuple[str, SeverityTargetTransform, SeveritySampleWeight, SegmentSide, object | None]] = [
        ("raw", "raw", "none", "all", training.severity_model),
        ("log1p", "log1p", "none", "all", None),
        ("weighted_sqrt", "raw", "sqrt", "all", None),
        ("weighted_linear", "raw", "linear", "all", None),
        ("raw_le50", "raw", "none", "le", None),
        ("log1p_le50", "log1p", "none", "le", None),
        ("raw_gt50", "raw", "none", "gt", None),
        ("log1p_gt50", "log1p", "none", "gt", None),
    ]
    # Имена le50/gt50 отражают порог в summary через value_threshold.

    models: dict[str, SeverityZooModel] = {}
    metric_rows: list[dict[str, object]] = []
    quantile_parts: list[pd.DataFrame] = []
    fin_rows: list[dict[str, object]] = []
    penalty_parts: list[pd.DataFrame] = []

    for name, transform, weight_mode, side, preset in specs:
        train_idx = _segment_indices(
            df, sev_train_idx, value_column, value_threshold, side
        )
        eval_idx = _segment_indices(
            df, sev_test_idx, value_column, value_threshold, side
        )
        effect_eval_idx = _segment_indices(
            df, effect_index, value_column, value_threshold, side
        )

        if preset is not None and side == "all":
            model = preset
        else:
            if len(train_idx) == 0:
                continue
            model = fit_severity_model(
                training,
                training_config,
                transform=transform,
                sample_weight=weight_mode,
                train_index=train_idx,
                eval_index=eval_idx if len(eval_idx) else None,
            )

        if len(eval_idx) == 0:
            continue

        pred_frame = _feature_rows_for_predict(training, eval_idx, df.loc[eval_idx])
        y_pred = severity_predict(model, pred_frame[features], cat, transform=transform)
        y_true = pd.to_numeric(df.loc[eval_idx, sev_target], errors="coerce").fillna(0.0)
        y_true_arr = y_true.to_numpy(dtype=float)
        err = y_pred - y_true_arr

        # Fin-effect на соответствующем scope (один порог).
        if len(effect_eval_idx) == 0:
            fe_result = None
        else:
            fe_frame = _feature_rows_for_predict(
                training, effect_eval_idx, df.loc[effect_eval_idx]
            )
            fe_sev = severity_predict(
                model, fe_frame[features], cat, transform=transform
            )
            if side == "all":
                fe_result = run_fin_effect_with_severity_predictions(
                    df,
                    training,
                    fe_sev,
                    split=split,
                    threshold=use_threshold,
                    config=fin_config,
                )
            else:
                fe_result = _fin_on_index(
                    df,
                    training,
                    effect_eval_idx,
                    fe_sev,
                    fin_config,
                    threshold=use_threshold,
                )

        scope = {
            "all": "all_test",
            "le": f"{value_column}<={value_threshold}",
            "gt": f"{value_column}>{value_threshold}",
        }[side]

        models[name] = SeverityZooModel(
            name=name,
            transform=transform,
            sample_weight=weight_mode,
            segment=side,
            model=model,
            eval_index=eval_idx,
            y_pred=y_pred,
            y_true=y_true_arr,
            fin_effect=fe_result,
        )

        metric_rows.append(
            {
                "variant": name,
                "scope": scope,
                "transform": transform,
                "sample_weight": weight_mode,
                "n": int(len(eval_idx)),
                "mae": float(np.nanmean(np.abs(err))),
                "bias": float(np.nanmean(err)),
                "pred_sum": float(np.nansum(y_pred)),
                "true_sum": float(np.nansum(y_true_arr)),
                "sum_bias": float(np.nansum(y_pred) - np.nansum(y_true_arr)),
            }
        )

        q_part = _quantile_table_for_preds(
            variant=name,
            y_true=y_true_arr,
            y_pred=y_pred,
            bin_axis=sev_target,
            bin_edges=bin_edges,
        )
        if not q_part.empty:
            q_part.insert(2, "scope", scope)
            quantile_parts.append(q_part)

        if fe_result is not None:
            fin_rows.append(
                {
                    "variant": name,
                    "scope": scope,
                    "n": int(len(effect_eval_idx)),
                    "threshold": use_threshold,
                    "net_effect": fe_result.net_effect,
                    "model_effect": fe_result.model_effect_total,
                    "fact_effect": fe_result.fact_effect_total,
                }
            )
            pen = fin_effect_penalty_table(fe_result, fin_config)
            if not pen.empty:
                pen = pen.copy()
                pen.insert(0, "variant", name)
                pen.insert(1, "scope", scope)
                penalty_parts.append(pen)

    return SeverityZooCompare(
        metrics_summary=plain_floats(pd.DataFrame(metric_rows)),
        quantile_summary=plain_floats(
            pd.concat(quantile_parts, ignore_index=True) if quantile_parts else pd.DataFrame()
        ),
        fin_effect_summary=plain_floats(pd.DataFrame(fin_rows)),
        penalty_summary=plain_floats(
            pd.concat(penalty_parts, ignore_index=True) if penalty_parts else pd.DataFrame()
        ),
        bin_axis=sev_target,
        value_column=value_column,
        value_threshold=value_threshold,
        models=models,
    )
