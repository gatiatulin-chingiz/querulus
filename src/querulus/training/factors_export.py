"""FactorsPlot / cohort для example: raw labels, CF fact/raw/cal, RG fact/raw/isotonic.

Без правок OutBoxML: свой Plotly + данные из DSM + сырой ``bundle.df``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from querulus.training.build_outboxml_configs import (
    ensure_predictable_model,
    unwrap_estimator,
)
from querulus.training.calibration import (
    SeverityCalibrator,
    apply_severity_calibrator,
    fit_severity_calibrator,
)
from querulus.training.calibration_compare import resolve_cal_index
from querulus.training.outboxml_metrics import prepare_dsm_features

__all__ = [
    "build_factors_figures",
    "fit_rg_severity_calibrator_isotonic",
    "dsm_feature_lists",
]


def dsm_feature_lists(dsm: Any, model_name: str) -> tuple[list[str], list[str]]:
    subset = dsm.get_result()[model_name].data_subset
    nums = [str(f) for f in (subset.features_numerical or [])]
    cats = [str(f) for f in (subset.features_categorical or [])]
    return nums, cats


def _positive_proba(estimator: Any, x: pd.DataFrame) -> pd.Series:
    if hasattr(estimator, "predict_proba"):
        scores = np.asarray(estimator.predict_proba(x)[:, 1], dtype=float)
    else:
        scores = np.asarray(estimator.predict(x), dtype=float)
    return pd.Series(scores, index=x.index, dtype=float)


def _raw_scores(
    dsm: Any,
    model_name: str,
    *,
    task: str,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, list[str], list[str]]:
    """y_true, raw pred, X_test (DSM), num, cat на test-индексе."""
    result = dsm.get_result()[model_name]
    subset = result.data_subset
    nums, cats = dsm_feature_lists(dsm, model_name)
    cols = [*nums, *cats]
    x = subset.X_test.loc[:, cols]
    y_true = subset.y_test.reindex(x.index).astype(float)
    estimator = unwrap_estimator(ensure_predictable_model(result.model))
    if task == "classification":
        raw = _positive_proba(estimator, x)
    else:
        pred = estimator.predict(x)
        raw = pd.Series(np.asarray(pred, dtype=float), index=x.index, dtype=float)
    return y_true, raw, x, nums, cats


def _axis_series(
    feature: str,
    *,
    is_categorical: bool,
    x_test: pd.DataFrame,
    df_raw: pd.DataFrame,
    index: pd.Index,
) -> pd.Series:
    """Ось X: для cat — сырые labels из df, иначе значения DSM X_test."""
    if is_categorical and feature in df_raw.columns:
        return df_raw.loc[index, feature].astype(str)
    if feature in x_test.columns:
        return x_test.loc[index, feature]
    if feature in df_raw.columns:
        return df_raw.loc[index, feature]
    raise KeyError(f"Фича {feature!r} нет ни в X_test, ни в df")


def _bin_groups(
    axis: pd.Series,
    *,
    is_categorical: bool,
    bins: int,
) -> pd.Series:
    if is_categorical:
        return axis.astype(str)
    vals = pd.to_numeric(axis, errors="coerce")
    nunique = int(vals.nunique(dropna=True))
    if nunique <= bins:
        return vals.astype(str)
    breakpoints = [np.nanpercentile(vals.dropna(), 100 * i / bins) for i in range(bins + 1)]
    breakpoints[0] = breakpoints[0] - 0.1
    # уникальные границы
    breakpoints = sorted(set(float(b) for b in breakpoints if np.isfinite(b)))
    if len(breakpoints) < 2:
        return vals.astype(str)
    return pd.cut(vals, bins=breakpoints, duplicates="drop").astype(str)


def _factor_figure(
    *,
    model_name: str,
    feature: str,
    groups: pd.Series,
    y_true: pd.Series,
    series: dict[str, pd.Series],
    colors: dict[str, str],
) -> Any:
    exposure = pd.Series(1.0, index=y_true.index)
    frame = pd.DataFrame({"_g": groups, "y_true": y_true, "exposure": exposure})
    for name, s in series.items():
        frame[name] = s.reindex(y_true.index)

    agg = frame.groupby("_g", dropna=False, observed=False).sum(numeric_only=True)
    agg["fact"] = agg["y_true"] / agg["exposure"]
    for name in series:
        agg[name] = agg[name] / agg["exposure"]

    x_labels = agg.index.astype(str).tolist()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            name="Exposure(Size)",
            x=x_labels,
            y=agg["exposure"].tolist(),
            marker=dict(color="rgba(255,180,180,0.6)"),
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            name="fact",
            x=x_labels,
            y=agg["fact"].tolist(),
            mode="lines",
            marker=dict(color="rgba(255,0,0,1)"),
        ),
        secondary_y=False,
    )
    for name, color in colors.items():
        if name not in agg.columns:
            continue
        fig.add_trace(
            go.Scatter(
                name=name,
                x=x_labels,
                y=agg[name].tolist(),
                mode="lines+markers",
                marker=dict(color=color),
            ),
            secondary_y=False,
        )
    fig.update_layout(title=model_name)
    fig.update_xaxes(title_text=f"Target groups for {feature}")
    fig.update_yaxes(title_text=model_name, secondary_y=False)
    fig.update_yaxes(title_text="Exposure(Size)", secondary_y=True)
    return fig


def _cohort_predict_gr(
    predict: pd.Series,
    *,
    cut_min_value: float = 0.1,
    cut_max_value: float = 0.9,
    samples: float = 100.0,
) -> pd.Series:
    """Бины 1:1 как OutBoxML ``MLPlot.cohort_plot`` (builtin ``round``, auto samples)."""
    column = predict.astype(float)
    samples_i = float(samples)
    cut_max_i = float(cut_max_value)
    # как в outboxml.plots: round(series * samples) / samples
    predict_gr = round(column * samples_i) / samples_i
    predict_gr = predict_gr.clip(
        lower=predict_gr.quantile(cut_min_value),
        upper=predict_gr.quantile(cut_max_i),
    )
    if len(predict_gr.unique()) > 20 and samples_i == 100.0:
        for _ in range(8):
            if len(predict_gr.unique()) < 20:
                break
            samples_i = samples_i / 5
            predict_gr = round(column * samples_i) / samples_i
            predict_gr = predict_gr.clip(
                lower=predict_gr.quantile(cut_min_value),
                upper=predict_gr.quantile(cut_max_i),
            )
    elif len(predict_gr.unique()) < 5 and cut_max_i == 0.9:
        for _ in range(8):
            if len(predict_gr.unique()) >= 5:
                break
            cut_max_i = cut_max_i * 1.1
            if cut_max_i > 1:
                cut_max_i = 1
            samples_i = samples_i * 2
            predict_gr = round(column * samples_i) / samples_i
            predict_gr = predict_gr.clip(
                lower=predict_gr.quantile(cut_min_value),
                upper=predict_gr.quantile(cut_max_i),
            )
    return predict_gr


def _cohort_figure(
    *,
    model_name: str,
    y_true: pd.Series,
    y_raw: pd.Series,
    y_cal: pd.Series | None = None,
    cut_min: float = 0.1,
    cut_max: float = 0.9,
    samples: float = 100.0,
) -> Any:
    """Cohort как исходный OutBoxML FactorsPlot/cohort + линия cal.

    Бины — ``cohort_plot`` OutBoxML (по raw / model score).
    Легенда: fact, model, cal. Exposure на secondary_y.
    """
    idx = y_true.index
    model = y_raw.reindex(idx).astype(float)
    frame = pd.DataFrame(
        {
            "y_true": y_true.astype(float),
            "model": model,
            "exposure": 1.0,
        },
        index=idx,
    )
    if y_cal is not None:
        frame["cal"] = y_cal.reindex(idx).astype(float)
    # OutBoxML: predict = y_prediction / exposure; group by predict
    frame["predict"] = frame["model"] / frame["exposure"]
    frame["predict_gr"] = _cohort_predict_gr(
        frame["predict"],
        cut_min_value=cut_min,
        cut_max_value=cut_max,
        samples=samples,
    )
    sum_cols = ["y_true", "model", "exposure"] + (
        ["cal"] if "cal" in frame.columns else []
    )
    gr = (
        frame[sum_cols + ["predict_gr"]]
        .groupby("predict_gr", observed=False)
        .sum()
        .reset_index()
    )
    gr["fact"] = gr["y_true"] / gr["exposure"]
    gr["model_mean"] = gr["model"] / gr["exposure"]
    if "cal" in gr.columns:
        gr["cal_mean"] = gr["cal"] / gr["exposure"]

    x = gr["predict_gr"].tolist()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            name="Exposure(Size)",
            x=x,
            y=gr["exposure"].tolist(),
            marker=dict(color="rgba(255,180,180,0.65)"),
            opacity=0.85,
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            name="fact",
            x=x,
            y=gr["fact"].tolist(),
            mode="lines",
            line=dict(color="rgba(255,0,0,1)", width=2),
            marker=dict(color="rgba(255,0,0,1)"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            name="model",
            x=x,
            y=gr["model_mean"].tolist(),
            mode="lines+markers",
            line=dict(color="rgba(60,255,60,1)", width=2),
            marker=dict(color="rgba(60,255,60,1)"),
        ),
        secondary_y=False,
    )
    if "cal_mean" in gr.columns:
        fig.add_trace(
            go.Scatter(
                name="cal",
                x=x,
                y=gr["cal_mean"].tolist(),
                mode="lines+markers",
                line=dict(color="rgba(60,60,255,1)", width=2),
                marker=dict(color="rgba(60,60,255,1)"),
            ),
            secondary_y=False,
        )
    fig.update_layout(
        title=f"{model_name} cohort",
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(title_text="Target groups")
    fig.update_yaxes(title_text=model_name, secondary_y=False)
    fig.update_yaxes(
        title_text="Exposure(Size)",
        secondary_y=True,
        showgrid=False,
        rangemode="tozero",
    )
    return fig


def fit_rg_severity_calibrator_isotonic(
    dsm_rg: Any,
    rg_name: str,
    df: pd.DataFrame,
    periods: dict[str, Any],
    *,
    prefer_prod_tau: bool = True,
    min_samples: int = 50,
) -> SeverityCalibrator:
    """Isotonic severity-calibrator на τ-cal / cal (fact > 0)."""
    cal_index = resolve_cal_index(periods, prefer_prod_tau=prefer_prod_tau)
    cal_index = cal_index.intersection(df.index)
    x_cal = prepare_dsm_features(
        dsm_rg, rg_name, df.loc[cal_index], ignore_row_filter=True
    )
    result = dsm_rg.get_result()[rg_name]
    estimator = unwrap_estimator(ensure_predictable_model(result.model))
    pred = pd.Series(
        np.asarray(estimator.predict(x_cal), dtype=float),
        index=x_cal.index,
        dtype=float,
    )
    y = pd.to_numeric(df.loc[pred.index, result.model_config.column_target], errors="coerce")
    return fit_severity_calibrator(y, pred, method="isotonic", min_samples=min_samples)


def build_factors_figures(
    *,
    dsm_cf: Any,
    cf_name: str,
    dsm_rg: Any,
    rg_name: str,
    df: pd.DataFrame,
    cf_calibrator: Any | None,
    rg_calibrator: SeverityCalibrator | None,
    bins: int = 5,
) -> dict[str, Any]:
    """Собрать FactorsPlot + cohort; ключи → Plotly figure.

    Легенда: fact / model / cal. Бины cohort — как OutBoxML ``cohort_plot``.
    Cat-ось X — сырые labels из ``df``, если колонка есть.
    """
    figures: dict[str, Any] = {}

    # --- CF ---
    y_cf, raw_cf, x_cf, nums_cf, cats_cf = _raw_scores(
        dsm_cf, cf_name, task="classification"
    )
    idx_cf = y_cf.index
    cal_cf = None
    if cf_calibrator is not None:
        cal_cf = _positive_proba(cf_calibrator, x_cf)
    cf_series: dict[str, pd.Series] = {"model": raw_cf}
    cf_colors = {"model": "rgba(60,255,60,1)"}
    if cal_cf is not None:
        cf_series["cal"] = cal_cf
        cf_colors["cal"] = "rgba(60,60,255,1)"

    for feat in [*nums_cf, *cats_cf]:
        is_cat = feat in cats_cf
        axis = _axis_series(
            feat, is_categorical=is_cat, x_test=x_cf, df_raw=df, index=idx_cf
        )
        groups = _bin_groups(axis, is_categorical=is_cat, bins=bins)
        fig = _factor_figure(
            model_name=cf_name,
            feature=feat,
            groups=groups,
            y_true=y_cf,
            series=cf_series,
            colors=cf_colors,
        )
        figures[f"{cf_name}__factors__{feat}"] = fig

    figures[f"{cf_name}__cohort"] = _cohort_figure(
        model_name=cf_name,
        y_true=y_cf,
        y_raw=raw_cf,
        y_cal=cal_cf,
    )

    # --- RG ---
    y_rg, raw_rg, x_rg, nums_rg, cats_rg = _raw_scores(
        dsm_rg, rg_name, task="regression"
    )
    idx_rg = y_rg.index
    cal_rg = None
    if rg_calibrator is not None:
        cal_rg = apply_severity_calibrator(rg_calibrator, raw_rg)
        if not isinstance(cal_rg, pd.Series):
            cal_rg = pd.Series(cal_rg, index=raw_rg.index, dtype=float)
    rg_series: dict[str, pd.Series] = {"model": raw_rg}
    rg_colors = {"model": "rgba(60,255,60,1)"}
    if cal_rg is not None:
        rg_series["cal"] = cal_rg
        rg_colors["cal"] = "rgba(60,60,255,1)"

    for feat in [*nums_rg, *cats_rg]:
        is_cat = feat in cats_rg
        axis = _axis_series(
            feat, is_categorical=is_cat, x_test=x_rg, df_raw=df, index=idx_rg
        )
        groups = _bin_groups(axis, is_categorical=is_cat, bins=bins)
        fig = _factor_figure(
            model_name=rg_name,
            feature=feat,
            groups=groups,
            y_true=y_rg,
            series=rg_series,
            colors=rg_colors,
        )
        figures[f"{rg_name}__factors__{feat}"] = fig

    figures[f"{rg_name}__cohort"] = _cohort_figure(
        model_name=rg_name,
        y_true=y_rg,
        y_raw=raw_rg,
        y_cal=cal_rg,
    )

    return figures
