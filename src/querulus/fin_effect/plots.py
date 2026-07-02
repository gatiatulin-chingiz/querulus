"""Визуализации финансового эффекта (Litigant fin_effect.py)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from querulus.fin_effect.config import FinEffectConfig


def _plot_deps():
    """Отложенный импорт matplotlib / seaborn / sklearn."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_score, recall_score

    return plt, sns, ConfusionMatrixDisplay, confusion_matrix, precision_score, recall_score


def plot_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    title: str = "Confusion Matrix",
) -> None:
    """Матрица ошибок классификации частоты."""
    plt, _, ConfusionMatrixDisplay, confusion_matrix, _, _ = _plot_deps()
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title(title)
    plt.show()


def plot_precision_recall_vs_threshold(
    model: Any,
    features: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
    ax: Any | None = None,
):
    """Precision и Recall в зависимости от порога классификации."""
    plt, _, _, _, precision_score, recall_score = _plot_deps()
    y_proba = model.predict_proba(features)[:, 1]
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)

    precisions: list[float] = []
    recalls: list[float] = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        precision = 1.0 if y_pred.sum() == 0 else precision_score(y_true, y_pred, zero_division=1)
        recall = recall_score(y_true, y_pred, zero_division=0)
        precisions.append(float(precision))
        recalls.append(float(recall))

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    ax.plot(thresholds, precisions, "b-", label="Precision", linewidth=2)
    ax.plot(thresholds, recalls, "r-", label="Recall", linewidth=2)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision и Recall в зависимости от порога классификации", fontsize=14)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    f1_scores = [
        2 * p * r / (p + r) if p + r > 0 else 0.0 for p, r in zip(precisions, recalls, strict=True)
    ]
    best_threshold = float(thresholds[int(np.argmax(f1_scores))])
    best_f1 = float(f1_scores[int(np.argmax(f1_scores))])
    ax.axvline(
        x=best_threshold,
        color="green",
        linestyle="--",
        label=f"Лучший threshold (F1={best_f1:.3f})",
        linewidth=2,
    )
    ax.legend(loc="best", fontsize=11)
    plt.tight_layout()
    return fig, ax, thresholds, precisions, recalls


def plot_cost_confusion_heatmaps(
    effect_df: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> None:
    """Суммы fin_effect_fact / fin_effect_model по квадрантам confusion matrix."""
    plt, sns, _, _, _, _ = _plot_deps()
    y_true = pd.Series(y_true).astype(int)
    y_pred = pd.Series(y_pred).astype(int)

    tn_mask = (y_true == 0) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    tp_mask = (y_true == 1) & (y_pred == 1)

    def get_costs(mask: pd.Series, column: str) -> float:
        return float(abs(effect_df.loc[mask, column].sum()))

    index = ["Нет иска (0)", "Иск (1)"]
    columns = ["Прогноз 0", "Прогноз 1"]
    fact_costs = [
        [get_costs(tn_mask, "fin_effect_fact"), get_costs(fp_mask, "fin_effect_fact")],
        [get_costs(fn_mask, "fin_effect_fact"), get_costs(tp_mask, "fin_effect_fact")],
    ]
    model_costs = [
        [get_costs(tn_mask, "fin_effect_model"), get_costs(fp_mask, "fin_effect_model")],
        [get_costs(fn_mask, "fin_effect_model"), get_costs(tp_mask, "fin_effect_model")],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(
        pd.DataFrame(fact_costs, index=index, columns=columns),
        annot=True,
        fmt=",.0f",
        cmap="Blues",
        ax=axes[0],
        cbar_kws={"label": "Сумма (₽)"},
        annot_kws={"size": 11},
    )
    axes[0].set_title("Фактические судебные расходы", fontsize=14, weight="bold")
    sns.heatmap(
        pd.DataFrame(model_costs, index=index, columns=columns),
        annot=True,
        fmt=",.0f",
        cmap="Greens",
        ax=axes[1],
        cbar_kws={"label": "Сумма (₽)"},
        annot_kws={"size": 11},
    )
    axes[1].set_title("Расходы по модели (Частота + Тяжесть)", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.show()


def _build_severity_bins(
    effect_df: pd.DataFrame,
    y_true_sev: pd.Series | np.ndarray,
    y_pred_sev: pd.Series | np.ndarray,
    *,
    mask_actual_claim: pd.Series | np.ndarray | None = None,
    config: FinEffectConfig | None = None,
    n_bins: int = 25,
) -> pd.DataFrame:
    """Агрегация fact/pred severity по бинам для строк с фактическим иском."""
    config = config or FinEffectConfig()
    if mask_actual_claim is None:
        mask_actual_claim = pd.Series(True, index=effect_df.index)
    mask_actual_claim = pd.Series(mask_actual_claim, index=effect_df.index).fillna(False)
    base_sum = effect_df["fin_effect_fact"].abs() if "fin_effect_fact" in effect_df.columns else effect_df.get(
        "base_sum", 0
    )

    plot_data = pd.DataFrame(
        {
            "fact_sev": pd.Series(y_true_sev, index=effect_df.index)[mask_actual_claim],
            "pred_sev": pd.Series(y_pred_sev, index=effect_df.index)[mask_actual_claim],
            "base_sum": pd.Series(base_sum, index=effect_df.index)[mask_actual_claim],
            "fact_surcharge": effect_df.loc[mask_actual_claim, config.base_payment_column],
        }
    )
    plot_data = plot_data[(plot_data["fact_sev"] > 0) & (plot_data["pred_sev"] >= 0)]
    plot_data["fact_sev_bin"] = pd.cut(
        plot_data["fact_sev"],
        bins=n_bins,
        include_lowest=True,
        right=False,
    ).apply(lambda value: value.left if pd.notna(value) else np.nan)
    plot_data = plot_data.dropna(subset=["fact_sev_bin"])
    plot_data["fact_sev_bin"] = plot_data["fact_sev_bin"].astype(float)
    return plot_data.groupby("fact_sev_bin").agg(
        fact_sev_median=("fact_sev", "median"),
        pred_sev=("pred_sev", "median"),
        base_sum=("base_sum", "median"),
        fact_surcharge=("fact_surcharge", "median"),
        n_claims=("fact_sev", "count"),
    ).reset_index()


def plot_severity_fact_vs_pred_binned(
    effect_df: pd.DataFrame,
    y_true_sev: pd.Series | np.ndarray,
    y_pred_sev: pd.Series | np.ndarray,
    *,
    mask_actual_claim: pd.Series | np.ndarray | None = None,
    config: FinEffectConfig | None = None,
    n_bins: int = 25,
    use_plotly: bool | None = None,
) -> Any | None:
    """Прогноз vs факт severity по бинам (plotly или matplotlib)."""
    grouped = _build_severity_bins(
        effect_df,
        y_true_sev,
        y_pred_sev,
        mask_actual_claim=mask_actual_claim,
        config=config,
        n_bins=n_bins,
    )

    if use_plotly is None:
        try:
            import plotly.graph_objects as go  # noqa: F401
            from plotly.subplots import make_subplots  # noqa: F401

            use_plotly = True
        except ImportError:
            use_plotly = False

    if use_plotly:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=grouped["fact_sev_bin"],
                y=grouped["fact_sev_median"] + grouped["fact_surcharge"],
                mode="lines+markers",
                name="Основной долг + УТС + Износ",
                line=dict(color="#FF6B6B", width=3),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=grouped["fact_sev_bin"],
                y=grouped["pred_sev"] + grouped["fact_surcharge"],
                mode="lines+markers",
                name="Прогноз модели",
                line=dict(color="#4ECDC4", width=3),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=grouped["fact_sev_bin"],
                y=grouped["base_sum"] + grouped["fact_surcharge"],
                mode="lines+markers",
                name="ПСР",
                line=dict(color="#9B5DE5", width=3),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=grouped["fact_sev_bin"],
                y=grouped["fact_surcharge"],
                mode="lines+markers",
                name="Фактические выплаты по убытку",
                line=dict(color="silver", width=3),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Bar(
                x=grouped["fact_sev_bin"],
                y=grouped["n_claims"],
                name="Количество исков",
                marker_color="rgba(100, 100, 100, 0.3)",
                opacity=0.6,
            ),
            secondary_y=True,
        )
        fig.update_layout(
            title="Прогноз vs Факт: суммы и экспозиция (равномерные бины)",
            width=1150,
            height=800,
            hovermode="x unified",
        )
        fig.show()
        return fig

    plt, _, _, _, _, _ = _plot_deps()
    fig, ax1 = plt.subplots(figsize=(15, 10))
    ax1.plot(
        grouped["fact_sev_bin"],
        grouped["fact_sev_median"],
        color="#FF6B6B",
        marker="o",
        linewidth=2,
        markersize=6,
        label="Основной долг + УТС + Износ",
    )
    ax1.plot(
        grouped["fact_sev_bin"],
        grouped["pred_sev"],
        color="#4ECDC4",
        marker="D",
        linewidth=2,
        markersize=6,
        label="Прогноз модели",
    )
    ax1.plot(
        grouped["fact_sev_bin"],
        grouped["base_sum"],
        color="#9B5DE5",
        marker="s",
        linewidth=2,
        markersize=6,
        label="ПСР",
    )
    ax1.set_xlabel("Сумма (бин, руб)", fontsize=12)
    ax1.set_ylabel("Сумма (руб)", fontsize=12)
    ax1.ticklabel_format(style="plain", axis="y")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    bin_width = np.diff(grouped["fact_sev_bin"]).mean() if len(grouped) > 1 else 100_000.0
    ax2.bar(
        grouped["fact_sev_bin"],
        grouped["n_claims"],
        color="gray",
        alpha=0.3,
        width=bin_width * 0.8,
        label="Количество исков",
    )
    ax2.set_ylabel("Количество исков", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", bbox_to_anchor=(0, -0.15), ncol=2)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()
    return fig


def plot_target_monthly_share(
    df: pd.DataFrame,
    *,
    config: FinEffectConfig | None = None,
) -> None:
    """Доля TARGET = 1 по месяцам."""
    plt, _, _, _, _, _ = _plot_deps()
    config = config or FinEffectConfig()
    data = df.copy()
    data[config.date_column] = pd.to_datetime(data[config.date_column])
    monthly = (
        data.assign(_month=data[config.date_column].dt.to_period("M"))
        .groupby("_month", as_index=False)
        .agg(
            target_share=(config.frequency_target_column, "mean"),
            total_count=(config.frequency_target_column, "count"),
        )
    )
    monthly["MONTH_START"] = monthly["_month"].dt.start_time

    plt.figure(figsize=(12, 5))
    plt.bar(monthly["MONTH_START"], monthly["target_share"] * 100, width=25, alpha=0.8)
    plt.xlabel("Месяц")
    plt.ylabel("Доля положительных случаев, %")
    plt.title("Доля TARGET = 1 в общем объёме (помесячно)")
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_positive_cases_by_month(
    df: pd.DataFrame,
    *,
    config: FinEffectConfig | None = None,
) -> None:
    """Количество положительных TARGET по месяцам."""
    plt, _, _, _, _, _ = _plot_deps()
    config = config or FinEffectConfig()
    data = df.copy()
    data[config.date_column] = pd.to_datetime(data[config.date_column])
    positive = (
        data[data[config.frequency_target_column] == 1]
        .assign(_month=data[config.date_column].dt.to_period("M"))
        .groupby("_month", as_index=False)
        .size()
        .rename(columns={"size": "positive_count"})
    )
    positive["MONTH_START"] = positive["_month"].dt.start_time

    plt.figure(figsize=(12, 5))
    plt.bar(positive["MONTH_START"], positive["positive_count"], width=25, alpha=0.8)
    plt.xlabel("Месяц")
    plt.ylabel("Количество положительных случаев (TARGET = 1)")
    plt.title("Положительные случаи по месяцам")
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def export_plot_html(fig: Any, path: str | Path) -> None:
    """Сохранить plotly-фигуру в HTML."""
    import plotly.io as pio

    pio.write_html(fig, file=str(path), auto_open=False)
