"""Расчёт и визуализация финансового эффекта querulus."""
from __future__ import annotations

import importlib
from typing import Any

from querulus.fin_effect.calculator import (
    FinEffectResult,
    ThresholdMetrics,
    add_premiums_column,
    apply_model_predictions,
    compute_fin_effect_fact,
    compute_fin_effect_model,
    evaluate_threshold,
    payments_fee,
    prepare_analytics_export,
    prepare_effect_frame,
    print_best_threshold_report,
    run_fin_effect_from_training,
    run_fin_effect_pipeline,
    search_best_threshold,
)
from querulus.fin_effect.config import ANALYTICS_RENAME_DICT, FinEffectConfig
from querulus.fin_effect.export import export_analytics_excel
from querulus.fin_effect.summary import (
    color_excel_table,
    create_summary_table,
    export_summary_excel,
)

_PLOT_EXPORTS = {
    "export_plot_html",
    "plot_confusion_matrix",
    "plot_cost_confusion_heatmaps",
    "plot_positive_cases_by_month",
    "plot_precision_recall_vs_threshold",
    "plot_severity_fact_vs_pred_binned",
    "plot_target_monthly_share",
}


def __getattr__(name: str) -> Any:
    if name in _PLOT_EXPORTS:
        plots = importlib.import_module("querulus.fin_effect.plots")
        return getattr(plots, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ANALYTICS_RENAME_DICT",
    "FinEffectConfig",
    "FinEffectResult",
    "ThresholdMetrics",
    "add_premiums_column",
    "apply_model_predictions",
    "color_excel_table",
    "compute_fin_effect_fact",
    "compute_fin_effect_model",
    "create_summary_table",
    "evaluate_threshold",
    "export_analytics_excel",
    "export_plot_html",
    "export_summary_excel",
    "payments_fee",
    "plot_confusion_matrix",
    "plot_cost_confusion_heatmaps",
    "plot_positive_cases_by_month",
    "plot_precision_recall_vs_threshold",
    "plot_severity_fact_vs_pred_binned",
    "plot_target_monthly_share",
    "prepare_analytics_export",
    "prepare_effect_frame",
    "print_best_threshold_report",
    "run_fin_effect_from_training",
    "run_fin_effect_pipeline",
    "search_best_threshold",
]
