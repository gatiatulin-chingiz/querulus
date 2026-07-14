"""Расчёт и визуализация финансового эффекта querulus."""
from __future__ import annotations

import importlib
from typing import Any

from querulus.fin_effect.calculator import (
    FinEffectResult,
    ThresholdMetrics,
    ThresholdStrategyResult,
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
    search_best_threshold_by_f1,
    search_threshold_strategies,
)
from querulus.fin_effect.compare_report import (
    StackCompareReport,
    compare_fact_bases,
    compare_premiums,
    compare_severity_predictions,
    compare_severity_targets,
    fact_only_compare_report,
    model_quadrant_breakdown,
    run_dual_stack_compare,
    summary_itogo_breakdown,
)
from querulus.fin_effect.config import ANALYTICS_RENAME_DICT, FinEffectConfig
from querulus.fin_effect.resolve import (
    CLAIMS_FREQUENCY_TARGETS,
    CLAIMS_SEVERITY_TARGETS,
    ICNL_FREQUENCY_TARGETS,
    ICNL_SEVERITY_TARGETS,
    LEGACY_FREQUENCY_TARGETS,
    LEGACY_SEVERITY_TARGETS,
    infer_legacy_dataset,
    resolve_fact_mode,
    resolve_fin_effect_config,
)
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
    "StackCompareReport",
    "ThresholdMetrics",
    "ThresholdStrategyResult",
    "add_premiums_column",
    "apply_model_predictions",
    "color_excel_table",
    "compare_fact_bases",
    "compare_premiums",
    "compare_severity_predictions",
    "compare_severity_targets",
    "compute_fin_effect_fact",
    "compute_fin_effect_model",
    "create_summary_table",
    "evaluate_threshold",
    "export_analytics_excel",
    "export_plot_html",
    "export_summary_excel",
    "fact_only_compare_report",
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
    "infer_legacy_dataset",
    "CLAIMS_FREQUENCY_TARGETS",
    "CLAIMS_SEVERITY_TARGETS",
    "ICNL_FREQUENCY_TARGETS",
    "ICNL_SEVERITY_TARGETS",
    "LEGACY_FREQUENCY_TARGETS",
    "LEGACY_SEVERITY_TARGETS",
    "model_quadrant_breakdown",
    "resolve_fact_mode",
    "resolve_fin_effect_config",
    "run_dual_stack_compare",
    "run_fin_effect_from_training",
    "run_fin_effect_pipeline",
    "search_best_threshold",
    "search_best_threshold_by_f1",
    "search_threshold_strategies",
    "summary_itogo_breakdown",
]
