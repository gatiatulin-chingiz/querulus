"""Расчёт и визуализация финансового эффекта querulus.

Тяжёлые подмодули (plots, segment_eval) — ленивые, без циклов с training.
"""
from __future__ import annotations

import importlib
from typing import Any

from querulus.fin_effect.calculator import (
    FinEffectResult,
    ThresholdMetrics,
    ThresholdStrategyResult,
    add_premiums_column,
    align_effect_inputs,
    apply_model_predictions,
    compute_fin_effect_fact,
    compute_fin_effect_model,
    economy_from_signed_effects,
    evaluate_threshold,
    payments_fee,
    prepare_analytics_export,
    prepare_effect_frame,
    print_best_threshold_report,
    recompute_fin_effect_model,
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
from querulus.fin_effect.threshold_policy import (
    COLLECT_PROD_THRESHOLD_JSON,
    COLLECT_VAL_THRESHOLD_JSON,
    ValThresholdResult,
    collect_prod_threshold_path,
    collect_val_threshold_path,
    load_collect_prod_threshold,
    load_collect_val_threshold,
    pick_threshold_on_val,
    pick_threshold_on_val_from_training,
    resolve_or_pick_val_threshold,
    resolve_val_threshold,
    save_collect_prod_threshold,
    save_collect_val_threshold,
    val_index_from_training,
    val_index_from_trainings,
)
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
from querulus.fin_effect.business_report import export_business_html
from querulus.fin_effect.export import export_analytics_excel
from querulus.fin_effect.summary import (
    color_excel_table,
    compare_formula_summaries,
    create_summary_table,
    export_summary_excel,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "export_plot_html": (".plots", "export_plot_html"),
    "plot_confusion_matrix": (".plots", "plot_confusion_matrix"),
    "plot_cost_confusion_heatmaps": (".plots", "plot_cost_confusion_heatmaps"),
    "plot_positive_cases_by_month": (".plots", "plot_positive_cases_by_month"),
    "plot_precision_recall_vs_threshold": (
        ".plots",
        "plot_precision_recall_vs_threshold",
    ),
    "plot_severity_fact_vs_pred_binned": (
        ".plots",
        "plot_severity_fact_vs_pred_binned",
    ),
    "plot_target_monthly_share": (".plots", "plot_target_monthly_share"),
    "SegmentFinEffectCompare": (".segment_eval", "SegmentFinEffectCompare"),
    "SeverityVariantsCompare": (".segment_eval", "SeverityVariantsCompare"),
    "compare_severity_fin_effect_variants": (
        ".segment_eval",
        "compare_severity_fin_effect_variants",
    ),
    "compare_value_before_segment_strategies": (
        ".segment_eval",
        "compare_value_before_segment_strategies",
    ),
    "fin_effect_penalty_table": (".segment_eval", "fin_effect_penalty_table"),
    "run_fin_effect_with_severity_predictions": (
        ".segment_eval",
        "run_fin_effect_with_severity_predictions",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "ANALYTICS_RENAME_DICT",
    "FinEffectConfig",
    "FinEffectResult",
    "StackCompareReport",
    "ThresholdMetrics",
    "ThresholdStrategyResult",
    "add_premiums_column",
    "align_effect_inputs",
    "apply_model_predictions",
    "color_excel_table",
    "compare_fact_bases",
    "compare_formula_summaries",
    "compare_premiums",
    "compare_severity_predictions",
    "compare_severity_targets",
    "compare_severity_fin_effect_variants",
    "compare_value_before_segment_strategies",
    "compute_fin_effect_fact",
    "compute_fin_effect_model",
    "create_summary_table",
    "economy_from_signed_effects",
    "evaluate_threshold",
    "export_analytics_excel",
    "export_business_html",
    "export_plot_html",
    "export_summary_excel",
    "fact_only_compare_report",
    "fin_effect_penalty_table",
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
    "recompute_fin_effect_model",
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
    "run_fin_effect_with_severity_predictions",
    "ValThresholdResult",
    "COLLECT_PROD_THRESHOLD_JSON",
    "COLLECT_VAL_THRESHOLD_JSON",
    "collect_prod_threshold_path",
    "collect_val_threshold_path",
    "load_collect_prod_threshold",
    "load_collect_val_threshold",
    "pick_threshold_on_val",
    "pick_threshold_on_val_from_training",
    "resolve_or_pick_val_threshold",
    "resolve_val_threshold",
    "save_collect_prod_threshold",
    "save_collect_val_threshold",
    "val_index_from_training",
    "val_index_from_trainings",
    "search_best_threshold",
    "search_best_threshold_by_f1",
    "search_threshold_strategies",
    "SegmentFinEffectCompare",
    "SeverityVariantsCompare",
    "summary_itogo_breakdown",
]
