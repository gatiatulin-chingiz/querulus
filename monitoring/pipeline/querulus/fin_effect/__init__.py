"""Расчёт и визуализация финансового эффекта querulus.

Все подмодули — ленивые: ``from querulus.fin_effect import X`` и
``querulus.fin_effect.monitoring_snapshots`` не тянут catboost, пока X
сам его не потребует. Нужно для monitoring pipeline в Airflow без ML-стека.
"""
from __future__ import annotations

import importlib
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # calculator (catboost)
    "FinEffectResult": (".calculator", "FinEffectResult"),
    "ThresholdMetrics": (".calculator", "ThresholdMetrics"),
    "ThresholdStrategyResult": (".calculator", "ThresholdStrategyResult"),
    "add_premiums_column": (".calculator", "add_premiums_column"),
    "align_effect_inputs": (".calculator", "align_effect_inputs"),
    "apply_model_predictions": (".calculator", "apply_model_predictions"),
    "compute_fin_effect_fact": (".calculator", "compute_fin_effect_fact"),
    "compute_fin_effect_model": (".calculator", "compute_fin_effect_model"),
    "evaluate_threshold": (".calculator", "evaluate_threshold"),
    "payments_fee": (".calculator", "payments_fee"),
    "prepare_analytics_export": (".calculator", "prepare_analytics_export"),
    "prepare_effect_frame": (".calculator", "prepare_effect_frame"),
    "print_best_threshold_report": (".calculator", "print_best_threshold_report"),
    "run_fin_effect_from_training": (".calculator", "run_fin_effect_from_training"),
    "run_fin_effect_pipeline": (".calculator", "run_fin_effect_pipeline"),
    "search_best_threshold": (".calculator", "search_best_threshold"),
    "search_best_threshold_by_f1": (".calculator", "search_best_threshold_by_f1"),
    "search_threshold_strategies": (".calculator", "search_threshold_strategies"),
    # signed / compare / config / threshold / resolve
    "economy_from_signed_effects": (".signed_effects", "economy_from_signed_effects"),
    "recompute_fin_effect_model": (".signed_effects", "recompute_fin_effect_model"),
    "StackCompareReport": (".compare_report", "StackCompareReport"),
    "compare_fact_bases": (".compare_report", "compare_fact_bases"),
    "compare_premiums": (".compare_report", "compare_premiums"),
    "compare_severity_predictions": (".compare_report", "compare_severity_predictions"),
    "compare_severity_targets": (".compare_report", "compare_severity_targets"),
    "fact_only_compare_report": (".compare_report", "fact_only_compare_report"),
    "model_quadrant_breakdown": (".compare_report", "model_quadrant_breakdown"),
    "run_dual_stack_compare": (".compare_report", "run_dual_stack_compare"),
    "summary_itogo_breakdown": (".compare_report", "summary_itogo_breakdown"),
    "ANALYTICS_RENAME_DICT": (".config", "ANALYTICS_RENAME_DICT"),
    "FinEffectConfig": (".config", "FinEffectConfig"),
    "COLLECT_PROD_THRESHOLD_JSON": (".threshold_policy", "COLLECT_PROD_THRESHOLD_JSON"),
    "COLLECT_VAL_THRESHOLD_JSON": (".threshold_policy", "COLLECT_VAL_THRESHOLD_JSON"),
    "DEFAULT_FIT_THRESHOLD": (".threshold_policy", "DEFAULT_FIT_THRESHOLD"),
    "ValThresholdResult": (".threshold_policy", "ValThresholdResult"),
    "collect_prod_threshold_path": (".threshold_policy", "collect_prod_threshold_path"),
    "collect_val_threshold_path": (".threshold_policy", "collect_val_threshold_path"),
    "load_collect_prod_threshold": (".threshold_policy", "load_collect_prod_threshold"),
    "load_collect_val_threshold": (".threshold_policy", "load_collect_val_threshold"),
    "pick_threshold_on_val": (".threshold_policy", "pick_threshold_on_val"),
    "pick_threshold_on_val_from_training": (
        ".threshold_policy",
        "pick_threshold_on_val_from_training",
    ),
    "resolve_or_pick_val_threshold": (
        ".threshold_policy",
        "resolve_or_pick_val_threshold",
    ),
    "resolve_val_threshold": (".threshold_policy", "resolve_val_threshold"),
    "save_collect_prod_threshold": (".threshold_policy", "save_collect_prod_threshold"),
    "save_collect_val_threshold": (".threshold_policy", "save_collect_val_threshold"),
    "val_index_from_training": (".threshold_policy", "val_index_from_training"),
    "val_index_from_trainings": (".threshold_policy", "val_index_from_trainings"),
    "CLAIMS_FREQUENCY_TARGETS": (".resolve", "CLAIMS_FREQUENCY_TARGETS"),
    "CLAIMS_SEVERITY_TARGETS": (".resolve", "CLAIMS_SEVERITY_TARGETS"),
    "ICNL_FREQUENCY_TARGETS": (".resolve", "ICNL_FREQUENCY_TARGETS"),
    "ICNL_SEVERITY_TARGETS": (".resolve", "ICNL_SEVERITY_TARGETS"),
    "LEGACY_FREQUENCY_TARGETS": (".resolve", "LEGACY_FREQUENCY_TARGETS"),
    "LEGACY_SEVERITY_TARGETS": (".resolve", "LEGACY_SEVERITY_TARGETS"),
    "infer_legacy_dataset": (".resolve", "infer_legacy_dataset"),
    "resolve_fact_mode": (".resolve", "resolve_fact_mode"),
    "resolve_fin_effect_config": (".resolve", "resolve_fin_effect_config"),
    "export_business_html": (".business_report", "export_business_html"),
    "DEFAULT_BOOTSTRAP_FOLDS": (".bootstrap", "DEFAULT_BOOTSTRAP_FOLDS"),
    "DEFAULT_BOOTSTRAP_SEED": (".bootstrap", "DEFAULT_BOOTSTRAP_SEED"),
    "BootstrapFinEffect": (".bootstrap", "BootstrapFinEffect"),
    "bootstrap_fin_effect": (".bootstrap", "bootstrap_fin_effect"),
    "export_analytics_excel": (".export", "export_analytics_excel"),
    "color_excel_table": (".summary", "color_excel_table"),
    "compare_formula_summaries": (".summary", "compare_formula_summaries"),
    "create_summary_table": (".summary", "create_summary_table"),
    "export_summary_excel": (".summary", "export_summary_excel"),
    # plots / segment (уже были ленивые)
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


__all__ = sorted(_LAZY_EXPORTS.keys())
