"""Обучение моделей querulus.

Реэкспорты ленивые: ``from querulus.training.X import Y`` не тянет весь пакет
и не создаёт циклы с ``fin_effect``.
"""
from __future__ import annotations

import importlib
from typing import Any

# name -> (submodule, attr)
_EXPORTS: dict[str, tuple[str, str]] = {
    "BackwardElimResult": (".backward_elim", "BackwardElimResult"),
    "DateSplitParts": (".splits", "DateSplitParts"),
    "HpoResult": (".hpo", "HpoResult"),
    "TARGET_STACKS": (".triple_stack", "TARGET_STACKS"),
    "SeverityLog1pCompare": (".severity_diagnostics", "SeverityLog1pCompare"),
    "SEVERITY_VARIANT_NAMES": (".severity_variant", "SEVERITY_VARIANT_NAMES"),
    "SeverityVariantSpec": (".severity_variant", "SeverityVariantSpec"),
    "SeverityZooCompare": (".severity_zoo", "SeverityZooCompare"),
    "StartYearEvalReport": (".start_year_eval", "StartYearEvalReport"),
    "StackEvalReport": (".stack_eval", "StackEvalReport"),
    "DEFAULT_START_YEARS": (".start_year_eval", "DEFAULT_START_YEARS"),
    "resolve_severity_variant": (".severity_variant", "resolve_severity_variant"),
    "with_periods": (".build_outboxml_configs", "with_periods"),
    "write_outboxml_configs": (".build_outboxml_configs", "write_outboxml_configs"),
    "load_outboxml_configs": (".build_outboxml_configs", "load_outboxml_configs"),
    "compute_period_windows": (".build_outboxml_configs", "compute_period_windows"),
    "default_model_version": (".build_outboxml_configs", "default_model_version"),
    "prepare_datasets_from_config": (
        ".build_outboxml_configs",
        "prepare_datasets_from_config",
    ),
    "display_dsm_collect_metrics": (".outboxml_metrics", "display_dsm_collect_metrics"),
    "enrich_dsm_model_metrics": (".outboxml_metrics", "enrich_dsm_model_metrics"),
    "predict_dsm_series": (".outboxml_metrics", "predict_dsm_series"),
    "prepare_dsm_features": (".outboxml_metrics", "prepare_dsm_features"),
    "TrainLoopFlags": (".train_loop", "TrainLoopFlags"),
    "TrainLoopResult": (".train_loop", "TrainLoopResult"),
    "TrainingArtifacts": (".pipeline", "TrainingArtifacts"),
    "TrainingConfig": (".config", "TrainingConfig"),
    "TripleStackResult": (".triple_stack", "TripleStackResult"),
    "backward_eliminate_by_metric": (".backward_elim", "backward_eliminate_by_metric"),
    "build_metrics_summary": (".triple_stack", "build_metrics_summary"),
    "compare_severity_log1p": (".severity_diagnostics", "compare_severity_log1p"),
    "drop_zero_importance_features": (
        ".feature_selection_io",
        "drop_zero_importance_features",
    ),
    "evaluate_legacy_vs_new": (".stack_eval", "evaluate_legacy_vs_new"),
    "export_leadership_html": (".leadership_report", "export_leadership_html"),
    "export_leadership_html_from_collect": (
        ".leadership_report",
        "export_leadership_html_from_collect",
    ),
    "evaluate_train_start_years": (".start_year_eval", "evaluate_train_start_years"),
    "ExampleDatasetBundle": (".example_pipeline", "ExampleDatasetBundle"),
    "ExampleDsmBundle": (".example_pipeline", "ExampleDsmBundle"),
    "ExamplePaths": (".example_pipeline", "ExamplePaths"),
    "ExampleThresholds": (".example_pipeline", "ExampleThresholds"),
    "export_prod_service_artifacts": (
        ".example_pipeline",
        "export_prod_service_artifacts",
    ),
    "create_querulus_automl": (".automl_fit", "create_querulus_automl"),
    "fit_automl_bundle": (".automl_fit", "fit_automl_bundle"),
    "fit_parity_models": (".example_pipeline", "fit_parity_models"),
    "fit_prod_models": (".example_pipeline", "fit_prod_models"),
    "load_example_dataset": (".example_pipeline", "load_example_dataset"),
    "load_example_thresholds": (".example_pipeline", "load_example_thresholds"),
    "patch_dsm_models": (".example_pipeline", "patch_dsm_models"),
    "predict_cf": (".example_pipeline", "predict_cf"),
    "predict_rg": (".example_pipeline", "predict_rg"),
    "resolve_example_paths": (".example_pipeline", "resolve_example_paths"),
    "run_parity_cross_test_metrics": (
        ".example_pipeline",
        "run_parity_cross_test_metrics",
    ),
    "run_prod_metrics": (".example_pipeline", "run_prod_metrics"),
    "run_prod_plots_and_email": (".example_pipeline", "run_prod_plots_and_email"),
    "run_test_fin_effect": (".example_pipeline", "run_test_fin_effect"),
    "run_test_prod_fin_effect": (".example_pipeline", "run_test_prod_fin_effect"),
    "payment_year_cohorts": (".start_year_eval", "payment_year_cohorts"),
    "score_stack_on_index": (".stack_eval", "score_stack_on_index"),
    "train_legacy_matching_new": (".stack_eval", "train_legacy_matching_new"),
    "apply_severity_calibrator": (".calibration", "apply_severity_calibrator"),
    "balance_binary_cal_frame": (".calibration", "balance_binary_cal_frame"),
    "CalibratorAbResult": (".calibration", "CalibratorAbResult"),
    "SeverityCalibrator": (".calibration", "SeverityCalibrator"),
    "compare_calibrator_ab": (".calibration", "compare_calibrator_ab"),
    "expected_calibration_error": (".calibration", "expected_calibration_error"),
    "feature_drift_report": (".drift", "feature_drift_report"),
    "filter_features_by_drift": (".drift", "filter_features_by_drift"),
    "filter_features_by_noise": (".noise_cut", "filter_features_by_noise"),
    "fit_probability_calibrator": (".calibration", "fit_probability_calibrator"),
    "fit_severity_calibrator": (".calibration", "fit_severity_calibrator"),
    "severity_mean_bias": (".calibration", "severity_mean_bias"),
    "format_features_table": (".pipeline", "format_features_table"),
    "format_metrics_table": (".pipeline", "format_metrics_table"),
    "format_training_summary": (".pipeline", "format_training_summary"),
    "frequency_metrics_table_at_threshold": (
        ".pipeline",
        "frequency_metrics_table_at_threshold",
    ),
    "frequency_predict_proba": (".pipeline", "frequency_predict_proba"),
    "load_feature_selection_latest": (
        ".feature_selection_io",
        "load_feature_selection_latest",
    ),
    "log_training_summary": (".pipeline", "log_training_summary"),
    "monthly_target_drift": (".drift", "monthly_target_drift"),
    "NoiseCutResult": (".noise_cut", "NoiseCutResult"),
    "rebuild_feature_selection_report": (
        ".feature_selection_report",
        "rebuild_feature_selection_report",
    ),
    "run_hpo": (".hpo", "run_hpo"),
    "run_model_diagnostics_visualizations": (
        ".plots",
        "run_model_diagnostics_visualizations",
    ),
    "run_mvp_frequency_eda": (".plots", "run_mvp_frequency_eda"),
    "run_severity_zoo_compare": (".severity_zoo", "run_severity_zoo_compare"),
    "run_train_loop_new": (".train_loop", "run_train_loop_new"),
    "run_training_visualizations": (".plots", "run_training_visualizations"),
    "run_triple_fin_effects": (".triple_stack", "run_triple_fin_effects"),
    "run_triple_stack": (".triple_stack", "run_triple_stack"),
    "save_feature_selection": (".feature_selection_io", "save_feature_selection"),
    "save_feature_selection_report": (
        ".feature_selection_report",
        "save_feature_selection_report",
    ),
    "severity_error_by_quantile": (".severity_diagnostics", "severity_error_by_quantile"),
    "split_by_date_periods": (".splits", "split_by_date_periods"),
    "train_models": (".pipeline", "train_models"),
    "train_triple_stacks": (".triple_stack", "train_triple_stacks"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
