"""Обучение моделей querulus."""

from querulus.training.calibration import expected_calibration_error, fit_probability_calibrator
from querulus.training.config import TrainingConfig
from querulus.training.corr_filter import CorrFilterResult, correlation_filter_features
from querulus.training.drift import (
    feature_drift_report,
    filter_features_by_drift,
    monthly_target_drift,
)
from querulus.training.feature_selection_io import (
    load_feature_selection_latest,
    save_feature_selection,
)
from querulus.training.hpo import HpoResult, run_hpo
from querulus.training.pipeline import (
    TrainingArtifacts,
    format_features_table,
    format_metrics_table,
    format_training_summary,
    frequency_predict_proba,
    log_training_summary,
    train_models,
)
from querulus.training.plots import (
    run_model_diagnostics_visualizations,
    run_mvp_frequency_eda,
    run_training_visualizations,
)
from querulus.training.severity_diagnostics import (
    SeverityLog1pCompare,
    compare_severity_log1p,
    severity_error_by_quantile,
)
from querulus.training.severity_zoo import SeverityZooCompare, run_severity_zoo_compare
from querulus.training.splits import DateSplitParts, split_by_date_periods
from querulus.training.train_loop import TrainLoopFlags, TrainLoopResult, run_train_loop_new
from querulus.training.triple_stack import (
    TARGET_STACKS,
    TripleStackResult,
    build_metrics_summary,
    run_triple_fin_effects,
    run_triple_stack,
    train_triple_stacks,
)

__all__ = [
    "CorrFilterResult",
    "DateSplitParts",
    "HpoResult",
    "TARGET_STACKS",
    "SeverityLog1pCompare",
    "SeverityZooCompare",
    "TrainLoopFlags",
    "TrainLoopResult",
    "TrainingArtifacts",
    "TrainingConfig",
    "TripleStackResult",
    "build_metrics_summary",
    "compare_severity_log1p",
    "correlation_filter_features",
    "expected_calibration_error",
    "feature_drift_report",
    "filter_features_by_drift",
    "fit_probability_calibrator",
    "format_features_table",
    "format_metrics_table",
    "format_training_summary",
    "frequency_predict_proba",
    "load_feature_selection_latest",
    "log_training_summary",
    "monthly_target_drift",
    "run_hpo",
    "run_model_diagnostics_visualizations",
    "run_mvp_frequency_eda",
    "run_severity_zoo_compare",
    "run_train_loop_new",
    "run_training_visualizations",
    "run_triple_fin_effects",
    "run_triple_stack",
    "save_feature_selection",
    "severity_error_by_quantile",
    "split_by_date_periods",
    "train_models",
    "train_triple_stacks",
]
