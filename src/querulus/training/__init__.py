"""Обучение моделей querulus."""

from querulus.training.config import TrainingConfig
from querulus.training.drift import feature_drift_report, monthly_target_drift
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
from querulus.training.feature_selection_io import (
    load_feature_selection_latest,
    save_feature_selection,
)
from querulus.training.severity_zoo import SeverityZooCompare, run_severity_zoo_compare
from querulus.training.triple_stack import (
    TARGET_STACKS,
    TripleStackResult,
    build_metrics_summary,
    run_triple_fin_effects,
    run_triple_stack,
    train_triple_stacks,
)

__all__ = [
    "TARGET_STACKS",
    "SeverityLog1pCompare",
    "SeverityZooCompare",
    "TrainingArtifacts",
    "TrainingConfig",
    "TripleStackResult",
    "build_metrics_summary",
    "compare_severity_log1p",
    "feature_drift_report",
    "format_features_table",
    "format_metrics_table",
    "format_training_summary",
    "frequency_predict_proba",
    "load_feature_selection_latest",
    "log_training_summary",
    "monthly_target_drift",
    "run_model_diagnostics_visualizations",
    "run_mvp_frequency_eda",
    "run_severity_zoo_compare",
    "run_training_visualizations",
    "run_triple_fin_effects",
    "run_triple_stack",
    "save_feature_selection",
    "severity_error_by_quantile",
    "train_models",
    "train_triple_stacks",
]
