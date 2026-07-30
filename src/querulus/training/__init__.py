"""Обучение моделей querulus."""

from querulus.training.backward_elim import (
    BackwardElimResult,
    backward_eliminate_by_metric,
)
from querulus.training.calibration import expected_calibration_error, fit_probability_calibrator
from querulus.training.config import TrainingConfig
from querulus.training.drift import (
    feature_drift_report,
    filter_features_by_drift,
    monthly_target_drift,
)
from querulus.training.feature_selection_io import (
    drop_zero_importance_features,
    load_feature_selection_latest,
    save_feature_selection,
)
from querulus.training.feature_selection_report import (
    rebuild_feature_selection_report,
    save_feature_selection_report,
)
from querulus.training.hpo import HpoResult, run_hpo
from querulus.training.noise_cut import NoiseCutResult, filter_features_by_noise
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
from querulus.training.severity_variant import (
    SEVERITY_VARIANT_NAMES,
    SeverityVariantSpec,
    resolve_severity_variant,
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
    "BackwardElimResult",
    "DateSplitParts",
    "HpoResult",
    "TARGET_STACKS",
    "SeverityLog1pCompare",
    "SEVERITY_VARIANT_NAMES",
    "SeverityVariantSpec",
    "SeverityZooCompare",
    "resolve_severity_variant",
    "TrainLoopFlags",
    "TrainLoopResult",
    "TrainingArtifacts",
    "TrainingConfig",
    "TripleStackResult",
    "backward_eliminate_by_metric",
    "build_metrics_summary",
    "compare_severity_log1p",
    "drop_zero_importance_features",
    "expected_calibration_error",
    "feature_drift_report",
    "filter_features_by_drift",
    "filter_features_by_noise",
    "fit_probability_calibrator",
    "format_features_table",
    "format_metrics_table",
    "format_training_summary",
    "frequency_predict_proba",
    "load_feature_selection_latest",
    "log_training_summary",
    "monthly_target_drift",
    "NoiseCutResult",
    "rebuild_feature_selection_report",
    "run_hpo",
    "run_model_diagnostics_visualizations",
    "run_mvp_frequency_eda",
    "run_severity_zoo_compare",
    "run_train_loop_new",
    "run_training_visualizations",
    "run_triple_fin_effects",
    "run_triple_stack",
    "save_feature_selection",
    "save_feature_selection_report",
    "severity_error_by_quantile",
    "split_by_date_periods",
    "train_models",
    "train_triple_stacks",
]
