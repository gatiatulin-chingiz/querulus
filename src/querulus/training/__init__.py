"""Обучение моделей querulus."""

from querulus.training.config import TrainingConfig
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
    "TrainingArtifacts",
    "TrainingConfig",
    "TripleStackResult",
    "build_metrics_summary",
    "format_features_table",
    "format_metrics_table",
    "format_training_summary",
    "frequency_predict_proba",
    "log_training_summary",
    "run_model_diagnostics_visualizations",
    "run_mvp_frequency_eda",
    "run_training_visualizations",
    "run_triple_fin_effects",
    "run_triple_stack",
    "train_models",
    "train_triple_stacks",
]
