"""Обучение моделей querulus."""

from querulus.training.config import TrainingConfig
from querulus.training.pipeline import TrainingArtifacts, train_models

__all__ = ["TrainingArtifacts", "TrainingConfig", "train_models"]
