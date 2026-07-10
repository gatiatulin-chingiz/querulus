"""Сборка и обогащение обучающего датасета."""

from querulus.dataset.artifacts import cleanup_legacy_artifacts
from querulus.dataset.pipeline import run_pipeline

__all__ = ["cleanup_legacy_artifacts", "run_pipeline"]
