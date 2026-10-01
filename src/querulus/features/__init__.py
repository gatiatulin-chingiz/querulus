"""Feature engineering: cleanup и derived FE_* колонки."""
from __future__ import annotations

from typing import Any

__all__ = ["run_features"]


def __getattr__(name: str) -> Any:
    """Ленивый реэкспорт: иначе цикл features.pipeline ↔ dataset.pipeline."""
    if name == "run_features":
        from querulus.features.pipeline import run_features as _run_features

        return _run_features
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
