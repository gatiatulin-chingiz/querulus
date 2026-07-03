"""Feature engineering: cleanup и derived FE_* колонки."""

__all__ = ["run_features"]


def run_features(*args, **kwargs):
    """Ленивый импорт, чтобы не тянуть dataset.pipeline при загрузке пакета."""
    from querulus.features.pipeline import run_features as _run_features

    return _run_features(*args, **kwargs)
