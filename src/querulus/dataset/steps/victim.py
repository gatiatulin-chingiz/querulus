"""Шаг пайплайна: victim."""
from __future__ import annotations

from querulus.dataset.filters import apply_victim_filters
from querulus.dataset.io import read_parquet_path
from querulus.dataset.paths import DataPaths


def load_victim(paths: DataPaths):
    """Загрузить victim parquet и применить фильтры из configs/dataset_filters.json."""
    df_victim = read_parquet_path(paths.victim_path, artifact="victim")
    return apply_victim_filters(df_victim)
