"""Сборка и обогащение обучающего датасета."""
from __future__ import annotations

from typing import Any

from querulus.dataset.schema import (
    DEFAULT_DATASET_SCHEMA,
    DatasetSchema,
    DatasetSchemaError,
)
from querulus.naming import DEFAULT_HIVE_TABLE

__all__ = [
    "DEFAULT_DATASET_SCHEMA",
    "DEFAULT_HIVE_TABLE",
    "DatasetSchema",
    "DatasetSchemaError",
    "cast_object_columns",
    "cleanup_legacy_artifacts",
    "hive_table_to_pandas",
    "load_df_final",
    "pandas_to_hive_table",
    "run_pipeline",
    "save_df_final",
]

_LAZY: dict[str, tuple[str, str]] = {
    "cast_object_columns": (".dtypes", "cast_object_columns"),
    "cleanup_legacy_artifacts": (".artifacts", "cleanup_legacy_artifacts"),
    "hive_table_to_pandas": (".hadoop", "hive_table_to_pandas"),
    "load_df_final": (".hadoop", "load_df_final"),
    "pandas_to_hive_table": (".hadoop", "pandas_to_hive_table"),
    "run_pipeline": (".pipeline", "run_pipeline"),
    "save_df_final": (".hadoop", "save_df_final"),
}


def __getattr__(name: str) -> Any:
    """Ленивый реэкспорт тяжёлых модулей (Spark/env) — schema импортируется без них."""
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        from importlib import import_module

        mod = import_module(module_name, __name__)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
