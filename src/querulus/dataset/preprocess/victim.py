"""Pandas-подготовка victim."""
from __future__ import annotations

from querulus.dataset.preprocess.filters import (
    VICTIM_OBJECT_TYPE_COLUMN,
    load_dataset_filters,
    merge_loss_object_types,
    victim_parquet_filter_query,
)


def prepare_victim(df_victim, df_loss_types):
    """Отфильтровать victim parquet, merge loss types, фильтр по object type."""
    df_victim = df_victim.query(victim_parquet_filter_query())
    df_victim = merge_loss_object_types(df_victim, df_loss_types)

    if VICTIM_OBJECT_TYPE_COLUMN not in df_victim.columns:
        raise KeyError(
            f"Колонка {VICTIM_OBJECT_TYPE_COLUMN!r} не найдена после merge с loss object types."
        )
    object_type = load_dataset_filters()["victim"]["victim_object_type"]
    return df_victim[
        df_victim[VICTIM_OBJECT_TYPE_COLUMN].astype(str) == object_type
    ].reset_index(drop=True)
