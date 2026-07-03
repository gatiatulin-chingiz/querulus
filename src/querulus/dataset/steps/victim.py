"""Шаг пайплайна: victim."""
from __future__ import annotations

from querulus.dataset.filters import (
    VICTIM_OBJECT_TYPE_COLUMN,
    load_dataset_filters,
    loss_object_types_sql,
    merge_loss_object_types,
    victim_parquet_filter_query,
)
from querulus.dataset.io import load_sql_artifact, read_parquet_path
from querulus.dataset.paths import DataPaths


def load_victim(
    paths: DataPaths,
    conn,
    *,
    use_sql: bool = False,
    save_checkpoint: bool = True,
):
    """Загрузить victim parquet, присоединить VictimObjectType из SQL и отфильтровать."""
    df_victim = read_parquet_path(paths.victim_path, artifact="victim")
    df_victim = df_victim.query(victim_parquet_filter_query())

    df_loss_types = load_sql_artifact(
        paths,
        conn,
        paths.raw_dir,
        "df_loss_object_types.parquet",
        loss_object_types_sql(),
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
    )
    df_victim = merge_loss_object_types(df_victim, df_loss_types)

    if VICTIM_OBJECT_TYPE_COLUMN not in df_victim.columns:
        raise KeyError(
            f"Колонка {VICTIM_OBJECT_TYPE_COLUMN!r} не найдена после merge с loss object types."
        )
    object_type = load_dataset_filters()["victim"]["victim_object_type"]
    return df_victim[
        df_victim[VICTIM_OBJECT_TYPE_COLUMN].astype(str) == object_type
    ].reset_index(drop=True)
