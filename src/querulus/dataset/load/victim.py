"""Fetch victim parquet и loss object types."""
from __future__ import annotations

from querulus.dataset.load.io import read_parquet_path
from querulus.dataset.load.sql import load_named_sql_artifact, loss_object_types_params
from querulus.dataset.paths import DataPaths


def fetch_victim_frame(paths: DataPaths):
    """Загрузить сырой victim parquet."""
    return read_parquet_path(paths.victim_path, artifact="victim")


def fetch_loss_object_types(
    paths: DataPaths,
    conn,
    *,
    use_sql: bool = False,
    save_checkpoint: bool = True,
):
    """SQL/parquet: VictimObjectType по LOSS_NUMBER."""
    return load_named_sql_artifact(
        paths,
        conn,
        paths.raw_dir,
        "df_loss_object_types.parquet",
        "loss_object_types.sql",
        sql_params=loss_object_types_params(),
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
    )
