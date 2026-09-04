"""Fetch pretensions SQL-артефакты."""
from __future__ import annotations

import pandas as pd

from querulus.dataset.load.sql import load_named_sql_artifact
from querulus.dataset.paths import DataPaths


def fetch_pretensions_base(
    paths: DataPaths,
    conn,
    *,
    use_sql: bool = False,
    save_checkpoint: bool = True,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    return load_named_sql_artifact(
        paths,
        conn,
        paths.raw_dir,
        "df_pretensions.parquet",
        "pretensions_base.sql",
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
        columns=columns,
    )


def fetch_pretension_fio_ids(
    paths: DataPaths,
    conn,
    *,
    use_sql: bool = False,
    save_checkpoint: bool = True,
) -> pd.DataFrame:
    return load_named_sql_artifact(
        paths,
        conn,
        paths.raw_dir,
        "pretension_fio_id.parquet",
        "pretension_fio_ids.sql",
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
    )


def fetch_pretensions_penalty(
    paths: DataPaths,
    conn,
    *,
    use_sql: bool = False,
    save_checkpoint: bool = True,
) -> pd.DataFrame:
    return load_named_sql_artifact(
        paths,
        conn,
        paths.raw_dir,
        "df_pretensions_3.parquet",
        "pretensions_penalty.sql",
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
        sql_reader=pd.read_sql_query,
    )
