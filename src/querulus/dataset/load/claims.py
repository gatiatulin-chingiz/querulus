"""Fetch claims SQL-артефакты (legacy enrich / person FE)."""
from __future__ import annotations

import pandas as pd

from querulus.dataset.load.sql import load_named_sql_artifact, render_claims_predicate
from querulus.dataset.paths import DataPaths


def fetch_claims_persons(
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
        "df_claims_persons.parquet",
        "claims_persons.sql",
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
        sql_reader=pd.read_sql_query,
    )


def fetch_claims_incoming(
    paths: DataPaths,
    conn,
    *,
    use_sql: bool = False,
    save_checkpoint: bool = True,
    claims_where: str | None = None,
) -> pd.DataFrame:
    predicate = claims_where or render_claims_predicate(icnl_alias="icnl", loss_alias="l")
    return load_named_sql_artifact(
        paths,
        conn,
        paths.raw_dir,
        "df_claims_incoming.parquet",
        "claims_incoming.sql",
        sql_params={"claims_where": predicate},
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
        sql_reader=pd.read_sql_query,
    )
