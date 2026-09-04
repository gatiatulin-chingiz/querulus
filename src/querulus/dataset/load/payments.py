"""Fetch payments SQL-артефакт."""
from __future__ import annotations

from querulus.dataset.load.sql import load_named_sql_artifact
from querulus.dataset.paths import DataPaths


def fetch_payments(
    paths: DataPaths,
    conn,
    *,
    use_sql: bool = False,
    save_checkpoint: bool = True,
    columns: list[str] | None = None,
):
    return load_named_sql_artifact(
        paths,
        conn,
        paths.raw_dir,
        "df_payments.parquet",
        "payments.sql",
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
        columns=columns or ["IncidentNumber", "PaymentDateTime", "PaymentValue"],
    )
