"""Fetch SQL-артефакты для targets."""
from __future__ import annotations

import pandas as pd

from querulus.dataset.load.sql import (
    load_named_sql_artifact,
    pretension_surcharge_params,
    render_claims_predicate,
    render_sql,
)
from querulus.dataset.paths import DataPaths


def fetch_calc_agg(
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
        "df_calc_agg.parquet",
        "calc_agg.sql",
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
    )


def fetch_target_psr(
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
        "target_2.parquet",
        "target_psr.sql",
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
    )


def fetch_target_3_pretensions(
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
        "target_3_pretensions.parquet",
        "pretension_surcharge_by_incident.sql",
        sql_params=pretension_surcharge_params(
            surcharge_alias="SurchargeValue_cumsum_by_incident",
            uts_alias="UTSSurchargeValue_cumsum_by_incident",
            pretension_types=(
                "Несогласие с суммой выплаты",
                "Претензия на принятое решение",
            ),
        ),
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
    )


def fetch_target_3_pretensions_all(
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
        "target_3_pretensions_all.parquet",
        "pretension_surcharge_by_incident.sql",
        sql_params=pretension_surcharge_params(
            surcharge_alias="SurchargeValue_cumsum_by_incident_all",
            uts_alias="UTSSurchargeValue_cumsum_by_incident_all",
            pretension_types=None,
        ),
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
    )


def fetch_target_3_claims(
    paths: DataPaths,
    conn,
    *,
    use_sql: bool = False,
    save_checkpoint: bool = True,
) -> pd.DataFrame:
    claims_where = render_claims_predicate(icnl_alias="icnl", loss_alias="l")
    return load_named_sql_artifact(
        paths,
        conn,
        paths.raw_dir,
        "target_3_claims.parquet",
        "claims_incoming_targets.sql",
        sql_params={"claims_where": claims_where},
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
    )
