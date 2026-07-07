"""Оркестратор person-feature engineering (static + history)."""

from __future__ import annotations

import logging

import pandas as pd

from querulus.dataset.filters import claims_sql_predicate
from querulus.dataset.io import LazyOisuuConnection
from querulus.dataset.paths import DataPaths
from querulus.features.person.history_court import add_person_court_history
from querulus.features.person.history_pretensions import add_person_pretension_history
from querulus.features.person.loaders import (
    load_claims_incoming,
    load_claims_persons,
    load_pretensions_base,
    load_pretensions_penalty_surcharge,
)
from querulus.features.person.static import add_person_static_features

logger = logging.getLogger("querulus.features.person")


def run_person_features(
    df: pd.DataFrame,
    paths: DataPaths,
    *,
    conn: LazyOisuuConnection | None,
    use_sql: bool,
    save_checkpoint: bool,
) -> pd.DataFrame:
    """Добавить person static + history признаки."""
    out = add_person_static_features(df)

    # Pretensions: base + penalty/surcharge merge (optional).
    pret = load_pretensions_base(paths, conn, use_sql=use_sql, save_checkpoint=save_checkpoint)
    try:
        pret_3 = load_pretensions_penalty_surcharge(paths, conn, use_sql=use_sql, save_checkpoint=save_checkpoint)
        if "PRETENSIONNUMBER" in pret_3.columns and "PRETENSION_NUMBER" not in pret_3.columns:
            pret_3["PRETENSION_NUMBER"] = pret_3["PRETENSIONNUMBER"]
        # keep only needed penalty fields
        keep = [c for c in ("PRETENSION_NUMBER", "PRETENSION_VALUE_PENALTY", "SURCHARGE_VALUE_PENALTY") if c in pret_3.columns]
        if keep:
            pret = pret.merge(pret_3[keep].drop_duplicates("PRETENSION_NUMBER"), on="PRETENSION_NUMBER", how="left")
    except FileNotFoundError:
        logger.info("df_pretensions_3.parquet not found; skip penalty/surcharge columns.")

    out = add_person_pretension_history(out, pret)

    # Court: incoming claims + persons
    claims_where = claims_sql_predicate(icnl_alias="icnl", loss_alias="l")
    df_claims_incoming = load_claims_incoming(
        paths,
        conn,
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
        claims_where_sql=claims_where,
    )
    df_claims_persons = load_claims_persons(paths, conn, use_sql=use_sql, save_checkpoint=save_checkpoint)
    out = add_person_court_history(out, df_claims_incoming, df_claims_persons)

    return out

