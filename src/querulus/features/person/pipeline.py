"""Оркестратор person-feature engineering (static + history)."""

from __future__ import annotations

import gc
import logging

import pandas as pd

from querulus.dataset.io import LazyOisuuConnection
from querulus.dataset.paths import DataPaths
from querulus.features.person.history_court import add_person_court_history
from querulus.features.person.history_pretensions import add_person_pretension_history
from querulus.features.person.loaders import (
    load_claims_persons,
    load_pretensions_base,
    load_pretensions_penalty_surcharge,
    load_target_claims_for_features,
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
    pretensions_base: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Добавить person static + history признаки."""
    out = add_person_static_features(df)

    # Pretensions: base + penalty/surcharge merge (optional).
    pret_raw = pretensions_base
    if pret_raw is None:
        pret_raw = load_pretensions_base(paths, conn, use_sql=use_sql, save_checkpoint=save_checkpoint)
    pret = pret_raw
    try:
        pret_3 = load_pretensions_penalty_surcharge(
            paths, conn, use_sql=use_sql, save_checkpoint=save_checkpoint
        )
        keep = [c for c in ("PRETENSION_NUMBER", "PRETENSION_VALUE_PENALTY", "SURCHARGE_VALUE_PENALTY") if c in pret_3.columns]
        if keep and "PRETENSION_NUMBER" in pret.columns:
            pret = pret.merge(pret_3[keep].drop_duplicates("PRETENSION_NUMBER"), on="PRETENSION_NUMBER", how="left")
        del pret_3
    except FileNotFoundError:
        logger.info("df_pretensions_3.parquet not found; skip penalty/surcharge columns.")

    out = add_person_pretension_history(out, pret)
    del pret
    gc.collect()

    # Court: target_3_claims (кэш targets) + persons
    df_claims_incoming = load_target_claims_for_features(
        paths,
        conn,
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
    )
    df_claims_persons = load_claims_persons(paths, conn, use_sql=use_sql, save_checkpoint=save_checkpoint)
    out = add_person_court_history(out, df_claims_incoming, df_claims_persons)
    del df_claims_persons
    gc.collect()

    return out

