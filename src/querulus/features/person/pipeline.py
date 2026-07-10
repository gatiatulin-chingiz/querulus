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

_PENALTY_COLS = ("PRETENSION_VALUE_PENALTY", "SURCHARGE_VALUE_PENALTY")


def _pretension_number_column(df: pd.DataFrame) -> str | None:
    for name in ("PRETENSION_NUMBER", "PRETENSIONNUMBER"):
        if name in df.columns:
            return name
    return None


def _merge_penalty_columns(pret: pd.DataFrame, pret_3: pd.DataFrame) -> pd.DataFrame:
    """Подтянуть penalty/surcharge по номеру претензии."""
    pn_pret = _pretension_number_column(pret)
    pn_3 = _pretension_number_column(pret_3)
    if pn_pret is None or pn_3 is None:
        logger.info("Penalty merge пропущен: нет колонки номера претензии.")
        return pret

    keep = [c for c in _PENALTY_COLS if c in pret_3.columns]
    if not keep:
        return pret

    extra = pret_3[[pn_3, *keep]].drop_duplicates(subset=[pn_3])
    if pn_pret == pn_3:
        return pret.merge(extra, on=pn_pret, how="left")

    out = pret.copy()
    out["_pn_merge"] = out[pn_pret]
    out = out.merge(extra, left_on="_pn_merge", right_on=pn_3, how="left")
    return out.drop(columns=["_pn_merge", pn_3], errors="ignore")


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
    logger.info("Person static FE готов, shape=%s", out.shape)

    # Pretensions: base + penalty/surcharge merge (optional).
    pret_raw = pretensions_base
    if pret_raw is None:
        pret_raw = load_pretensions_base(paths, conn, use_sql=use_sql, save_checkpoint=save_checkpoint)
    pret = pret_raw
    try:
        pret_3 = load_pretensions_penalty_surcharge(
            paths, conn, use_sql=use_sql, save_checkpoint=save_checkpoint
        )
        pret = _merge_penalty_columns(pret, pret_3)
        del pret_3
        logger.info("Penalty merge готов, pret shape=%s", pret.shape)
    except FileNotFoundError:
        logger.info("df_pretensions_3.parquet not found; skip penalty/surcharge columns.")

    out = add_person_pretension_history(out, pret)
    del pret
    gc.collect()
    logger.info("Person pretension history готов, shape=%s", out.shape)

    # Court: target_3_claims (кэш targets) + persons
    df_claims_incoming = load_target_claims_for_features(
        paths,
        conn,
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
    )
    logger.info("Claims для court history загружены, shape=%s", df_claims_incoming.shape)
    df_claims_persons = load_claims_persons(paths, conn, use_sql=use_sql, save_checkpoint=save_checkpoint)
    out = add_person_court_history(out, df_claims_incoming, df_claims_persons)
    del df_claims_incoming, df_claims_persons
    gc.collect()
    logger.info("Person court history готов, shape=%s", out.shape)

    return out

