"""Оркестратор feature engineering (этапы 0–1)."""
from __future__ import annotations

import gc
import logging

import pandas as pd

from querulus.dataset.io import checkpoint
from querulus.dataset.paths import DataPaths
from querulus.features.cleanup import cleanup_merge_columns
from querulus.features.config import FeatureConfig, load_feature_config
from querulus.features.derived import add_derived_features
from querulus.features.data_quality import clip_negative_value_before_diff
from querulus.features.integer_casts import cast_integer_like_columns

logger = logging.getLogger("querulus.features")

_INCIDENT_COLUMN = "INCIDENT_NUMBER"


def _filter_pretensions_for_incidents(df: pd.DataFrame, pret: pd.DataFrame) -> pd.DataFrame:
    """Оставить претензии только по инцидентам из обучающей выборки."""
    if pret.empty or _INCIDENT_COLUMN not in df.columns:
        return pret
    incident_col = None
    for name in ("INCIDENT_NUMBER", "INCIDENTNUMBER"):
        if name in pret.columns:
            incident_col = name
            break
    if incident_col is None:
        return pret
    inc_set = set(pd.to_numeric(df[_INCIDENT_COLUMN], errors="coerce").dropna().astype("int64"))
    mask = pd.to_numeric(pret[incident_col], errors="coerce").isin(inc_set)
    filtered = pret.loc[mask]
    logger.info(
        "Претензии для incident-FE: %s → %s строк (инцидентов в df=%s)",
        len(pret),
        len(filtered),
        len(inc_set),
    )
    return filtered


def run_features(
    df: pd.DataFrame,
    paths: DataPaths,
    *,
    conn=None,
    use_sql: bool = False,
    config: FeatureConfig | None = None,
    save_checkpoint: bool = True,
    include_fe_features: bool = True,
    include_person_features: bool = False,
) -> pd.DataFrame:
    """Этап 0 (cleanup) + опционально derived/incident FE + person-history.

    ``include_fe_features=False``: только cleanup, без ``FE_*`` derived/incident.
    ``include_person_features`` имеет смысл только при ``include_fe_features=True``.
    """
    feature_config = config or load_feature_config()
    rows_before = len(df)
    cols_before = df.shape[1]

    df = cleanup_merge_columns(df, feature_config)
    # Возраст/год и 0/1-флаги часто float из SQL — приводим к Int64 (in-place)
    df = cast_integer_like_columns(df)
    gc.collect()

    pret_base = None
    if include_fe_features:
        df = add_derived_features(df, feature_config)
        gc.collect()

        from querulus.features.incident_pretensions import add_incident_pretension_features
        from querulus.features.person.loaders import load_pretensions_base

        pret_base = load_pretensions_base(
            paths, conn, use_sql=use_sql, save_checkpoint=save_checkpoint
        )
        pret_incident = _filter_pretensions_for_incidents(df, pret_base)
        # Полный pret не нужен без person-FE — сразу освобождаем пик ОЗУ
        if not include_person_features:
            del pret_base
            pret_base = None
            gc.collect()
        df = add_incident_pretension_features(df, pret_incident, feature_config)
        del pret_incident
        gc.collect()

        if include_person_features:
            from querulus.features.person.pipeline import run_person_features

            logger.info(
                "Person FE включены — пик ОЗУ высокий; "
                "при нехватке памяти поставьте INCLUDE_PERSON_FEATURES=False"
            )
            df = run_person_features(
                df,
                paths,
                conn=conn,
                use_sql=use_sql,
                save_checkpoint=save_checkpoint,
                pretensions_base=pret_base,
            )
            del pret_base
            pret_base = None
            gc.collect()
            # Person money-FE появляются только здесь → REAL после сборки.
            from querulus.features.inflation import (
                MONETARY_COLUMNS_FOR_REAL,
                add_real_monetary_columns,
            )

            person_money = tuple(
                c for c in MONETARY_COLUMNS_FOR_REAL if c.startswith("FE_PERSON_")
            )
            df = add_real_monetary_columns(
                df,
                df[feature_config.t0_column],
                person_money,
                base_year=feature_config.inflation_base_year,
            )
        else:
            logger.info("Person features пропущены (include_person_features=False).")
        if pret_base is not None:
            del pret_base
        gc.collect()
    else:
        logger.info("Derived/incident FE пропущены (include_fe_features=False).")
        if include_person_features:
            logger.info("Person FE пропущены: нужен include_fe_features=True.")

    # Аномалии: WITHOUT < WITH → отрицательный «износ» → клип в 0 (без drop)
    if "FE_VALUE_BEFORE_DIFF" in df.columns:
        before = pd.to_numeric(df["FE_VALUE_BEFORE_DIFF"], errors="coerce")
        n_neg = int((before < 0).sum())
        df = clip_negative_value_before_diff(df)
        if n_neg:
            logger.info(
                "FE_VALUE_BEFORE_DIFF < 0: clipped %s cells to 0",
                n_neg,
            )

    fe_added = [col for col in feature_config.fe_columns if col in df.columns]
    logger.info(
        "FE завершён: rows=%s (было %s), cols=%s (+%s FE, всего FE=%s)",
        len(df),
        rows_before,
        df.shape[1],
        df.shape[1] - cols_before,
        len(fe_added),
    )

    return checkpoint(
        df,
        paths,
        paths.processed_dir,
        "df_final_3.parquet",
        save=save_checkpoint,
    )
