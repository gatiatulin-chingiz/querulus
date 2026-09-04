"""Оркестратор пайплайна сборки датасета."""
from __future__ import annotations

import gc
import logging

import pandas as pd

from querulus.dataset.load.io import LazyOisuuConnection, checkpoint, setup_notebook_logging
from querulus.dataset.preprocess.enrich import enrich_dataset
from querulus.dataset.preprocess.maturity import apply_target_maturity_from_paths
from querulus.dataset.preprocess.targets import build_targets
from querulus.dataset.preprocess.victim import prepare_victim
from querulus.dataset.load.victim import fetch_loss_object_types, fetch_victim_frame
from querulus.dataset.paths import DataPaths
from querulus.dataset.steps.claims import load_claims
from querulus.dataset.steps.payments import load_claims_payments
from querulus.dataset.steps.pretensions import load_pretensions
from querulus.features.pipeline import run_features

logger = logging.getLogger("querulus.dataset")


def run_pipeline(
    *,
    use_sql: bool = False,
    save_checkpoint: bool = True,
    include_enrich: bool = False,
    include_fe_features: bool = True,
    include_person_features: bool = False,
    resume_from_targets: bool = False,
    maturity_enabled: bool | None = None,
) -> pd.DataFrame:
    """Собрать обучающий датасет.

    По умолчанию (include_enrich=False): victim → targets.
    При include_enrich=True дополнительно выполняются legacy-шаги claims/payments/
    pretensions/enrich (см. комментарии в соответствующих модулях); в обучении не
    используются из-за утечки ПСР в колонках *_FTRS_*.

    resume_from_targets=True: пропустить victim/targets, загрузить df_after_targets.parquet.
    include_fe_features=False: без derived/incident ``FE_*``.
    include_person_features=False: без FE_PERSON_* (экономия ОЗУ, по умолчанию).
    maturity_enabled: None — configs/dataset_filters.json; False/True — override.
    """
    setup_notebook_logging()

    paths = DataPaths.from_config()
    conn = LazyOisuuConnection()
    df: pd.DataFrame | None = None

    try:
        if resume_from_targets:
            df = checkpoint(
                pd.DataFrame(),
                paths,
                paths.processed_dir,
                "df_after_targets.parquet",
                save=False,
            )
            logger.info("Продолжение с df_after_targets.parquet, shape=%s", df.shape)
            df = apply_target_maturity_from_paths(
                df, paths, maturity_enabled=maturity_enabled
            )
        else:
            df_victim = fetch_victim_frame(paths)
            df_loss_types = fetch_loss_object_types(
                paths, conn, use_sql=use_sql, save_checkpoint=save_checkpoint
            )
            df_victim = prepare_victim(df_victim, df_loss_types)

            if include_enrich:
                df_claims_persons, df_claims, df_claims_ = load_claims(
                    paths, conn, use_sql=use_sql, save_checkpoint=save_checkpoint
                )
                df_claims_payments = load_claims_payments(
                    paths, conn, df_claims, use_sql=use_sql, save_checkpoint=save_checkpoint
                )
                del df_claims
                gc.collect()
                df_pretensions, pretension_fio_id = load_pretensions(
                    paths, conn, use_sql=use_sql, save_checkpoint=save_checkpoint
                )
                df = enrich_dataset(
                    paths,
                    df_victim,
                    df_claims_payments,
                    df_claims_,
                    df_claims_payments,
                    df_pretensions,
                    df_claims_persons,
                    pretension_fio_id,
                    save_checkpoint=save_checkpoint,
                )
                del df_victim, df_claims_payments, df_pretensions, df_claims_persons
                del pretension_fio_id, df_claims_
                gc.collect()
            else:
                df = df_victim

            df = build_targets(
                paths,
                conn,
                df,
                save_checkpoint=save_checkpoint,
                use_sql=use_sql,
                maturity_enabled=maturity_enabled,
            )
            df = checkpoint(
                df,
                paths,
                paths.processed_dir,
                "df_after_targets.parquet",
                save=save_checkpoint,
            )
            gc.collect()

        df = run_features(
            df,
            paths,
            conn=conn,
            use_sql=use_sql,
            save_checkpoint=save_checkpoint,
            include_fe_features=include_fe_features,
            include_person_features=include_person_features,
        )
    finally:
        conn.close()

    if df is not None:
        logger.info("Пайплайн завершён: итоговый shape=%s", df.shape)
    return df
