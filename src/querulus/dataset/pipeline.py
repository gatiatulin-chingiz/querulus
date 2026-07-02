"""Оркестратор пайплайна сборки датасета."""
from __future__ import annotations

import gc
import logging

import pandas as pd

from querulus.dataset.io import LazyOisuuConnection, setup_notebook_logging
from querulus.dataset.paths import DataPaths
from querulus.dataset.steps.claims import load_claims
from querulus.dataset.steps.enrich import enrich_dataset
from querulus.dataset.steps.payments import load_claims_payments
from querulus.dataset.steps.pretensions import load_pretensions
from querulus.dataset.steps.targets import build_targets
from querulus.dataset.steps.victim import load_victim

logger = logging.getLogger("querulus.dataset")


def run_pipeline(
    *,
    use_sql: bool = False,
    save_checkpoint: bool = True,
) -> pd.DataFrame:
    """Собрать обучающий датасет: victim → claims → payments → pretensions → enrich → targets."""
    setup_notebook_logging()

    paths = DataPaths.from_config()
    conn = LazyOisuuConnection()

    try:
        df_victim = load_victim(paths)
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
            None,
            df_claims_,
            df_claims_payments,
            df_pretensions,
            df_claims_persons,
            pretension_fio_id,
            save_checkpoint=save_checkpoint,
        )
        del df_victim, df_claims_payments, df_pretensions, df_claims_persons
        del pretension_fio_id
        gc.collect()
        df = build_targets(
            paths, conn, df, df_claims_, save_checkpoint=save_checkpoint, use_sql=use_sql
        )
    finally:
        conn.close()

    logger.info("Пайплайн завершён: итоговый shape=%s", df.shape)
    return df
