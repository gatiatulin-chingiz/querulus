"""Списки parquet-артефактов и очистка legacy-кэша."""
from __future__ import annotations

import logging
from pathlib import Path

from querulus.dataset.paths import DataPaths

logger = logging.getLogger("querulus.dataset")

# Только include_enrich=True (Litigant legacy, в обучении не используются).
LEGACY_ENRICH_ARTIFACTS: tuple[str, ...] = (
    "df_pre_final.parquet",
    "df_claims_pre_final.parquet",
    "df_pretensions_pre_final.parquet",
    "pre_final.parquet",
    "df_pretensions_enriched.parquet",
    "df_claims_.parquet",
    "df_claims.parquet",
    "df_claims_payments.parquet",
)

# Дублируют target_3_claims / enrich; feature-пайплайн их больше не читает.
REDUNDANT_RAW_ARTIFACTS: tuple[str, ...] = (
    "df_claims_incoming.parquet",
)

# Промежуточные processed: удалять при полной пересборке (не resume).
STALE_PROCESSED_ARTIFACTS: tuple[str, ...] = (
    "df_after_targets.parquet",
    "df_final_3.parquet",
)


def _unlink_if_exists(path: Path) -> bool:
    if not path.is_file():
        return False
    path.unlink()
    return True


def cleanup_legacy_artifacts(
    paths: DataPaths | None = None,
    *,
    enrich_only: bool = True,
    redundant_raw: bool = True,
    stale_processed: bool = False,
) -> list[str]:
    """Удалить лишние parquet-кэши.

    enrich_only: артефакты legacy enrich (include_enrich).
    redundant_raw: df_claims_incoming и пр. (заменены target_3_claims).
    stale_processed: df_after_targets / df_final_3 перед полной пересборкой.
    """
    paths = paths or DataPaths.from_config()
    removed: list[str] = []

    names: list[str] = []
    if enrich_only:
        names.extend(LEGACY_ENRICH_ARTIFACTS)
    if redundant_raw:
        names.extend(REDUNDANT_RAW_ARTIFACTS)
    if stale_processed:
        names.extend(STALE_PROCESSED_ARTIFACTS)

    for name in names:
        directory = paths.processed_dir if name in STALE_PROCESSED_ARTIFACTS else paths.raw_dir
        for candidate in paths.artifact_candidates(directory, name):
            if _unlink_if_exists(candidate):
                removed.append(str(candidate))
                logger.info("Удалён legacy-артефакт: %s", candidate)
                break

    if removed:
        logger.info("Очистка: удалено %s файлов", len(removed))
    else:
        logger.info("Очистка: нечего удалять")
    return removed
