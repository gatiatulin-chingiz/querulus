"""Стабильный нейминг Querulus (semver + партиции данных, без даты в имени модели).

См. ``chat-Нейминг моделей МЛ_ практика и стандарты.txt``:
имя модели стабильное; ``2.0.0`` — релиз; дата/версия датасета — партиции/meta.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from querulus import PROJECT_ROOT

# Стабильные имена (snake_case для Hive / Python / MLflow).
MODEL_NAME: str = "querulus"
MODEL_CF_NAME: str = "querulus_cf"
MODEL_RG_NAME: str = "querulus_rg"

# SemVer релиза контракта/рецепта (не дата переобучения).
MODEL_VERSION: str = "2.0.0"

# Hive: одна таблица + партиции model_version / data_date / dataset_version.
HIVE_SCHEMA: str = "models"
HIVE_TABLE_NAME: str = "querulus_train_dataset"
DEFAULT_HIVE_TABLE: str = f"{HIVE_SCHEMA}.{HIVE_TABLE_NAME}"
DEFAULT_APP_NAME: str = "querulus_train_dataset"

PARTITION_MODEL_VERSION: str = "model_version"
PARTITION_DATA_DATE: str = "data_date"
PARTITION_DATASET_VERSION: str = "dataset_version"
HIVE_PARTITION_COLUMNS: tuple[str, ...] = (
    PARTITION_MODEL_VERSION,
    PARTITION_DATA_DATE,
    PARTITION_DATASET_VERSION,
)

DEFAULT_DATASET_VERSION: str = "1"

DEFAULT_PARQUET_NAME: str = "querulus_train_dataset.parquet"
LEGACY_PARQUET_NAME: str = "df_final_3.parquet"
DEFAULT_PARQUET_PATH: Path = PROJECT_ROOT / "data" / "processed" / DEFAULT_PARQUET_NAME
LEGACY_PARQUET_PATH: Path = PROJECT_ROOT / "data" / "processed" / LEGACY_PARQUET_NAME
LATEST_DATASET_POINTER: Path = (
    PROJECT_ROOT / "data" / "processed" / "querulus_train_dataset_latest.json"
)


def default_model_version(**_kwargs: Any) -> str:
    """SemVer модели (``2.0.0``). Аргументы игнорируются (совместимость вызовов)."""
    return MODEL_VERSION


def today_data_date(*, now: datetime | date | None = None) -> str:
    """Дата среза данных ``YYYY-MM-DD`` (UTC, если now не передан)."""
    if now is None:
        now = datetime.now(timezone.utc)
    if isinstance(now, datetime):
        return now.date().isoformat()
    return now.isoformat()


def hive_table(*, schema: str = HIVE_SCHEMA, table: str = HIVE_TABLE_NAME) -> str:
    return f"{schema}.{table}"


def dataset_partition_values(
    *,
    model_version: str | None = None,
    data_date: str | None = None,
    dataset_version: str | None = None,
) -> dict[str, str]:
    return {
        PARTITION_MODEL_VERSION: model_version or MODEL_VERSION,
        PARTITION_DATA_DATE: data_date or today_data_date(),
        PARTITION_DATASET_VERSION: dataset_version or DEFAULT_DATASET_VERSION,
    }


def configs_dir_for_version(
    model_version: str | None = None,
    *,
    configs_root: Path | str | None = None,
) -> Path:
    root = Path(configs_root) if configs_root is not None else PROJECT_ROOT / "configs"
    return root / MODEL_NAME / (model_version or MODEL_VERSION)


def artifacts_dir_for_version(
    model_version: str | None = None,
    *,
    results_root: Path | str | None = None,
) -> Path:
    root = (
        Path(results_root)
        if results_root is not None
        else PROJECT_ROOT / "integration" / "results"
    )
    return root / MODEL_NAME / (model_version or MODEL_VERSION)


def write_latest_dataset_pointer(
    path: Path | str | None = None,
    *,
    hive_table_name: str | None = None,
    model_version: str | None = None,
    data_date: str | None = None,
    dataset_version: str | None = None,
    parquet_path: str | Path | None = None,
) -> Path:
    """Указатель на последнюю записанную партицию датасета."""
    parts = dataset_partition_values(
        model_version=model_version,
        data_date=data_date,
        dataset_version=dataset_version,
    )
    out = Path(path) if path is not None else LATEST_DATASET_POINTER
    payload = {
        "model_name": MODEL_NAME,
        "hive_table": hive_table_name or DEFAULT_HIVE_TABLE,
        **parts,
        "parquet_path": str(parquet_path) if parquet_path is not None else None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def read_latest_dataset_pointer(
    path: Path | str | None = None,
) -> dict[str, Any] | None:
    pointer = Path(path) if path is not None else LATEST_DATASET_POINTER
    if not pointer.is_file():
        return None
    return json.loads(pointer.read_text(encoding="utf-8"))


def resolve_dataset_partitions(
    *,
    model_version: str | None = None,
    data_date: str | None = None,
    dataset_version: str | None = None,
    pointer_path: Path | str | None = None,
) -> dict[str, str]:
    """Явные аргументы → иначе latest pointer → иначе сегодня / defaults."""
    pointer = read_latest_dataset_pointer(pointer_path)
    return {
        PARTITION_MODEL_VERSION: model_version
        or (pointer or {}).get(PARTITION_MODEL_VERSION)
        or MODEL_VERSION,
        PARTITION_DATA_DATE: data_date
        or (pointer or {}).get(PARTITION_DATA_DATE)
        or today_data_date(),
        PARTITION_DATASET_VERSION: dataset_version
        or (pointer or {}).get(PARTITION_DATASET_VERSION)
        or DEFAULT_DATASET_VERSION,
    }
