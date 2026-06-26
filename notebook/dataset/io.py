"""I/O: checkpoint parquet и подключение к OISUU."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TypeVar

import pandas as pd
import pymssql

import config
from dataset.paths import DataPaths

T = TypeVar("T", bound=pd.DataFrame)

logger = logging.getLogger("querulus.dataset")


def setup_notebook_logging(level: int = logging.INFO) -> None:
    """Настроить вывод логов в stdout (видно в ячейках Jupyter)."""
    logger.setLevel(level)
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[dataset] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False


def _log_parquet(action: str, path: Path, df: pd.DataFrame, artifact: str = "") -> None:
    suffix = f"  (artifact={artifact})" if artifact else ""
    logger.info("%s: %s  shape=%s%s", action, path, df.shape, suffix)


def read_parquet_path(path: Path, *, artifact: str = "") -> pd.DataFrame:
    """Загрузить parquet по точному пути."""
    df = pd.read_parquet(str(path))
    _log_parquet("LOAD", path, df, artifact)
    return df


def read_artifact(paths: DataPaths, directory: Path, name: str) -> pd.DataFrame:
    """Загрузить parquet с учётом legacy-путей и вариантов имени."""
    path = paths.resolve_artifact(directory, name)
    if path is None:
        logger.error("Не найден parquet: %s", name)
        for candidate in paths.artifact_candidates(directory, name):
            logger.error("  проверен: %s", candidate)
        write_path = paths.artifact(directory, name)
        raise FileNotFoundError(
            f"Артефакт {name!r} не найден. "
            f"Задайте USE_SQL=True для загрузки из БД или положите файл в legacy "
            f"({config.litigant_legacy_data_root}/data/raw|processed|parquet). "
            f"Путь записи при save_checkpoint: {write_path}"
        )
    df = pd.read_parquet(str(path))
    _log_parquet("LOAD", path, df, name)
    return df


def checkpoint(
    df: T,
    paths: DataPaths,
    directory: Path,
    name: str,
    *,
    save: bool = True,
) -> T:
    """Сохранить или загрузить датафрейм (SAVE_CHECKPOINT-паттерн)."""
    if save:
        path = paths.artifact(directory, name)
        df.to_parquet(str(path))
        _log_parquet("SAVE", path, df, name)
        return df
    return read_artifact(paths, directory, name)


def checkpoint_local(
    df: T,
    path: Path,
    *,
    save: bool = True,
) -> T:
    """Checkpoint в локальную папку notebook/data (без legacy-поиска)."""
    path = Path(path)
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(str(path))
        _log_parquet("SAVE", path, df, path.name)
        return df
    return read_parquet_path(path, artifact=path.name)


def connect_oisuu() -> pymssql.Connection:
    """Подключение к OISUU через pymssql."""
    return pymssql.connect(
        server=config.oisuu_db_server,
        user=config.oisuu_db_user,
        password=config.oisuu_db_password,
        database=config.oisuu_db_database,
        port=config.oisuu_db_port,
    )
