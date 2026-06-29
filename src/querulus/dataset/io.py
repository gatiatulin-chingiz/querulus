"""I/O: checkpoint parquet и подключение к OISUU."""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import pandas as pd
import pymssql

from querulus.dataset import config
from querulus.dataset.paths import DataPaths

T = TypeVar("T", bound=pd.DataFrame)

logger = logging.getLogger("querulus.dataset")


class LazyOisuuConnection:
    """Подключение к OISUU по требованию (без кредов, если SQL не нужен)."""

    def __init__(self) -> None:
        self._conn: pymssql.Connection | None = None

    def get(self) -> pymssql.Connection:
        if self._conn is None:
            logger.info("Подключение к OISUU (%s)...", config.oisuu_db_server or "?")
            self._conn = connect_oisuu()
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


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
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Parquet для {artifact or path.name!r} не найден: {path}. "
            "Проверьте путь в VICTIM_PARQUET или configs/dataset_sources.json."
        )
    if path.is_file() and path.suffix.lower() != ".parquet":
        raise ValueError(
            f"Ожидался parquet для {artifact or path.name!r}, но указан файл: {path}. "
            "Укажите путь к .parquet в VICTIM_PARQUET или artifact_paths.victim."
        )
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
            f"Задайте USE_SQL=True для загрузки из БД, путь в configs/dataset_sources.json "
            f"или положите файл в legacy "
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


def load_sql_artifact(
    paths: DataPaths,
    conn: LazyOisuuConnection,
    directory: Path,
    name: str,
    query: str,
    *,
    use_sql: bool = False,
    save_checkpoint: bool = True,
    sql_reader: Callable | None = None,
) -> pd.DataFrame:
    """Сырой SQL-дамп: parquet из directory или выгрузка в SQL с сохранением в raw."""
    label = Path(name).stem

    if not use_sql:
        path = paths.resolve_artifact(directory, name)
        if path is not None:
            logger.info("LOAD parquet: %s (use_sql=False)", path)
            return read_parquet_path(path, artifact=label)
        logger.info("Parquet %r не найден — выгрузка из SQL", name)

    reader = sql_reader or (lambda q, c: pd.read_sql(q, c))
    logger.info("LOAD sql: %s", label)
    df = reader(query, conn.get())

    if save_checkpoint:
        out = paths.artifact(directory, name)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(str(out))
        logger.info("SAVE raw: %s  shape=%s", out, df.shape)

    return df


def connect_oisuu() -> pymssql.Connection:
    """Подключение к OISUU через pymssql."""
    return pymssql.connect(
        server=config.oisuu_db_server,
        user=config.oisuu_db_user,
        password=config.oisuu_db_password,
        database=config.oisuu_db_database,
        port=config.oisuu_db_port,
    )
