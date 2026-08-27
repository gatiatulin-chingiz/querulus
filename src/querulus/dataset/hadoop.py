"""Запись / чтение итогового датафрейма Querulus через Spark → Hive.

Пример использования (как в ``comments.txt``)::

    from querulus.dataset.hadoop import pandas_to_hive_table, hive_table_to_pandas

    pandas_to_hive_table(df, \"models.querulus_df_final_3\")
    df = hive_table_to_pandas(\"models.querulus_df_final_3\")
"""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("querulus.dataset.hadoop")

DEFAULT_HIVE_TABLE = "models.querulus_df_final_3"
DEFAULT_APP_NAME = "querulus_df_final"


def _patch_pandas_iteritems() -> None:
    """Pandas 2.x: Spark иногда ждёт устаревший ``DataFrame.iteritems``."""
    if not hasattr(pd.DataFrame, "iteritems"):
        pd.DataFrame.iteritems = pd.DataFrame.items  # type: ignore[attr-defined, method-assign]


def build_spark_session(
    app_name: str = DEFAULT_APP_NAME,
    *,
    executor_memory: str = "6g",
    driver_memory: str = "24g",
    driver_cores: int = 2,
    executor_cores: int = 1,
    executor_instances: int = 2,
) -> Any:
    """SparkSession с Hive (как в рабочем примере на jovyan)."""
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName(app_name)
        .enableHiveSupport()
        .config("spark.executor.memory", executor_memory)
        .config("spark.driver.memory", driver_memory)
        .config("spark.driver.cores", driver_cores)
        .config("spark.executor.cores", executor_cores)
        .config("spark.executor.instances", str(executor_instances))
        .config("spark.dynamicAllocation.enabled", "false")
        .getOrCreate()
    )


def _is_nested_value(value: Any) -> bool:
    if value is None:
        return False
    return isinstance(value, (np.ndarray, list, tuple, dict, set))


def _cell_to_spark_scalar(value: Any) -> Any:
    """Скаляр, который Spark умеет инферить; nested → JSON-строка."""
    if value is None:
        return None
    try:
        if value is pd.NA or value is pd.NaT:
            return None
    except Exception:  # noqa: BLE001
        pass
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, dict, set)):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        x = float(value)
        return None if np.isnan(x) else x
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.to_pydatetime()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _object_column_has_nested(series: pd.Series, *, sample_n: int = 200) -> bool:
    sample = series.dropna().head(sample_n)
    return any(_is_nested_value(v) for v in sample)


def _prepare_frame_for_spark(df: pd.DataFrame) -> pd.DataFrame:
    """Копия кадра: index → колонка; nested/ndarray → JSON string (иначе Spark падает)."""
    out = df.copy()
    if out.index.name or not isinstance(out.index, pd.RangeIndex):
        idx_name = out.index.name or "index"
        if idx_name in out.columns:
            idx_name = f"{idx_name}_ix"
        out = out.reset_index(names=idx_name)

    nested_cols: list[str] = []
    for col in list(out.columns):
        series = out[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            out[col] = pd.to_datetime(series, errors="coerce")
            continue
        if isinstance(series.dtype, pd.CategoricalDtype):
            out[col] = series.astype(object)
            series = out[col]
        if pd.api.types.is_bool_dtype(series) and str(series.dtype) != "boolean":
            out[col] = series.astype("boolean")
            continue
        # object / string: проверить nested (begin_calc и т.п. с ndarray)
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            if _object_column_has_nested(series):
                nested_cols.append(str(col))
                out[col] = series.map(_cell_to_spark_scalar)
            else:
                out[col] = series.map(_cell_to_spark_scalar)

    if nested_cols:
        logger.warning(
            "Hive: колонки с nested/ndarray сериализованы в JSON-строку: %s",
            nested_cols,
        )
        print(f"[hive] nested->json columns ({len(nested_cols)}): {nested_cols[:20]}")

    out.columns = [str(c) for c in out.columns]
    return out


def pandas_to_hive_table(
    df: pd.DataFrame,
    table_name: str = DEFAULT_HIVE_TABLE,
    *,
    mode: str = "overwrite",
    spark: Any | None = None,
    stop_spark: bool = True,
    app_name: str = DEFAULT_APP_NAME,
) -> str:
    """Pandas → Spark DataFrame → ``saveAsTable`` (Hive).

    Returns:
        Полное имя таблицы.
    """
    if df is None or len(df) == 0:
        raise ValueError("Пустой DataFrame — в Hive не пишем")

    _patch_pandas_iteritems()
    own_spark = spark is None
    spark = spark or build_spark_session(app_name=app_name)
    try:
        prepared = _prepare_frame_for_spark(df)
        logger.info(
            "Hive WRITE start: table=%s mode=%s shape=%s",
            table_name,
            mode,
            prepared.shape,
        )
        sdf = spark.createDataFrame(prepared)
        sdf.write.mode(mode).saveAsTable(table_name)
        logger.info("Hive WRITE done: %s", table_name)
        print(f"[hive] wrote {table_name} shape={prepared.shape} mode={mode}")
        return table_name
    finally:
        if own_spark and stop_spark:
            spark.stop()


def hive_table_to_pandas(
    table_name: str = DEFAULT_HIVE_TABLE,
    *,
    spark: Any | None = None,
    stop_spark: bool = True,
    app_name: str = DEFAULT_APP_NAME,
) -> pd.DataFrame:
    """Hive-таблица → Pandas DataFrame."""
    own_spark = spark is None
    spark = spark or build_spark_session(app_name=app_name)
    try:
        logger.info("Hive READ start: table=%s", table_name)
        sdf = spark.table(table_name)
        pdf = sdf.toPandas()
        logger.info("Hive READ done: %s shape=%s", table_name, pdf.shape)
        print(f"[hive] loaded {table_name} shape={pdf.shape}")
        return pdf
    finally:
        if own_spark and stop_spark:
            spark.stop()
