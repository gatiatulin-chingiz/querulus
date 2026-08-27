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


def _to_spark_string_or_none(value: Any) -> str | None:
    """Значение для StringType: None или str (без nested)."""
    scalar = _cell_to_spark_scalar(value)
    if scalar is None:
        return None
    if isinstance(scalar, str):
        return scalar
    if isinstance(scalar, (bool, int, float)):
        if isinstance(scalar, float) and np.isnan(scalar):
            return None
        return str(scalar)
    return str(scalar)


def _prepare_frame_for_spark(df: pd.DataFrame) -> pd.DataFrame:
    """Копия кадра: index → колонка; nested/ndarray → JSON; типы под Spark schema."""
    out = df.copy()
    if out.index.name or not isinstance(out.index, pd.RangeIndex):
        idx_name = out.index.name or "index"
        if idx_name in out.columns:
            idx_name = f"{idx_name}_ix"
        out = out.reset_index(names=idx_name)

    nested_cols: list[str] = []
    all_null_as_str: list[str] = []
    for col in list(out.columns):
        series = out[col]
        # Все NULL → StringType + None (иначе Spark: CANNOT_DETERMINE_TYPE / NullType)
        if series.isna().all():
            all_null_as_str.append(str(col))
            out[col] = pd.Series([None] * len(out), index=out.index, dtype=object)
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            out[col] = pd.to_datetime(series, errors="coerce")
            continue
        if isinstance(series.dtype, pd.CategoricalDtype):
            out[col] = series.astype(object).map(_to_spark_string_or_none)
            continue
        if pd.api.types.is_bool_dtype(series):
            out[col] = series.map(lambda x: None if pd.isna(x) else bool(x))
            continue
        # Nullable Int64 / все NA уже отсечены: NA в int → float64, иначе Long schema
        if pd.api.types.is_integer_dtype(series):
            if series.isna().any():
                out[col] = series.astype("float64")
            else:
                out[col] = series.astype("int64")
            continue
        if pd.api.types.is_float_dtype(series):
            out[col] = series.astype("float64")
            continue
        # object / string: nested → JSON, остальное → строка/скаляр для StringType
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            if _object_column_has_nested(series):
                nested_cols.append(str(col))
            out[col] = series.map(_to_spark_string_or_none)
            continue
        # неизвестный dtype → строка
        out[col] = series.map(_to_spark_string_or_none)

    if nested_cols:
        logger.warning(
            "Hive: колонки с nested/ndarray сериализованы в JSON-строку: %s",
            nested_cols,
        )
        print(f"[hive] nested->json columns ({len(nested_cols)}): {nested_cols[:20]}")
    if all_null_as_str:
        logger.warning(
            "Hive: полностью пустые колонки → StringType: %s",
            all_null_as_str,
        )
        print(
            f"[hive] all-null→string columns ({len(all_null_as_str)}): "
            f"{all_null_as_str[:30]}"
        )

    out.columns = [str(c) for c in out.columns]
    return out


def _spark_schema_for_pandas(df: pd.DataFrame) -> Any:
    """Явная схема: без inference NullType (CANNOT_DETERMINE_TYPE)."""
    from pyspark.sql.types import (
        BooleanType,
        DoubleType,
        LongType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    fields: list[Any] = []
    for col in df.columns:
        name = str(col)
        series = df[col]
        if series.isna().all():
            fields.append(StructField(name, StringType(), True))
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            fields.append(StructField(name, TimestampType(), True))
            continue
        if pd.api.types.is_bool_dtype(series):
            fields.append(StructField(name, BooleanType(), True))
            continue
        # после prepare bool → object с True/False/None
        sample = series.dropna().head(50)
        if len(sample) > 0 and all(isinstance(x, bool) for x in sample):
            fields.append(StructField(name, BooleanType(), True))
            continue
        if pd.api.types.is_integer_dtype(series) and not series.isna().any():
            fields.append(StructField(name, LongType(), True))
            continue
        if pd.api.types.is_float_dtype(series) or pd.api.types.is_integer_dtype(series):
            fields.append(StructField(name, DoubleType(), True))
            continue
        fields.append(StructField(name, StringType(), True))
    return StructType(fields)


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
        schema = _spark_schema_for_pandas(prepared)
        logger.info(
            "Hive WRITE start: table=%s mode=%s shape=%s",
            table_name,
            mode,
            prepared.shape,
        )
        sdf = spark.createDataFrame(prepared, schema=schema)
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
