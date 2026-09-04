"""Запись / чтение итогового датафрейма Querulus через Spark → Hive.

Стабильная таблица ``models.querulus_train_dataset`` + партиции
``model_version`` / ``data_date`` / ``dataset_version`` (см. ``querulus.naming``).

Пример::

    from querulus.dataset.hadoop import load_df_final, save_df_final
    from querulus.naming import DEFAULT_HIVE_TABLE, DEFAULT_PARQUET_PATH

    dest, hive_ok = save_df_final(
        df,
        parquet_path=DEFAULT_PARQUET_PATH,
        hive_table=DEFAULT_HIVE_TABLE,
    )
    df, source = load_df_final(
        hive_table=DEFAULT_HIVE_TABLE,
        parquet_path=DEFAULT_PARQUET_PATH,
    )
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from querulus.naming import (
    DEFAULT_APP_NAME,
    DEFAULT_HIVE_TABLE,
    DEFAULT_PARQUET_PATH,
    HIVE_PARTITION_COLUMNS,
    LEGACY_PARQUET_PATH,
    dataset_partition_values,
    resolve_dataset_partitions,
    write_latest_dataset_pointer,
)

logger = logging.getLogger("querulus.dataset.hadoop")


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

    spark = (
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
    logger.info("SparkSession ready: app=%s enableHiveSupport=True", app_name)
    print(f"[hive] SparkSession ready app={app_name} enableHiveSupport=True")
    return spark


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


def _datetime_to_iso_or_none(value: Any) -> str | None:
    """Datetime → ISO-строка без tz (обход Spark DST NonExistentTimeError)."""
    if value is None:
        return None
    try:
        if value is pd.NaT or pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.isoformat(sep=" ", timespec="seconds")


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
    datetime_as_str: list[str] = []
    for col in list(out.columns):
        series = out[col]
        # Все NULL → StringType + None (иначе Spark: CANNOT_DETERMINE_TYPE / NullType)
        if series.isna().all():
            all_null_as_str.append(str(col))
            out[col] = pd.Series([None] * len(out), index=out.index, dtype=object)
            continue
        # datetime64 → ISO string: Spark tz_localize падает на DST gaps (1984-04-01 и т.п.)
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_as_str.append(str(col))
            out[col] = pd.to_datetime(series, errors="coerce").map(_datetime_to_iso_or_none)
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
    if datetime_as_str:
        logger.warning(
            "Hive: datetime → ISO-строка (обход DST/tz Spark): %s",
            datetime_as_str,
        )
        print(
            f"[hive] datetime→string columns ({len(datetime_as_str)}): "
            f"{datetime_as_str[:30]}"
        )
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
    )

    fields: list[Any] = []
    for col in df.columns:
        name = str(col)
        series = df[col]
        if series.isna().all():
            fields.append(StructField(name, StringType(), True))
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
        # datetime уже в ISO-строках на этапе prepare; object/string → StringType
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
    partition_by: tuple[str, ...] | list[str] | None = HIVE_PARTITION_COLUMNS,
) -> str:
    """Pandas → Spark DataFrame → ``saveAsTable`` (Hive).

    При ``partition_by`` включает dynamic partition overwrite, чтобы
    ``overwrite`` затирал только затронутые партиции.

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
        parts = list(partition_by) if partition_by else []
        logger.info(
            "Hive WRITE start: table=%s mode=%s shape=%s partitions=%s",
            table_name,
            mode,
            prepared.shape,
            parts or None,
        )
        sdf = spark.createDataFrame(prepared, schema=schema)
        writer = sdf.write.mode(mode)
        if parts:
            spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
            missing = [c for c in parts if c not in prepared.columns]
            if missing:
                raise ValueError(f"Нет partition-колонок в df: {missing}")
            writer = writer.partitionBy(*parts)
        writer.saveAsTable(table_name)
        logger.info("Hive WRITE done: %s", table_name)
        print(
            f"[hive] wrote {table_name} shape={prepared.shape} "
            f"mode={mode} parts={parts or '-'}"
        )
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
    partition_filters: dict[str, str] | None = None,
    drop_partition_columns: bool = True,
) -> pd.DataFrame:
    """Hive-таблица → Pandas DataFrame."""
    own_spark = spark is None
    spark = spark or build_spark_session(app_name=app_name)
    try:
        logger.info(
            "Hive READ start: table=%s filters=%s",
            table_name,
            partition_filters or {},
        )
        sdf = spark.table(table_name)
        for col, value in (partition_filters or {}).items():
            sdf = sdf.filter(sdf[col] == value)
        pdf = sdf.toPandas()
        if drop_partition_columns:
            drop_cols = [c for c in HIVE_PARTITION_COLUMNS if c in pdf.columns]
            if drop_cols:
                pdf = pdf.drop(columns=drop_cols)
        logger.info("Hive READ done: %s shape=%s", table_name, pdf.shape)
        print(
            f"[hive] loaded {table_name} shape={pdf.shape} "
            f"filters={partition_filters or {}}"
        )
        return pdf
    finally:
        if own_spark and stop_spark:
            spark.stop()


def load_df_final(
    *,
    hive_table: str = DEFAULT_HIVE_TABLE,
    parquet_path: str | Path | None = None,
    prefer_hive: bool = True,
    spark: Any | None = None,
    stop_spark: bool = True,
    app_name: str = DEFAULT_APP_NAME,
    fallback_parquet_path: str | Path | None = None,
    generate_synthetic_if_missing: bool = False,
    synthetic_n_rows: int = 600,
    model_version: str | None = None,
    data_date: str | None = None,
    dataset_version: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Читает итоговый df: Hive (партиция) → при сбое локальный parquet."""
    path = Path(parquet_path) if parquet_path is not None else DEFAULT_PARQUET_PATH
    parts = resolve_dataset_partitions(
        model_version=model_version,
        data_date=data_date,
        dataset_version=dataset_version,
    )
    hive_error: str | None = None
    if prefer_hive:
        try:
            pdf = hive_table_to_pandas(
                hive_table,
                spark=spark,
                stop_spark=stop_spark,
                app_name=app_name,
                partition_filters=parts,
            )
            source = (
                f"hive:{hive_table}"
                f"|model_version={parts['model_version']}"
                f"|data_date={parts['data_date']}"
                f"|dataset_version={parts['dataset_version']}"
            )
            logger.info("dataset source=%s shape=%s", source, pdf.shape)
            print("=" * 72)
            print("ИСТОЧНИК ДАТАСЕТА: Hive (Hadoop)")
            print(f"  таблица: {hive_table}")
            print(f"  партиции: {parts}")
            print(f"  shape:   {pdf.shape}")
            print("=" * 72)
            return pdf, source
        except Exception as exc:  # noqa: BLE001
            hive_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Hive недоступен (%s) — fallback на parquet: %s",
                hive_error.splitlines()[0][:200],
                path,
            )

    candidates: list[Path] = [path]
    if fallback_parquet_path is not None:
        fb = Path(fallback_parquet_path)
        if fb not in candidates:
            candidates.append(fb)
    if LEGACY_PARQUET_PATH not in candidates:
        candidates.append(LEGACY_PARQUET_PATH)

    resolved: Path | None = next((c for c in candidates if c.is_file()), None)
    if resolved is None and generate_synthetic_if_missing:
        target = candidates[0]
        from querulus.synthetic_dataset import write_synthetic_final_dataset

        resolved = write_synthetic_final_dataset(target, n_rows=synthetic_n_rows)
        print(f"[dataset] синтетика сгенерирована: {resolved}")

    if resolved is None:
        tried = ", ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            f"Нет Hive и нет локального parquet (пробовали: {tried})."
        )
    if resolved != path:
        logger.warning("parquet fallback: %s → %s", path, resolved)
        print(f"[dataset] parquet fallback: {resolved}")

    pdf = pd.read_parquet(resolved)
    drop_cols = [c for c in HIVE_PARTITION_COLUMNS if c in pdf.columns]
    if drop_cols:
        pdf = pdf.drop(columns=drop_cols)
    source = f"parquet:{resolved}"
    logger.info("dataset source=%s shape=%s", source, pdf.shape)
    print("=" * 72)
    if hive_error is not None:
        print("ИСТОЧНИК ДАТАСЕТА: локальный parquet (Hive недоступен)")
        print(f"  причина: {hive_error.splitlines()[0][:240]}")
    elif prefer_hive:
        print("ИСТОЧНИК ДАТАСЕТА: локальный parquet")
    else:
        print("ИСТОЧНИК ДАТАСЕТА: локальный parquet (Hive не запрашивался)")
    print(f"  файл:  {resolved}")
    print(f"  shape: {pdf.shape}")
    print("=" * 72)
    return pdf, source


def save_df_final(
    df: pd.DataFrame,
    *,
    parquet_path: str | Path | None = None,
    hive_table: str = DEFAULT_HIVE_TABLE,
    prefer_hive: bool = True,
    mode: str = "overwrite",
    spark: Any | None = None,
    stop_spark: bool = True,
    app_name: str = DEFAULT_APP_NAME,
    model_version: str | None = None,
    data_date: str | None = None,
    dataset_version: str | None = None,
) -> tuple[str, bool]:
    """Всегда пишет локальный parquet; Hive — best-effort с партициями."""
    if df is None or len(df) == 0:
        raise ValueError("Пустой DataFrame — нечего сохранять")

    from querulus.dataset.dtypes import cast_object_columns
    from querulus.features.inflation import ensure_legacy_real_column_aliases

    parts = dataset_partition_values(
        model_version=model_version,
        data_date=data_date,
        dataset_version=dataset_version,
    )
    df = cast_object_columns(ensure_legacy_real_column_aliases(df))

    path = Path(parquet_path) if parquet_path is not None else DEFAULT_PARQUET_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Parquet WRITE done: %s shape=%s", path, df.shape)
    print(f"[dataset] wrote parquet {path} shape={df.shape}")

    write_latest_dataset_pointer(
        hive_table_name=hive_table,
        model_version=parts["model_version"],
        data_date=parts["data_date"],
        dataset_version=parts["dataset_version"],
        parquet_path=path,
    )

    if not prefer_hive:
        dest = f"parquet:{path}"
        print(f"[dataset] destination={dest} (Hive skipped)")
        return dest, False

    try:
        hive_df = df.copy()
        for col, value in parts.items():
            hive_df[col] = value
        pandas_to_hive_table(
            hive_df,
            hive_table,
            mode=mode,
            spark=spark,
            stop_spark=stop_spark,
            app_name=app_name,
            partition_by=HIVE_PARTITION_COLUMNS,
        )
        dest = (
            f"hive:{hive_table}"
            f"|model_version={parts['model_version']}"
            f"|data_date={parts['data_date']}"
            f"|dataset_version={parts['dataset_version']}"
        )
        print(f"[dataset] destination={dest} (parquet cache={path})")
        return dest, True
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "Hive WRITE failed (%s) — остаётся parquet: %s",
            msg.splitlines()[0][:200],
            path,
        )
        print(
            f"[dataset] Hive WRITE failed ({type(exc).__name__}) -> keep parquet\n"
            f"  detail: {msg.splitlines()[0][:240]}\n"
            f"[dataset] destination=parquet:{path}"
        )
        return f"parquet:{path}", False
