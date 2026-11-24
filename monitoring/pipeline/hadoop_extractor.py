"""Чтение reference-датасета из Hive/Hadoop (как BUU monitoring_buu-master).

Контурный образ Airflow/Spark: Hive metastore + hdfs dfs -get.
Паттерн: spark.table / spark.sql → parquet на HDFS → hdfs get → pd.read_parquet.

``pyspark`` в venv обычно нет: берём модули из ``$SPARK_HOME/python``
(в DAG: ``SPARK_HOME=/usr/local/spark``).
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from typing import List, Optional

import pandas as pd


def _ensure_pyspark() -> None:
    """Импорт pyspark из venv или из SPARK_HOME (контурный образ)."""
    try:
        import pyspark  # noqa: F401

        return
    except ImportError:
        pass

    spark_home = os.environ.get("SPARK_HOME", "/usr/local/spark")
    py_dir = os.path.join(spark_home, "python")
    lib_dir = os.path.join(py_dir, "lib")
    for path in (py_dir, lib_dir):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
    for z in sorted(glob.glob(os.path.join(lib_dir, "py4j-*.zip"))):
        if z not in sys.path:
            sys.path.insert(0, z)

    try:
        import pyspark  # noqa: F401
    except ImportError as exc:
        raise ModuleNotFoundError(
            "pyspark not in venv and not found under "
            f"SPARK_HOME={spark_home!r} (expected {py_dir}). "
            "Set SPARK_HOME or pip install pyspark."
        ) from exc


_ensure_pyspark()
from pyspark.sql import SparkSession  # noqa: E402


class HadoopExtractor:
    """Выгрузка таблицы/запроса из Hive в pandas.DataFrame."""

    def __init__(
        self,
        source: Optional[str] = None,
        sql_query: Optional[str] = None,
        columns: Optional[List[str]] = None,
        to_df: str = "hive_reference.parquet",
        app_name: str = "querulus_monitoring",
        metastore_jars: str = "path:/usr/local/apache-hive-4.0.1-bin/lib/*",
        metastore_version: str = "4.0.1",
        java_home: str = "/usr/lib/jvm/java-17-openjdk-amd64",
    ):
        self.source = source
        self.sql_query = sql_query
        self.columns = columns
        self.to_df = to_df
        self.app_name = app_name
        self.metastore_jars = metastore_jars
        self.metastore_version = metastore_version
        self.java_home = java_home

        if source is None and sql_query is None:
            raise ValueError("Either 'sql_query' or 'source' must be provided.")

    def _drop_local(self, path: str) -> None:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    def spark_to_pandas_parquet(self) -> pd.DataFrame:
        if "JAVA_HOME" not in os.environ:
            os.environ["JAVA_HOME"] = self.java_home
        os.environ["PATH"] = f"{os.environ['JAVA_HOME']}/bin:{os.environ['PATH']}"

        spark = (
            SparkSession.builder.appName(self.app_name)
            .enableHiveSupport()
            .config("spark.executor.memory", "8g")
            .config("spark.executor.cores", 1)
            .config("spark.driver.core", "1")
            .config("spark.driver.memory", "8g")
            .config("spark.executor.instances", "4")
            .config("spark.dynamicAllocation.enabled", "false")
            .config("spark.hadoop.hive.exec.dynamic.partition", "true")
            .config(
                "spark.hadoop.hive.exec.dynamic.partition.mode=nonstrict",
                "nonstrict",
            )
            .config("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED")
            .config("spark.sql.ansi.enabled", "false")
            .config("spark.sql.legacy.allowNonEmptyLocationInCTAS", "true")
            .config("spark.sql.hive.metastore.jars", self.metastore_jars)
            .config("spark.sql.hive.metastore.version", self.metastore_version)
            .getOrCreate()
        )

        try:
            if self.sql_query:
                df = spark.sql(self.sql_query)
            else:
                df = spark.table(self.source)

            if self.columns:
                available = [f.name for f in df.schema]
                cols = [c for c in self.columns if c in available]
                if cols:
                    df = df.select(*cols)

            self._drop_local(self.to_df)
            hdfs_path = f"/tmp/{self.to_df}"
            df.write.mode("overwrite").parquet(hdfs_path)
            subprocess.run(f"hdfs dfs -get {hdfs_path}", shell=True, check=True)
            return pd.read_parquet(self.to_df)
        finally:
            spark.stop()
            self._drop_local(self.to_df)

    def extract_dataset(self) -> pd.DataFrame:
        return self.spark_to_pandas_parquet()
