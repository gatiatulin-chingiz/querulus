"""Querulus monitoring DAG for contour Airflow.

Deploy:
  1) this file → Airflow dags folder (e.g. dags/BUU_projects/ or custom dags)
  2) ../pipeline → path available to workers (PIPELINE_DIR)

Env (secrets via Airflow Connections / Variables / pod env, not in code):
  SOURCE_MODE=prod
  PIPELINE_DIR=/path/to/monitoring/pipeline
  GRAFANA_SQLALCHEMY_URL or MSSQL_* (see pipeline/config.py)
  PYTHONPATH must include querulus package root (…/examples/querulus/src)
    for fin_effect (imports querulus.fin_effect)

Optional local/synthetic:
  SOURCE_MODE=synthetic
  SYNTHETIC_DATA_DIR=…
  GENERATE_SCRIPT=…/local/synthetic/generate.py
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PIPELINE = os.environ.get("PIPELINE_DIR", "/opt/querulus/monitoring/pipeline")
SOURCE_MODE = os.environ.get("SOURCE_MODE", "prod").strip().lower()
SYNTH_DIR = os.environ.get("SYNTHETIC_DATA_DIR", "")
GENERATE = os.environ.get("GENERATE_SCRIPT", "")

default_args = {
    "owner": "querulus",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="querulus_monitoring",
    default_args=default_args,
    description="Querulus errors + datadrift + fin_effect → MSSQL Motor.agrafana",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["querulus", "monitoring", "BUU_projects"],
) as dag:
    run_errors = BashOperator(
        task_id="errors",
        bash_command=f"python {PIPELINE}/errors.py",
    )
    run_datadrift = BashOperator(
        task_id="datadrift",
        bash_command=f"python {PIPELINE}/datadrift.py",
    )
    run_fin_effect = BashOperator(
        task_id="fin_effect",
        bash_command=f"python {PIPELINE}/fin_effect.py",
    )

    if SOURCE_MODE == "synthetic" and GENERATE and SYNTH_DIR:
        generate_synthetic = BashOperator(
            task_id="generate_synthetic",
            bash_command=f"python {GENERATE} --out {SYNTH_DIR}",
        )
        generate_synthetic >> [run_errors, run_datadrift, run_fin_effect]
    else:
        [run_errors, run_datadrift, run_fin_effect]
