"""Querulus monitoring DAG for contour Airflow.

Deploy:
  1) this file → Airflow dags folder (e.g. dags/BUU_projects/ or custom dags)
  2) ../pipeline → path available to workers (PIPELINE_DIR)

Schedule: every day 10:00. Metrics always; email alert only Monday 10:00
(after metrics), ShortCircuit on weekday.

Env (secrets via Airflow Connections / Variables / pod env, not in code):
  SOURCE_MODE=prod
  PIPELINE_DIR=/path/to/monitoring/pipeline
  GRAFANA_SQLALCHEMY_URL or MSSQL_* (see pipeline/config.py)
  PYTHONPATH must include querulus package root (…/examples/querulus/src)
    for fin_effect (imports querulus.fin_effect)
  Email (Monday only): EMAIL_LOGIN, EMAIL_PASSWORD, EMAIL_RECEIVERS,
    EMAIL_SMTP_SERVER, EMAIL_PORT, EMAIL_SENDER, GRAFANA_DASHBOARD_URL
  ALERT_WEEKDAY=0 (Mon), ALERT_FORCE=1 to always run alert task,
  ALERT_DRY_RUN=1 to skip SMTP

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
from airflow.operators.python import ShortCircuitOperator

PIPELINE = os.environ.get("PIPELINE_DIR", "/opt/querulus/monitoring/pipeline")
SOURCE_MODE = os.environ.get("SOURCE_MODE", "prod").strip().lower()
SYNTH_DIR = os.environ.get("SYNTHETIC_DATA_DIR", "")
GENERATE = os.environ.get("GENERATE_SCRIPT", "")


def _is_monday_alert(**context) -> bool:
    """True on Monday (ALERT_WEEKDAY, default 0) or when ALERT_FORCE is set."""
    force = os.environ.get("ALERT_FORCE", "").strip().lower() in {"1", "true", "yes"}
    if force:
        return True
    weekday = int(os.environ.get("ALERT_WEEKDAY", "0") or "0")
    when = context.get("data_interval_end") or context["logical_date"]
    return int(when.weekday()) == weekday


default_args = {
    "owner": "querulus",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="querulus_monitoring",
    default_args=default_args,
    description="Querulus errors + datadrift + fin_effect → MSSQL; weekly email Mon 10:00",
    schedule="0 10 * * *",
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

    is_monday_alert = ShortCircuitOperator(
        task_id="is_monday_alert",
        python_callable=_is_monday_alert,
    )
    alert_email = BashOperator(
        task_id="alert_email",
        bash_command=f"python {PIPELINE}/alert_email.py",
    )

    metrics = [run_errors, run_datadrift, run_fin_effect]
    if SOURCE_MODE == "synthetic" and GENERATE and SYNTH_DIR:
        generate_synthetic = BashOperator(
            task_id="generate_synthetic",
            bash_command=f"python {GENERATE} --out {SYNTH_DIR}",
        )
        generate_synthetic >> metrics
    metrics >> is_monday_alert >> alert_email
