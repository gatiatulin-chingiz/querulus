"""Querulus monitoring DAG — контурный стиль как BUU_projects / MONITORING_BUU.

На контуре в Airflow dags лежит только этот файл.
Код пайплайна каждый run: git clone → make venv/test/automl в pod.

GitLab path: Variable ``querulus_git_path``
  default ``/mlops/losses_projects/monitoring/querulus_monitoring.git``

Локально (docker, без K8s): ``QUERULUS_DAG_MODE=bash`` + mount PIPELINE_DIR.
Synthetic: ``QUERULUS_USE_SYNTHETIC=1`` → local/synthetic/run_*.py
"""
from __future__ import annotations

import datetime
import os

from airflow import DAG
from airflow.hooks.base import BaseHook
from airflow.models import Variable

tags = ["BUU_projects", "querulus", "monitoring"]
task_name = "querulus_monitoring"
DEBUG = False  # True → sleep 3600 в конце pod (отладка)

# Contour GitLab: clone → каталог querulus_monitoring/ (= корень pipeline)
git_path = Variable.get(
    "querulus_git_path",
    default_var="/mlops/losses_projects/monitoring/querulus_monitoring.git",
)
git_folder = "querulus_monitoring"
dag_schedule_interval = "0 9 * * *"  # ежедневно 09:00; письмо — только в понедельник

default_args = {
    "owner": "querulus",
    "start_date": datetime.datetime(2026, 1, 1, 9, 0, 0),
    "retries": 1,
    "provide_context": True,
}

# k8s (контур) | bash (local SequentialExecutor)
DAG_MODE = os.environ.get("QUERULUS_DAG_MODE", "k8s").strip().lower()


def _build_bash_dag():
    """Local: скрипты уже на диске (volume), без clone."""
    from airflow.operators.bash import BashOperator
    from airflow.operators.python import ShortCircuitOperator

    pipeline = os.environ.get("PIPELINE_DIR", "/opt/querulus/monitoring/pipeline")
    local_synth = os.environ.get(
        "LOCAL_SYNTHETIC_DIR", "/opt/querulus/monitoring/local/synthetic"
    )
    # Synthetic не в pipeline/: отдельные run_*.py под local docker
    use_synth = os.environ.get("QUERULUS_USE_SYNTHETIC", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    synth_dir = os.environ.get("SYNTHETIC_DATA_DIR", f"{local_synth}/data")
    generate = os.environ.get("GENERATE_SCRIPT", f"{local_synth}/generate.py")

    def _is_monday_alert(**context) -> bool:
        """ShortCircuit: письмо только в ALERT_WEEKDAY (0=пн) или ALERT_FORCE."""
        force = os.environ.get("ALERT_FORCE", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if force:
            return True
        weekday = int(os.environ.get("ALERT_WEEKDAY", "0") or "0")
        when = context.get("data_interval_end") or context["logical_date"]
        return int(when.weekday()) == weekday

    if use_synth:
        errors_cmd = f"python {local_synth}/run_errors.py"
        drift_cmd = f"python {local_synth}/run_datadrift.py"
        fin_cmd = f"python {local_synth}/run_fin_effect.py"
    else:
        errors_cmd = f"python {pipeline}/errors.py"
        drift_cmd = f"python {pipeline}/datadrift.py"
        fin_cmd = f"python {pipeline}/fin_effect.py"

    with DAG(
        dag_id=task_name,
        default_args=default_args,
        description="Querulus monitoring (local bash mode)",
        schedule=dag_schedule_interval,
        catchup=False,
        max_active_runs=1,
        tags=tags + ["bash"],
    ) as dag:
        run_errors = BashOperator(task_id="errors", bash_command=errors_cmd)
        run_datadrift = BashOperator(task_id="datadrift", bash_command=drift_cmd)
        run_fin_effect = BashOperator(task_id="fin_effect", bash_command=fin_cmd)
        is_monday_alert = ShortCircuitOperator(
            task_id="is_monday_alert",
            python_callable=_is_monday_alert,
        )
        alert_email = BashOperator(
            task_id="alert_email",
            bash_command=f"python {pipeline}/alert_email.py",
        )
        metrics = [run_errors, run_datadrift, run_fin_effect]
        if use_synth and generate and synth_dir:
            generate_synthetic = BashOperator(
                task_id="generate_synthetic",
                bash_command=f"python {generate} --out {synth_dir}",
            )
            generate_synthetic >> metrics
        metrics >> is_monday_alert >> alert_email
    return dag


def _build_k8s_dag():
    """Контур: KubernetesPodOperator — clone GitLab + make automl.

    Секреты БД не в git: BaseHook.get_connection → env_vars пода.
    Имена Connections как у MONITORING_BUU (уже заведены на инстансе).
    """
    from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
    from kubernetes.client import models as k8s

    # Volumes: dags PVC, рабочий /data_jcs, kerberos/hive/hadoop конфиги ADH
    volume_mount = [
        k8s.V1VolumeMount(
            mount_path="/opt/airflow/dags", name="dags", sub_path=None, read_only=True
        ),
        k8s.V1VolumeMount(
            mount_path="/data_jcs", name="dag-venv", sub_path=None, read_only=False
        ),
        k8s.V1VolumeMount(
            mount_path="/opt/adh-etl-user.keytab",
            name="dag-keytab",
            sub_path="adh-etl-user.airflow.keytab",
            read_only=True,
        ),
        k8s.V1VolumeMount(
            mount_path="/etc/krb5.conf",
            name="dag-krb5",
            sub_path="krb5.conf",
            read_only=True,
        ),
        k8s.V1VolumeMount(
            mount_path="/usr/local/spark/conf/hive-site.xml",
            name="hive-site-volume",
            sub_path="hive-site.xml",
            read_only=True,
        ),
        k8s.V1VolumeMount(
            mount_path="/usr/local/hadoop/etc/hadoop/hive-site.xml",
            name="hive-site-volume",
            sub_path="hive-site.xml",
            read_only=True,
        ),
        k8s.V1VolumeMount(
            mount_path="/usr/local/hadoop/etc/hadoop/core-site.xml",
            name="core-site-volume",
            sub_path="core-site.xml",
            read_only=True,
        ),
        k8s.V1VolumeMount(
            mount_path="/usr/local/hadoop/etc/hadoop/hdfs-site.xml",
            name="hdfs-site-volume",
            sub_path="hdfs-site.xml",
            read_only=True,
        ),
        k8s.V1VolumeMount(
            mount_path="/etc/hive/conf/beeline-hs2-connection.xml",
            name="beeline-hs2-connection-volume",
            sub_path="beeline-hs2-connection.xml",
            read_only=True,
        ),
        k8s.V1VolumeMount(
            mount_path="/etc/pki/ca-trust/extracted/java/truststore.jks",
            name="truststore-adh-volume",
            sub_path="truststore.jks",
            read_only=True,
        ),
        k8s.V1VolumeMount(
            mount_path="/spark/truststore.jks",
            name="truststore-adh-volume",
            sub_path="truststore.jks",
            read_only=True,
        ),
    ]

    volume = [
        k8s.V1Volume(
            name="dags",
            persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
                claim_name="airflow-dags"
            ),
        ),
        k8s.V1Volume(
            name="dag-venv",
            persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
                claim_name="airflow-dag-venv"
            ),
        ),
        k8s.V1Volume(
            name="dag-keytab",
            config_map=k8s.V1ConfigMapVolumeSource(name="airflow-adh-keytab"),
        ),
        k8s.V1Volume(
            name="dag-krb5",
            config_map=k8s.V1ConfigMapVolumeSource(name="krb-conf"),
        ),
        k8s.V1Volume(
            name="hive-site-volume",
            config_map=k8s.V1ConfigMapVolumeSource(name="hive-site"),
        ),
        k8s.V1Volume(
            name="core-site-volume",
            config_map=k8s.V1ConfigMapVolumeSource(name="core-site"),
        ),
        k8s.V1Volume(
            name="hdfs-site-volume",
            config_map=k8s.V1ConfigMapVolumeSource(name="hdfs-site"),
        ),
        k8s.V1Volume(
            name="beeline-hs2-connection-volume",
            config_map=k8s.V1ConfigMapVolumeSource(name="beeline-hs2-connection"),
        ),
        k8s.V1Volume(
            name="truststore-adh-volume",
            config_map=k8s.V1ConfigMapVolumeSource(name="truststore-adh"),
        ),
    ]

    k8s_base = Variable.get(key="k8s_base", deserialize_json=True)
    compute_resources = k8s.V1ResourceRequirements(
        requests=k8s_base["resources"]["requests"],
        limits=k8s_base["resources"]["limits"],
    )

    gl_token = BaseHook.get_connection("gitlab_token")
    git_url = "https://{login}:{token}@gitlab.vsk.ru{git_path}".format(
        login=gl_token.login,
        token=gl_token.password,
        git_path=git_path,
    )

    # Connections → env (pipeline/config.py читает те же ключи, что BUU)
    rtdm = BaseHook.get_connection("sandbox_rtdm")  # Oracle RTDM / spr_uu_json_log
    oisuu = BaseHook.get_connection("OISUU-DB-BIZ_monitoring")  # MatemModelsLogs + витрина
    motor = BaseHook.get_connection("ACTUAR2_monitoring")  # Motor.agrafana write
    email = BaseHook.get_connection("DCO_Email_Notification")

    user_env = {
        # Motor / Actuar — запись метрик под Grafana
        "MSSQL_HOST": motor.host or "",
        "MSSQL_PORT": str(motor.port) if motor.port else "1433",
        "MSSQL_DATABASE": motor.schema or "Motor",
        "MSSQL_USER": motor.login or "",
        "MSSQL_SA_PASSWORD": motor.password or "",
        "MSSQL_ODBC_DRIVER": "ODBC Driver 18 for SQL Server",
        "motor_host": motor.host or "",
        "motor_name": motor.schema or "Motor",
        "motor_user_name": motor.login or "",
        "motor_user_pass": motor.password or "",
        # OISUU — errors + fin_effect monitoring
        "database_host": oisuu.host or "",
        "database_port": str(oisuu.port) if oisuu.port else "1433",
        "database_name": oisuu.schema or "OISUU_report",
        "database_user_name": oisuu.login or "",
        "database_user_pass": oisuu.password or "",
        # Oracle RTDM — DataDrift prod-логи
        "SERVER_PROD8": rtdm.host or "",
        "PORT_PROD8": str(rtdm.port) if rtdm.port else "1521",
        "USER_PROD8": rtdm.login or "",
        "PASSWORD_PROD8": rtdm.password or "",
        "SERVICE_NAME_PROD8": rtdm.schema or "rtdmdb08",
        # Email: логин/пароль из Connection; получатели — DEFAULT_EMAIL_RECEIVERS в config
        "EMAIL_SMTP_SERVER": "internalsmtp.vsk.ru",
        "EMAIL_PORT": "2525",
        "EMAIL_SENDER": "artificial.intelligence@vsk.ru",
        "EMAIL_LOGIN": email.login or "",
        "EMAIL_PASSWORD": email.password or "",
    }
    # Hive table / окна train — в pipeline/config.py (DEFAULT_REFERENCE_HIVE_TABLE).
    # FIN_EFFECT_RETRO_PATH — если нужен файл на PVC, задать Variable (опционально).
    retro_path = Variable.get("querulus_fin_effect_retro_path", default_var="")
    if retro_path:
        user_env["FIN_EFFECT_RETRO_PATH"] = retro_path

    # bash в pod: kinit (Kerberos для HDFS/Hive) → clone → make
    bcom = """
set -e
echo "=== kinit ===" &&
kinit -F -k -t /opt/adh-etl-user.keytab adh-etl-user &&
echo "=== prepare dir ===" &&
rm -rf /data_jcs/schedule/{task_name} &&
mkdir -p /data_jcs/schedule/{task_name} &&
cd /data_jcs/schedule/{task_name} &&
export GIT_SSL_NO_VERIFY=true &&
echo "=== git clone ===" &&
git clone {git_url} &&
cd {git_folder} &&
ls -la &&
echo "=== make venv ===" &&
make venv &&
echo "=== make test ===" &&
make test &&
echo "=== make automl ===" &&
make automl &&
echo "=== make clean ===" &&
make clean &&
echo "=== DONE ==="
""".format(
        task_name=task_name,
        git_url=git_url,
        git_folder=git_folder,
    )
    if DEBUG:
        bcom += " && sleep 3600"

    # Системные пути ADH/Hadoop внутри образа (как у BUU)
    system_env = {
        "HIVE_CONF_DIR": "/etc/hive/conf",
        "SPARK_HOME": "/usr/local/spark",
        "HADOOP_HOME": "/usr/local/hadoop",
        "HADOOP_HDFS_HOME": "/usr/local/hadoop",
        "HADOOP_CONF_DIR": "/usr/local/hadoop/etc/hadoop",
        "HADOOP_VERSION": "3",
        "HADOOP_OPTS": "-Djava.library.path=/usr/local/hadoop/lib/native",
    }

    with DAG(
        task_name,
        default_args=default_args,
        schedule=dag_schedule_interval,
        catchup=False,
        max_active_runs=1,
        tags=tags,
    ) as dag:
        KubernetesPodOperator(
            name=task_name,
            container_resources=compute_resources,
            image=k8s_base["image"],
            volumes=volume,
            volume_mounts=volume_mount,
            env_vars={**system_env, **user_env},
            cmds=["bash", "-cx", bcom],
            task_id="querulus_monitoring",
            on_finish_action="delete_pod",
            in_cluster=True,
            do_xcom_push=True,
            get_logs=True,
        )
    return dag


if DAG_MODE == "bash":
    dag = _build_bash_dag()
else:
    dag = _build_k8s_dag()
