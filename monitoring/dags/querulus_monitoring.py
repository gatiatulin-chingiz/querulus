"""Querulus monitoring DAG — контурный стиль как BUU_projects / MONITORING_BUU.

На контуре в Airflow dags лежит только этот файл.
Код пайплайна: каждый pod сам делает git clone в ``/tmp`` (как MONITORING_BUU).
PVC ``airflow-dag-venv`` не используем — его нет на кластере.

GitLab path: Variable ``querulus_git_path``
  default ``/mlops/losses_projects/monitoring/querulus_monitoring.git``

Локально (docker, без K8s): ``QUERULUS_DAG_MODE=bash`` + mount PIPELINE_DIR.
Synthetic: ``QUERULUS_USE_SYNTHETIC=1`` → local/synthetic/run_*.py

Схема K8s-рана (каждая таска = отдельный pod, свой clone в /tmp):

    prepare  →  errors
             →  datadrift   →  is_monday_alert  →  alert_email
             →  fin_effect

  prepare          — clone + venv + test (ранний fail без записи в Motor)
  errors/…         — снова clone + venv + make <этап> (диск между pod не шарится)
  is_monday_alert  — ShortCircuit на scheduler (без pod)
  alert_email      — clone + venv + make alert_email
"""
from __future__ import annotations

import datetime
import os

from airflow import DAG
from airflow.hooks.base import BaseHook
from airflow.models import Variable
from airflow.operators.python import ShortCircuitOperator

tags = ["BUU_projects", "querulus", "monitoring"]
task_name = "querulus_monitoring"
# True → после prepare pod спит 1ч (можно зайти kubectl exec и посмотреть файлы)
DEBUG = False
# True → alert_email в любой день (без Airflow Variable / env на scheduler).
# После проверки верни False — иначе письмо будет каждый daily run.
FORCE_ALERT = True

# Contour GitLab: clone → каталог querulus_monitoring/ (= корень pipeline)
git_path = Variable.get(
    "querulus_git_path",
    default_var="/mlops/losses_projects/monitoring/querulus_monitoring.git",
)
git_folder = "querulus_monitoring"

# Как MONITORING_BUU: workdir в /tmp внутри pod (не PVC). Между тасками не шарится.
_work_root = f"/tmp/schedule/{task_name}"
_work_dir = f"{_work_root}/{git_folder}"

dag_schedule_interval = "0 9 * * *"  # ежедневно 09:00; письмо — только в понедельник

default_args = {
    "owner": "querulus",
    "start_date": datetime.datetime(2026, 1, 1, 9, 0, 0),
    "retries": 0,  # как MONITORING_BUU: упала — не гоняем второй раз
    "provide_context": True,
}

# k8s (контур) | bash (local SequentialExecutor)
DAG_MODE = os.environ.get("QUERULUS_DAG_MODE", "k8s").strip().lower()


def _truthy(raw: str) -> bool:
    return (raw or "").strip().lower() in {"1", "true", "yes"}


def _is_monday_alert(**context) -> bool:
    """ShortCircuit: письмо только в ALERT_WEEKDAY (0=пн) или force.

    Force (любой weekday), в порядке приоритета:
      1) константа ``FORCE_ALERT`` в этом файле (без доступа к Variables)
      2) env ``ALERT_FORCE`` на scheduler/worker

    Если False — Airflow НЕ запустит downstream (alert_email).
    Это не pod: выполняется на scheduler/worker Airflow.
    """
    if FORCE_ALERT or _truthy(os.environ.get("ALERT_FORCE", "")):
        return True
    weekday = int(os.environ.get("ALERT_WEEKDAY", "0") or "0")
    when = context.get("data_interval_end") or context["logical_date"]
    return int(when.weekday()) == weekday


def _build_bash_dag():
    """Local: скрипты уже на диске (volume), без clone и без PVC."""
    from airflow.operators.bash import BashOperator

    pipeline = os.environ.get("PIPELINE_DIR", "/opt/querulus/monitoring/pipeline")
    local_synth = os.environ.get(
        "LOCAL_SYNTHETIC_DIR", "/opt/querulus/monitoring/local/synthetic"
    )
    use_synth = os.environ.get("QUERULUS_USE_SYNTHETIC", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    synth_dir = os.environ.get("SYNTHETIC_DATA_DIR", f"{local_synth}/data")
    generate = os.environ.get("GENERATE_SCRIPT", f"{local_synth}/generate.py")

    if use_synth:
        # Локальная отладка без контурных БД: synthetic wrappers
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
            # Сначала генерим фиктивные данные, потом считаем метрики
            generate_synthetic = BashOperator(
                task_id="generate_synthetic",
                bash_command=f"python {generate} --out {synth_dir}",
            )
            generate_synthetic >> metrics
        # Три метрики параллельно → гейт понедельника → письмо
        metrics >> is_monday_alert >> alert_email
    return dag


def _build_k8s_dag():
    """Контур: несколько KubernetesPodOperator = несколько Airflow-тасок.

    Как MONITORING_BUU: clone+venv в /tmp внутри pod.
    PVC airflow-dag-venv нет на кластере — не монтируем.
    Каждый этапный pod заново клонирует репо (диск между pod не общий).
    """
    from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
    from kubernetes.client import models as k8s

    # Volumes 1:1 как MONITORING_BUU (иначе pod висит Pending на отсутствующих CM)
    volume_mount = [
        k8s.V1VolumeMount(
            mount_path="/opt/airflow/dags", name="dags", sub_path=None, read_only=True
        ),
        k8s.V1VolumeMount(
            mount_path="/etc/krb5.conf",
            name="dag-krb5",
            sub_path="krb5.conf",
            read_only=True,
        ),
        k8s.V1VolumeMount(
            mount_path="/usr/local/apache-hive-4.0.1-bin/conf/hive-site.xml",
            name="adh-configs",
            sub_path="hive.hive-site.xml",
            read_only=True,
        ),
        k8s.V1VolumeMount(
            mount_path="/usr/local/apache-hive-4.0.1-bin/conf/beeline-hs2-connection.xml",
            name="adh-configs",
            sub_path="hive.beeline-hs2-connection.xml",
            read_only=True,
        ),
        k8s.V1VolumeMount(
            mount_path="/usr/local/hadoop-3.3.6/etc/hadoop/hive-site.xml",
            name="adh-configs",
            sub_path="hive.hive-site.xml",
            read_only=True,
        ),
        k8s.V1VolumeMount(
            mount_path="/usr/local/hadoop-3.3.6/etc/hadoop/core-site.xml",
            name="adh-configs",
            sub_path="core.core-site.xml",
            read_only=True,
        ),
        k8s.V1VolumeMount(
            mount_path="/usr/local/hadoop-3.3.6/etc/hadoop/hdfs-site.xml",
            name="adh-configs",
            sub_path="hdfs.hdfs-site.xml",
            read_only=True,
        ),
        k8s.V1VolumeMount(
            mount_path="/opt/adh-etl-user.keytab",
            name="adh-etl-user-airflow-keytab",
            sub_path="adh-etl-user.airflow.keytab",
            read_only=True,
        ),
    ]

    volume = [
        k8s.V1Volume(
            name="dags",
            persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
                claim_name="airflow-custom-dags"
            ),
        ),
        k8s.V1Volume(
            name="dag-krb5",
            config_map=k8s.V1ConfigMapVolumeSource(name="krb-conf"),
        ),
        k8s.V1Volume(
            name="adh-configs",
            config_map=k8s.V1ConfigMapVolumeSource(name="adh-configs"),
        ),
        k8s.V1Volume(
            name="adh-etl-user-airflow-keytab",
            secret=k8s.V1SecretVolumeSource(secret_name="adh-etl-user.airflow.keytab"),
        ),
    ]

    # Как MONITORING_BUU: образ из Variable k8s_base_b
    k8s_base = Variable.get(key="k8s_base_b", deserialize_json=True)
    compute_resources = k8s.V1ResourceRequirements(
        requests=k8s_base["resources"]["requests"],
        limits=k8s_base["resources"]["limits"],
    )

    # Токен GitLab → URL clone (логин/пароль Connection не логируем)
    gl_token = BaseHook.get_connection("gitlab_token")
    git_url = "https://{login}:{token}@gitlab.vsk.ru{git_path}".format(
        login=gl_token.login,
        token=gl_token.password,
        git_path=git_path,
    )

    # Секреты БД / почты → env pod'а (скрипты читают MONITORING_BUU-имена)
    rtdm = BaseHook.get_connection("sandbox_rtdm")
    oisuu = BaseHook.get_connection("OISUU-DB-BIZ_monitoring")
    motor = BaseHook.get_connection("ACTUAR2_monitoring")
    email = BaseHook.get_connection("DCO_Email_Notification")

    user_env = {
        # Motor.agrafana — куда пишем метрики
        "motor_host": motor.host or "",
        "motor_name": motor.schema or "Motor",
        "motor_user_name": motor.login or "",
        "motor_user_pass": motor.password or "",
        # OISUU — витрина / MatemModelsLogs (errors, fin_effect)
        "database_host": oisuu.host or "",
        "database_port": str(oisuu.port) if oisuu.port else "1433",
        "database_name": oisuu.schema or "OISUU_report",
        "database_user_name": oisuu.login or "",
        "database_user_pass": oisuu.password or "",
        # Oracle RTDM (PROD8) — prod-фичи для datadrift
        "SERVER_PROD8": rtdm.host or "",
        "PORT_PROD8": str(rtdm.port) if rtdm.port else "1521",
        "USER_PROD8": rtdm.login or "",
        "PASSWORD_PROD8": rtdm.password or "",
        "SERVICE_NAME_PROD8": rtdm.schema or "rtdmdb08",
        # SMTP для alert_email
        "EMAIL_SMTP_SERVER": "internalsmtp.vsk.ru",
        "EMAIL_PORT": "2525",
        "EMAIL_SENDER": "artificial.intelligence@vsk.ru",
        "EMAIL_LOGIN": email.login or "",
        "EMAIL_PASSWORD": email.password or "",
    }
    # Опционально: файл-override ретро (по умолчанию Hive train main = datadrift)
    retro_path = Variable.get("querulus_fin_effect_retro_path", default_var="")
    if retro_path:
        user_env["FIN_EFFECT_RETRO_PATH"] = retro_path

    # Пути Hadoop/Hive внутри образа контура (нужны datadrift → Hive)
    system_env = {
        "HIVE_CONF_DIR": "/etc/hive/conf",
        "SPARK_HOME": "/usr/local/spark",
        "HADOOP_HOME": "/usr/local/hadoop",
        "HADOOP_HDFS_HOME": "/usr/local/hadoop",
        "HADOOP_CONF_DIR": "/usr/local/hadoop/etc/hadoop",
        "HADOOP_VERSION": "3",
        "HADOOP_OPTS": "-Djava.library.path=/usr/local/hadoop/lib/native",
    }
    env_vars = {**system_env, **user_env}

    # --- bash: как MONITORING_BUU — rm/mkdir/clone/venv в /tmp, потом make -----
    def _bootstrap_cmd(*, make_targets: str, label: str) -> str:
        """Полный bootstrap в /tmp + make-цели (через пробел). Не трогает БД сам по себе."""
        return f"""
set -e
echo "=== [{label}] kinit ==="
kinit -F -k -t /opt/adh-etl-user.keytab adh-etl-user

echo "=== [{label}] чистим /tmp workdir (только FS pod, не БД) ==="
rm -rf {_work_root}
mkdir -p {_work_root}
cd {_work_root}

export GIT_SSL_NO_VERIFY=true
echo "=== [{label}] git clone ==="
git clone {git_url}
cd {git_folder}
ls -la

echo "=== [{label}] make venv ==="
make venv

echo "=== [{label}] make {make_targets} ==="
make {make_targets}

echo "=== [{label}] DONE ==="
"""

    prepare_cmd = _bootstrap_cmd(make_targets="test", label="prepare")
    if DEBUG:
        prepare_cmd += "\necho 'DEBUG sleep 3600'; sleep 3600\n"

    def _k8s_task(task_id: str, bash: str) -> KubernetesPodOperator:
        """Один Airflow-столбик = один pod."""
        return KubernetesPodOperator(
            task_id=task_id,
            name=f"{task_name}-{task_id}",
            container_resources=compute_resources,
            image=k8s_base["image"],
            volumes=volume,
            volume_mounts=volume_mount,
            env_vars=env_vars,
            cmds=["bash", "-cx", bash],
            # Удаляем только успешные: при fail pod остаётся → kubectl logs
            # (UI часто отдаёт пустой http://:8793 — hostname log-server не задан)
            on_finish_action="delete_succeeded_pod",
            # Clear/retry не должен цепляться к старому Failed-pod (иначе нет [base] логов)
            reattach_on_restart=False,
            in_cluster=True,
            do_xcom_push=False,
            get_logs=True,
            # Pull образа / mount иногда > default 120s
            startup_timeout_seconds=600,
        )

    with DAG(
        dag_id=task_name,
        default_args=default_args,
        description=(
            "Querulus monitoring: prepare → errors → datadrift → fin_effect "
            "→ (Mon) alert_email"
        ),
        schedule=dag_schedule_interval,
        catchup=False,
        max_active_runs=1,
        tags=tags,
    ) as dag:
        prepare = _k8s_task("prepare", prepare_cmd)
        errors = _k8s_task(
            "errors", _bootstrap_cmd(make_targets="errors", label="errors")
        )
        datadrift = _k8s_task(
            "datadrift", _bootstrap_cmd(make_targets="datadrift", label="datadrift")
        )
        fin_effect = _k8s_task(
            "fin_effect",
            _bootstrap_cmd(make_targets="fin_effect", label="fin_effect"),
        )
        is_monday_alert = ShortCircuitOperator(
            task_id="is_monday_alert",
            python_callable=_is_monday_alert,
        )
        alert_email = _k8s_task(
            "alert_email",
            _bootstrap_cmd(make_targets="alert_email", label="alert_email"),
        )

        prepare >> [errors, datadrift, fin_effect] >> is_monday_alert >> alert_email
    return dag


if DAG_MODE == "bash":
    dag = _build_bash_dag()
else:
    dag = _build_k8s_dag()
