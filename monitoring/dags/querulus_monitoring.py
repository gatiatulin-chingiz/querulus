"""Querulus monitoring DAG — контурный стиль как BUU_projects / MONITORING_BUU.

На контуре в Airflow dags лежит только этот файл.
Код пайплайна: git clone на PVC → отдельные Airflow-таски (errors / datadrift / …).

GitLab path: Variable ``querulus_git_path``
  default ``/mlops/losses_projects/monitoring/querulus_monitoring.git``

Локально (docker, без K8s): ``QUERULUS_DAG_MODE=bash`` + mount PIPELINE_DIR.
Synthetic: ``QUERULUS_USE_SYNTHETIC=1`` → local/synthetic/run_*.py

Схема K8s-рана (каждая таска = отдельный pod):

    prepare  →  errors
             →  datadrift   →  is_monday_alert  →  alert_email
             →  fin_effect

  prepare          — СТИРАЕТ старый каталог на PVC, клонирует репо заново, ставит venv
  errors/…         — НЕ клонируют снова: ходят в уже подготовленный каталог на PVC
  is_monday_alert  — ShortCircuit на scheduler (без pod): False → alert_email не стартует
  alert_email      — отдельный pod с make alert_email
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

# Contour GitLab: clone → каталог querulus_monitoring/ (= корень pipeline)
git_path = Variable.get(
    "querulus_git_path",
    default_var="/mlops/losses_projects/monitoring/querulus_monitoring.git",
)
git_folder = "querulus_monitoring"

# Общий workdir на PVC между тасками ОДНОГО DAG run.
# Pod'ы эфемерные, а /data_jcs — постоянный диск (PVC airflow-dag-venv),
# поэтому prepare пишет сюда clone+venv, а следующие таски их читают.
_work_root = f"/data_jcs/schedule/{task_name}"  # весь «рабочий стол» DAG
_work_dir = f"{_work_root}/{git_folder}"  # корень клона (= Makefile, *.py)

dag_schedule_interval = "0 9 * * *"  # ежедневно 09:00; письмо — только в понедельник

default_args = {
    "owner": "querulus",
    "start_date": datetime.datetime(2026, 1, 1, 9, 0, 0),
    "retries": 1,
    "provide_context": True,
}

# k8s (контур) | bash (local SequentialExecutor)
DAG_MODE = os.environ.get("QUERULUS_DAG_MODE", "k8s").strip().lower()


def _is_monday_alert(**context) -> bool:
    """ShortCircuit: письмо только в ALERT_WEEKDAY (0=пн) или ALERT_FORCE.

    Если вернёт False — Airflow НЕ запустит downstream (alert_email).
    Это не pod: выполняется на scheduler/worker Airflow.
    """
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

    Важно про жизненный цикл файлов:
      - каждый KubernetesPodOperator поднимает НОВЫЙ pod и после успеха
        УДАЛЯЕТ его (on_finish_action=\"delete_pod\");
      - поэтому код нельзя класть «внутрь» ephemeral container FS;
      - общий диск — PVC airflow-dag-venv → mount /data_jcs;
      - prepare ПЕРЕЗАПИСЫВАЕТ /data_jcs/schedule/querulus_monitoring
        (rm -rf + mkdir + git clone + make venv);
      - errors / datadrift / fin_effect / alert_email только cd туда и make <цель>.
    """
    from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
    from kubernetes.client import models as k8s

    # --- куда монтируем тома внутри каждого pod ---------------------------------
    volume_mount = [
        # DAG-файлы Airflow (read-only): сам этот .py уже загружен scheduler'ом
        k8s.V1VolumeMount(
            mount_path="/opt/airflow/dags", name="dags", sub_path=None, read_only=True
        ),
        # РАБОЧИЙ диск (read-write): сюда prepare кладёт clone + venv
        k8s.V1VolumeMount(
            mount_path="/data_jcs", name="dag-venv", sub_path=None, read_only=False
        ),
        # Kerberos keytab для kinit (Hive / Hadoop)
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
        # Hive / HDFS конфиги (один ConfigMap hive-site монтируется в 2 пути)
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

    # --- сами тома (PVC / ConfigMap) --------------------------------------------
    volume = [
        k8s.V1Volume(
            name="dags",
            persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
                claim_name="airflow-dags"
            ),
        ),
        # Общий PVC для clone+venv между тасками (не путать с airflow-dags)
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

    # Образ и CPU/RAM из Airflow Variable k8s_base (общий для контура)
    k8s_base = Variable.get(key="k8s_base", deserialize_json=True)
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
    # Опционально: путь к ретро-файлу фин.эффекта (Variable → env)
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

    # --- bash bodies: что реально выполняется внутри pod ------------------------
    #
    # prepare: ЕДИНСТВЕННОЕ место, где каталог на PVC уничтожается и создаётся
    # заново. Без rm -rf остался бы вчерашний clone/venv → stale-код / битый venv.
    prepare_cmd = f"""
set -e
echo "=== kinit (тикет Kerberos для Hive/HDFS) ==="
kinit -F -k -t /opt/adh-etl-user.keytab adh-etl-user

echo "=== УДАЛЯЕМ старый workdir на PVC (если был) ==="
# ВНИМАНИЕ: стирает ВЕСЬ /data_jcs/schedule/querulus_monitoring
# (прошлый clone + venv). Нужно, чтобы каждый run брал свежий код из GitLab.
rm -rf {_work_root}

echo "=== СОЗДАЁМ пустой workdir заново ==="
mkdir -p {_work_root}
cd {_work_root}

export GIT_SSL_NO_VERIFY=true
echo "=== git clone pipeline из GitLab ==="
# Создаёт подкаталог {_work_root}/{git_folder}/
git clone {git_url}
cd {git_folder}
ls -la

echo "=== make venv (создаёт ./venv и ставит зависимости) ==="
make venv

echo "=== make test (py_compile, без запросов к БД) ==="
make test

echo "=== prepare DONE — clone+venv лежат на PVC, pod можно убивать ==="
"""
    if DEBUG:
        # Не удаляем файлы — только держим pod живым для ручного захода
        prepare_cmd += "\necho 'DEBUG sleep 3600'; sleep 3600\n"

    def _step_cmd(make_target: str) -> str:
        """Bash для этапной таски: НЕ трогает clone, только запускает make.

        Каждый такой вызов — новый pod. kinit делаем снова, потому что
        Kerberos-тикет жил в предыдущем (уже удалённом) контейнере.
        venv на PVC уже есть после prepare — его не пересоздаём.
        """
        return f"""
set -e
echo "=== kinit (новый pod = новый тикет) ==="
kinit -F -k -t /opt/adh-etl-user.keytab adh-etl-user

echo "=== переходим в clone на PVC (prepare уже положил сюда код+venv) ==="
cd {_work_dir}

# Страховка: если prepare упал / PVC пустой — падаем сразу, а не на pip
test -d venv || {{ echo "venv missing — did prepare run?" >&2; exit 1; }}

echo "=== make {make_target} (этап пайплайна, пишет в Motor) ==="
make {make_target}

echo "=== {make_target} DONE ==="
"""

    def _k8s_task(task_id: str, bash: str) -> KubernetesPodOperator:
        """Один Airflow-столбик = один pod с общим PVC и одними и теми же env."""
        return KubernetesPodOperator(
            task_id=task_id,
            name=f"{task_name}-{task_id}",
            container_resources=compute_resources,
            image=k8s_base["image"],
            volumes=volume,
            volume_mounts=volume_mount,
            env_vars=env_vars,
            cmds=["bash", "-cx", bash],
            # После успеха/фейла pod УДАЛЯЕТСЯ; файлы на PVC остаются
            on_finish_action="delete_pod",
            in_cluster=True,
            do_xcom_push=False,
            get_logs=True,
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
        max_active_runs=1,  # не запускать второй run, пока живёт первый (общий PVC)
        tags=tags,
    ) as dag:
        # 1) стереть/пересоздать clone+venv
        prepare = _k8s_task("prepare", prepare_cmd)
        # 2) три независимых этапа (можно параллельно после prepare)
        errors = _k8s_task("errors", _step_cmd("errors"))
        datadrift = _k8s_task("datadrift", _step_cmd("datadrift"))
        fin_effect = _k8s_task("fin_effect", _step_cmd("fin_effect"))
        # 3) гейт на scheduler: не понедельник → alert_email skipped
        is_monday_alert = ShortCircuitOperator(
            task_id="is_monday_alert",
            python_callable=_is_monday_alert,
        )
        # 4) письмо (отдельный pod)
        alert_email = _k8s_task("alert_email", _step_cmd("alert_email"))

        # Граф: prepare → {errors, datadrift, fin_effect} → гейт → письмо
        prepare >> [errors, datadrift, fin_effect] >> is_monday_alert >> alert_email
    return dag


if DAG_MODE == "bash":
    dag = _build_bash_dag()
else:
    dag = _build_k8s_dag()
