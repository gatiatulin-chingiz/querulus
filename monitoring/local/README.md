# Local Querulus monitoring stack

1:1 с контуром: **Airflow → MSSQL (`Motor.agrafana`) → Grafana**.  
Дашборд — **ручной Import JSON** (не auto-provision).

**Боевой контур:**
- [`../dags/`](../dags/) — DAG в Airflow (по умолчанию **K8s + git clone**, как BUU)
- [`../pipeline/`](../pipeline/) — `.py` + `Makefile` + `.env` + JSON; код подтягивается clone’ом

**Локально:** `QUERULUS_DAG_MODE=bash` в compose; DAG из `../dags/`, скрипты volume.  
**Legacy:** [`../archive/`](../archive/) (gitignored).

**Pipeline:**
- `pipeline/errors.py` → `Querulus_errors`
- `pipeline/datadrift.py` → `Querulus_DataDrift` (OutBoxML)
- `pipeline/fin_effect.py` → `Querulus_FinEffect` (wide: `week_start, stem, slot, filial`)
- `pipeline/alert_email.py` → weekly mail (DAG: пн 09:00 после metrics; локально `ALERT_DRY_RUN=1`)

DAG: ежедневно `0 9 * * *`; письмо только в понедельник (`is_monday_alert`).

**FinEffect:** расчёт в `examples/querulus/src/querulus/fin_effect/`.  
В таблице: `filial='(все)'` — ITT/annual; `filial=<имя>` — coverage/paths/attn.

## Quick start

```bash
cd examples/querulus/monitoring/local
cp .env.example .env   # заполни пароли (motor_* + MSSQL_SA_PASSWORD)
docker compose up -d --build
```

Логины/пароли только в `.env` (не в compose/config/yaml).  
UI: порты и users — из `.env` (`GF_SECURITY_ADMIN_*`, `_AIRFLOW_WWW_USER_*`).

После `airflow-init` синтетика уже сгенерирована. В Airflow UI: DAG `querulus_monitoring` → Trigger.

**Дашборд:** Grafana → Dashboards → Import → файл  
`../pipeline/querulus_monitoring.json` → datasource **Querulus MSSQL**.

Фильтры:
- **Модель** — multi: main_* и second_* (весь дашборд)
- **Филиал** — доля потока + gap

Финэффект в данных только для stem в слоте main (second в фильтре есть, на финэффекте пусто).

## Без Docker (smoke)

Нужен доступный MSSQL (`Motor` + schema `agrafana`) и ODBC Driver 18:

```bash
export SOURCE_MODE=synthetic
export motor_host=localhost
export motor_name=Motor
export motor_user_name=sa
export motor_user_pass='...'
export SYNTHETIC_DATA_DIR="$(pwd)/synthetic/data"
export PYTHONPATH="$(pwd)/.."

python synthetic/generate.py
python ../pipeline/errors.py
python ../pipeline/datadrift.py
python ../pipeline/fin_effect.py
```

## Env

| Variable | Meaning |
|----------|---------|
| `motor_host` / `motor_name` / `motor_user_name` / `motor_user_pass` | Motor для pipeline (стиль BUU) |
| `MSSQL_*` | только Docker-образ MSSQL + Grafana datasource |
| `database_*` | OISUU (errors / fin_effect) |
| `SERVER_PROD8` / `USER_PROD8` / … | Oracle RTDM (datadrift) |
| `GF_SECURITY_ADMIN_USER` / `GF_SECURITY_ADMIN_PASSWORD` | Grafana UI |
| `_AIRFLOW_WWW_USER_USERNAME` / `_AIRFLOW_WWW_USER_PASSWORD` | Airflow UI |
| `SOURCE_MODE` | `synthetic` / `prod` |

## Tables (`Motor.agrafana`)

- `Querulus_errors` — replace
- `Querulus_DataDrift` — append (guard `date+model_name+track`)
- `Querulus_FinEffect` — replace, grain `(week_start, stem, slot, filial)`

## Notes

- Airflow meta: **SQLite** (`SequentialExecutor`), не Postgres.
- Image: ODBC Driver 18 + `pyodbc` ([`requirements-airflow.txt`](../pipeline/requirements-airflow.txt)).
- В бой тащить только `monitoring/pipeline/`.
