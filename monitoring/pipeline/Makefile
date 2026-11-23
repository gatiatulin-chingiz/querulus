# Querulus monitoring pipeline (contour / BUU-style Airflow pod)
#
# Airflow K8s DAG calls targets one-by-one (separate pods / Airflow tasks):
#   make venv | test | errors | datadrift | fin_effect | alert_email
# Local one-shot still works:
#   make automl   # all stages; Mon (or ALERT_FORCE=1) -> alert_email
#
# Что трогает диск:
#   make venv   — СОЗДАЁТ ./venv (поверх, если уже есть — python -m venv обновит)
#   make clean  — УДАЛЯЕТ ./venv, .pytest_cache, __pycache__
#   этапы       — только читают код/venv и пишут в БД Motor (локально файлы не чистят)
# Каталог clone на PVC стирает НЕ Makefile, а таска prepare в DAG (rm -rf).

# Standalone clone (querulus_monitoring.git): PIPE=repo root.
# Nested in monorepo: also add ../../src (examples/querulus/src).
PIPE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
export PYTHONPATH := $(PIPE):$(PIPE)/src:$(PIPE)/../..:$(PIPE)/../../src:$(PYTHONPATH)

# Создать виртуальное окружение и поставить зависимости с VSK PyPI
venv:
	echo $@ processing...
	python3 -m venv ./venv
	venv/bin/python3 -m pip config set global.index-url 'http://repo.vsk.ru/repository/pypi.org/simple'
	venv/bin/python3 -m pip config set global.extra-index-url 'https://repo.vsk.ru/repository/pipdiligence/simple'
	venv/bin/python3 -m pip config set global.trusted-host 'repo.vsk.ru'
	venv/bin/python3 -m pip install --upgrade pip
	./venv/bin/pip3 install --extra-index-url https://repo.vsk.ru/repository/pipdiligence/simple --trusted-host repo.vsk.ru -r requirements.txt

# Заглушка под BUU-контракт (у нас данные не из git-fetch)
fetch:
	@echo "No fetch step - sources: OISUU + Oracle RTDM + Hive reference"

# Быстрая проверка синтаксиса без хождения в БД
test:
	@echo "Running syntax checks..."
	./venv/bin/python3 -m py_compile config.py settings.py db.py hadoop_extractor.py write_sql.py errors.py datadrift.py fin_effect.py alert_email.py
	@echo "Syntax checks passed."

# --- этапы пайплайна (каждая цель = отдельная Airflow-таска на контуре) -------

# Ошибки модели: MatemModelsLogs (MSSQL) -> Motor.agrafana
errors:
	@echo "=== errors ==="
	./venv/bin/python3 errors.py

# DataDrift: Oracle prod + Hive train -> PSI/метрики в Motor
datadrift:
	@echo "=== datadrift ==="
	./venv/bin/python3 datadrift.py

# Фин.эффект: OISUU-витрина (+ retro path) -> Motor
fin_effect:
	@echo "=== fin_effect ==="
	./venv/bin/python3 fin_effect.py

# Еженедельное письмо (DAG сам решает, звать ли цель — ShortCircuit в понедельник)
alert_email:
	@echo "=== alert_email ==="
	./venv/bin/python3 alert_email.py

# Локальный one-shot: все метрики подряд; письмо — только пн / ALERT_FORCE
automl: errors datadrift fin_effect
	@dow=$$(date +%u); \
	if [ "$$dow" = "1" ] || [ "$${ALERT_FORCE}" = "1" ] || [ "$${ALERT_FORCE}" = "true" ]; then \
		echo "Weekly alert (Monday or ALERT_FORCE)..."; \
		$(MAKE) alert_email; \
	else \
		echo "Skip alert_email (weekday=$$dow, ALERT_FORCE=$${ALERT_FORCE:-})"; \
	fi

# Заглушка под BUU-контракт (результаты уже в Motor, отдельного push нет)
push:
	@echo "No push step - results written to MSSQL agrafana"

# УДАЛЯЕТ локальный venv и кэши Python. На контуре обычно не зовётся:
# prepare в DAG делает rm -rf всего workdir целиком перед новым clone.
clean:
	@echo "Cleaning up..."
	rm -rf venv
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Done."
