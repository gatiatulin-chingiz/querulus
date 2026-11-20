# Querulus monitoring pipeline (contour / BUU-style Airflow pod)
#
#   make venv     — venv + deps (VSK PyPI)
#   make test     — syntax check
#   make automl   — errors + datadrift + fin_effect; Mon (or ALERT_FORCE=1) → alert_email
#   make clean    — remove venv / caches

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
export PYTHONPATH := $(ROOT)/src:$(ROOT):$(PYTHONPATH)

venv:
	echo $@ processing...
	python3 -m venv ./venv
	venv/bin/python3 -m pip config set global.index-url 'http://repo.vsk.ru/repository/pypi.org/simple'
	venv/bin/python3 -m pip config set global.extra-index-url 'https://repo.vsk.ru/repository/pipdiligence/simple'
	venv/bin/python3 -m pip config set global.trusted-host 'repo.vsk.ru'
	venv/bin/python3 -m pip install --upgrade pip
	./venv/bin/pip3 install --extra-index-url https://repo.vsk.ru/repository/pipdiligence/simple --trusted-host repo.vsk.ru -r requirements.txt

fetch:
	@echo "No fetch step — metrics sources via SOURCE_MODE / extractors"

test:
	@echo "Running syntax checks..."
	./venv/bin/python3 -m py_compile config.py write_sql.py errors.py datadrift.py fin_effect.py alert_email.py
	@echo "Syntax checks passed."

automl:
	@echo "Running Querulus monitoring pipeline..."
	./venv/bin/python3 errors.py
	./venv/bin/python3 datadrift.py
	./venv/bin/python3 fin_effect.py
	@dow=$$(date +%u); \
	if [ "$$dow" = "1" ] || [ "$${ALERT_FORCE}" = "1" ] || [ "$${ALERT_FORCE}" = "true" ]; then \
		echo "Weekly alert (Monday or ALERT_FORCE)..."; \
		./venv/bin/python3 alert_email.py; \
	else \
		echo "Skip alert_email (weekday=$$dow, ALERT_FORCE=$${ALERT_FORCE:-})"; \
	fi

push:
	@echo "No push step — results written to MSSQL agrafana"

clean:
	@echo "Cleaning up..."
	rm -rf venv
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Done."
