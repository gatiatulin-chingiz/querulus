"""Один раз создать Motor.agrafana таблицы мониторинга (IF NOT EXISTS).

  python ensure_tables.py
  make tables
"""
from __future__ import annotations

import logging
import sys

from pipeline.settings import load_settings
from pipeline.write_sql import ensure_monitoring_tables, get_engine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("ensure_tables")


def main() -> int:
    settings = load_settings()
    engine = get_engine(settings.grafana_sqlalchemy_url)
    ensure_monitoring_tables(engine, schema=settings.schema)
    logger.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
