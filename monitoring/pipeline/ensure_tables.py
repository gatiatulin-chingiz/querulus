"""Один раз создать таблицы мониторинга в Motor.agrafana (IF NOT EXISTS).

БД Motor и schema agrafana должны уже существовать.

  python ensure_tables.py
  make tables
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_PIPELINE_ROOT = Path(__file__).resolve().parent
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))
import _bootstrap_path  # noqa: F401 — monorepo + standalone GitLab clone

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
