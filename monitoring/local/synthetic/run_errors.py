"""Local-only: synthetic errors → pipeline aggregate + Motor write."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pandas as pd

_SYNTH = Path(__file__).resolve().parent
_MONITORING = _SYNTH.parents[1]
if str(_MONITORING) not in sys.path:
    sys.path.insert(0, str(_MONITORING))

from pipeline.settings import load_settings
from pipeline.errors import aggregate_errors
from pipeline.write_sql import get_engine, write_append_guarded

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("local.errors")


def _load_synthetic(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "errors.parquet"
    if not path.is_file():
        path = data_dir / "errors.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Synthetic errors not found under {data_dir}. Run generate.py"
        )
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    if "РезультатПроверки" in df.columns:
        df = df.loc[df["РезультатПроверки"] == -1].copy()
    if "check_result" in df.columns:
        df = df.loc[df["check_result"] == -1].copy()
    return df


def main() -> int:
    data_dir = Path(
        os.environ.get("SYNTHETIC_DATA_DIR", str(_SYNTH / "data"))
    )
    settings = load_settings()
    raw = _load_synthetic(data_dir)
    daily = aggregate_errors(raw)
    engine = get_engine(settings.grafana_sqlalchemy_url)
    n = write_append_guarded(
        daily,
        engine=engine,
        table=settings.table_errors,
        schema=settings.schema,
        guard_cols=[
            "run_date",
            "stem",
            "slot",
            "model_version",
            "error_fingerprint",
        ],
    )
    logger.info("local synthetic errors done || rows=%s", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
