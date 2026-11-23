"""Local-only: synthetic train/prod features → pipeline datadrift + Motor write."""
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
from pipeline.datadrift import META_COLS, compute_datadrift_by_meta
from pipeline.write_sql import get_engine, write_append_guarded

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("local.datadrift")


def _read_features(data_dir: Path, stem: str) -> pd.DataFrame:
    for suffix in (".parquet", ".csv"):
        path = data_dir / f"{stem}{suffix}"
        if path.is_file():
            return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    raise FileNotFoundError(
        f"Synthetic {stem} not found under {data_dir}. Run generate.py"
    )


def main() -> int:
    data_dir = Path(
        os.environ.get("SYNTHETIC_DATA_DIR", str(_SYNTH / "data"))
    )
    settings = load_settings()
    train = _read_features(data_dir, "train_features")
    prod = _read_features(data_dir, "prod_features")
    features = [
        c for c in train.columns if c in prod.columns and c not in META_COLS
    ]
    if not features:
        raise ValueError("No shared feature columns between train and prod")
    report = compute_datadrift_by_meta(train, prod, features)
    if report.empty:
        logger.warning("Empty datadrift report")
        return 0
    engine = get_engine(settings.grafana_sqlalchemy_url)
    n = write_append_guarded(
        report,
        engine=engine,
        table=settings.table_drift,
        schema=settings.schema,
        guard_cols=["date", "model_name", "track"],
    )
    logger.info("local synthetic datadrift done || rows=%s", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
