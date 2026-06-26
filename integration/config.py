"""Конфигурация сервиса querulus (main, integration_tests)."""
from pathlib import Path

from environs import Env

env_reader = Env()
_env_file = Path(__file__).resolve().parent / ".env"
if _env_file.exists():
    env_reader.read_env(_env_file)
else:
    env_reader.read_env(Path(__file__).resolve().parent / "env_template")

base_path = Path(__file__).resolve().parent
project_root = base_path.parent
prod_models_path = project_root / "data" / "processed"
prod_date_format = "%d.%m.%Y"

outboxml_train_df_path = env_reader.str(
    "OUTBOXML_TRAIN_DF_PATH",
    str(project_root / "data/processed/df_for_service_test_with_framework_preds.parquet"),
)
outboxml_model_group = env_reader.str(
    "OUTBOXML_MODEL_GROUP", "querulus_ansamble_2026_04_09_v1"
)
outboxml_preds_cf_col = "preds_cf"
outboxml_preds_rg_col = "preds_rg"
