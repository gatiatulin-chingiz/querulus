"""Конфигурация сервиса querulus (main, tests)."""
from pathlib import Path

from querulus import PROJECT_ROOT
from querulus.env import load_project_env

env_reader = load_project_env()
project_root = PROJECT_ROOT
base_path = Path(__file__).resolve().parent
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
