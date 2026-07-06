"""Конфигурация сервиса querulus (main, tests)."""
from pathlib import Path

from environs import Env

env_reader = Env()
env_reader.read_env("env_template")

base_path = Path(__file__).resolve().parent
prod_models_path = "./results"
prod_date_format = "%d.%m.%Y"

# --- Examples / integration tests (querulus, etc.) ---
# Keep these in env to avoid hardcoding paths/model names in code.
outboxml_train_df_path = "/home/jovyan/old_home/Litigant/data/processed/df_for_service_test_with_framework_preds_3.parquet"
outboxml_model_group = "querulus_ansamble_2026_04_18_v1"
outboxml_preds_cf_col = "preds_cf"
outboxml_preds_rg_col = "preds_rg"
