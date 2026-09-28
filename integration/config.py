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
outboxml_train_df_path = env_reader.str(
    "OUTBOXML_TRAIN_DF_PATH",
    "/home/jovyan/old_home/Litigant/data/processed/df_for_service_test_with_framework_preds_3.parquet",
)
outboxml_model_group = env_reader.str(
    "OUTBOXML_MODEL_GROUP", "querulus_ansamble_2026_04_18_v1"
)
outboxml_preds_cf_col = env_reader.str("OUTBOXML_PREDS_CF_COL", "preds_cf")
outboxml_preds_rg_col = env_reader.str("OUTBOXML_PREDS_RG_COL", "preds_rg")

# --- Shadow 2.0.0 (second_ only; # CUTOVER → main_) ---
SHADOW_NEW_AS_SECOND = True
shadow_models_subdir = env_reader.str("SHADOW_MODELS_SUBDIR", "querulus/2.0.0")
shadow_model_group = env_reader.str("SHADOW_MODEL_GROUP", "querulus_ansamble")
shadow_meta_filename = env_reader.str("SHADOW_META_FILENAME", "metadata.json")
shadow_dq_bounds_filename = env_reader.str("SHADOW_DQ_BOUNDS_FILENAME", "dq_bounds.json")
shadow_train_df_path = env_reader.str("SHADOW_TRAIN_DF_PATH", "")
