"""Конфигурация сервиса querulus (main, tests)."""
from pathlib import Path

from environs import Env

env_reader = Env()
env_reader.read_env("env_template")

base_path = Path(__file__).resolve().parent
prod_models_path = "./results"
prod_date_format = "%d.%m.%Y"

# --- Legacy main_ ---
outboxml_train_df_path = env_reader.str(
    "OUTBOXML_TRAIN_DF_PATH",
    "",
)
outboxml_model_group = env_reader.str(
    "OUTBOXML_MODEL_GROUP", "querulus_ansamble_legacy"
)
outboxml_preds_cf_col = env_reader.str("OUTBOXML_PREDS_CF_COL", "preds_cf")
outboxml_preds_rg_col = env_reader.str("OUTBOXML_PREDS_RG_COL", "preds_rg")

# --- Second 2.0.0 ---
SHADOW_NEW_AS_SECOND = True
shadow_models_subdir = env_reader.str("SHADOW_MODELS_SUBDIR", "")
shadow_model_group = env_reader.str(
    "SHADOW_MODEL_GROUP", "UU_querulus_2026_10_09_16_51_20"
)
shadow_meta_filename = env_reader.str("SHADOW_META_FILENAME", "metadata.json")
second_cf_calibrator_filename = env_reader.str(
    "SECOND_CF_CALIBRATOR_FILENAME", "cf_calibrator.pickle"
)
second_rg_calibrator_filename = env_reader.str(
    "SECOND_RG_CALIBRATOR_FILENAME", "rg_calibrator.pickle"
)
shadow_train_df_path = env_reader.str("SHADOW_TRAIN_DF_PATH", "")
