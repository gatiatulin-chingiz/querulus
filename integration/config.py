"""Конфигурация сервиса querulus (пути к артефактам).

Stem модели — из запроса. THRESHOLD — из вектора; иначе default ниже.
"""
from pathlib import Path

from environs import Env

env_reader = Env()
env_reader.read_env("env_template")

base_path = Path(__file__).resolve().parent
prod_models_path = "./results"

# Fallback τ, если в векторе нет THRESHOLD (main_ и second_). Не читать metadata.json.
default_classification_threshold = env_reader.float(
    "DEFAULT_CLASSIFICATION_THRESHOLD", 0.6
)

# --- Second 2.0.0 (пути рядом с pickle; stem — из second_model в запросе) ---
shadow_models_subdir = env_reader.str("SHADOW_MODELS_SUBDIR", "")
second_cf_calibrator_filename = env_reader.str(
    "SECOND_CF_CALIBRATOR_FILENAME", "cf_calibrator.pickle"
)
second_rg_calibrator_filename = env_reader.str(
    "SECOND_RG_CALIBRATOR_FILENAME", "rg_calibrator.pickle"
)
