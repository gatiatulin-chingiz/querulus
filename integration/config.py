"""Конфигурация сервиса querulus (пути к артефактам).

Stem модели — из запроса. THRESHOLD — из вектора; иначе default ниже
(отдельно для main_ и second_). Версии треков — в main_/second_response.
"""
from pathlib import Path

from environs import Env

env_reader = Env()
env_reader.read_env("env_template")

base_path = Path(__file__).resolve().parent
prod_models_path = "./results"

# Версии треков (в ответах и /api/health).
main_version = env_reader.str("MAIN_VERSION", "1.4.0")
second_version = env_reader.str("SECOND_VERSION", "2.0.0")

# Fallback τ, если в векторе нет THRESHOLD. Не читать metadata.json.
default_main_threshold = env_reader.float("DEFAULT_MAIN_THRESHOLD", 0.6)
default_second_threshold = env_reader.float("DEFAULT_SECOND_THRESHOLD", 0.6)

# --- Second 2.0.0 (пути рядом с pickle; stem — из second_model в запросе) ---
shadow_models_subdir = env_reader.str("SHADOW_MODELS_SUBDIR", "")
second_cf_calibrator_filename = env_reader.str(
    "SECOND_CF_CALIBRATOR_FILENAME", "cf_calibrator.pickle"
)
second_rg_calibrator_filename = env_reader.str(
    "SECOND_RG_CALIBRATOR_FILENAME", "rg_calibrator.pickle"
)
