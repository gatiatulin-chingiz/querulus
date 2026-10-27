"""Конфигурация сервиса querulus (пути к артефактам).

Имена моделей (stem) и THRESHOLD приходят в запросе / векторе, не из этого файла.
"""
from pathlib import Path

from environs import Env

env_reader = Env()
env_reader.read_env("env_template")

base_path = Path(__file__).resolve().parent
prod_models_path = "./results"

# --- Second 2.0.0 (пути рядом с pickle; stem — из second_model в запросе) ---
shadow_models_subdir = env_reader.str("SHADOW_MODELS_SUBDIR", "")
shadow_meta_filename = env_reader.str("SHADOW_META_FILENAME", "metadata.json")
second_cf_calibrator_filename = env_reader.str(
    "SECOND_CF_CALIBRATOR_FILENAME", "cf_calibrator.pickle"
)
second_rg_calibrator_filename = env_reader.str(
    "SECOND_RG_CALIBRATOR_FILENAME", "rg_calibrator.pickle"
)
