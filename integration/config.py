"""Конфигурация HTTP-сервиса Querulus.

Читает ``env_template`` через ``environs.Env``. Stem модели приходит в
запросе; порог — из колонки ``THRESHOLD`` вектора, иначе default ниже
(отдельно для main_ и second_). Версии треков уходят в ответы и
``GET /api/health``.

Атрибуты модуля:
    base_path: каталог ``integration/`` (якорь относительных путей).
    prod_models_path: каталог pickle (по умолчанию ``./results``).
    main_version / second_version: строки версий треков.
    default_main_threshold / default_second_threshold: fallback τ.
    second_cf_calibrator_filename / second_rg_calibrator_filename:
        имена pickle калибраторов рядом с моделями в prod_models.
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
default_second_threshold = env_reader.float("DEFAULT_SECOND_THRESHOLD", 0.37)

# Stem pickle, которые сервис обязан загрузить до приёма запросов.
# В каждом файле пара [классификация, регрессия] — четыре модели.
# Файлы лежат в prod_models_path рядом с калибраторами.
main_model_stem = env_reader.str("MAIN_MODEL_STEM", "querulus_ansamble_2026_04_18_v1")
second_model_stem = env_reader.str("SECOND_MODEL_STEM", "UU_querulus_2026_11_05_12_44_24")

second_cf_calibrator_filename = env_reader.str(
    "SECOND_CF_CALIBRATOR_FILENAME", "cf_calibrator.pickle"
)
second_rg_calibrator_filename = env_reader.str(
    "SECOND_RG_CALIBRATOR_FILENAME", "rg_calibrator.pickle"
)
