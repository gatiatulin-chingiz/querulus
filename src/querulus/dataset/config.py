"""Конфигурация пайплайна сборки датасета."""
from pathlib import Path

from querulus import PROJECT_ROOT
from querulus.env import load_project_env

base_path = Path(__file__).resolve().parent
project_root = PROJECT_ROOT

env_reader = load_project_env()

# --- Dataset paths (collect pipeline) ---
litigant_data_root = env_reader.str("LITIGANT_DATA_ROOT", str(project_root))
litigant_legacy_data_root = env_reader.str("LITIGANT_LEGACY_DATA_ROOT", "")
victim_parquet_path = env_reader.str("VICTIM_PARQUET", "")
litigant_artifact_version = env_reader.str("LITIGANT_ARTIFACT_VERSION", "")

# --- OISUU / pymssql ---
oisuu_db_server = env_reader.str("OISUU_DB_SERVER", "")
oisuu_db_user = env_reader.str("OISUU_DB_USER", "")
oisuu_db_password = env_reader.str("OISUU_DB_PASSWORD", "")
oisuu_db_database = env_reader.str("OISUU_DB_DATABASE", "")
oisuu_db_port = env_reader.int("OISUU_DB_PORT", 1433)
