"""Конфигурация пайплайна сборки датасета."""
import json
from pathlib import Path

from querulus import PROJECT_ROOT
from querulus.env import load_project_env

base_path = Path(__file__).resolve().parent
project_root = PROJECT_ROOT

env_reader = load_project_env()

_DATASET_SOURCES_PATH = project_root / "configs" / "dataset_sources.json"


def _load_artifact_overrides() -> dict[str, str]:
    """Явные пути к parquet из configs/dataset_sources.json."""
    if not _DATASET_SOURCES_PATH.exists():
        return {}
    data = json.loads(_DATASET_SOURCES_PATH.read_text(encoding="utf-8"))
    paths = data.get("artifact_paths", {})
    return {
        key: value.strip()
        for key, value in paths.items()
        if key and not key.startswith("_") and isinstance(value, str) and value.strip()
    }


# --- Dataset paths (collect pipeline) ---
litigant_data_root = env_reader.str("LITIGANT_DATA_ROOT", str(project_root))
litigant_legacy_data_root = env_reader.str("LITIGANT_LEGACY_DATA_ROOT", "")
victim_parquet_path = env_reader.str("VICTIM_PARQUET", "")
litigant_artifact_version = env_reader.str("LITIGANT_ARTIFACT_VERSION", "")
artifact_overrides = _load_artifact_overrides()

# --- OISUU / pymssql ---
oisuu_db_server = env_reader.str("OISUU_DB_SERVER", "")
oisuu_db_user = env_reader.str("OISUU_DB_USER", "")
oisuu_db_password = env_reader.str("OISUU_DB_PASSWORD", "")
oisuu_db_database = env_reader.str("OISUU_DB_DATABASE", "")
oisuu_db_port = env_reader.int("OISUU_DB_PORT", 1433)
