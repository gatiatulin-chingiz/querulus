"""Загрузка переменных окружения из корня проекта."""
from pathlib import Path

from environs import Env

from querulus import PROJECT_ROOT


def load_project_env() -> Env:
    """Читает `.env` или `env_template` из корня репозитория."""
    env_reader = Env()
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        env_reader.read_env(env_file)
    else:
        env_reader.read_env(PROJECT_ROOT / "env_template")
    return env_reader
