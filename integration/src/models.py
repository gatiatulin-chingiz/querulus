"""Пути и кэш pickle: модели и калибраторы — один контракт.

Все артефакты лежат в ``config.prod_models_path``. Загрузка только через
``ArtifactCache.get(path)``: один раз с диска, дальше из памяти.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from loguru import logger

try:
    from integration import config
except ImportError:
    import config


# reviewed
class ArtifactCache:
    """Кэш pickle по абсолютному пути. Модели и калибраторы грузятся одинаково."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def get(self, path: Path) -> Any:
        """Вернуть объект из кэша или прочитать файл.

        Args:
            path: путь к pickle.

        Returns:
            Десериализованный объект.

        Raises:
            FileNotFoundError: файла нет.
            ValueError: pickle пуст (``None``).
        """
        key = str(path.resolve())
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if not path.is_file():
            raise FileNotFoundError(f"pickle не найден: {path}")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if obj is None:
            raise ValueError(f"pickle пуст (None): {key}")
        self._cache[key] = obj
        logger.info("pickle loaded || path={}", key)
        return obj


# reviewed
def prod_models_dir() -> Path:
    """Каталог pickle моделей и калибраторов.

    Returns:
        Абсолютный Path к каталогу моделей.
    """
    models_path = Path(config.prod_models_path)
    if models_path.is_absolute():
        return models_path
    return (Path(config.base_path) / models_path).resolve()


# reviewed
def model_pickle_path(stem: str) -> Path:
    """Путь к pickle модели: ``prod_models_dir / {stem}.pickle``."""
    return prod_models_dir() / f"{stem.strip()}.pickle"


# reviewed
def calibrator_paths() -> tuple[Path, Path]:
    """Пути CF и RG калибраторов second_ в каталоге моделей.

    Returns:
        ``(cf_path, rg_path)``.

    Raises:
        RuntimeError: имя калибратора в config пустое.
    """
    cf = str(config.second_cf_calibrator_filename).strip()
    rg = str(config.second_rg_calibrator_filename).strip()
    if not cf or not rg:
        raise RuntimeError("SECOND_CF/RG_CALIBRATOR_FILENAME обязательны")
    root = prod_models_dir()
    return root / cf, root / rg


# reviewed
def feature_names_from_model_result(model_result: dict[str, Any]) -> list[str]:
    """Имена фич из ``model_config.features`` элемента pickle-группы.

    Args:
        model_result: элемент группы ``{"model_config", "model"}``.

    Returns:
        Непустой список имён признаков по порядку из config.

    Raises:
        TypeError / ValueError / KeyError: битый или пустой ``model_config``.
    """
    # Пустой model_config уже отсечён в _load_model_pair.
    cfg = model_result["model_config"]
    if not isinstance(cfg, dict):
        raise TypeError(
            f"model_config должен быть dict, получено {type(cfg).__name__}"
        )
    features = cfg["features"]
    if not features:
        raise ValueError("model_config.features пуст")
    names: list[str] = []
    for i, item in enumerate(features):
        if isinstance(item, dict):
            name = item.get("name")
        else:
            name = getattr(item, "name", None)
        if not name:
            raise ValueError(f"model_config.features[{i}] без name")
        names.append(str(name))
    return names
