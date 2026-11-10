"""Смоук на старте: один вектор через score_main и score_second.

Файл ``startup_check.json`` (только ``vector``) лежит в каталоге моделей.
Исключение скоринга, пустой ответ или неконечное число — процесс не поднимается.
Значения с эталоном не сравниваются.
"""
from __future__ import annotations

import json
import math
from typing import Any

from loguru import logger

import config
from src.models import prod_models_dir
from src.predictions import score_main, score_second

_CHECK_NAME = "startup_check.json"
_FIELDS = ("classification_proba", "prediction", "threshold")


# reviewed
def _check_result(name: str, response: dict[str, Any]) -> None:
    result = response.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"startup check: {name} без result")
    for field in _FIELDS:
        if field not in result:
            raise RuntimeError(f"startup check: {name} не вернул {field}")
        value = result[field]
        values = value if isinstance(value, list) else [value]
        if isinstance(value, list) and not value:
            raise RuntimeError(f"startup check: {name}.{field} пустой")
        for item in values:
            label = f"{name}.{field}"
            if item is None or isinstance(item, bool):
                raise RuntimeError(f"startup check: {label} не число ({item!r})")
            number = float(item)
            if not math.isfinite(number):
                raise RuntimeError(
                    f"startup check: {label} не конечное число ({item!r})"
                )


# reviewed
def assert_startup_scores() -> None:
    """Прогнать контрольный вектор. Исключение или неконечные числа — ошибка старта."""
    path = prod_models_dir() / _CHECK_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Нет контрольного вектора: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    vector = payload.get("vector") if isinstance(payload, dict) else None
    if not isinstance(vector, dict):
        raise RuntimeError(f"startup check: в {path.name} нет объекта vector")
    features = [vector]
    main_stem = str(config.main_model_stem).strip()
    second_stem = str(config.second_model_stem).strip()
    _check_result("main", score_main(main_stem, features))
    _check_result("second", score_second(second_stem, features))
    logger.info("Startup check ok || path={}", str(path))
