"""Единственная точка входа HTTP-сервиса скоринга Querulus.

Запуск: ``python main.py`` из каталога ``integration/``
(Docker ``ENTRYPOINT`` делает то же самое).

Содержит только FastAPI-приложение и оркестрацию dual-track.
Скоринг — ``src.predictions.score_main`` / ``score_second``.
"""
from __future__ import annotations

import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from mldataworker.core.pydantic_models import ServiceRequest

import config
from src.models import prod_models_dir
from src.predictions import preload_models, score_main, score_second
from src.startup_check import assert_startup_scores


# reviewed
def _startup() -> None:
    """Лог старта, загрузка моделей и контрольный скоринг. Ошибка — процесс не поднимается."""
    # Пишем в лог версии и каталог моделей.
    models_dir = prod_models_dir()
    logger.info(
        "Service startup || main_version={} || second_version={} || main_stem={} || second_stem={} || models_dir={} || cwd={}",
        config.main_version,
        config.second_version,
        config.main_model_stem,
        config.second_model_stem,
        str(models_dir),
        str(Path.cwd()),
    )
    # Без каталога моделей процесс не поднимается.
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Каталог моделей отсутствует: {models_dir}")
    # Грузим четыре модели и калибраторы.
    preload_models()
    # Прогоняем контрольный вектор.
    assert_startup_scores()


# reviewed
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """До yield — модели в памяти. Исключение из ``_startup`` останавливает uvicorn."""
    # Старт до приёма запросов. Ошибка здесь останавливает uvicorn.
    _startup()
    # Дальше сервис принимает запросы.
    yield


app = FastAPI(lifespan=lifespan)
version = config.main_version


# reviewed
def main_predict(
    group_name: str,
    features_values: List[Dict],
    second_group_name: Optional[str] = None,
    second_features_values: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Скоринг dual-track: обязательный main_ + опциональный second_.

    Args:
        group_name: stem pickle main_ (``ServiceRequest.main_model``).
        features_values: вектор main_ (``main_request``).
        second_group_name: stem pickle second_ или None.
        second_features_values: вектор second_ или None.

    Returns:
        ``{"main_response": ..., "second_response": ...}``.
        ``second_response`` — ``{}``, если второй трек не запрошен.

    Raises:
        Любая ошибка score_main / score_second пробрасывается наверх
        (HTTP-слой превратит её в 500).
    """
    # Считаем main_.
    main_response = score_main(group_name, features_values)

    # Считаем second_, если он есть в запросе.
    second_response: Dict[str, Any] = {}
    if second_group_name and second_features_values:
        try:
            second_response = score_second(
                second_group_name,
                second_features_values,
            )
        except Exception:
            logger.exception(
                "main_predict: second_ failed || group={}",
                second_group_name,
            )
            raise

    # Собираем dual-track ответ.
    return {
        "main_response": main_response,
        "second_response": second_response,
    }


# reviewed
@app.get("/api/health")
def health_route():
    """Проверка живости процесса и версии треков.

    Returns:
        JSON: ``health=True``, ``main_version``, ``second_version``.
    """
    logger.debug("GET /api/health")
    return JSONResponse(
        content={
            "health": True,
            "main_version": config.main_version,
            "second_version": config.second_version,
        }
    )


# reviewed
@app.post("/api/predict")
def predict_route(service_request: ServiceRequest):
    """HTTP-обёртка над ``main_predict``.

    Args:
        service_request: тело ОИСУУ (main_model/request, second_*).

    Returns:
        200 + JSON dual-track ответа либо HTTP 500 с error/traceback.
    """
    try:
        # Считаем ответ.
        body = main_predict(
            service_request.main_model,
            service_request.main_request,
            service_request.second_model,
            service_request.second_request,
        )
        return JSONResponse(content=body, status_code=200)
    # Ошибку скоринга отдаём как 500.
    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception(
            "Predict failed || type={} || error={}",
            type(exc).__name__,
            str(exc),
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(exc),
                "traceback": tb.splitlines(),
            },
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
