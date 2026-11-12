import aiohttp
import argparse
import asyncio
from fastapi import FastAPI, status
from fastapi.responses import ORJSONResponse
from fastapi.encoders import jsonable_encoder
import os
import pandas as pd
import pickle
from pydantic import RootModel
from typing import Dict, List, Optional, Union, Any
import uvicorn
import time
import functools
import traceback
from loguru import logger

from mldataworker.core.predict import ensemble_predict
# from mldataworker.core.pydantic_models import InputRequest
from mldataworker.core.utils import ResultPickle
from mldataworker.core.validators import GroupValidator


app = FastAPI(default_response_class=ORJSONResponse)

all_groups: Dict = {}

http_session: Optional[aiohttp.ClientSession] = None

class InputRequest(RootModel):
    root: Dict


class Config:
    def __init__(self):
        self.prod_models_path: Optional[str] = None
        self.url_one: Optional[str] = None

    def init_config(self, prod_models_path: str, url_one: Optional[str] = None):
        self.prod_models_path = prod_models_path
        self.url_one = url_one


cnfg = Config()


rename_dict = {
}


@app.on_event("startup")
async def startup_event():
    global http_session
    timeout = aiohttp.ClientTimeout(total=10)
    connector = aiohttp.TCPConnector(limit=200, ttl_dns_cache=300)
    http_session = aiohttp.ClientSession(timeout=timeout, connector=connector)


@app.on_event("shutdown")
async def shutdown_event():
    global http_session
    if http_session:
        await http_session.close()


def log_async_call():
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()

            group_name = None
            submodel = None

            try:
                input_request = kwargs.get("input_request")
                if input_request is not None:
                    root = getattr(input_request, "root", None)
                    if isinstance(root, dict):
                        group_name = root.get("model")
                        submodel = root.get("submodel")
            except Exception:
                pass

            logger.info(
                "CALL {} model={} submodel={}",
                func.__qualname__,
                group_name,
                submodel,
            )

            try:
                result = await func(*args, **kwargs)
                elapsed = round((time.perf_counter() - start) * 1000, 2)

                logger.info(
                    "OK {} model={} submodel={} time={}ms",
                    func.__qualname__,
                    group_name,
                    submodel,
                    elapsed,
                )
                return result

            except Exception:
                elapsed = round((time.perf_counter() - start) * 1000, 2)

                logger.error(
                    "ERROR {} model={} submodel={} time={}ms\n{}",
                    func.__qualname__,
                    group_name,
                    submodel,
                    elapsed,
                    traceback.format_exc(),
                )
                raise

        return wrapper
    return decorator


def rename_features(features_values: Optional[List[Dict]]) -> Optional[List[Dict]]:
    if features_values is None:
        return None
    for i in range(len(features_values)):
        features_values[i].update({rename_dict.get(k, k): v for k, v in features_values[i].items()})
    return features_values


def update_all_groups(config: Config, group_name: str):
    group = all_groups.get(group_name)
    if not group:
        logger.info("Loading pickle || " + group_name)
        with open(os.path.join(config.prod_models_path, f"{group_name}.pickle"), "rb") as f:
            group = pickle.load(f)
            GroupValidator(group).validate()
            for model in group:
                if isinstance(model, dict): model["model_name"] = model["model"].model_name
            all_groups.update({group_name: group})
    return group


async def fetch_predict(
        config: Config, session, model: str, request: List[Dict[str, Any]], date: str, submodel: str, type_: str
):
    """Функция для отправки запроса на /predict_one"""

    payload = {
        "model": model,
        "request": request if not rename_dict else rename_features(request),
        "date": date,
        "submodel": submodel
    }

    async with session.post(
            f"{config.url_one}/api/predict_one", json=payload
    ) as response:
        return await response.json(), model, submodel, type_


async def main_predict(
        config,
        group_name: Optional[str],
        features_values: Union[List[Dict], pd.DataFrame],
        async_mode: bool = True,
        submodel: Optional[str] = None,
) -> Dict:
    """
    :param config: `config` should contain `prod_models_path`
    :param group_name: name of the main model
    :param features_values: data for main model
    :param async_mode:
    :param submodel: name of the submodel
    """

    predict_tasks = []

    if not group_name:
        # TODO: add date
        group_name = ResultPickle(config).get_last_group_name()

    group = update_all_groups(config, group_name)

    for model in group:
        if submodel and (model["model_name"] if isinstance(model, dict) else model.model_name) != submodel:
            print(f'{submodel} not found')
            pass
        else:
            print(f'{submodel} found')
            predict_tasks.append(ensemble_predict(
                group_name, model, features_values, log=False, modify_dtypes=False, raise_on_encoding_error=True
            ))

    if async_mode:
        predictions = await asyncio.gather(*predict_tasks)
    else:
        predictions = []
        for task in predict_tasks:
            predictions.append(await task)

    main_response = {
        "usage_model": group_name,
        "result": {},
        "version_model": {},
        "df": {},
        "default": {}
    }
    for predict in predictions:
        main_response["result"].update(predict["result"])
        main_response["version_model"].update(predict["version_model"])
        main_response["df"].update(predict["df"])
        main_response["default"].update(predict["default"])

    return {"main_response": main_response}


@app.get("/api/health")
async def health_route():
    return ORJSONResponse(content={"health": True}, status_code=status.HTTP_200_OK)


@app.post("/api/predict_one")
#@log_async_call()
async def predict_one(input_request: InputRequest):

    try:
        group_name = input_request.root.get("model")
        features_values = input_request.root.get("request")
        submodel = input_request.root.get("submodel")
        # TODO: add date

        prediction = await main_predict(
            config=cnfg,
            group_name=group_name,
            features_values=features_values,
            submodel=submodel,
            async_mode=True
        )

        response = prediction
        status_code = status.HTTP_200_OK

    except Exception as exc:
        traceback.print_exc()
        response = {"error": traceback.format_exc()}
        status_code = status.HTTP_400_BAD_REQUEST

    return ORJSONResponse(content=response, status_code=status_code)


@app.post("/api/predict")
@log_async_call()
async def process_request(input_request: InputRequest):
    async with aiohttp.ClientSession() as session:
        tasks = []
        output_response = {}

        # Итерация по полям в запросе
        for key, value in input_request.root.items():
            if key.endswith("_model"):
                group_name = value
                group = update_all_groups(cnfg, group_name)
                request_key = f"{key.replace('_model', '_request')}"
                date_key = f"{key.replace('_model', '_date')}"
                type_ = f"{key.replace('_model', '')}"

                # Проверяем наличие соответствующих ключей
                if request_key in input_request.root and date_key in input_request.root:
                    request_data = input_request.root[request_key]
                    date = input_request.root[date_key]

                    # Итерация по нужным моделям в пикле
                    for model in group:
                        tasks.append(
                            fetch_predict(
                                config=cnfg,
                                session=session,
                                model=group_name,
                                request=request_data,
                                date=date,
                                submodel=model["model_name"] if isinstance(model, dict) else model.model_name,
                                type_=type_,
                            )
                        )

                    output_response[f"{key.replace('_model', '_response')}"] = {
                        "usage_model": group_name,
                        "result": {},
                        "version_model": {},
                        "df": {},
                        "default": {}
                    }

        responses = await asyncio.gather(*tasks)

        # Формируем объединенный ответ
        for key in input_request.root:
            if key.endswith("_model"):
                for response, model, submodel, type_ in responses:
                    output_key = f"{type_ + '_response'}"
                    output_response[output_key]["result"].update(response["main_response"]["result"])
                    output_response[output_key]["version_model"].update(response["main_response"]["version_model"])
                    output_response[output_key]["df"].update(response["main_response"]["df"])
                    output_response[output_key]["default"].update(response["main_response"]["default"])

        return output_response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prod_models_path", default="./models")
    parser.add_argument("--url_one", default="http://localhost:9999")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int,default=9999)
    args = vars(parser.parse_args())

    cnfg.init_config(prod_models_path=args["prod_models_path"], url_one=args["url_one"])

    uvicorn.run(app, host=args["host"], port=args["port"])
