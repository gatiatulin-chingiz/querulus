"""Flat FastAPI-сервис Querulus (как example_1).

Запуск: ``python main.py`` (registry по умолчанию — ``<dir(main.py)>/results``).
В Docker путь задаёт ``make_service.sh`` → ``/home/model_user/list_models/``.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import pickle
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

from outboxml.core.predict import ensemble_predict
from outboxml.core.pydantic_models import ModelConfig, ServiceRequest

MODELS_JSON = "models.json"
PIPELINES_DIR = "pipelines"
SHAP_MODULE = "shap_comment"
# stem → [pipeline, threshold, version, cf|null, rg|null]
_MODELS_ENTRY_LEN = 5
_INTEGRATION_DIR = Path(__file__).resolve().parent
_DEFAULT_MODELS_PATH = str(_INTEGRATION_DIR / "results")


# ---------------------------------------------------------------------------
# Helpers (бывший preprocessing.py)
# ---------------------------------------------------------------------------

# reviewed
def read_json_object(models_dir: Path, filename: str) -> Dict[str, Any]:
    """Прочитать непустой JSON-object из ``models_dir / filename``."""
    path = models_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Нет {filename}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Пустой {filename}: {path}")
    return payload

# reviewed
def resolve_threshold(df: pd.DataFrame, default: float) -> float:
    if "THRESHOLD" not in df.columns:
        return default
    value = pd.to_numeric(df["THRESHOLD"].iloc[0], errors="coerce")
    if pd.isna(value):
        return default
    return float(value)

# reviewed
def df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = json.loads(df.to_json(orient="records", force_ascii=False))
    return [{k: row[k] for k in sorted(row.keys())} for row in rows]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Registry:
    """Всё состояние сервиса по моделям: ``models.json``, пайплайны, pickle."""

    pipelines: Dict[str, str]
    modules: Dict[str, ModuleType]
    shap: ModuleType
    thresholds: Dict[str, float]
    versions: Dict[str, str]
    models: Dict[str, Tuple[Any, Any]]
    calibrators: Dict[str, Tuple[Any, Any]]
    ts_country_by_brand: Dict[str, Dict[str, str]]
    submodel_names: Dict[str, Dict[str, str]]
    models_json_mtime: float
    loaded_at: str


class Config:
    def __init__(self) -> None:
        self.prod_models_path: Path = Path(_DEFAULT_MODELS_PATH)
        self.registry: Optional[Registry] = None
        self.url_one: str = "http://localhost:8080"
        self.reload_interval: int = 30
        self.loaded_at: Optional[str] = None
        self.last_reload_error: Optional[str] = None


cnfg = Config()


def _registry() -> Registry:
    registry = cnfg.registry
    if registry is None:
        raise RuntimeError("registry не загружен — сервис ещё не стартовал")
    return registry

# reviewed
def _models_dir() -> Path:
    models_dir = cnfg.prod_models_path.resolve()
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Каталог моделей отсутствует: {models_dir}")
    return models_dir

# reviewed
def _set_registry(registry: Registry) -> None:
    cnfg.registry = registry
    cnfg.loaded_at = registry.loaded_at
    cnfg.last_reload_error = None


# reviewed
def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info("pickle loaded || path={}", path)
    return obj

# reviewed
def _load_model_pair(path: Path) -> Tuple[Any, Any]:
    group = _load_pickle(path)
    if not isinstance(group, (list, tuple)) or len(group) < 2:
        raise ValueError(f"{path.name}: ожидается [clf, rg], получено {type(group).__name__}")
    return group[0], group[1]

# reviewed
def _model_name(item: Dict[str, Any]) -> str:
    return ModelConfig.model_validate(item["model_config"]).name

# reviewed
def _load_module(path: Path, module_name: str) -> ModuleType:
    """Загрузить ``.py`` через importlib без регистрации в ``sys.modules``."""
    if not path.is_file():
        raise FileNotFoundError(f"Нет модуля пайплайна: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    loader = None if spec is None else spec.loader
    if spec is None or loader is None:
        raise ImportError(f"Не удалось создать spec для {path}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module

# reviewed
def _load_pipeline_module(models_dir: Path, pipeline_name: str, stem: str) -> ModuleType:
    path = models_dir / PIPELINES_DIR / f"{pipeline_name}.py"
    return _load_module(path, f"querulus_pipeline_{stem}_{pipeline_name}")


# reviewed
def load_registry(models_dir: Path) -> Registry:
    """Прочитать ``models.json`` и загрузить пайплайны / pickle."""
    models_dir = models_dir.resolve()
    models_json_path = models_dir / MODELS_JSON
    if not models_json_path.is_file():
        raise FileNotFoundError(f"Нет {MODELS_JSON}: {models_json_path}")
    models_json_mtime = models_json_path.stat().st_mtime

    shap_path = models_dir / PIPELINES_DIR / f"{SHAP_MODULE}.py"
    shap = _load_module(shap_path, f"querulus_{SHAP_MODULE}_{models_json_mtime}")
    if not callable(getattr(shap, "shap_top_features_comment", None)):
        raise ValueError(f"{SHAP_MODULE}.py: нет shap_top_features_comment")

    pipelines: Dict[str, str] = {}
    modules: Dict[str, ModuleType] = {}
    thresholds: Dict[str, float] = {}
    versions: Dict[str, str] = {}
    models: Dict[str, Tuple[Any, Any]] = {}
    calibrators: Dict[str, Tuple[Any, Any]] = {}
    ts_country_by_brand: Dict[str, Dict[str, str]] = {}
    submodel_names: Dict[str, Dict[str, str]] = {}

    for stem, raw in read_json_object(models_dir, MODELS_JSON).items():
        if not isinstance(raw, list) or len(raw) != _MODELS_ENTRY_LEN:
            raise ValueError(
                f"{MODELS_JSON}[{stem!r}]: ожидается list длины {_MODELS_ENTRY_LEN} "
                f"[pipeline, threshold, version, cf|null, rg|null], получено {raw!r}"
            )
        pipeline_name, threshold, version, cf, rg = raw
        if not isinstance(pipeline_name, str) or not pipeline_name:
            raise ValueError(f"{MODELS_JSON}[{stem!r}]: пустое имя pipeline")
        if not (0.0 <= float(threshold) <= 1.0):
            raise ValueError(f"{MODELS_JSON}[{stem!r}]: threshold вне [0, 1]: {threshold}")
        if not version:
            raise ValueError(f"{MODELS_JSON}[{stem!r}]: пустая version")

        module = _load_pipeline_module(models_dir, pipeline_name, stem)

        if bool(cf) != bool(rg):
            raise ValueError(
                f"{MODELS_JSON}[{stem!r}]: cf и rg задаются вместе или оба должны быть null"
            )
        if cf and rg:
            calibrators[stem] = (
                _load_pickle(models_dir / cf),
                _load_pickle(models_dir / rg),
            )

        clf_model, rg_model = _load_model_pair(models_dir / f"{stem}.pickle")
        pipelines[stem] = pipeline_name
        modules[stem] = module
        thresholds[stem] = float(threshold)
        versions[stem] = str(version)
        models[stem] = (clf_model, rg_model)
        ts_country_by_brand[stem] = module.load_ts_country_by_brand(models_dir)
        submodel_names[stem] = {
            "clf": _model_name(clf_model),
            "rg": _model_name(rg_model),
        }

    loaded_at = datetime.now(timezone.utc).isoformat()
    registry = Registry(
        pipelines=pipelines,
        modules=modules,
        shap=shap,
        thresholds=thresholds,
        versions=versions,
        models=models,
        calibrators=calibrators,
        ts_country_by_brand=ts_country_by_brand,
        submodel_names=submodel_names,
        models_json_mtime=models_json_mtime,
        loaded_at=loaded_at,
    )
    logger.info(
        "registry loaded || models={} || calibrators={} || pipelines={}",
        len(registry.models),
        len(registry.calibrators),
        sorted(set(registry.pipelines.values())),
    )
    return registry


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

# reviewed
def _round2(values: List[float]) -> List[float]:
    return [round(v, 2) for v in values]

# reviewed
def _combine_output_matrices(clf_t: pd.DataFrame, reg_t: pd.DataFrame) -> pd.DataFrame:
    shared = [c for c in reg_t.columns if c in clf_t.columns]
    if not shared:
        return pd.concat([clf_t, reg_t], axis=1)
    reg_part = reg_t.rename(columns={c: f"{c}__rg" for c in shared})
    return pd.concat([clf_t, reg_part], axis=1)

# reviewed
def _preds_from_result(predict_out: Dict[str, Any]) -> List[float]:
    result_map = predict_out["result"]
    model_name = next(iter(result_map))
    series_dict = result_map[model_name]
    return [float(series_dict[k]) for k in sorted(series_dict)]

# reviewed
def _df_from_result(predict_out: Dict[str, Any]) -> pd.DataFrame:
    df_map = predict_out["df"]
    model_name = next(iter(df_map))
    return pd.DataFrame(df_map[model_name])

# reviewed
def _preprocess(
    registry: Registry,
    stem: str,
    features_values: List[Dict],
) -> pd.DataFrame:
    return registry.modules[stem].preprocess(
        features_values,
        registry.ts_country_by_brand[stem],
    )

# reviewed
def _http_500(exc: Exception, *, where: str) -> HTTPException:
    logger.exception(
        "{} failed || type={} || error={}",
        where,
        type(exc).__name__,
        exc,
    )
    return HTTPException(
        status_code=500,
        detail={"error": f"{type(exc).__name__}: {exc}"},
    )

# reviewed
async def score_track(
    registry: Registry,
    stem: str,
    features_values: List[Dict],
    *,
    async_mode: bool = True,
) -> Dict[str, Any]:
    """preprocess → ensemble_predict → postprocess → SHAP → df snapshots."""
    module = registry.modules[stem]
    df_pre = _preprocess(registry, stem, features_values)
    threshold = resolve_threshold(df_pre, registry.thresholds[stem])
    clf_model, rg_model = registry.models[stem]

    if async_mode:
        clf_out, rg_out = await asyncio.gather(
            ensemble_predict(stem, clf_model, df_pre),
            ensemble_predict(stem, rg_model, df_pre),
        )
    else:
        clf_out = await ensemble_predict(stem, clf_model, df_pre)
        rg_out = await ensemble_predict(stem, rg_model, df_pre)

    raw_clf = _preds_from_result(clf_out)
    raw_reg = _preds_from_result(rg_out)
    clf_t = _df_from_result(clf_out)
    reg_t = _df_from_result(rg_out)

    post = module.postprocess(
        df_pre,
        raw_clf,
        raw_reg,
        registry.calibrators.get(stem),
        threshold,
    )
    proba = post["proba"]
    labels = post["labels"]
    masked = post["regression"]

    cpm_comment = (
        registry.shap.shap_top_features_comment(
            clf_model["model"], clf_t, prefix="shap_clf"
        )
        + " | "
        + registry.shap.shap_top_features_comment(
            rg_model["model"], reg_t, prefix="shap_reg"
        )
    ).strip()

    output_outboxml = _combine_output_matrices(clf_t, reg_t)
    return {
        "usage_model": stem,
        "version": registry.versions[stem],
        "CPMComment": cpm_comment,
        "result": {
            "classification_proba": _round2(proba),
            "classification_predictions": _round2(labels),
            "regression_predictions": _round2(masked),
            "prediction": _round2(masked),
            "threshold": round(threshold, 2),
        },
        "df": {
            "input_raw": features_values,
            "output_preprocessed": df_to_records(df_pre),
            "output_outboxml": df_to_records(output_outboxml),
        },
    }

# reviewed
async def score_submodel(
    registry: Registry,
    stem: str,
    features_values: List[Dict],
    submodel: str,
) -> Dict[str, Any]:
    """Сырой результат одной модели (clf или rg) без калибровки/маски/SHAP."""
    names = registry.submodel_names[stem]
    name_to_model = {
        names["clf"]: registry.models[stem][0],
        names["rg"]: registry.models[stem][1],
    }
    if submodel not in name_to_model:
        raise HTTPException(
            status_code=422,
            detail=(
                f"submodel {submodel!r} нет у stem {stem!r}; "
                f"доступны: {sorted(name_to_model)}"
            ),
        )
    df_pre = _preprocess(registry, stem, features_values)
    predict_out = await ensemble_predict(stem, name_to_model[submodel], df_pre)
    return {
        "usage_model": stem,
        "version": registry.versions[stem],
        "submodel": submodel,
        "result": predict_out.get("result", {}),
        "version_model": predict_out.get("version_model", {}),
        "df": {
            "input_raw": features_values,
            "output_preprocessed": df_to_records(df_pre),
            "output_outboxml": df_to_records(_df_from_result(predict_out)),
        },
    }


# ---------------------------------------------------------------------------
# main_predict (сигнатура как example_1)
# ---------------------------------------------------------------------------

# reviewed
async def main_predict(
    config: Config,
    group_name: Optional[str],
    features_values: Union[List[Dict], pd.DataFrame],
    async_mode: bool = True,
    submodel: Optional[str] = None,
) -> Dict[str, Any]:
    # registry гарантирует Depends(_registry) на роуте
    registry = config.registry
    assert registry is not None
    if not group_name:
        raise HTTPException(
            status_code=422,
            detail="Нужно указать model — имя stem из models.json",
        )
    if group_name not in registry.pipelines:
        raise HTTPException(
            status_code=422,
            detail=(
                f"stem нет в models.json: {[group_name]}; "
                f"известны: {sorted(registry.pipelines)}"
            ),
        )
    if isinstance(features_values, pd.DataFrame):
        features = features_values.to_dict(orient="records")
    else:
        features = features_values
    if submodel:
        response = await score_submodel(registry, group_name, features, submodel)
    else:
        response = await score_track(
            registry, group_name, features, async_mode=async_mode
        )
    return {"main_response": response}


# ---------------------------------------------------------------------------
# Reload watcher
# ---------------------------------------------------------------------------

# reviewed
async def _watch_registry(interval: int) -> None:
    while True:
        await asyncio.sleep(max(1, interval))
        try:
            models_dir = _models_dir()
            models_json_path = models_dir / MODELS_JSON
            if not models_json_path.is_file():
                continue
            mtime = models_json_path.stat().st_mtime
            current = cnfg.registry
            if current is not None and mtime == current.models_json_mtime:
                continue
            logger.info("models.json changed || reloading registry")
            new_registry = await asyncio.to_thread(load_registry, models_dir)
            _set_registry(new_registry)
            logger.info("registry reloaded || loaded_at={}", new_registry.loaded_at)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            cnfg.last_reload_error = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "registry reload failed || keep old || error={}",
                cnfg.last_reload_error,
            )

# reviewed
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    _set_registry(load_registry(_models_dir()))
    watcher = asyncio.create_task(_watch_registry(cnfg.reload_interval))
    try:
        yield
    finally:
        watcher.cancel()
        try:
            await watcher
        except asyncio.CancelledError:
            pass


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

# reviewed
class PredictOneRequest(BaseModel):
    model: Optional[str] = None
    request: List[Dict[str, Any]] = Field(default_factory=list)
    submodel: Optional[str] = None

# reviewed
@app.get("/api/health")
def health_route(registry: Registry = Depends(_registry)):
    return JSONResponse(
        content={
            "health": True,
            "pipelines": registry.pipelines,
            "versions": registry.versions,
            "thresholds": registry.thresholds,
            "calibrators": sorted(registry.calibrators),
            "submodels": registry.submodel_names,
            "loaded_at": cnfg.loaded_at or registry.loaded_at,
            "last_reload_error": cnfg.last_reload_error,
        }
    )


# reviewed
@app.post("/api/predict_one")
async def predict_one_route(
    input_request: PredictOneRequest,
    _: Registry = Depends(_registry),
):
    try:
        body = await main_predict(
            config=cnfg,
            group_name=input_request.model,
            features_values=input_request.request,
            submodel=input_request.submodel,
            async_mode=True,
        )
        return JSONResponse(content=body, status_code=200)
    except HTTPException:
        raise
    except Exception as exc:
        raise _http_500(exc, where="PredictOne")

# reviewed
async def _fetch_predict_one(
    session: aiohttp.ClientSession,
    *,
    model: str,
    request: List[Dict[str, Any]],
    type_: str,
) -> Tuple[Dict[str, Any], str, int]:
    url = f"{cnfg.url_one.rstrip('/')}/api/predict_one"
    payload = {"model": model, "request": request, "submodel": None}
    async with session.post(url, json=payload) as response:
        data = await response.json(content_type=None)
        return data, type_, response.status


# reviewed
@app.post("/api/predict")
async def predict_route(
    service_request: ServiceRequest,
    registry: Registry = Depends(_registry),
):
    known = registry.pipelines
    unknown = [
        s
        for s in (service_request.main_model, service_request.second_model)
        if s and s not in known
    ]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"stem нет в models.json: {unknown}; известны: {sorted(known)}",
        )
    try:
        tasks = []
        async with aiohttp.ClientSession() as session:
            if service_request.main_model and service_request.main_request is not None:
                tasks.append(
                    _fetch_predict_one(
                        session,
                        model=service_request.main_model,
                        request=service_request.main_request,
                        type_="main",
                    )
                )
            if service_request.second_model and service_request.second_request is not None:
                tasks.append(
                    _fetch_predict_one(
                        session,
                        model=service_request.second_model,
                        request=service_request.second_request,
                        type_="second",
                    )
                )
            responses = await asyncio.gather(*tasks)

        output: Dict[str, Any] = {"main_response": {}, "second_response": {}}
        for data, type_, status_code in responses:
            if status_code != 200:
                raise HTTPException(status_code=status_code, detail=data)
            main_resp = data.get("main_response", data)
            output[f"{type_}_response"] = main_resp
        return JSONResponse(content=output, status_code=200)
    except HTTPException:
        raise
    except Exception as exc:
        raise _http_500(exc, where="Predict")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# reviewed
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Querulus scoring service")
    parser.add_argument(
        "--prod_models_path",
        default=_DEFAULT_MODELS_PATH,
        help="Каталог pickle и registry (models.json, pipelines/, Other_dicts.json)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--url_one",
        default="http://localhost:8080",
        help="Базовый URL для /api/predict → /api/predict_one",
    )
    parser.add_argument(
        "--reload_interval",
        type=int,
        default=30,
        help="Интервал проверки mtime models.json (сек)",
    )
    return parser.parse_args()

# reviewed
def main() -> None:
    args = parse_args()
    cnfg.prod_models_path = Path(args.prod_models_path)
    cnfg.url_one = args.url_one
    cnfg.reload_interval = args.reload_interval
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
