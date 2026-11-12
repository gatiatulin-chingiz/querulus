"""Flat FastAPI-сервис Querulus (как example_1).

Запуск: ``python main.py --prod_models_path ./results``
"""
from __future__ import annotations

import argparse
import pickle
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

from outboxml.core.predict import ensemble_predict
from outboxml.core.pydantic_models import ServiceRequest

from preprocessing import (
    df_to_records,
    load_ts_country_by_brand,
    preprocess,
    read_json_object,
    resolve_threshold,
)

from catboost import Pool  # type: ignore

MODELS_JSON = "models.json"
# stem → [pipeline, threshold, version, cf|null, rg|null]
_MODELS_ENTRY_LEN = 5

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Registry:
    """Всё состояние сервиса по моделям: ``models.json``, загруженные pickle, бренды."""

    pipelines: Dict[str, str]
    thresholds: Dict[str, float]
    versions: Dict[str, str]
    models: Dict[str, Tuple[Any, Any]]       # stem → (clf, rg)
    calibrators: Dict[str, Tuple[Any, Any]]  # v2-stem → (cf, rg)
    ts_country_by_brand: Dict[str, str]


class Config:
    def __init__(self) -> None:
        self.prod_models_path: Path = Path("./results")
        self.registry: Optional[Registry] = None


cnfg = Config()


def _registry() -> Registry:
    if cnfg.registry is None:
        raise RuntimeError("registry не загружен — сервис ещё не стартовал")
    return cnfg.registry


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info("pickle loaded || path={}", path)
    return obj


def _load_model_pair(path: Path) -> Tuple[Any, Any]:
    group = _load_pickle(path)
    if not isinstance(group, (list, tuple)) or len(group) < 2:
        raise ValueError(f"{path.name}: ожидается [clf, rg], получено {type(group).__name__}")
    for i, item in enumerate(group[:2]):
        if not isinstance(item, dict) or item.get("model") is None or "model_config" not in item:
            raise ValueError(f"{path.name}: элемент[{i}] без model/model_config")
    return group[0], group[1]

# reviewed
def load_registry(models_dir: Path) -> Registry:
    """Прочитать ``models.json`` и загрузить все pickle моделей/калибраторов."""
    pipelines: Dict[str, str] = {}
    thresholds: Dict[str, float] = {}
    versions: Dict[str, str] = {}
    models: Dict[str, Tuple[Any, Any]] = {}
    calibrators: Dict[str, Tuple[Any, Any]] = {}

    for stem, raw in read_json_object(models_dir, MODELS_JSON).items():
        if not isinstance(raw, list) or len(raw) != _MODELS_ENTRY_LEN:
            raise ValueError(
                f"{MODELS_JSON}[{stem!r}]: ожидается list длины {_MODELS_ENTRY_LEN} "
                f"[pipeline, threshold, version, cf|null, rg|null], получено {raw!r}"
            )
        kind, threshold, version, cf, rg = raw
        if kind not in ("legacy", "v2"):
            raise ValueError(f"{MODELS_JSON}[{stem!r}]: unknown pipeline {kind!r}")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"{MODELS_JSON}[{stem!r}]: threshold вне [0, 1]: {threshold}")
        if not version:
            raise ValueError(f"{MODELS_JSON}[{stem!r}]: пустая version")
        if kind == "v2":
            if not cf or not rg:
                raise ValueError(f"{MODELS_JSON}[{stem!r}]: для pipeline=v2 нужны cf и rg")
            calibrators[stem] = (_load_pickle(models_dir / cf), _load_pickle(models_dir / rg))

        pipelines[stem] = kind
        thresholds[stem] = threshold
        versions[stem] = version
        models[stem] = _load_model_pair(models_dir / f"{stem}.pickle")

    registry = Registry(
        pipelines=pipelines,
        thresholds=thresholds,
        versions=versions,
        models=models,
        calibrators=calibrators,
        ts_country_by_brand=load_ts_country_by_brand(models_dir),
    )
    logger.info(
        "registry loaded || models={} || calibrators={} || brands={}",
        len(registry.models),
        len(registry.calibrators),
        len(registry.ts_country_by_brand),
    )
    return registry

# reviewed
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    models_dir = cnfg.prod_models_path.resolve()
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Каталог моделей отсутствует: {models_dir}")
    cnfg.registry = load_registry(models_dir)
    yield

# reviewed
app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------


def _round2(values: List[float]) -> List[float]:
    return [round(v, 2) for v in values]


def apply_cf_calibrator(calibrator: Any, raw_scores: List[float]) -> List[float]:
    arr = np.asarray(
        calibrator.predict(np.asarray(raw_scores, dtype=float)), dtype=float
    )
    return [float(x) for x in arr.ravel()]


def apply_rg_calibrator(calibrator: Any, raw_reg: List[float]) -> List[float]:
    arr = np.asarray(
        calibrator.predict(np.asarray(raw_reg, dtype=float)), dtype=float
    )
    return [float(max(0.0, x)) for x in arr.ravel()]


def masked_predictions(
    labels: List[float],
    raw_reg: List[float],
    df_common: pd.DataFrame,
) -> List[float]:
    if "IS_LAWYER" in df_common.columns:
        is_lawyer = (
            pd.to_numeric(df_common["IS_LAWYER"], errors="coerce").fillna(0).eq(1)
        ).tolist()
    else:
        is_lawyer = [False] * len(labels)
    masked: List[float] = []
    for c, r, lawyer in zip(labels, raw_reg, is_lawyer):
        if c or lawyer:
            masked.append(r)
        else:
            masked.append(0.0)
    return masked


def unwrap_model(model: Any) -> Any:
    inner = model
    for _ in range(2):
        if hasattr(inner, "model"):
            inner = getattr(inner, "model")
        else:
            break
    return inner


def _catboost_shap_values(base_model: Any, t: pd.DataFrame) -> Any:
    feature_names = list(t.columns)
    model_feature_names = list(base_model.feature_names_)
    model_cat_idxs = list(base_model.get_cat_feature_indices())
    cat_names = [
        model_feature_names[i]
        for i in model_cat_idxs
        if i < len(model_feature_names)
    ]
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    cat_idxs = [name_to_idx[n] for n in cat_names if n in name_to_idx]

    df_pool = t.copy()
    for i, col in enumerate(feature_names):
        if i in cat_idxs:
            df_pool[col] = df_pool[col].astype("string").fillna("NA")
        else:
            df_pool[col] = pd.to_numeric(df_pool[col], errors="coerce").fillna(0.0)

    pool = Pool(df_pool, feature_names=feature_names, cat_features=cat_idxs)
    return base_model.get_feature_importance(pool, type="ShapValues")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-x)))


def shap_top_features_comment(model: Any, t: pd.DataFrame, *, prefix: str) -> str:
    top_n = 5
    try:
        feature_names = list(t.columns)
        n_with_bias = len(feature_names) + 1
        v = np.asarray(_catboost_shap_values(unwrap_model(model), t))
        if v.ndim == 3:
            class_idx = 1 if v.shape[1] == n_with_bias and v.shape[2] > 1 else 0
            v = v[:, :, class_idx]
        if v.ndim != 2 or v.shape[0] < 1:
            return f"{prefix}: SHAP empty"

        proba_prefix = ""
        base_raw: Optional[float] = None
        if v.shape[1] == n_with_bias:
            if prefix.startswith("shap_clf"):
                base_raw = float(v[0, -1])
                p = _sigmoid(base_raw + float(np.sum(v[0, :-1])))
                proba_prefix = f"p={p:.4f} | "
            v = v[:, :-1]
        row = v[0]

        if base_raw is not None:
            base_p = _sigmoid(base_raw)
            rel = np.array([(_sigmoid(base_raw + float(c)) / base_p - 1.0) * 100.0 for c in row])
            idx = np.argsort(np.abs(rel))[::-1][:top_n]
            parts = [f"{feature_names[i]}={rel[i]:+.2f}%" for i in idx]
        else:
            idx = np.argsort(np.abs(row))[::-1][:top_n]
            parts = [f"{feature_names[i]}={row[i]:+.4f}" for i in idx]
        return f"{prefix}: " + proba_prefix + ", ".join(parts)
    except Exception as e:
        msg = str(e).replace("\n", " ").strip()
        if len(msg) > 120:
            msg = msg[:120] + "..."
        return f"{prefix}: SHAP error ({type(e).__name__}: {msg})"


def _combine_output_matrices(clf_t: pd.DataFrame, reg_t: pd.DataFrame) -> pd.DataFrame:
    shared = [c for c in reg_t.columns if c in clf_t.columns]
    if not shared:
        return pd.concat([clf_t, reg_t], axis=1)
    reg_part = reg_t.rename(columns={c: f"{c}__rg" for c in shared})
    return pd.concat([clf_t, reg_part], axis=1)


def _preds_from_result(predict_out: Dict[str, Any]) -> List[float]:
    result_map = predict_out["result"]
    model_name = next(iter(result_map))
    series_dict = result_map[model_name]
    return [float(series_dict[k]) for k in sorted(series_dict, key=lambda x: (str(type(x)), x))]


def _df_from_result(predict_out: Dict[str, Any]) -> pd.DataFrame:
    df_map = predict_out["df"]
    model_name = next(iter(df_map))
    return pd.DataFrame(df_map[model_name])


# ---------------------------------------------------------------------------
# Score one track
# ---------------------------------------------------------------------------


async def score_track(stem: str, features_values: List[Dict]) -> Dict[str, Any]:
    """preprocess → ensemble_predict → (cal if v2) → mask → SHAP → df snapshots."""
    registry = _registry()
    kind = registry.pipelines[stem]
    df_pre = preprocess(features_values, registry.ts_country_by_brand, kind)
    threshold = resolve_threshold(df_pre, registry.thresholds[stem])

    clf_model, rg_model = registry.models[stem]

    clf_out = await ensemble_predict(stem, clf_model, df_pre)
    rg_out = await ensemble_predict(stem, rg_model, df_pre)

    raw_clf = _preds_from_result(clf_out)
    raw_reg = _preds_from_result(rg_out)
    clf_t = _df_from_result(clf_out)
    reg_t = _df_from_result(rg_out)

    if kind == "v2":
        cf_cal, rg_cal = registry.calibrators[stem]
        proba = apply_cf_calibrator(cf_cal, raw_clf)
        cal_reg = apply_rg_calibrator(rg_cal, raw_reg)
    else:
        proba = raw_clf
        cal_reg = raw_reg

    labels = [1.0 if s >= threshold else 0.0 for s in proba]
    masked = masked_predictions(labels, cal_reg, df_pre)

    cpm_comment = (
        shap_top_features_comment(clf_model["model"], clf_t, prefix="shap_clf")
        + " | "
        + shap_top_features_comment(rg_model["model"], reg_t, prefix="shap_reg")
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


async def main_predict(
    group_name: str,
    features_values: List[Dict],
    second_group_name: Optional[str] = None,
    second_features_values: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    main_response = await score_track(group_name, features_values)
    second_response: Dict[str, Any] = {}
    if second_group_name and second_features_values:
        second_response = await score_track(second_group_name, second_features_values)
    return {
        "main_response": main_response,
        "second_response": second_response,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

# reviewed
@app.get("/api/health")
def health_route():
    registry = _registry()
    return JSONResponse(
        content={
            "health": True,
            "pipelines": registry.pipelines,
            "versions": registry.versions,
            "thresholds": registry.thresholds,
            "calibrators": sorted(registry.calibrators),
        }
    )

# reviewed
@app.post("/api/predict")
async def predict_route(service_request: ServiceRequest):
    known = _registry().pipelines
    unknown = [
        s for s in (service_request.main_model, service_request.second_model)
        if s and s not in known
    ]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"stem нет в models.json: {unknown}; известны: {sorted(known)}",
        )
    try:
        body = await main_predict(
            service_request.main_model,
            service_request.main_request,
            service_request.second_model,
            service_request.second_request,
        )
        return JSONResponse(content=body, status_code=200)
    except Exception as exc:
        logger.exception("Predict failed || type={} || error={}", type(exc).__name__, exc)
        raise HTTPException(
            status_code=500,
            detail={"error": f"{type(exc).__name__}: {exc}"},
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# reviewed
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Querulus scoring service")
    parser.add_argument(
        "--prod_models_path",
        default="./results",
        help="Каталог pickle и registry JSON (models.json, Other_dicts.json)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()

# reviewed
if __name__ == "__main__":
    args = parse_args()
    cnfg.prod_models_path = Path(args.prod_models_path)
    uvicorn.run(app, host=args.host, port=args.port)
