"""Скоринг second_ (модель 2.0.0): FE_* → predict → калибраторы → τ cal → IS_LAWYER."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    from features.derived import add_second_model_features
    from scoring.profiles import (
        feature_names_from_model_result,
        load_optional_pickle,
        second_cf_calibrator_path,
        second_meta_path,
        second_pickle_path,
        second_rg_calibrator_path,
        threshold_from_meta,
    )
    from service.masking import masked_predictions
except ImportError:
    from integration.features.derived import add_second_model_features
    from integration.scoring.profiles import (
        feature_names_from_model_result,
        load_optional_pickle,
        second_cf_calibrator_path,
        second_meta_path,
        second_pickle_path,
        second_rg_calibrator_path,
        threshold_from_meta,
    )
    from integration.service.masking import masked_predictions

_APPLICANT_FORM_REMAP_UPPER: dict[str, str] = {
    "СКРЫТЫЙ ЮРИСТ": "ЮРИСТ С ПОТЕРПЕВШИМ",
    "ПРЕДСТАВИТЕЛЬ (АВТОЮРИСТ)": "ПРЕДСТАВИТЕЛЬ (ПО ДОВЕРЕННОСТИ)",
    "ПРЕДСТАВИТЕЛЬ (НЕ АВТОЮРИСТ)": "ВЫГОДОПРИОБРЕТАТЕЛЬ",
}

DEFAULT_THRESHOLD_FALLBACK = 0.6


#reviewed
def prepare_vector_upper(features_values: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(features_values)
    for column in df.columns:
        df[column] = df[column].map(
            lambda v: np.nan if isinstance(v, str) and v == "" else v
        )
        s = df[column]
        if s.dtype == object and s.map(lambda v: isinstance(v, str)).any():
            df[column] = s.map(lambda v: v.upper() if isinstance(v, str) else v)
    return df


#reviewed
def enrich_dates_and_forms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    event_dt = pd.to_datetime(out["EVENT_DATE"], errors="coerce")
    loss_dt = pd.to_datetime(out["PAYMENT_ORDER_DATE_TIME"], errors="coerce")
    out["EVENT_YEAR"] = event_dt.dt.year
    out["APPLY_DELAY"] = (
        loss_dt.dt.normalize() - event_dt.dt.normalize()
    ).dt.days.fillna(-1).astype(int)

    if "APPLICANT_FORM" in out.columns:
        out["APPLICANT_FORM"] = out["APPLICANT_FORM"].map(
            lambda x: _APPLICANT_FORM_REMAP_UPPER.get(x, x)
            if isinstance(x, str)
            else x
        )
    return out


#new
def preprocess_second_vector(features_values: List[Dict]) -> pd.DataFrame:
    """UPPER → dates/forms → FE_* (CPI REAL + derived). Clip — в prepare_dataset из JSON."""
    df = prepare_vector_upper(features_values)
    df = enrich_dates_and_forms(df)
    if df["EVENT_DATE"].isna().any() or df["PAYMENT_ORDER_DATE_TIME"].isna().any():
        raise ValueError("Second: EVENT_DATE or PAYMENT_ORDER_DATE_TIME is NaN")
    event_d = pd.to_datetime(df["EVENT_DATE"], errors="coerce").dt.normalize()
    loss_d = pd.to_datetime(df["PAYMENT_ORDER_DATE_TIME"], errors="coerce").dt.normalize()
    if (event_d > loss_d).any():
        raise ValueError("Second: EVENT_DATE > PAYMENT_ORDER_DATE_TIME")
    return add_second_model_features(df)


#reviewed
def _load_group(group_name: str) -> Any:
    path = second_pickle_path(group_name)
    if not path.is_file():
        logger.error("Second pickle не найден || path={}", str(path))
        raise FileNotFoundError(f"Second pickle не найден: {path}")
    with open(path, "rb") as f:
        group = pickle_load(f)
    logger.info("Second pickle loaded || model={} || path={}", group_name, str(path))
    return group


#new
def pickle_load(f):
    import pickle

    return pickle.load(f)


#reviewed
def _prepare_matrix(model_result: Dict[str, Any], df: pd.DataFrame) -> tuple[Any, pd.DataFrame]:
    from mldataworker.core.data_prepare import prepare_dataset
    from mldataworker.core.pydantic_models import ModelConfig

    cfg = ModelConfig.model_validate(model_result["model_config"])
    model = model_result["model"]
    matrix = prepare_dataset(
        cfg.name,
        df,
        train_ind=df.index,
        test_ind=pd.Index([]),
        model_config=cfg,
        check_prepared=False,
        calc_corr=False,
        save_data=False,
        log=False,
    ).data
    return model, matrix


#reviewed
def _require_columns(df: pd.DataFrame, columns: List[str], role: str) -> pd.DataFrame:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Second: в векторе нет полей для {role}: {missing}")
    return df.loc[:, columns]


#reviewed
def _unwrap_model(model: Any) -> Any:
    if model is None:
        return model
    if hasattr(model, "predict") and hasattr(model, "model"):
        return model
    current = model
    for _ in range(4):
        if current is None:
            break
        if hasattr(current, "predict_proba") or (
            hasattr(current, "predict") and not hasattr(current, "model")
        ):
            return current
        nested = getattr(current, "model", None)
        if nested is None or nested is current:
            break
        current = nested
    return model


#new
def _raw_classification_scores(model: Any, t: pd.DataFrame) -> List[float]:
    m = _unwrap_model(model)
    if hasattr(m, "model") and hasattr(m, "predict"):
        return [float(x) for x in np.asarray(m.predict(t), dtype=float).ravel()]
    if hasattr(m, "predict_proba"):
        proba = m.predict_proba(t)
        return [float(row[1]) for row in np.asarray(proba)]
    pred = m.predict(t)
    return [float(x) for x in np.asarray(pred).ravel()]


#new
def apply_cf_calibrator(
    calibrator: Any | None,
    raw_scores: List[float],
    prepared_matrix: pd.DataFrame,
) -> List[float]:
    """Откалибровать CF proba. Поддержка CCCV / Isotonic / predict_proba(X)."""
    if calibrator is None:
        return list(raw_scores)
    if hasattr(calibrator, "predict_proba"):
        try:
            proba = calibrator.predict_proba(prepared_matrix)
            return [float(row[1]) for row in np.asarray(proba)]
        except Exception:
            logger.exception("CF calibrator.predict_proba failed — fallback isotonic/raw")
    if hasattr(calibrator, "predict"):
        arr = np.asarray(calibrator.predict(np.asarray(raw_scores, dtype=float)), dtype=float)
        return [float(x) for x in arr.ravel()]
    logger.warning("CF calibrator без predict/predict_proba — raw scores")
    return list(raw_scores)


#new
def apply_rg_calibrator(
    calibrator: Any | None,
    raw_reg: List[float],
) -> List[float]:
    if calibrator is None:
        return list(raw_reg)
    if hasattr(calibrator, "predict"):
        arr = np.asarray(calibrator.predict(np.asarray(raw_reg, dtype=float)), dtype=float)
        return [float(max(0.0, x)) for x in arr.ravel()]
    logger.warning("RG calibrator без predict — raw")
    return list(raw_reg)


#reviewed
def resolve_second_threshold(df_common: pd.DataFrame, group_name: str) -> float:
    if "THRESHOLD" in df_common.columns:
        value = pd.to_numeric(df_common["THRESHOLD"].iloc[0], errors="coerce")
        if not pd.isna(value):
            return float(value)
        logger.warning("Second: THRESHOLD в векторе битый — берём meta/default")
    meta_thr = threshold_from_meta(second_meta_path(group_name))
    if meta_thr is not None:
        return float(meta_thr)
    logger.warning(
        "Second: нет best_threshold в meta — fallback {}",
        DEFAULT_THRESHOLD_FALLBACK,
    )
    return float(DEFAULT_THRESHOLD_FALLBACK)


#reviewed
def _round_numbers(values: Optional[List[float]], ndigits: int = 2) -> Optional[List[float]]:
    if values is None:
        return None
    return [round(float(v), ndigits) for v in values]


#reviewed
def json_loads_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = json.loads(df.to_json(orient="records", force_ascii=False))
    return [{k: row[k] for k in sorted(row.keys())} for row in rows]


#new
def score_second(
    group_name: str,
    features_values: List[Dict],
) -> Dict[str, Any]:
    """Полный скоринг 2.0.0 → структура как main_response."""
    group = _load_group(group_name)
    clf_model, rg_model = group[0], group[1]
    df_common = preprocess_second_vector(features_values)
    threshold_value = resolve_second_threshold(df_common, group_name)

    clf_feats = feature_names_from_model_result(clf_model)
    rg_feats = feature_names_from_model_result(rg_model)
    if not clf_feats or not rg_feats:
        raise ValueError("Second: пустой список features в model_config pickle")

    clf_df = _require_columns(df_common, clf_feats, "Классификации(v2)")
    rg_df = _require_columns(df_common, rg_feats, "Регрессии(v2)")

    model_c, t_c = _prepare_matrix(clf_model, clf_df)
    raw_proba = _raw_classification_scores(model_c, t_c)
    cf_cal = load_optional_pickle(second_cf_calibrator_path())
    proba = apply_cf_calibrator(cf_cal, raw_proba, t_c)
    labels = [1.0 if s >= threshold_value else 0.0 for s in proba]

    model_r, t_r = _prepare_matrix(rg_model, rg_df)
    raw_reg = [float(x) for x in np.asarray(model_r.predict(t_r)).ravel()]
    rg_cal = load_optional_pickle(second_rg_calibrator_path())
    cal_reg = apply_rg_calibrator(rg_cal, raw_reg)
    masked = masked_predictions(labels, cal_reg, df_common)

    rounded_proba = _round_numbers(proba)
    rounded_labels = _round_numbers(labels)
    rounded_masked = _round_numbers(masked)

    prediction_metrics = {
        "classification_proba": rounded_proba,
        "classification_predictions": rounded_labels,
        "regression_predictions": rounded_masked,
    }
    result_body = {
        **prediction_metrics,
        "prediction": rounded_masked,
        "threshold": round(float(threshold_value), 2),
    }
    output_df = pd.concat([t_c, t_r], axis=1)
    if output_df.columns.duplicated().any():
        output_df = output_df.loc[:, ~output_df.columns.duplicated()]

    return {
        "usage_model": group_name,
        "result": result_body,
        "df": {
            "input": json_loads_records(df_common),
            "output": json_loads_records(output_df),
        },
    }
