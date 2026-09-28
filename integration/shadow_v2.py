"""Shadow scoring 2.0.0 для second_response (без импорта querulus.src).

# CUTOVER: после 2 недель перенести этот пайплайн в main_ и удалить legacy.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    from features_cpi import add_real_monetary_columns
    from features_dq import apply_frozen_dq_from_path
    from model_profiles import (
        feature_names_from_model_result,
        shadow_dq_bounds_path,
        shadow_meta_path,
        shadow_pickle_path,
        threshold_from_meta,
    )
except ImportError:  # python -m integration / unittest package path
    from integration.features_cpi import add_real_monetary_columns
    from integration.features_dq import apply_frozen_dq_from_path
    from integration.model_profiles import (
        feature_names_from_model_result,
        shadow_dq_bounds_path,
        shadow_meta_path,
        shadow_pickle_path,
        threshold_from_meta,
    )

# Remap после UPPER() — ключи и значения в верхнем регистре (как replace в JSON моделей).
_APPLICANT_FORM_REMAP_UPPER: dict[str, str] = {
    "СКРЫТЫЙ ЮРИСТ": "ЮРИСТ С ПОТЕРПЕВШИМ",
    "ПРЕДСТАВИТЕЛЬ (АВТОЮРИСТ)": "ПРЕДСТАВИТЕЛЬ (ПО ДОВЕРЕННОСТИ)",
    "ПРЕДСТАВИТЕЛЬ (НЕ АВТОЮРИСТ)": "ВЫГОДОПРИОБРЕТАТЕЛЬ",
}

DEFAULT_THRESHOLD_FALLBACK = 0.6


def _prepare_vector_upper(features_values: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(features_values)
    for column in df.columns:
        df[column] = df[column].map(
            lambda v: np.nan if isinstance(v, str) and v == "" else v
        )
        s = df[column]
        if s.dtype == object and s.map(lambda v: isinstance(v, str)).any():
            df[column] = s.map(lambda v: v.upper() if isinstance(v, str) else v)
    return df


def _enrich_dates_and_forms(df: pd.DataFrame) -> pd.DataFrame:
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


def preprocess_shadow_vector(features_values: List[Dict]) -> pd.DataFrame:
    """UPPER → dates/forms → CPI REAL → frozen DQ."""
    df = _prepare_vector_upper(features_values)
    df = _enrich_dates_and_forms(df)
    if df["EVENT_DATE"].isna().any() or df["PAYMENT_ORDER_DATE_TIME"].isna().any():
        raise ValueError("Shadow: EVENT_DATE or PAYMENT_ORDER_DATE_TIME is NaN")
    event_d = pd.to_datetime(df["EVENT_DATE"], errors="coerce").dt.normalize()
    loss_d = pd.to_datetime(df["PAYMENT_ORDER_DATE_TIME"], errors="coerce").dt.normalize()
    if (event_d > loss_d).any():
        raise ValueError("Shadow: EVENT_DATE > PAYMENT_ORDER_DATE_TIME")

    df = add_real_monetary_columns(df, event_date_column="EVENT_DATE")
    df = apply_frozen_dq_from_path(df, shadow_dq_bounds_path())
    return df


def _load_group(group_name: str) -> Any:
    path = shadow_pickle_path(group_name)
    if not path.is_file():
        logger.error("Shadow pickle не найден || path={}", str(path))
    with open(path, "rb") as f:
        group = pickle.load(f)
    logger.info("Shadow pickle loaded || model={} || path={}", group_name, str(path))
    return group


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


def _require_columns(df: pd.DataFrame, columns: List[str], role: str) -> pd.DataFrame:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Shadow: в векторе нет полей для {role}: {missing}")
    return df.loc[:, columns]


def _unwrap_model(model: Any) -> Any:
    """Достать оценку только если сверху нет своего predict (как у GLM-обёртки)."""
    if model is None:
        return model
    # OutBoxML GLMCatboostCombineModel / CatboostOverGLM: predict сам готовит вход.
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


def _classification_proba_and_labels(
    model: Any, t: pd.DataFrame, threshold: float
) -> tuple[Optional[List[float]], List[float]]:
    m = _unwrap_model(model)
    # Обёртки OutBoxML: predict → P(class=1); сырой CatBoost — predict_proba.
    if hasattr(m, "model") and hasattr(m, "predict"):
        scores_arr = np.asarray(m.predict(t), dtype=float).ravel()
        scores = [float(x) for x in scores_arr]
        labels = [1.0 if s >= threshold else 0.0 for s in scores]
        return scores, labels
    if hasattr(m, "predict_proba"):
        proba = m.predict_proba(t)
        scores = [float(row[1]) for row in np.asarray(proba)]
        labels = [1.0 if s >= threshold else 0.0 for s in scores]
        return scores, labels
    pred = m.predict(t)
    scores_list = [float(x) for x in np.asarray(pred).ravel()]
    labels = [1.0 if s >= threshold else 0.0 for s in scores_list]
    return scores_list, labels


def _masked_predictions(
    labels: List[float],
    raw_reg: List[float],
    df_common: pd.DataFrame,
) -> List[float]:
    if "IS_LAWYER" in df_common.columns:
        is_lawyer = (
            pd.to_numeric(df_common["IS_LAWYER"], errors="coerce")
            .fillna(0)
            .eq(1)
            .tolist()
        )
    else:
        is_lawyer = [False] * len(labels)
    masked: List[float] = []
    for c, r, lawyer in zip(labels, raw_reg, is_lawyer):
        reg = float(r)
        if c or lawyer:
            masked.append(reg)
        else:
            masked.append(0.0)
    return masked


def _round_numbers(values: Optional[List[float]], ndigits: int = 2) -> Optional[List[float]]:
    if values is None:
        return None
    return [round(float(v), ndigits) for v in values]


def _df_to_json(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = json_loads_records(df)
    return rows


def json_loads_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    import json

    rows = json.loads(df.to_json(orient="records", force_ascii=False))
    return [{k: row[k] for k in sorted(row.keys())} for row in rows]


def resolve_shadow_threshold(df_common: pd.DataFrame, group_name: str) -> float:
    if "THRESHOLD" in df_common.columns:
        value = pd.to_numeric(df_common["THRESHOLD"].iloc[0], errors="coerce")
        if not pd.isna(value):
            return float(value)
        logger.warning("Shadow: THRESHOLD в векторе битый — берём meta/default")
    meta_thr = threshold_from_meta(shadow_meta_path(group_name))
    if meta_thr is not None:
        return float(meta_thr)
    logger.warning(
        "Shadow: нет best_threshold в meta — fallback {}",
        DEFAULT_THRESHOLD_FALLBACK,
    )
    return float(DEFAULT_THRESHOLD_FALLBACK)


def score_shadow_v2(
    group_name: str,
    features_values: List[Dict],
) -> Dict[str, Any]:
    """Полный скоринг 2.0.0 → структура как main_response."""
    group = _load_group(group_name)
    clf_model, rg_model = group[0], group[1]
    df_common = preprocess_shadow_vector(features_values)
    threshold_value = resolve_shadow_threshold(df_common, group_name)

    clf_feats = feature_names_from_model_result(clf_model)
    rg_feats = feature_names_from_model_result(rg_model)
    if not clf_feats or not rg_feats:
        raise ValueError("Shadow: пустой список features в model_config pickle")

    clf_df = _require_columns(df_common, clf_feats, "Классификации(v2)")
    rg_df = _require_columns(df_common, rg_feats, "Регрессии(v2)")

    model_c, t_c = _prepare_matrix(clf_model, clf_df)
    proba, labels = _classification_proba_and_labels(model_c, t_c, threshold_value)

    model_r, t_r = _prepare_matrix(rg_model, rg_df)
    raw_reg = [float(x) for x in np.asarray(model_r.predict(t_r)).ravel()]
    masked = _masked_predictions(labels, raw_reg, df_common)

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
