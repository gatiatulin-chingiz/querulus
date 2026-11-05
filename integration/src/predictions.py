"""Скоринг main_ и second_: predict → калибровка (second) → маска → ответ.

Публичный API: ``score_main``, ``score_second``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    from integration import config
    from integration.src.calibration import apply_cf_calibrator, apply_rg_calibrator
    from integration.src.explain import shap_top_features_comment
    from integration.src.is_lawyer import masked_predictions
    from integration.src.models import (
        ArtifactCache,
        calibrator_paths,
        feature_names_from_model_result,
        model_pickle_path,
    )
    from integration.src.preprocessing.common import (
        df_to_records,
        prepare_matrix,
        preprocess_input_vector,
        require_columns,
        resolve_threshold,
    )
    from integration.src.preprocessing.second import preprocess_second_vector
except ImportError:
    import config
    from src.calibration import apply_cf_calibrator, apply_rg_calibrator
    from src.explain import shap_top_features_comment
    from src.is_lawyer import masked_predictions
    from src.models import (
        ArtifactCache,
        calibrator_paths,
        feature_names_from_model_result,
        model_pickle_path,
    )
    from src.preprocessing.common import (
        df_to_records,
        prepare_matrix,
        preprocess_input_vector,
        require_columns,
        resolve_threshold,
    )
    from src.preprocessing.second import preprocess_second_vector

# Состав и порядок колонок main_ (не берём из pickle).
CLASSIFICATION_FEATURES: List[str] = [
    "FILIAL",
    "EVENT_CREATED_BY_GIBDD_FLAG",
    "VICTIM_MAX_WEIGHT",
    "RECIEVE_METHOD",
    "APPLICANT_FORM",
    "APPLICANT_AGE",
    "VICTIM_VEHICLE_CATEGORY",
    "GUILTY_CAPACITY_ENGINE",
    "VICTIM_VEHICLE_AGE",
    "EVENT_YEAR",
]

REGRESSION_FEATURES: List[str] = [
    "AMOUNT_REPAIR",
    "LOSS_UNIT_ZONE",
    "VICTIM_VEHICLE_COUNTRY",
    "APPLY_DELAY",
]

# reviewed
_CACHE = ArtifactCache()


# reviewed
def _load_model_pair(stem: str, track: str) -> Any:
    """Загрузить pickle трека и проверить пару [clf, rg] с model_config/model."""
    group = _CACHE.get(model_pickle_path(stem))
    if not isinstance(group, (list, tuple)) or len(group) < 2:
        raise ValueError(
            f"{track}: pickle должен содержать [clf, rg], получено {type(group).__name__}"
        )
    for i, item in enumerate(group[:2]):
        if not isinstance(item, dict):
            raise ValueError(
                f"{track}: элемент[{i}] должен быть dict, получено {type(item).__name__}"
            )
        if "model_config" not in item or "model" not in item:
            raise ValueError(
                f"{track}: элемент[{i}] без ключей model_config/model"
            )
        if item["model"] is None:
            raise ValueError(f"{track}: элемент[{i}].model is None")
        cfg = item["model_config"]
        if cfg is None or (isinstance(cfg, dict) and not cfg):
            raise ValueError(f"{track}: элемент[{i}].model_config пуст")
    return group


# reviewed
def preload_models() -> None:
    """Загрузить 4 модели и оба калибратора до приёма запросов.

    Нет файла, пустой pickle или битая пара — процесс не поднимается.
    """
    main_stem = str(config.main_model_stem).strip()
    second_stem = str(config.second_model_stem).strip()
    if not main_stem or not second_stem:
        raise RuntimeError("MAIN_MODEL_STEM и SECOND_MODEL_STEM обязательны")
    _load_model_pair(main_stem, "main")
    _load_model_pair(second_stem, "second")
    # Нет файла / пустой pickle — ArtifactCache.get падает здесь же.
    cf_path, rg_path = calibrator_paths()
    _CACHE.get(cf_path)
    _CACHE.get(rg_path)
    logger.info(
        "Models preloaded || main={} || second={} || cf={} || rg={}",
        main_stem,
        second_stem,
        cf_path.name,
        rg_path.name,
    )


# reviewed
def _classification_proba_and_labels(
    model: Any, t: pd.DataFrame, threshold: float
) -> tuple[Optional[List[float]], List[float]]:
    """
    Считает список вероятностей (если доступно) и бинарные метки 0.0/1.0 по порогу.

    Вход:
        model: модель классификации;
        t: признаки после prepare_dataset;
        threshold: порог сравнения с вероятностью класса 1.

    Возвращает:
        tuple:
            - первый элемент: список вероятностей P(класс 1) или None, если модель отдаёт
              только жёсткие метки без proba;
            - второй элемент: список меток после порога (float 0/1) либо сырые predict,
              если это не вероятности.
    """
    scores = model.predict(t)
    # `predict()` может вернуть numpy array или pandas Series — приводим к JSON-совместимому list[float].
    if isinstance(scores, pd.Series):
        scores_list = scores.astype(float).tolist()
    elif isinstance(scores, np.ndarray):
        scores_list = scores.astype(float).tolist()
    elif isinstance(scores, list):
        scores_list = [float(x) for x in scores]
    else:
        scores_list = [float(scores)]

    labels = [1.0 if s >= threshold else 0.0 for s in scores_list]
    return scores_list, labels


# reviewed
def _round_numbers(obj: Any) -> Any:
    """Рекурсивно приводит и округляет числа для JSON до 2 знаков.

    Вход:
        obj: скаляр, список, dict, ndarray/Series или None.

    Возвращает:
        list/dict/float с округлёнными значениями; int/str/bool без изменений;
        ndarray/Series → ``list[float]``.
    """
    if obj is None:
        return None
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), 2)
    if isinstance(obj, (pd.Series, np.ndarray)):
        return [
            _round_numbers(x)
            for x in np.asarray(obj, dtype=float).ravel().tolist()
        ]
    if isinstance(obj, list):
        return [_round_numbers(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _round_numbers(v) for k, v in obj.items()}
    return obj


# reviewed
def run_classification_model(
    model_result: Dict[str, Any],
    df: pd.DataFrame,
    threshold: float,
) -> Dict[str, Any]:
    """
    Классификация: prepare_dataset → proba/метки.

    Возвращает:
        dict: "labels" — метки после порога; "proba" — P(класс 1) или None;
        "t" — матрица после prepare_dataset.
    """
    try:
        # Готовим матрицу и считаем proba с метками.
        model, t = prepare_matrix(model_result, df)
        proba_list, labels = _classification_proba_and_labels(model, t, threshold)
        return {"labels": labels, "proba": proba_list, "t": t}
    except Exception:
        logger.exception(
            "run_classification_model: ошибка || df_shape={} || threshold={}",
            getattr(df, "shape", None),
            threshold,
        )
        raise


# reviewed
def run_regression_model(
    model_result: Dict[str, Any],
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Регрессия: prepare_dataset → ``model.predict`` (маска — в ``score_main``).

    Возвращает:
        dict: "raw_predictions" — вывод регрессии по строкам;
        "t" — матрица после prepare_dataset.
    """
    try:
        model, t = prepare_matrix(model_result, df)
        raw = _round_numbers(model.predict(t))
        return {"raw_predictions": raw, "t": t}
    except Exception:
        logger.exception(
            "run_regression_model: ошибка || df_shape={}",
            getattr(df, "shape", None),
        )
        raise


# reviewed
def score_main(group_name: str, features_values: List[Dict]) -> Dict[str, Any]:
    """
    Полный скоринг main_ → ``main_response``.

    Вход:
        group_name: stem ``*.pickle`` (классификация + регрессия);
        features_values: вектор признаков (список объектов).

    Возвращает:
        dict: ``usage_model``, ``version``, ``CPMComment``, ``result``, ``df``.
        В ``result``: classification_proba / classification_predictions /
        regression_predictions / prediction / threshold.
    """
    # Грузим пару clf+rg через общий кэш.
    try:
        group = _load_model_pair(group_name, "main")
        clf_model, rg_model = group[0], group[1]
    except Exception:
        logger.exception("score_main: загрузка группы моделей || group={}", group_name)
        raise

    # Общий вход: UPPER → касты/derived → validate.
    try:
        df_common = preprocess_input_vector(features_values)
    except Exception:
        logger.exception("score_main: входной вектор некорректен")
        raise

    # Берём порог из вектора или дефолт main_.
    threshold_value = resolve_threshold(df_common, config.default_main_threshold)

    # Считаем классификацию.
    clf_df = require_columns(df_common, CLASSIFICATION_FEATURES, "Классификации")
    clf = run_classification_model(clf_model, clf_df, threshold_value)
    labels = clf["labels"]
    proba = clf["proba"]
    clf_t = clf["t"]

    # SHAP: unwrap внутри shap_top_features_comment.
    cpm_comment = shap_top_features_comment(
        clf_model["model"],
        clf_t,
        prefix="shap_clf",
    )

    # Считаем регрессию и дописываем SHAP.
    reg_df = require_columns(df_common, REGRESSION_FEATURES, "Регрессии")
    reg = run_regression_model(rg_model, reg_df)
    raw_reg = reg["raw_predictions"]
    reg_t = reg["t"]
    cpm_comment = (
        cpm_comment
        + " | "
        + shap_top_features_comment(
            rg_model["model"],
            reg_t,
            prefix="shap_reg",
        )
    ).strip()

    if len(labels) != len(raw_reg):
        logger.error(
            "score_main: несовпадение длин clf/reg || len_labels={} || len_raw_reg={}",
            len(labels),
            len(raw_reg),
        )

    # Обнуляем регрессию, если класс 0 и заявитель не юрист.
    masked_reg = masked_predictions(labels, raw_reg, df_common)

    # Округляем числа ответа до 2 знаков.
    rounded_masked_reg = _round_numbers(masked_reg)
    result_body: Dict[str, Any] = {
        "classification_proba": _round_numbers(proba),
        "classification_predictions": _round_numbers(labels),
        "regression_predictions": rounded_masked_reg,
        "prediction": rounded_masked_reg,
        "threshold": round(float(threshold_value), 2),
    }

    # Отладочная таблица «что реально подали в модели» (после всех трансформаций).
    output_df = pd.concat([clf_t, reg_t], axis=1)
    if output_df.columns.duplicated().any():
        dupes = output_df.columns[output_df.columns.duplicated()].tolist()
        raise ValueError(f"Дубли имён колонок в output после объединения clf/reg: {dupes}")

    # Собираем ответ main_.
    return {
        "usage_model": group_name,
        "version": config.main_version,
        "CPMComment": cpm_comment,
        "result": result_body,
        "df": {
            "input": df_to_records(df_common),
            "output": df_to_records(output_df),
        },
    }


# reviewed
def _raw_classification_scores(model: Any, t: pd.DataFrame) -> List[float]:
    """Сырые скоры CF до калибратора: ``model.predict(t)`` (контракт pickle).

    Args:
        model: модель классификации после prepare_matrix.
        t: матрица признаков.

    Returns:
        list[float] скоров по строкам.
    """
    return [float(x) for x in np.asarray(model.predict(t), dtype=float).ravel()]


# reviewed
def score_second(
    group_name: str,
    features_values: List[Dict],
) -> Dict[str, Any]:
    """Полный скоринг 2.0.0 → структура как ``main_response``.

    Args:
        group_name: stem ``second_model`` (файл ``{stem}.pickle``).
        features_values: ``second_request``.

    Returns:
        dict: ``usage_model``, ``version``, ``CPMComment``, ``result``, ``df``.
        В ``result``: classification_proba / predictions, regression /
        prediction (после маски), threshold.
    """
    # Грузим пару clf+rg и готовим вектор с FE_* (validate внутри preprocess).
    group = _load_model_pair(group_name, "second")
    clf_model, rg_model = group[0], group[1]
    df_common = preprocess_second_vector(features_values)
    # Берём порог из вектора или дефолт second_.
    threshold_value = resolve_threshold(df_common, config.default_second_threshold)

    # Имена фич берём из pickle (битый/пустой config → ошибка в feature_names).
    clf_feats = feature_names_from_model_result(clf_model)
    rg_feats = feature_names_from_model_result(rg_model)

    clf_df = require_columns(df_common, clf_feats, "Классификации(v2)")
    rg_df = require_columns(df_common, rg_feats, "Регрессии(v2)")

    # Считаем классификацию и калибруем вероятность.
    model_c, t_c = prepare_matrix(clf_model, clf_df)
    raw_proba = _raw_classification_scores(model_c, t_c)
    cf_path, rg_path = calibrator_paths()
    cf_cal, rg_cal = _CACHE.get(cf_path), _CACHE.get(rg_path)
    proba = apply_cf_calibrator(cf_cal, raw_proba)
    labels = [1.0 if s >= threshold_value else 0.0 for s in proba]

    # Считаем регрессию, калибруем и применяем маску юриста.
    model_r, t_r = prepare_matrix(rg_model, rg_df)
    raw_reg = [float(x) for x in np.asarray(model_r.predict(t_r)).ravel()]
    cal_reg = apply_rg_calibrator(rg_cal, raw_reg)
    masked = masked_predictions(labels, cal_reg, df_common)

    # SHAP: unwrap внутри shap_top_features_comment.
    cpm_comment = (
        shap_top_features_comment(model_c, t_c, prefix="shap_clf")
        + " | "
        + shap_top_features_comment(model_r, t_r, prefix="shap_reg")
    ).strip()

    # Округляем числа ответа до 2 знаков (тот же хелпер, что у main_).
    rounded_proba = _round_numbers(proba)
    rounded_labels = _round_numbers(labels)
    rounded_masked = _round_numbers(masked)

    result_body = {
        "classification_proba": rounded_proba,
        "classification_predictions": rounded_labels,
        "regression_predictions": rounded_masked,
        "prediction": rounded_masked,
        "threshold": round(float(threshold_value), 2),
    }
    # Собираем ответ second_ (дубли колонок — как у main_: ошибка, не soft-drop).
    output_df = pd.concat([t_c, t_r], axis=1)
    if output_df.columns.duplicated().any():
        dupes = output_df.columns[output_df.columns.duplicated()].tolist()
        raise ValueError(
            f"Дубли имён колонок в output после объединения clf/reg: {dupes}"
        )

    return {
        "usage_model": group_name,
        "version": config.second_version,
        "CPMComment": cpm_comment,
        "result": result_body,
        "df": {
            "input": df_to_records(df_common),
            "output": df_to_records(output_df),
        },
    }
