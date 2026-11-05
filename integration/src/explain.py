"""SHAP top-фичи → строка для поля CPMComment (main_ и second_).

Модели треков — CatBoost: ``get_feature_importance(..., type="ShapValues")``.
Ошибки глотаются: в комментарий пишется причина, запрос скоринга не падает.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from catboost import Pool  # type: ignore
except Exception:  # noqa: BLE001
    Pool = None  # type: ignore[assignment,misc]


# reviewed
def unwrap_mldataworker_model(model: Any) -> Any:
    """Снять 1–2 уровня обёртки mldataworker (``model.model``).

    Нужен «голый» CatBoost для feature importance.

    Args:
        model: объект из pickle (возможно обёрнутый).

    Returns:
        Вложенный estimator либо исходный объект.
    """
    inner = model
    for _ in range(2):
        if hasattr(inner, "model"):
            inner = getattr(inner, "model")
        else:
            break
    return inner


# reviewed
def _catboost_shap_values(base_model: Any, t: pd.DataFrame) -> Any:
    """SHAP через CatBoost ``get_feature_importance(..., type='ShapValues')``."""
    if Pool is None:
        raise ImportError("catboost.Pool unavailable")

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


# reviewed
def shap_top_features_comment(
    model: Any,
    t: pd.DataFrame,
    *,
    prefix: str,
) -> str:
    """Построить краткий SHAP-комментарий по одной строке матрицы ``t``.

    Для ``prefix``, начинающегося с ``shap_clf``, дополнительно считает
    относительный вклад в вероятность (%). Иначе — сырые SHAP-значения.
    В комментарий входят топ-5 фич.

    Args:
        model: CatBoost (обёртку снимает сама функция).
        t: матрица признаков после prepare_dataset (обычно 1 строка).
        prefix: метка в начале строки (``shap_clf`` / ``shap_reg``).

    Returns:
        Строка вида ``"shap_clf: p=… | FEAT=+x%, …"`` либо сообщение об ошибке.
        Исключения наружу не пробрасывает.
    """
    top_n = 5
    try:
        base_model = unwrap_mldataworker_model(model)
        feature_names = list(t.columns)
        proba_prefix = ""
        base_raw_for_pct: Optional[float] = None

        values = _catboost_shap_values(base_model, t)

        if isinstance(values, list) and values:
            v = np.asarray(values[0])
        else:
            v = np.asarray(values)
        if v.ndim == 3 and v.shape[1] == len(feature_names) + 1:
            class_idx = 1 if v.shape[2] > 1 else 0
            if prefix.startswith("shap_clf"):
                base_raw = float(v[0, -1, class_idx])
                raw = base_raw + float(np.sum(v[0, :-1, class_idx]))
                p = 1.0 / (1.0 + float(np.exp(-raw)))
                proba_prefix = f"p={p:.4f} | "
                base_raw_for_pct = base_raw
            v = v[:, :, class_idx]
        if v.ndim == 2 and v.shape[1] == len(feature_names) + 1:
            if prefix.startswith("shap_clf"):
                base_raw = float(v[0, -1])
                raw = base_raw + float(np.sum(v[0, :-1]))
                p = 1.0 / (1.0 + float(np.exp(-raw)))
                proba_prefix = f"p={p:.4f} | "
                base_raw_for_pct = base_raw
            v = v[:, :-1]
        if v.ndim == 3:
            v = v[:, :, 0]
        if v.ndim != 2 or v.shape[0] < 1:
            return f"{prefix}: SHAP empty"

        row = v[0]
        if len(feature_names) != row.shape[0]:
            feature_names = feature_names[: row.shape[0]]

        if prefix.startswith("shap_clf") and base_raw_for_pct is not None:
            base_p = 1.0 / (1.0 + float(np.exp(-base_raw_for_pct)))
            rel = []
            for c in row:
                p_i = 1.0 / (
                    1.0 + float(np.exp(-(base_raw_for_pct + float(c))))
                )
                if base_p == 0:
                    rel.append(0.0)
                else:
                    rel.append((p_i / base_p - 1.0) * 100.0)
            rel_arr = np.asarray(rel, dtype=float)
            idx = np.argsort(np.abs(rel_arr))[::-1][:top_n]
            parts = [f"{feature_names[i]}={rel_arr[i]:+.2f}%" for i in idx]
            return f"{prefix}: " + proba_prefix + ", ".join(parts)

        idx = np.argsort(np.abs(row))[::-1][:top_n]
        parts = [f"{feature_names[i]}={row[i]:+.4f}" for i in idx]
        return f"{prefix}: " + proba_prefix + ", ".join(parts)
    except Exception as e:
        msg = str(e).replace("\n", " ").strip()
        if len(msg) > 120:
            msg = msg[:120] + "..."
        return f"{prefix}: SHAP error ({type(e).__name__}: {msg})"
