"""SHAP-комментарий для ответа сервиса (общий для всех stem)."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from catboost import Pool  # type: ignore

# reviewed
def unwrap_model(model: Any) -> Any:
    inner = model
    for _ in range(2):
        if hasattr(inner, "model"):
            inner = getattr(inner, "model")
        else:
            break
    return inner

# reviewed
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

# reviewed
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-x)))

# reviewed
def shap_top_features_comment(
    model: Any,
    t: pd.DataFrame,
    *,
    prefix: str,
    top_n: int = 5,
) -> str:
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
            rel = np.array(
                [(_sigmoid(base_raw + float(c)) / base_p - 1.0) * 100.0 for c in row]
            )
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
