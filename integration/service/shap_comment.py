"""SHAP top-фичи → строка для CPMComment (main_ и second_)."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


#reviewed
def unwrap_mldataworker_model(model: Any) -> Any:
    """Достаёт «сырой» estimator из обёрток mldataworker (model.model.model)."""
    inner = model
    for _ in range(2):
        if hasattr(inner, "model"):
            inner = getattr(inner, "model")
        else:
            break
    return inner


#reviewed
def shap_top_features_comment(
    model: Any,
    t: pd.DataFrame,
    *,
    prefix: str,
    top_n: int = 5,
) -> str:
    """SHAP для запроса → короткая строка CPMComment; ошибки не поднимает."""
    try:
        import shap  # type: ignore
    except Exception as e:
        return f"{prefix}: SHAP unavailable ({type(e).__name__})"

    try:
        base_model = unwrap_mldataworker_model(model)
        feature_names = list(t.columns)
        proba_prefix = ""
        base_raw_for_pct: Optional[float] = None

        def _is_catboost_model(m: Any) -> bool:
            mod = getattr(m, "__module__", "") or ""
            name = getattr(m, "__class__", type(m)).__name__
            return mod.startswith("catboost") or name.startswith("CatBoost")

        if _is_catboost_model(base_model):
            if not hasattr(base_model, "get_feature_importance"):
                raise ValueError("CatBoost model without get_feature_importance()")
            try:
                from catboost import Pool  # type: ignore
            except Exception:
                raise

            cat_idxs: list[int] = []
            try:
                model_feature_names = list(getattr(base_model, "feature_names_", None) or [])
                if not model_feature_names and hasattr(base_model, "get_feature_names"):
                    model_feature_names = list(base_model.get_feature_names() or [])
                model_cat_idxs = []
                if hasattr(base_model, "get_cat_feature_indices"):
                    model_cat_idxs = list(base_model.get_cat_feature_indices() or [])
                cat_names = [
                    model_feature_names[i]
                    for i in model_cat_idxs
                    if i < len(model_feature_names)
                ]
                name_to_idx = {name: i for i, name in enumerate(feature_names)}
                cat_idxs = [name_to_idx[n] for n in cat_names if n in name_to_idx]
            except Exception:
                cat_idxs = []

            df_pool = t.copy()
            for i, col in enumerate(feature_names):
                if i in cat_idxs:
                    df_pool[col] = df_pool[col].astype("string").fillna("NA")
                else:
                    df_pool[col] = pd.to_numeric(df_pool[col], errors="coerce").fillna(0.0)

            pool = Pool(df_pool, feature_names=feature_names, cat_features=cat_idxs)
            values = base_model.get_feature_importance(pool, type="ShapValues")
            expected_value = None
        else:
            try:
                tree_explainer = shap.TreeExplainer(
                    base_model,
                    feature_perturbation="tree_path_dependent",
                )
                values = tree_explainer.shap_values(t)
                expected_value = getattr(tree_explainer, "expected_value", None)
            except Exception as e:
                if type(e).__name__ == "CatBoostError" and hasattr(
                    base_model, "get_feature_importance"
                ):
                    from catboost import Pool  # type: ignore

                    cat_idxs = []
                    try:
                        model_feature_names = list(
                            getattr(base_model, "feature_names_", None) or []
                        )
                        if not model_feature_names and hasattr(
                            base_model, "get_feature_names"
                        ):
                            model_feature_names = list(
                                base_model.get_feature_names() or []
                            )
                        model_cat_idxs = []
                        if hasattr(base_model, "get_cat_feature_indices"):
                            model_cat_idxs = list(
                                base_model.get_cat_feature_indices() or []
                            )
                        cat_names = [
                            model_feature_names[i]
                            for i in model_cat_idxs
                            if i < len(model_feature_names)
                        ]
                        name_to_idx = {name: i for i, name in enumerate(feature_names)}
                        cat_idxs = [
                            name_to_idx[n] for n in cat_names if n in name_to_idx
                        ]
                    except Exception:
                        cat_idxs = []

                    df_pool = t.copy()
                    for i, col in enumerate(feature_names):
                        if i in cat_idxs:
                            df_pool[col] = df_pool[col].astype("string").fillna("NA")
                        else:
                            df_pool[col] = pd.to_numeric(
                                df_pool[col], errors="coerce"
                            ).fillna(0.0)

                    pool = Pool(
                        df_pool, feature_names=feature_names, cat_features=cat_idxs
                    )
                    values = base_model.get_feature_importance(pool, type="ShapValues")
                    expected_value = None
                else:
                    raise

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
        if (
            not proba_prefix
            and prefix.startswith("shap_clf")
            and expected_value is not None
            and v.ndim == 2
        ):
            try:
                ev = expected_value
                if isinstance(ev, (list, tuple, np.ndarray)):
                    class_idx = 1 if len(ev) > 1 else 0
                    ev = ev[class_idx]
                base_raw = float(ev)
                raw = base_raw + float(np.sum(v[0]))
                p = 1.0 / (1.0 + float(np.exp(-raw)))
                proba_prefix = f"p={p:.4f} | "
                base_raw_for_pct = base_raw
            except Exception:
                proba_prefix = ""
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
