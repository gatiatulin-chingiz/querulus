"""CatBoost fit: tree_count, gap-penalty для HPO, финальный fit без ES."""
from __future__ import annotations

from typing import Any, Literal

TaskType = Literal["classification", "regression"]


def catboost_fit_stats(model: object, *, iterations_cap: int | None = None) -> dict[str, int]:
    """Фактическое число деревьев после fit (с ES или без)."""
    tree_count = int(getattr(model, "tree_count_", 0) or 0)
    best_iteration = int(getattr(model, "get_best_iteration", lambda: tree_count - 1)())
    if best_iteration < 0:
        best_iteration = max(tree_count - 1, 0)
    out: dict[str, int] = {
        "tree_count": tree_count,
        "best_iteration": best_iteration,
    }
    if iterations_cap is not None:
        out["iterations_cap"] = int(iterations_cap)
    return out


def train_val_gap(train_metric: float, val_metric: float, *, task_type: TaskType) -> float:
    """Разрыв train–val (>0 = val хуже train, типичное переобучение)."""
    if task_type == "classification":
        return float(train_metric - val_metric)
    return float(val_metric - train_metric)


def apply_gap_penalty(
    val_metric: float,
    gap: float,
    *,
    task_type: TaskType,
    gap_lambda: float,
) -> float:
    """Objective с штрафом λ·gap (classification: max, regression MAE: min)."""
    penalty = float(gap_lambda) * float(gap)
    if task_type == "classification":
        return float(val_metric - penalty)
    return float(val_metric + penalty)


def strip_hpo_meta(params: dict[str, Any]) -> dict[str, Any]:
    """Убрать служебные ключи перед передачей в CatBoost."""
    skip = {
        "iterations_cap",
        "tree_count",
        "best_iteration",
        "early_stopping_rounds",
        "gap_lambda",
        "best_value_raw",
        "best_value",
        "mean_train_val_gap",
    }
    return {key: value for key, value in params.items() if key not in skip and value is not None}
