"""Сохранение результатов CatBoost feature selection на диск."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from querulus import PROJECT_ROOT

DEFAULT_FEATURE_SELECTION_DIR = PROJECT_ROOT / "data" / "processed" / "feature_selection"


def _importance_records(importance: pd.DataFrame | None) -> list[dict[str, Any]]:
    """Список {feature, importance} для JSON."""
    if importance is None or getattr(importance, "empty", True):
        return []
    if "feature" not in importance.columns or "importance" not in importance.columns:
        return []
    return [
        {"feature": str(row.feature), "importance": float(row.importance)}
        for row in importance.itertuples(index=False)
    ]


def save_feature_selection(
    *,
    stack: str,
    task: str,
    selected_features: list[str],
    summary: dict[str, Any] | None = None,
    directory: Path | str | None = None,
    importance: pd.DataFrame | None = None,
    categorical_features: list[str] | None = None,
) -> Path:
    """Сохранить отобранные фичи в JSON (не зависит от outputs ноутбука).

    Путь: ``data/processed/feature_selection/{stack}_{task}_{timestamp}.json``
    + актуальный ``{stack}_{task}_latest.json``.
    ``importance`` / ``categorical_features`` нужны для пересборки HTML без FS.
    """
    out_dir = Path(directory) if directory is not None else DEFAULT_FEATURE_SELECTION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "stack": stack,
        "task": task,
        "saved_at_utc": stamp,
        "n_selected": len(selected_features),
        "selected_features": list(selected_features),
        "eliminated_features": list(
            (summary or {}).get("eliminated_features_names") or []
        ),
        "importance": _importance_records(importance),
        "categorical_features": list(categorical_features or []),
    }
    stamped = out_dir / f"{stack}_{task}_{stamp}.json"
    latest = out_dir / f"{stack}_{task}_latest.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    stamped.write_text(text, encoding="utf-8")
    latest.write_text(text, encoding="utf-8")
    return latest


def load_feature_selection_latest(
    stack: str,
    task: str,
    *,
    directory: Path | str | None = None,
) -> dict[str, Any] | None:
    """Загрузить последний JSON отбора (или None)."""
    out_dir = Path(directory) if directory is not None else DEFAULT_FEATURE_SELECTION_DIR
    path = out_dir / f"{stack}_{task}_latest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def drop_zero_importance_features(
    features: list[str] | tuple[str, ...],
    importance: pd.DataFrame | None,
    *,
    eps: float = 0.0,
) -> tuple[list[str], list[str], pd.DataFrame | None]:
    """Убрать фичи с importance ≤ ``eps`` (нет в таблице — оставляем).

    Возвращает ``(kept, dropped, importance_kept)``.
    """
    feature_list = list(features)
    if importance is None or getattr(importance, "empty", True):
        return feature_list, [], importance
    if "feature" not in importance.columns or "importance" not in importance.columns:
        return feature_list, [], importance

    weights = {
        str(row.feature): float(row.importance)
        for row in importance.itertuples(index=False)
    }
    kept: list[str] = []
    dropped: list[str] = []
    for name in feature_list:
        weight = weights.get(name)
        if weight is None or weight > eps:
            kept.append(name)
        else:
            dropped.append(name)

    if not dropped:
        return feature_list, [], importance

    kept_set = set(kept)
    trimmed = importance[importance["feature"].astype(str).isin(kept_set)].copy()
    return kept, dropped, trimmed.reset_index(drop=True)
