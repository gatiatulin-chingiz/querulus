"""Сохранение результатов CatBoost feature selection на диск."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from querulus import PROJECT_ROOT

DEFAULT_FEATURE_SELECTION_DIR = PROJECT_ROOT / "data" / "processed" / "feature_selection"


def save_feature_selection(
    *,
    stack: str,
    task: str,
    selected_features: list[str],
    summary: dict[str, Any] | None = None,
    directory: Path | str | None = None,
) -> Path:
    """Сохранить отобранные фичи в JSON (не зависит от outputs ноутбука).

    Путь: ``data/processed/feature_selection/{stack}_{task}_{timestamp}.json``
    + актуальный ``{stack}_{task}_latest.json``.
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
