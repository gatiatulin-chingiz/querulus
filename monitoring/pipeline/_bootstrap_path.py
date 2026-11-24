"""Сделать ``import pipeline.*`` рабочим в monorepo и в standalone GitLab-clone.

Monorepo: ``monitoring/pipeline/*.py`` → в ``sys.path`` корень ``monitoring/``.
Contour (querulus_monitoring.git): файлы в корне clone → регистрируем этот каталог
как пакет ``pipeline`` (имя папки на диске другое).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_PARENT = _ROOT.parent


def ensure() -> None:
    """Идемпотентно: после вызова ``from pipeline.settings`` резолвится."""
    monorepo = _PARENT / "pipeline" / "settings.py"
    if monorepo.is_file():
        parent = str(_PARENT)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        return

    if "pipeline" in sys.modules and getattr(sys.modules["pipeline"], "__path__", None):
        return

    init = _ROOT / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "pipeline",
        init if init.is_file() else None,
        submodule_search_locations=[str(_ROOT)],
    )
    if spec is None:
        raise RuntimeError(f"cannot register pipeline package from {_ROOT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline"] = mod
    if spec.loader is not None and init.is_file():
        spec.loader.exec_module(mod)


ensure()
