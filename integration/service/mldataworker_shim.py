"""Локальный shim: mldataworker → outboxml (в контуре outboxml ставится через pip).

Нужен, чтобы:
  - ``from mldataworker.core…`` работал без отдельного пакета mldataworker;
  - pickle legacy-моделей с ``mldataworker.models.*`` грузился через ``outboxml.models``.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path


#reviewed
def _ensure_pkg(name: str) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    # pickle ``import mldataworker.models`` требует package (есть __path__)
    if not hasattr(mod, "__path__"):
        mod.__path__ = []  # type: ignore[attr-defined]
    return mod


#reviewed
def _alias_models(root: types.ModuleType) -> None:
    if "mldataworker.models" in sys.modules:
        return
    try:
        import outboxml.models as _models
    except ModuleNotFoundError:
        return
    sys.modules["mldataworker.models"] = _models
    root.models = _models


#reviewed
def install_mldataworker_shim() -> None:
    # Сначала пробуем настоящий пакет, не перекрывая его фейком.
    if "mldataworker" not in sys.modules:
        try:
            import mldataworker  # noqa: F401
        except ModuleNotFoundError:
            pass

    root = _ensure_pkg("mldataworker")
    _alias_models(root)

    if "mldataworker.core.data_prepare" in sys.modules:
        return

    try:
        from mldataworker.core import data_prepare as _  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    try:
        from outboxml.core import data_prepare as _dp
        from outboxml.core import pydantic_models as _pm
        from outboxml.core import utils as _utils
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", "") != "outboxml" and "outboxml" not in str(exc):
            raise
        raise ModuleNotFoundError(
            "Нужен пакет outboxml для mldataworker shim. "
            "В контуре: pip install outboxml (или корпоративный wheel)."
        ) from exc

    core = _ensure_pkg("mldataworker.core")
    data_prepare = types.ModuleType("mldataworker.core.data_prepare")
    pydantic_models = types.ModuleType("mldataworker.core.pydantic_models")
    utils = types.ModuleType("mldataworker.core.utils")

    data_prepare.prepare_dataset = _dp.prepare_dataset
    pydantic_models.ModelConfig = _pm.ModelConfig
    pydantic_models.ServiceRequest = _pm.ServiceRequest
    utils.ResultPickle = _utils.ResultPickle

    sys.modules["mldataworker.core.data_prepare"] = data_prepare
    sys.modules["mldataworker.core.pydantic_models"] = pydantic_models
    sys.modules["mldataworker.core.utils"] = utils
    root.core = core
    core.data_prepare = data_prepare
    core.pydantic_models = pydantic_models
    core.utils = utils


if __name__ == "__main__":
    install_mldataworker_shim()
    print("mldataworker shim OK ->", Path(__file__).resolve())
