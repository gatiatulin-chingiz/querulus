"""Локальный shim: mldataworker → outboxml (в контуре outboxml ставится через pip)."""
from __future__ import annotations

import sys
import types
from pathlib import Path


#reviewed
def install_mldataworker_shim() -> None:
    if "mldataworker" in sys.modules:
        return
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

    root = types.ModuleType("mldataworker")
    core = types.ModuleType("mldataworker.core")
    data_prepare = types.ModuleType("mldataworker.core.data_prepare")
    pydantic_models = types.ModuleType("mldataworker.core.pydantic_models")
    utils = types.ModuleType("mldataworker.core.utils")

    data_prepare.prepare_dataset = _dp.prepare_dataset
    pydantic_models.ModelConfig = _pm.ModelConfig
    pydantic_models.ServiceRequest = _pm.ServiceRequest
    utils.ResultPickle = _utils.ResultPickle

    sys.modules["mldataworker"] = root
    sys.modules["mldataworker.core"] = core
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
