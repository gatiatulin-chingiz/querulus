"""Локальный shim: mldataworker → outboxml (корпоративный пакет в venv нет)."""
from __future__ import annotations

import sys
import types
from pathlib import Path


def install_mldataworker_shim() -> None:
    if "mldataworker" in sys.modules:
        return
    from outboxml.core import data_prepare as _dp
    from outboxml.core import pydantic_models as _pm
    from outboxml.core import utils as _utils

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
    print("mldataworker shim OK →", Path(__file__).resolve())
