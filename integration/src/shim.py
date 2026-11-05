"""Локальный shim: пакетное имя ``mldataworker`` → реализация ``outboxml``.

В контуре ОИСУУ ставится настоящий ``mldataworker``. Локально / в тестах
достаточно outboxml: этот модуль:

* создаёт package-заглушки ``mldataworker.core.*`` с символами из outboxml;
* алиасит ``mldataworker.models`` → ``outboxml.models``, чтобы pickle
  моделей грузились без ModuleNotFoundError.

Вызывать ``install_mldataworker_shim()`` до импорта ``src.predictions`` / ``main`` в тестах.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

try:
    import mldataworker as _mldataworker_pkg  # noqa: F401
except ModuleNotFoundError:
    _mldataworker_pkg = None

try:
    import outboxml.models as _outboxml_models
except ModuleNotFoundError:
    _outboxml_models = None

try:
    from mldataworker.core import data_prepare as _mldataworker_data_prepare
except ModuleNotFoundError:
    _mldataworker_data_prepare = None

try:
    from outboxml.core import data_prepare as _outboxml_data_prepare
    from outboxml.core import pydantic_models as _outboxml_pydantic_models
    from outboxml.core import utils as _outboxml_utils
except ModuleNotFoundError:
    _outboxml_data_prepare = None
    _outboxml_pydantic_models = None
    _outboxml_utils = None


# new
def _ensure_pkg(name: str) -> types.ModuleType:
    """Гарантировать, что в ``sys.modules`` есть package с ``__path__``.

    Pickle ``import mldataworker.models`` требует, чтобы родитель был package.

    Args:
        name: dotted-имя пакета (``mldataworker``, ``mldataworker.core``, …).

    Returns:
        Существующий или только что созданный ModuleType.
    """
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    # pickle ``import mldataworker.models`` требует package (есть __path__)
    if not hasattr(mod, "__path__"):
        mod.__path__ = []  # type: ignore[attr-defined]
    return mod


# new
def _alias_models(root: types.ModuleType) -> None:
    """Подменить ``mldataworker.models`` на ``outboxml.models``, если доступен.

    Args:
        root: модуль ``mldataworker`` в sys.modules.
    """
    if "mldataworker.models" in sys.modules:
        return
    if _outboxml_models is None:
        return
    sys.modules["mldataworker.models"] = _outboxml_models
    root.models = _outboxml_models


# new
def install_mldataworker_shim() -> None:
    """Установить shim, если настоящего mldataworker нет.

    Порядок:
        1. попробовать импортировать реальный ``mldataworker``;
        2. алиаснуть ``.models`` на outboxml;
        3. если ``mldataworker.core.data_prepare`` уже есть — выйти;
        4. иначе пробросить символы prepare_dataset / ModelConfig /
           ServiceRequest / ResultPickle из outboxml.

    Raises:
        ModuleNotFoundError: нет ни mldataworker.core, ни outboxml.
    """
    # Сначала пробуем настоящий пакет, не перекрывая его фейком.
    if "mldataworker" not in sys.modules and _mldataworker_pkg is not None:
        sys.modules["mldataworker"] = _mldataworker_pkg

    root = _ensure_pkg("mldataworker")
    _alias_models(root)

    if "mldataworker.core.data_prepare" in sys.modules:
        return

    if _mldataworker_data_prepare is not None:
        return

    if (
        _outboxml_data_prepare is None
        or _outboxml_pydantic_models is None
        or _outboxml_utils is None
    ):
        raise ModuleNotFoundError(
            "Нужен пакет outboxml для mldataworker shim. "
            "В контуре: pip install outboxml (или корпоративный wheel)."
        )

    core = _ensure_pkg("mldataworker.core")
    data_prepare = types.ModuleType("mldataworker.core.data_prepare")
    pydantic_models = types.ModuleType("mldataworker.core.pydantic_models")
    utils = types.ModuleType("mldataworker.core.utils")

    data_prepare.prepare_dataset = _outboxml_data_prepare.prepare_dataset
    pydantic_models.ModelConfig = _outboxml_pydantic_models.ModelConfig
    pydantic_models.ServiceRequest = _outboxml_pydantic_models.ServiceRequest
    utils.ResultPickle = _outboxml_utils.ResultPickle

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
