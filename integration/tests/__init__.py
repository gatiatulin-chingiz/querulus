"""Тесты сервиса querulus (unit + integration).

Импорты сервиса — как в Docker: ``config`` / ``src.*`` / ``main``
от корня ``integration/``. Пакет моделирования — отдельный ``querulus/src``.
"""
import sys
from pathlib import Path

_INTEGRATION = Path(__file__).resolve().parents[1]
_QUERULUS_SRC = _INTEGRATION.parent / "src"

if str(_INTEGRATION) not in sys.path:
    sys.path.insert(0, str(_INTEGRATION))
if _QUERULUS_SRC.is_dir() and str(_QUERULUS_SRC) not in sys.path:
    sys.path.insert(0, str(_QUERULUS_SRC))
