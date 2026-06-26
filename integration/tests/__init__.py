"""Тесты сервиса querulus (unit + integration)."""
import sys
from pathlib import Path

# Пакет querulus лежит в src/ (layout как в cookiecutter-data-science)
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
