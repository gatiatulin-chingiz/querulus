"""vector_checker — сверка вектора 1С (Сутяжность) с df_for_service.

Команды::

    python -m integration.tests.vector_checker prepare
    python -m integration.tests.vector_checker compare --excel path/to/export.xlsx
    python -m integration.tests.vector_checker demo
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
