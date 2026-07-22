"""Логи этапов train-loop для ноутбука."""
from __future__ import annotations


def stage_start(name: str, *, detail: str = "") -> None:
    """Печать старта этапа."""
    suffix = f" — {detail}" if detail else ""
    print(f"[B] >>> STAGE {name} START{suffix}")


def stage_skipped(name: str, flag: str) -> None:
    """Печать пропуска этапа по флагу."""
    print(f"[B] >>> STAGE {name} SKIPPED ({flag}=False)")


def stage_done(name: str, *, detail: str = "") -> None:
    """Печать завершения этапа."""
    suffix = f" → {detail}" if detail else ""
    print(f"[B] >>> STAGE {name} DONE{suffix}")
