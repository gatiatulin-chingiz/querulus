"""Разбор имени severity-варианта (zoo) → transform / weight / сегмент."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SeverityTargetTransform = Literal["raw", "log1p"]
SeveritySampleWeight = Literal["none", "sqrt", "linear"]
SeveritySegmentSide = Literal["all", "le", "gt"]

SEVERITY_VARIANT_NAMES: tuple[str, ...] = (
    "raw",
    "log1p",
    "weighted_sqrt",
    "weighted_linear",
    "raw_le50",
    "log1p_le50",
    "raw_gt50",
    "log1p_gt50",
)


@dataclass(frozen=True)
class SeverityVariantSpec:
    """Параметры severity для train-loop / zoo."""

    name: str
    transform: SeverityTargetTransform
    sample_weight: SeveritySampleWeight
    segment: SeveritySegmentSide


_SPECS: dict[str, SeverityVariantSpec] = {
    "raw": SeverityVariantSpec("raw", "raw", "none", "all"),
    "log1p": SeverityVariantSpec("log1p", "log1p", "none", "all"),
    "weighted_sqrt": SeverityVariantSpec("weighted_sqrt", "raw", "sqrt", "all"),
    "weighted_linear": SeverityVariantSpec("weighted_linear", "raw", "linear", "all"),
    "raw_le50": SeverityVariantSpec("raw_le50", "raw", "none", "le"),
    "log1p_le50": SeverityVariantSpec("log1p_le50", "log1p", "none", "le"),
    "raw_gt50": SeverityVariantSpec("raw_gt50", "raw", "none", "gt"),
    "log1p_gt50": SeverityVariantSpec("log1p_gt50", "log1p", "none", "gt"),
}


def resolve_severity_variant(name: str) -> SeverityVariantSpec:
    """Разобрать имя варианта zoo → transform / weight / segment."""
    key = str(name).strip()
    if key not in _SPECS:
        known = ", ".join(SEVERITY_VARIANT_NAMES)
        raise ValueError(f"Неизвестный SEVERITY_VARIANT={name!r}. Допустимо: {known}")
    return _SPECS[key]
