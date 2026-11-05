"""Реэкспорт общих валидаций для импортов ``preprocessing.main``."""
from __future__ import annotations

try:
    from integration.src.preprocessing.common import (
        INPUT_REQUIRED_COLUMNS,
        MODEL_FEATURES,
        validate_derived_dates_strict,
        validate_input_vector,
        validate_model_features,
        validate_no_duplicate_columns,
        validate_preprocessed_vector,
    )
except ImportError:
    from src.preprocessing.common import (
        INPUT_REQUIRED_COLUMNS,
        MODEL_FEATURES,
        validate_derived_dates_strict,
        validate_input_vector,
        validate_model_features,
        validate_no_duplicate_columns,
        validate_preprocessed_vector,
    )

__all__ = [
    "INPUT_REQUIRED_COLUMNS",
    "MODEL_FEATURES",
    "validate_derived_dates_strict",
    "validate_input_vector",
    "validate_model_features",
    "validate_no_duplicate_columns",
    "validate_preprocessed_vector",
]
