"""Точка входа Docker / прямой импорт: re-export FastAPI app из service."""
from __future__ import annotations

try:
    from service.main import (  # noqa: F401
        CLASSIFICATION_FEATURES,
        REGRESSION_FEATURES,
        app,
        enrich_common_vector_dataframe,
        main_predict,
        prepare_common_vector_dataframe,
        preprocessing_and_validate_vector,
        validate_derived_dates_strict,
        validate_exact_columns_order,
        validate_no_duplicate_columns,
        version,
    )
except ImportError:
    from integration.service.main import (  # noqa: F401
        CLASSIFICATION_FEATURES,
        REGRESSION_FEATURES,
        app,
        enrich_common_vector_dataframe,
        main_predict,
        prepare_common_vector_dataframe,
        preprocessing_and_validate_vector,
        validate_derived_dates_strict,
        validate_exact_columns_order,
        validate_no_duplicate_columns,
        version,
    )

__all__ = [
    "CLASSIFICATION_FEATURES",
    "REGRESSION_FEATURES",
    "app",
    "enrich_common_vector_dataframe",
    "main_predict",
    "prepare_common_vector_dataframe",
    "preprocessing_and_validate_vector",
    "validate_derived_dates_strict",
    "validate_exact_columns_order",
    "validate_no_duplicate_columns",
    "version",
]

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
