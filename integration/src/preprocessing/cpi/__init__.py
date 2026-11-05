"""CPI / дефляция денежных полей для second_."""
from .cpi import (
    INFLATION_BASE_YEAR,
    MONETARY_COLUMNS_FOR_REAL,
    RU_CPI_LEVEL_VS_BASE,
    add_real_monetary_columns,
    deflate_to_base_year,
    real_feature_name,
    resolve_cpi_year,
)

__all__ = [
    "INFLATION_BASE_YEAR",
    "MONETARY_COLUMNS_FOR_REAL",
    "RU_CPI_LEVEL_VS_BASE",
    "add_real_monetary_columns",
    "deflate_to_base_year",
    "real_feature_name",
    "resolve_cpi_year",
]
