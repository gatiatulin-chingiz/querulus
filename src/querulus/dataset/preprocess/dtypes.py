"""Нормализация dtypes перед сохранением итогового датасета."""
from __future__ import annotations

import pandas as pd


def cast_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Categorical / string[pyarrow] → plain object.

    Нужно для OutBoxML ``prepare_categorical``: присвоение default вне
    ``categories`` pandas.Categorical падает. Делается при сохранении
    ``df_final`` (collect), не в example/DSM.
    """
    out = df.copy()
    for name in out.columns:
        series = out[name]
        dtype = series.dtype
        if isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_categorical_dtype(dtype):
            out[name] = series.astype(object)
        elif pd.api.types.is_string_dtype(dtype) and not pd.api.types.is_object_dtype(dtype):
            out[name] = series.astype(object)
    return out
