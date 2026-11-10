"""Входной препроцессинг вектора second_ до predict."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.preprocessing.common import (
    preprocess_input_vector,
    validate_model_features,
)
from src.preprocessing.second.features import add_second_model_features


# reviewed
def preprocess_second_vector(features_values: List[Dict]) -> pd.DataFrame:
    """Подготовить кадр second_: общий вход → FE_* → validate model.

    Args:
        features_values: ``second_request`` (list[dict]).

    Returns:
        DataFrame с входными колонками + engineered-фичами.
    """
    df = preprocess_input_vector(features_values)
    df = add_second_model_features(df)
    validate_model_features(df)
    return df