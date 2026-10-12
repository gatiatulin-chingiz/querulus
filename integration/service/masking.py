"""Единая маска IS_LAWYER для main_ и second_."""
from __future__ import annotations

from typing import List

import pandas as pd


#reviewed
def masked_predictions(
    labels: List[float],
    raw_reg: List[float],
    df_common: pd.DataFrame,
) -> List[float]:
    """Маскированные значения регрессии / итога.

    Правила:
        - класс 1 → регрессия;
        - класс 0 и IS_LAWYER 1 → регрессия;
        - иначе 0.
    """
    if "IS_LAWYER" in df_common.columns:
        is_lawyer = (
            pd.to_numeric(df_common["IS_LAWYER"], errors="coerce")
            .fillna(0)
            .eq(1)
            .tolist()
        )
    else:
        is_lawyer = [False] * len(labels)

    masked: List[float] = []
    for c, r, lawyer in zip(labels, raw_reg, is_lawyer):
        reg = float(r)
        if c or lawyer:
            masked.append(reg)
        else:
            masked.append(0.0)
    return masked
