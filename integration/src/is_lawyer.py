"""Маска итогового prediction по классу и флагу IS_LAWYER.

Одинаковая семантика для main_ и second_: если классификация = 0 и
заявитель не юрист — в ответе регрессия обнуляется.
"""
from __future__ import annotations

from typing import List

import pandas as pd


# reviewed
def masked_predictions(
    labels: List[float],
    raw_reg: List[float],
    df_common: pd.DataFrame,
) -> List[float]:
    """Применить бизнес-маску к сырой / откалиброванной регрессии.

    Правила построчно:
        * метка класса 1 → отдать регрессию;
        * метка 0 и ``IS_LAWYER == 1`` → отдать регрессию;
        * иначе → 0.

    Args:
        labels: бинарные метки классификации (0/1) после порога.
        raw_reg: значения регрессии той же длины.
        df_common: кадр после preprocess; ``IS_LAWYER`` опционален
            (нет колонки → как 0, старые запросы ОИСУУ).

    Returns:
        Список маскированных float для полей
        ``regression_predictions`` / ``prediction``.
    """
    if "IS_LAWYER" in df_common.columns:
        is_lawyer = (
            pd.to_numeric(df_common["IS_LAWYER"], errors="coerce").fillna(0).eq(1)
        ).tolist()
    else:
        is_lawyer = [False] * len(labels)

    # Класс 0 и не юрист — регрессию обнуляем.
    # raw_reg уже list[float] из run_regression_model / калибратора.
    masked: List[float] = []
    for c, r, lawyer in zip(labels, raw_reg, is_lawyer):
        if c or lawyer:
            masked.append(r)
        else:
            masked.append(0.0)
    return masked