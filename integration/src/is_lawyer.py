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
        df_common: кадр с обязательной колонкой ``IS_LAWYER``.

    Returns:
        Список маскированных float для полей
        ``regression_predictions`` / ``prediction``.
    """
    is_lawyer = df_common["IS_LAWYER"].eq(1).tolist()

    # Класс 0 и не юрист — регрессию обнуляем.
    # raw_reg уже list[float] из run_regression_model / калибратора.
    masked: List[float] = []
    for c, r, lawyer in zip(labels, raw_reg, is_lawyer):
        if c or lawyer:
            masked.append(r)
        else:
            masked.append(0.0)
    return masked