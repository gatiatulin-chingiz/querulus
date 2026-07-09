"""Утилиты person-features (память, фильтрация истории)."""

from __future__ import annotations

import pandas as pd

from querulus.features.person.config import ROLES
from querulus.features.person.loaders import normalize_person_id_series


def collect_person_ids(df: pd.DataFrame) -> frozenset[str]:
    """Все person_id из ролей victim-строки (для фильтрации истории)."""
    ids: set[str] = set()
    for role in ROLES:
        column = role.person_id_column
        if column not in df.columns:
            continue
        values = normalize_person_id_series(df[column]).dropna()
        ids.update(values.tolist())
    return frozenset(ids)
