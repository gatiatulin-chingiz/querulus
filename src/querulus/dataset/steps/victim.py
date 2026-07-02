"""Шаг пайплайна: victim."""
from __future__ import annotations

from querulus.dataset.io import read_parquet_path
from querulus.dataset.paths import DataPaths


def load_victim(paths: DataPaths):
    """Загрузить victim parquet целиком с фильтром по строкам."""
    df_victim = read_parquet_path(paths.victim_path, artifact="victim")

    return df_victim.query(
        'REFUND_FORM_DETAILED in ["Ремонт","Денежная","Денежная. Отказ от ремонта","Ремонт. Смена СТОА"]'
        'and LOSS_DATE_TIME >= "2022-01-01"'
        'and LOSS_DATE_TIME <= "2025-06-30"'
        'and LOSS_PROCESS in ["Прямое ОСАГО (с 1 марта 2009)","Традиционное ОСАГО"]'
        'and RISK == "Ущерб имуществу третьих лиц"'
    )
