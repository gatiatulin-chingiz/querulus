"""Утилиты дедупликации претензий."""
from __future__ import annotations

import pandas as pd


def dedupe_pretension_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Убрать дубликаты строк претензий после JOIN с IncidentToLoss."""
    result = df.loc[:, ~df.columns.duplicated()].copy()
    for key in ("PRETENSION_ID", "PRETENSIONID"):
        if key in result.columns:
            return result.drop_duplicates(subset=[key])
    if "PRETENSION_NUMBER" in result.columns:
        subset = ["PRETENSION_NUMBER"]
        if "LOSS_ID" in result.columns:
            subset.append("LOSS_ID")
        elif "LOSSID" in result.columns:
            subset.append("LOSSID")
        return result.drop_duplicates(subset=subset)
    return result.drop_duplicates()
