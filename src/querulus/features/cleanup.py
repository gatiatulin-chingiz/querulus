"""Этап 0: dedup merge-колонок после join."""
from __future__ import annotations

import pandas as pd

from querulus.features.config import FeatureConfig


def cleanup_merge_columns(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Свести _x/_y merge-колонки в одну целевую без удаления исходников."""
    target = config.dedup_target
    sources = [col for col in config.dedup_sources if col in df.columns]
    if not sources:
        return df

    out = df.copy()
    merged = out[target] if target in out.columns else pd.Series(pd.NA, index=out.index)

    for col in sources:
        merged = merged.fillna(out[col])

    out[target] = merged
    return out
