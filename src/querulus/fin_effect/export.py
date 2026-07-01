"""Экспорт таблиц финансового эффекта."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from querulus.fin_effect.calculator import prepare_analytics_export
from querulus.fin_effect.config import FinEffectConfig
from querulus.fin_effect.summary import export_summary_excel


def export_analytics_excel(
    df: pd.DataFrame,
    path: str | Path,
    *,
    config: FinEffectConfig | None = None,
    rename: bool = True,
) -> Path:
    """Сохранить построчную аналитику в Excel (Аналитика.xlsx)."""
    config = config or FinEffectConfig()
    output_path = Path(path)
    export_df = prepare_analytics_export(df, config, rename=rename)
    export_df.to_excel(output_path, index=False)
    return output_path
