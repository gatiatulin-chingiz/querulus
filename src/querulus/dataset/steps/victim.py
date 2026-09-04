"""Шаг пайплайна: victim (обратная совместимость)."""
from __future__ import annotations

from querulus.dataset.load.victim import fetch_loss_object_types, fetch_victim_frame
from querulus.dataset.preprocess.victim import prepare_victim


def load_victim(
    paths,
    conn,
    *,
    use_sql: bool = False,
    save_checkpoint: bool = True,
):
    """Загрузить victim parquet, присоединить VictimObjectType из SQL и отфильтровать."""
    df_victim = fetch_victim_frame(paths)
    df_loss_types = fetch_loss_object_types(
        paths, conn, use_sql=use_sql, save_checkpoint=save_checkpoint
    )
    return prepare_victim(df_victim, df_loss_types)
