"""Re-export фильтров из preprocess (обратная совместимость)."""
from querulus.dataset.load.sql import loss_object_types_params, render_claims_predicate, render_sql
from querulus.dataset.preprocess.filters import *  # noqa: F403

__all__ = [
    "VICTIM_OBJECT_TYPE_COLUMN",
    "claims_sql_predicate",
    "ensure_victim_object_type_column",
    "load_dataset_filters",
    "loss_object_types_sql",
    "merge_loss_object_types",
    "select_primary_loss_per_incident",
    "victim_parquet_filter_query",
]

claims_sql_predicate = render_claims_predicate


def loss_object_types_sql(filters=None) -> str:
    """Deprecated: используйте ``render_sql('loss_object_types.sql', ...)``."""
    return render_sql("loss_object_types.sql", **loss_object_types_params(filters))
