"""Re-export pretension utils (обратная совместимость)."""
from querulus.dataset.load.sql import pretension_surcharge_params, render_sql
from querulus.dataset.preprocess.pretension import *  # noqa: F403

__all__ = [
    "PRETENSION_PAID_ANSWER_TYPES",
    "collapse_pretension_surcharge_to_incident",
    "dedupe_pretension_rows",
    "pretension_surcharge_by_incident_sql",
]


def pretension_surcharge_by_incident_sql(
    *,
    surcharge_alias: str,
    uts_alias: str,
    pretension_types: tuple[str, ...] | None = None,
    answer_types: tuple[str, ...] = PRETENSION_PAID_ANSWER_TYPES,
) -> str:
    """Deprecated: используйте ``render_sql('pretension_surcharge_by_incident.sql', ...)``."""
    return render_sql(
        "pretension_surcharge_by_incident.sql",
        **pretension_surcharge_params(
            surcharge_alias=surcharge_alias,
            uts_alias=uts_alias,
            pretension_types=pretension_types,
            answer_types=answer_types,
        ),
    )
