"""Шаг пайплайна: targets (обратная совместимость)."""
from querulus.dataset.preprocess.targets import (  # noqa: F401
    CLAIM_PERIOD_COL,
    TARGET_3_SEV_COMPONENT_COLS,
    TARGET_FREQ_CLAIMS_COMPONENT_COLS,
    TARGET_FREQ_CLAIMS_GROUP,
    TARGET_FREQ_COMPONENT_COLS,
    TARGET_SEV_CLAIMS_COMPONENT_COLS,
    TARGET_SEV_COMPONENT_COLS,
    build_targets,
    ensure_claims_targets,
    is_void_claim_instance,
    pick_last_claim_instances,
)

__all__ = [
    "CLAIM_PERIOD_COL",
    "TARGET_3_SEV_COMPONENT_COLS",
    "TARGET_FREQ_CLAIMS_COMPONENT_COLS",
    "TARGET_FREQ_CLAIMS_GROUP",
    "TARGET_FREQ_COMPONENT_COLS",
    "TARGET_SEV_CLAIMS_COMPONENT_COLS",
    "TARGET_SEV_COMPONENT_COLS",
    "build_targets",
    "ensure_claims_targets",
    "is_void_claim_instance",
    "pick_last_claim_instances",
]
