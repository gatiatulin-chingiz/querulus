"""Static person features from victim row (no external history)."""

from __future__ import annotations

import pandas as pd

from querulus.features.person.config import PERSON_PREFIX, ROLES


def add_person_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить FE_PERSON_STATIC_* признаки на одной строке инцидента."""
    out = df.copy()

    # Equality flags between role person ids.
    role_cols = [(role.suffix, role.person_id_column) for role in ROLES if role.person_id_column in out.columns]
    for i in range(len(role_cols)):
        left_name, left_col = role_cols[i]
        for j in range(i + 1, len(role_cols)):
            right_name, right_col = role_cols[j]
            col_name = f"{PERSON_PREFIX}STATIC_EQ_{left_name}_{right_name}"
            left = out[left_col]
            right = out[right_col]
            both = left.notna() & right.notna()
            out[col_name] = (left == right).where(both).astype("Int64")

    # Age differences (where present).
    age_cols = {
        "APPLICANT": "APPLICANT_AGE",
        "VICTIM": "VICTIM_AGE",
        "GUILTY": "GUILTY_AGE",
        "DRIVER": "DRIVER_AGE",
        "PAYMENT_RECIPIENT": "PAYMENT_RECIPIENT_AGE",
    }
    for left_role, left_col in age_cols.items():
        if left_col not in out.columns:
            continue
        for right_role, right_col in age_cols.items():
            if right_col not in out.columns or right_role <= left_role:
                continue
            name = f"{PERSON_PREFIX}STATIC_DIFF_{left_role}_AGE_{right_role}_AGE"
            out[name] = pd.to_numeric(out[left_col], errors="coerce") - pd.to_numeric(out[right_col], errors="coerce")

    return out

