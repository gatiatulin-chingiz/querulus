"""Конфиг person-feature engineering (v1).

Принципы:
- Якорь времени: T0 = LOSS_DATE_TIME текущей строки (инцидента).
- История: берём только события по предыдущим инцидентам того же человека:
  `event_date < T0` и `INCIDENT_NUMBER != текущий`.
- ID используются только как ключи join; в обучение не идут (TO_DROP).
- Претензии: join по ApplicantPersonID и/или RecipientPersonID (oisuu81_t_Pretensions).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PersonRole:
    """Роль, колонка person_id в victim и ключи join к претензиям."""

    name: str
    person_id_column: str
    pretension_person_columns: tuple[str, ...]

    @property
    def suffix(self) -> str:
        return self.name.upper()


T0_COLUMN: str = "LOSS_DATE_TIME"
INCIDENT_COLUMN: str = "INCIDENT_NUMBER"

# Колонки person_id в oisuu81_t_Pretensions (после нормализации).
PRETENSION_APPLICANT_COL: str = "APPLICANT_PERSON_ID"
PRETENSION_RECIPIENT_COL: str = "RECIPIENT_PERSON_ID"
PRETENSION_BOTH_COLS: tuple[str, ...] = (PRETENSION_APPLICANT_COL, PRETENSION_RECIPIENT_COL)


ROLES: tuple[PersonRole, ...] = (
    PersonRole("APPLICANT", "APPLICANT_ID", (PRETENSION_APPLICANT_COL,)),
    PersonRole("VICTIM_PH", "VICTIM_POLICYHOLDER_PERSON_ID", PRETENSION_BOTH_COLS),
    PersonRole("VICTIM", "VICTIM_PERSON_ID", PRETENSION_BOTH_COLS),
    PersonRole("GUILTY", "GUILTY_PERSON_ID", PRETENSION_BOTH_COLS),
    PersonRole("DRIVER", "DRIVER_PERSON_ID", PRETENSION_BOTH_COLS),
    PersonRole("PAYMENT_RECIPIENT", "PAYMENT_RECIPIENT_PERSON_ID", (PRETENSION_RECIPIENT_COL,)),
    PersonRole("VICTIM_OWNER", "VICTIM_OBJECT_OWNER_PERSON_ID", PRETENSION_BOTH_COLS),
    PersonRole("GUILTY_OWNER", "GUILTY_OBJECT_OWNER_PERSON_ID", PRETENSION_BOTH_COLS),
    PersonRole("POLICYHOLDER", "POLICYHOLDER_PERSON_ID", PRETENSION_BOTH_COLS),
    PersonRole("POLICYHOLDER_OWNER", "POLICYHOLDER_OBJECT_OWNER_PERSON_ID", PRETENSION_BOTH_COLS),
)


PERSON_PREFIX: str = "FE_PERSON_"
