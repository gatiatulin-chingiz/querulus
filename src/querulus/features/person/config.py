"""Конфиг person-feature engineering (v1).

Принципы:
- Якорь времени: T0 = PAYMENT_ORDER_DATE_TIME текущей строки (инцидента).
- История: берём только события по предыдущим инцидентам того же человека:
  `event_date < T0` и `INCIDENT_NUMBER != текущий`.
- ID используются только как ключи join; в обучение не идут (TO_DROP).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PersonRole:
    """Роль и колонка person_id в victim-датасете."""

    name: str
    person_id_column: str

    @property
    def suffix(self) -> str:
        return self.name.upper()


T0_COLUMN: str = "PAYMENT_ORDER_DATE_TIME"
INCIDENT_COLUMN: str = "INCIDENT_NUMBER"


ROLES: tuple[PersonRole, ...] = (
    PersonRole("APPLICANT", "APPLICANT_ID"),
    PersonRole("VICTIM_PH", "VICTIM_POLICYHOLDER_PERSON_ID"),
    PersonRole("VICTIM", "VICTIM_PERSON_ID"),
    PersonRole("GUILTY", "GUILTY_PERSON_ID"),
    PersonRole("DRIVER", "DRIVER_PERSON_ID"),
    PersonRole("PAYMENT_RECIPIENT", "PAYMENT_RECIPIENT_PERSON_ID"),
    PersonRole("VICTIM_OWNER", "VICTIM_OBJECT_OWNER_PERSON_ID"),
    PersonRole("GUILTY_OWNER", "GUILTY_OBJECT_OWNER_PERSON_ID"),
    PersonRole("POLICYHOLDER", "POLICYHOLDER_PERSON_ID"),
    PersonRole("POLICYHOLDER_OWNER", "POLICYHOLDER_OBJECT_OWNER_PERSON_ID"),
)


PERSON_PREFIX: str = "FE_PERSON_"

