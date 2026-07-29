"""Русские названия признаков для бизнес-отчётов."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from querulus import PROJECT_ROOT

# Сырые колонки (не FE_*), которых нет как отдельных строк в catalog.
RAW_FEATURE_RU: dict[str, str] = {
    "EVENT_CREATED_BY_GIBDD_FLAG": "Оформление ДТП через ГИБДД",
    "FILIAL": "Филиал",
    "VICTIM_VEHICLE_CATEGORY": "Категория ТС потерпевшего",
    "APPLICANT_FORM": "Форма заявителя",
    "RECIEVE_METHOD": "Способ подачи заявления",
    "LOSS_UNIT_ZONE": "Зона урегулирующего подразделения",
    "LOSS_UNIT": "Урегулирующее подразделение",
    "VICTIM_VEHICLE_COUNTRY": "Страна ТС потерпевшего",
    "VICTIM_VEHICLE_AGE": "Возраст ТС потерпевшего",
    "VICTIM_VEHICLE_BRAND": "Бренд ТС потерпевшего",
    "VICTIM_MAX_WEIGHT": "Макс. масса ТС потерпевшего",
    "VICTIM_CAPACITY_ENGINE": "Мощность двигателя потерпевшего",
    "GUILTY_CAPACITY_ENGINE": "Мощность двигателя виновника",
    "GUILTY_MAX_WEIGHT": "Макс. масса ТС виновника",
    "GUILTY_VEHICLE_AGE": "Возраст ТС виновника",
    "GUILTY_VEHICLE_CATEGORY": "Категория ТС виновника",
    "GUILTY_VEHICLE_COUNTRY": "Страна ТС виновника",
    "APPLICANT_AGE": "Возраст заявителя",
    "APPLICANT_SEX": "Пол заявителя",
    "EVENT_YEAR": "Год ДТП",
    "EVENT_MONTH": "Месяц ДТП",
    "EVENT_HOUR": "Час ДТП",
    "APPLY_DELAY": "Задержка подачи заявления (дни)",
    "PARTICIPANTS_COUNT": "Число участников ДТП",
    "REGION": "Регион убытка",
    "REGION_EVENT": "Регион ДТП",
    "ACCEPTED_UNIT": "Принявшее подразделение",
    "NOT_NOTIFICATION": "Нет уведомления о ДТП",
    "CUSTOMER_IMPORTANCE": "Важность клиента",
    "INSURANCE_AMOUNT": "Страховая сумма",
    "FRANCHISE_VALUE": "Франшиза",
    "RSA_POLICY_KBM": "КБМ",
    "RSAPolicyKBM": "КБМ",
    "USED_AS_TAXI": "Использование как такси",
    "USED_AS_CARSH": "Использование как каршеринг",
    "DTPOSAGO_TYPE": "Тип ДТП ОСАГО",
    "DTPOSAGOType": "Тип ДТП ОСАГО",
}

_CATALOG_ROW_RE = re.compile(
    r"\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|",
)
_DEFAULT_CATALOG = PROJECT_ROOT / "configs" / "features_catalog.md"


@lru_cache(maxsize=4)
def load_catalog_ru_labels(catalog_path: str | None = None) -> dict[str, str]:
    """Разобрать ``features_catalog.md``: feature → краткое русское описание."""
    path = Path(catalog_path) if catalog_path else _DEFAULT_CATALOG
    if not path.exists():
        return {}
    labels: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _CATALOG_ROW_RE.search(line)
        if not match:
            continue
        name, description = match.group(1).strip(), match.group(2).strip()
        # Пропуск шаблонов вида FE_*_{FIELD}_*
        if "{" in name or name.lower() in {"фича", "колонки", "фича / шаблон", "шаблон фичи"}:
            continue
        if description.lower() in {"описание", "------"}:
            continue
        labels[name] = description
    return labels


def feature_ru_name(feature: str, catalog_path: str | None = None) -> str:
    """Русское имя фичи: catalog → RAW → человекочитаемый fallback."""
    catalog = load_catalog_ru_labels(catalog_path)
    if feature in catalog:
        return catalog[feature]
    if feature in RAW_FEATURE_RU:
        return RAW_FEATURE_RU[feature]
    # Префиксные шаблоны person / incident
    for prefix, title in (
        ("FE_PERSON_STATIC_", "Person static"),
        ("FE_PERSON_PRET_", "Person pretensions"),
        ("FE_INCIDENT_DECLARED_", "Сумма Declared по претензиям инцидента"),
        ("FE_INCIDENT_", "Агрегат претензий инцидента"),
    ):
        if feature.startswith(prefix):
            return f"{title}: {feature[len(prefix):]}"
    return feature.replace("_", " ").strip()
