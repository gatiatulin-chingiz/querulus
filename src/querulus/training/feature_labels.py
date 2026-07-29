"""Русские названия признаков для бизнес-отчётов."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from querulus import PROJECT_ROOT

# Сырые колонки и частые FE без отдельной строки в catalog.
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
    "VICTIM_VEHICLE_MADE_IN_RF": "ТС потерпевшего произведено в РФ",
    "VICTIM_VEHICLE_IS_JAPAN": "Японское ТС потерпевшего",
    "VICTIM_MAX_WEIGHT": "Макс. масса ТС потерпевшего",
    "VICTIM_CAPACITY_ENGINE": "Мощность двигателя потерпевшего",
    "VICTIM_NUM_DOORS": "Число дверей ТС потерпевшего",
    "VICTIM_NUM_PLACE": "Число мест ТС потерпевшего",
    "VICTIM_TYPE_ENGINE": "Тип двигателя потерпевшего",
    "VICTIM_TYPE_BODY": "Тип кузова потерпевшего",
    "VICTIM_AGE": "Возраст потерпевшего",
    "GUILTY_CAPACITY_ENGINE": "Мощность двигателя виновника",
    "GUILTY_MAX_WEIGHT": "Макс. масса ТС виновника",
    "GUILTY_VEHICLE_AGE": "Возраст ТС виновника",
    "GUILTY_VEHICLE_CATEGORY": "Категория ТС виновника",
    "GUILTY_VEHICLE_COUNTRY": "Страна ТС виновника",
    "GUILTY_OBJECT_YEAR": "Год выпуска ТС виновника",
    "GUILTY_AGE": "Возраст виновника",
    "DRIVER_AGE": "Возраст водителя",
    "APPLICANT_AGE": "Возраст заявителя",
    "APPLICANT_SEX": "Пол заявителя",
    "PAYMENT_RECIPIENT_AGE": "Возраст получателя выплаты",
    "EVENT_YEAR": "Год ДТП",
    "EVENT_MONTH": "Месяц ДТП",
    "EVENT_HOUR": "Час ДТП",
    "EVENT_DAY": "День недели ДТП",
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
    "FE_VALUE_BEFORE_WITHOUT_REAL_2020": "Калькуляция без износа (руб. 2020)",
    "FE_VALUE_BEFORE_WITH_REAL_2020": "Калькуляция с износом (руб. 2020)",
    "FE_PREMIUM_SUM_ALL_REAL_2020": "Сумма премий (руб. 2020)",
}

_TOKEN_RU: dict[str, str] = {
    "FE": "признак",
    "PERSON": "лицо",
    "STATIC": "статический",
    "PRET": "претензии",
    "INCIDENT": "инцидент",
    "DECLARED": "заявлено",
    "DIFF": "разница",
    "SUM": "сумма",
    "COUNT": "число",
    "AGE": "возраст",
    "YEAR": "год",
    "APPLICANT": "заявитель",
    "VICTIM": "потерпевший",
    "GUILTY": "виновник",
    "DRIVER": "водитель",
    "OWNER": "собственник",
    "POLICYHOLDER": "страхователь",
    "PAYMENT": "выплата",
    "RECIPIENT": "получатель",
    "PH": "страхователь",
    "COURT": "суд",
    "SURCHARGE": "доплата",
    "PRETENSION": "претензия",
    "VALUE": "сумма",
    "UTSVALUE": "УТС",
    "UTS": "УТС",
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
        if "{" in name or name.lower() in {"фича", "колонки", "фича / шаблон", "шаблон фичи"}:
            continue
        if description.lower() in {"описание", "------"}:
            continue
        labels[name] = description
    return labels


def _humanize_ru(feature: str) -> str:
    """Собрать русское имя из токенов, если нет готового словаря."""
    parts = [p for p in feature.split("_") if p]
    translated = [_TOKEN_RU.get(p.upper(), p.lower()) for p in parts]
    return " ".join(translated)


def feature_ru_name(feature: str, catalog_path: str | None = None) -> str:
    """Русское имя фичи: RAW → catalog → шаблоны → токены (без английского fallback)."""
    if feature in RAW_FEATURE_RU:
        return RAW_FEATURE_RU[feature]
    catalog = load_catalog_ru_labels(catalog_path)
    if feature in catalog:
        return catalog[feature]
    if feature.startswith("FE_PERSON_STATIC_"):
        return "Статический признак лица: " + _humanize_ru(feature[len("FE_PERSON_STATIC_") :])
    if feature.startswith("FE_PERSON_PRET_"):
        return "История претензий лица: " + _humanize_ru(feature[len("FE_PERSON_PRET_") :])
    if feature.startswith("FE_PERSON_COURT_"):
        return "Судебная история лица: " + _humanize_ru(feature[len("FE_PERSON_COURT_") :])
    if feature.startswith("FE_INCIDENT_DECLARED_"):
        return "Сумма заявленного по претензиям инцидента: " + _humanize_ru(
            feature[len("FE_INCIDENT_DECLARED_") :]
        )
    if feature.startswith("FE_INCIDENT_"):
        return "Агрегат претензий инцидента: " + _humanize_ru(feature[len("FE_INCIDENT_") :])
    return _humanize_ru(feature)
