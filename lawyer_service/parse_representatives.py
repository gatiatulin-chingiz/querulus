"""Раскладка «Представители автоюристов, все филиалы.xlsx» в построчный шаблон."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path

import openpyxl
from openpyxl.styles import Font
from openpyxl.utils.datetime import from_excel

SRC = Path(__file__).with_name("Представители автоюристов, все филиалы.xlsx")
OUT = Path(__file__).with_name("Представители автоюристов_разложенные.xlsx")

ORG_TYPES = r"ООО|ОАО|ЗАО|ПАО|АО|НАО|НКО|ФГУП|ГУП|МУП|АНО|КФХ"
ORG_RE = re.compile(rf"\b(?:{ORG_TYPES})\b", re.IGNORECASE)
IP_RE = re.compile(r"\bИП\b", re.IGNORECASE)
DIR_RE = re.compile(
    r"(?:[-–—,]?\s*)?(?:директор|руководитель)\s*-?\s*",
    re.IGNORECASE,
)
DATE_RE = re.compile(r"\b(\d{2}\.\d{2}\.\d{4})(?!\d)")
INN_INLINE_RE = re.compile(r"\(\s*ИНН\s*([0-9]{10,12})\s*\)", re.IGNORECASE)
INN_TOKEN_RE = re.compile(r"(?:ИНН\s*)?([0-9]{10,12})", re.IGNORECASE)
PHONE_RE = re.compile(
    r"(?:тел\.?\s*)(?:\+?7|8)[\s\-\(\)]*\d{3}[\s\-\(\)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}"
    r"|(?<!\d)(?:\+7|8)(?:[\s\-\(\)]+\d{3}){2}[\s\-]+\d{2}[\s\-]+\d{2}(?!\d)",
    re.IGNORECASE,
)
URL_RE = re.compile(r"https?://\S+|сайт\s*-?\s*", re.IGNORECASE)
JUNK_PREFIX_RE = re.compile(
    r"^(?:независимая\s+оценка|автоюрист|адвокат|оценка)\b.*?",
    re.IGNORECASE,
)

OUT_COLS = [
    "ФИЛИАЛ",
    "ФИО",
    "ДАТА РОЖДЕНИЯ",
    "ИНН",
    "ТЕЛЕФОН",
    "E-MAIL",
    "АДРЕС",
    "КАТЕГОРИЯ",
    "ИСТОЧНИК (№ СТРОКИ)",
    "AHANTER_CHECK",
]

KIND_TO_CATEGORY = {
    "person": "физ.лицо",
    "ul": "юр.лицо",
    "ip": "ИП",
}

# «Аванесов Артур Михайлович ИП» → нужно переставить маркер в начало
TRAILING_IP_RE = re.compile(
    r"([А-Яа-яЁёA-Za-z][А-Яа-яЁёA-Za-z\-\.]*(?:\s+[А-Яа-яЁёA-Za-z][А-Яа-яЁёA-Za-z\-\.]*){0,3})\s+ИП\b",
    re.IGNORECASE,
)


@dataclass
class Entity:
    name: str
    kind: str  # person | ul | ip
    dob: str = ""
    inn: str = ""
    phone: str = ""
    source: str = ""  # B | C

    @property
    def ahanter_check(self) -> int:
        return 0 if self.kind in {"ul", "ip"} else 1

    @property
    def category(self) -> str:
        return KIND_TO_CATEGORY.get(self.kind, "физ.лицо")


def _norm_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip(" ,;.-")


def _norm_key(name: str) -> str:
    text = name.upper()
    text = re.sub(r"^ИП\s+", "", text)
    text = text.replace("«", '"').replace("»", '"').replace("“", '"').replace("”", '"')
    text = re.sub(r'["\'«»]', "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_person_name(text: str) -> bool:
    text = _norm_space(text)
    if not text or ORG_RE.search(text) or IP_RE.match(text):
        return False
    if re.search(r"\d", text):
        return False
    words = text.split()
    if not (1 <= len(words) <= 4):
        return False
    # Отсекаем явно не-ФИО
    if words[0].lower() in {"сайт", "тел", "инн", "г"}:
        return False
    return bool(re.match(r"^[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-\.]*$", words[0]))


def _cell_to_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%d.%m.%Y")
    if isinstance(value, date):
        return value.strftime("%d.%m.%Y")
    if isinstance(value, bool):
        return ""
    if isinstance(value, int):
        # Excel serial date, попавший как int
        if 20000 <= value <= 60000:
            try:
                return from_excel(value).strftime("%d.%m.%Y")
            except Exception:
                return str(value)
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return _cell_to_str(int(value))
        return str(value)
    text = str(value).strip()
    if text.upper() in {"#VALUE!", "#REF!", "#N/A", "NONE", "NAN"}:
        return ""
    return text


def _extract_inline_inn(text: str) -> tuple[str, str]:
    inn = ""
    match = INN_INLINE_RE.search(text)
    if match:
        inn = match.group(1)
        text = INN_INLINE_RE.sub("", text)
    return _norm_space(text), inn


def _flip_trailing_ip(text: str) -> str:
    """«ФИО ИП» → «ИП ФИО», чтобы split не оставлял голое «ИП»."""

    def _repl(match: re.Match) -> str:
        fio = _norm_space(match.group(1))
        if _is_person_name(fio):
            return f"ИП {fio}"
        return match.group(0)

    return TRAILING_IP_RE.sub(_repl, text)


def _clean_person(text: str) -> str:
    text, _ = _extract_inline_inn(text)
    text = URL_RE.sub(" ", text)
    text = re.sub(r"\bг\.\s*[А-Яа-яЁё\-]+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.strip(" ,;.-")
    # хвостовое ИП убираем из ФИО — статус обрабатывается отдельно
    text = re.sub(r"\s+ИП\s*$", "", text, flags=re.IGNORECASE)
    return _norm_space(text)


def _clean_org_name(raw: str) -> str:
    text, _ = _extract_inline_inn(raw)
    text = URL_RE.sub(" ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = _norm_space(text)
    # ООО "Название" / ООО «Название»
    m = re.match(
        rf"^((?:{ORG_TYPES}))\s*[\"«](.+?)[\"»]\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return f'{m.group(1).upper()} "{m.group(2).strip()}"'
    m = re.match(rf"^((?:{ORG_TYPES}))\s+(.+)$", text, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()} {m.group(2).strip()}"
    return text


def _clean_ip_name(raw: str) -> str:
    text, inn = _extract_inline_inn(raw)
    text = URL_RE.sub(" ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\bг\.\s*[А-Яа-яЁё\-]+\b", " ", text, flags=re.IGNORECASE)
    text = _norm_space(text)
    # голое «ИП» без ФИО — мусор от неудачного split
    if re.fullmatch(r"ИП", text, flags=re.IGNORECASE):
        return ""
    # ИП «Алеварский» / ИП Фамилия Имя Отчество
    m = re.match(r"^ИП\s*[\"«](.+?)[\"»]\s*$", text, flags=re.IGNORECASE)
    if m:
        return f'ИП "{m.group(1).strip()}"'
    m = re.match(r"^ИП\s+(.+)$", text, flags=re.IGNORECASE)
    if m:
        return f"ИП {_norm_space(m.group(1))}"
    return text


def _parse_lawyer_cell(text: str) -> list[Entity]:
    """Разбирает колонку «Автоюрист» на последовательность сущностей."""
    text = _cell_to_str(text)
    if not text:
        return []

    text = text.replace("\n", " ")
    text = URL_RE.sub(" ", text)
    text = _flip_trailing_ip(text)
    entities: list[Entity] = []

    # Режем по маркерам юрлиц / ИП / директор|руководитель, сохраняя разделители
    parts = re.split(
        rf"(?=(?:\b(?:{ORG_TYPES}|ИП)\b)|(?:(?:директор|руководитель)\s*-?\s*))",
        text,
        flags=re.IGNORECASE,
    )
    parts = [_norm_space(p) for p in parts if _norm_space(p)]

    for part in parts:
        if not part:
            continue

        # директор / руководитель ФИО
        if re.match(r"^(?:директор|руководитель)\b", part, flags=re.IGNORECASE):
            person_raw = DIR_RE.sub("", part, count=1)
            person_raw, inn = _extract_inline_inn(person_raw)
            # отрезаем хвост, если вдруг прилипло следующее ООО
            person_raw = re.split(rf"\b(?:{ORG_TYPES}|ИП)\b", person_raw, maxsplit=1)[0]
            person_raw = _clean_person(person_raw)
            if _is_person_name(person_raw):
                entities.append(Entity(person_raw, "person", inn=inn, source="C"))
            continue

        if IP_RE.match(part):
            chunk = part
            # ИП ... до следующего маркера уже отрезано split'ом, но подчистим хвост
            chunk = re.split(r"\b(?:директор|руководитель)\b", chunk, maxsplit=1)[0]
            chunk, inn = _extract_inline_inn(chunk)
            name = _clean_ip_name(chunk)
            # не сохраняем голое «ИП»
            if name and not re.fullmatch(r"ИП", name, flags=re.IGNORECASE):
                entities.append(Entity(name, "ip", inn=inn, source="C"))
            continue

        if ORG_RE.match(part):
            chunk = part
            # ООО "X" - директор Y  → org до директора
            chunk = re.split(r"\b(?:директор|руководитель)\b", chunk, maxsplit=1)[0]
            chunk = chunk.rstrip(" -–—,;")
            chunk, inn = _extract_inline_inn(chunk)
            # «ООО Автоправо Федченко Виталий Валерьевич» → org + person
            trailing_person = ""
            if not re.search(r"[\"«»]", chunk):
                words = _norm_space(chunk).split()
                if len(words) >= 4:
                    maybe_fio = " ".join(words[-3:]).rstrip(".")
                    if _is_person_name(maybe_fio):
                        trailing_person = _clean_person(maybe_fio)
                        chunk = " ".join(words[:-3])
            name = _clean_org_name(chunk)
            if name:
                entities.append(Entity(name, "ul", inn=inn, source="C"))
            if trailing_person:
                entities.append(Entity(trailing_person, "person", source="C"))
            continue

        # Прочий текст: возможно ФИО или мусор вроде «независимая оценка "Гарант", Божко Назар»
        chunk, inn = _extract_inline_inn(part)
        chunk = chunk.strip(" ,;.-")
        if "," in chunk:
            # Берём куски через запятую
            for sub in chunk.split(","):
                sub = _norm_space(sub)
                sub = JUNK_PREFIX_RE.sub("", sub).strip()
                # если осталось название в кавычках без типа — пропускаем как не-сущность
                if _is_person_name(sub):
                    entities.append(Entity(_clean_person(sub), "person", source="C"))
            continue

        chunk = JUNK_PREFIX_RE.sub("", chunk).strip()
        if _is_person_name(chunk):
            entities.append(Entity(_clean_person(chunk), "person", inn=inn, source="C"))

    return entities


def _parse_ids(value) -> tuple[list[str], list[str], list[str]]:
    """Возвращает (dates, inns, phones) из колонки D."""
    dates: list[str] = []
    inns: list[str] = []
    phones: list[str] = []

    if isinstance(value, datetime):
        return [value.strftime("%d.%m.%Y")], [], []
    if isinstance(value, date):
        return [value.strftime("%d.%m.%Y")], [], []

    text = _cell_to_str(value)
    if not text:
        return dates, inns, phones

    for m in PHONE_RE.finditer(text):
        phones.append(_norm_space(m.group(0)))
    text_wo_phone = PHONE_RE.sub(" ", text)

    for m in DATE_RE.finditer(text_wo_phone):
        dates.append(m.group(1))

    # ИНН: явные «ИНН N» и просто 10/12 цифр
    for m in re.finditer(r"ИНН\s*([0-9]{10,12})", text_wo_phone, flags=re.IGNORECASE):
        inns.append(m.group(1))
    if not inns:
        for m in re.finditer(r"\b([0-9]{10,12})\b", text_wo_phone):
            inns.append(m.group(1))

    return dates, inns, phones


def _person_key_from_ip(ip_name: str) -> str:
    return _norm_key(re.sub(r"^ИП\s+", "", ip_name, flags=re.IGNORECASE))


def _entity_from_rep(rep_raw) -> Entity | None:
    """Представитель из колонки B, с учётом хвоста «… ИП»."""
    text = _norm_space(_cell_to_str(rep_raw)).rstrip(",")
    if not text:
        return None
    text = _flip_trailing_ip(text)
    if IP_RE.match(text):
        name = _clean_ip_name(text)
        if name and not re.fullmatch(r"ИП", name, flags=re.IGNORECASE):
            return Entity(name, "ip", source="B")
        return None
    if _is_person_name(text):
        return Entity(_clean_person(text), "person", source="B")
    return Entity(_norm_space(text), "person", source="B")


def _merge_entities(rep: Entity | None, lawyer_entities: list[Entity]) -> list[Entity]:
    """Собирает итоговый список: представитель + автоюристы, с дедупом и приоритетом ИП."""
    result: list[Entity] = []

    if rep is not None:
        result.append(rep)

    for ent in lawyer_entities:
        key = _norm_key(ent.name)
        if ent.kind == "ip":
            pkey = _person_key_from_ip(ent.name)
            # человек из B совпал с ИП → оставляем ИП
            merged = False
            for i, existing in enumerate(result):
                if existing.kind == "person" and _norm_key(existing.name) == pkey:
                    result[i] = Entity(
                        ent.name,
                        "ip",
                        dob=existing.dob or ent.dob,
                        inn=existing.inn or ent.inn,
                        phone=existing.phone or ent.phone,
                        source=existing.source,
                    )
                    merged = True
                    break
                if existing.kind == "ip" and _norm_key(existing.name) == key:
                    if not existing.inn and ent.inn:
                        existing.inn = ent.inn
                    if not existing.dob and ent.dob:
                        existing.dob = ent.dob
                    merged = True
                    break
            if not merged:
                result.append(ent)
            continue

        if ent.kind == "person":
            merged = False
            for existing in result:
                if existing.kind == "person" and _norm_key(existing.name) == key:
                    if not existing.inn and ent.inn:
                        existing.inn = ent.inn
                    if not existing.dob and ent.dob:
                        existing.dob = ent.dob
                    merged = True
                    break
                if existing.kind == "ip" and _person_key_from_ip(existing.name) == key:
                    if not existing.inn and ent.inn:
                        existing.inn = ent.inn
                    if not existing.dob and ent.dob:
                        existing.dob = ent.dob
                    merged = True
                    break
            if not merged:
                result.append(ent)
            continue

        # ul — добавляем, если такого ещё нет
        for existing in result:
            if existing.kind == "ul" and _norm_key(existing.name) == key:
                if not existing.inn and ent.inn:
                    existing.inn = ent.inn
                break
        else:
            result.append(ent)

    return result


def _assign_ids(entities: list[Entity], dates: list[str], inns: list[str], phones: list[str]) -> None:
    # даты — сначала ИП без даты, затем физлица (см. строки вроде «Сизова + ИП Шутов»)
    date_targets = [e for e in entities if e.kind == "ip" and not e.dob]
    date_targets += [e for e in entities if e.kind == "person" and not e.dob]
    for dob, ent in zip(dates, date_targets):
        ent.dob = dob

    # ИНН
    remaining = list(inns)
    for inn in list(remaining):
        assigned = False
        if len(inn) == 10:
            for ent in entities:
                if ent.kind == "ul" and not ent.inn:
                    ent.inn = inn
                    assigned = True
                    break
        elif len(inn) == 12:
            # 12 знаков — сначала ИП, затем физлица
            for kind in ("ip", "person"):
                for ent in entities:
                    if ent.kind == kind and not ent.inn:
                        ent.inn = inn
                        assigned = True
                        break
                if assigned:
                    break
        if assigned:
            remaining.remove(inn)

    # хвост нестандартной длины — ЮЛ, затем ИП, затем физлица
    for inn in remaining:
        placed = False
        for kind in ("ul", "ip", "person"):
            for ent in entities:
                if ent.kind == kind and not ent.inn:
                    ent.inn = inn
                    placed = True
                    break
            if placed:
                break

    # телефон — первому лицу/ИП без телефона, иначе первой сущности
    for phone in phones:
        for ent in entities:
            if ent.kind in {"person", "ip"} and not ent.phone:
                ent.phone = phone
                break
        else:
            if entities and not entities[0].phone:
                entities[0].phone = phone


def parse_row(filial, rep_raw, lawyer_raw, ids_raw, source_row: int) -> list[dict]:
    rep = _entity_from_rep(rep_raw)
    lawyer_entities = _parse_lawyer_cell(lawyer_raw)
    entities = _merge_entities(rep, lawyer_entities)
    dates, inns, phones = _parse_ids(ids_raw)

    # инлайн-ИНН из текста автоюриста уже лежат на сущностях;
    # ИНН из D не должны дублировать уже проставленные
    already = {e.inn for e in entities if e.inn}
    inns = [i for i in inns if i not in already]

    _assign_ids(entities, dates, inns, phones)

    rows = []
    for ent in entities:
        if not ent.name or re.fullmatch(r"ИП", ent.name, flags=re.IGNORECASE):
            continue
        rows.append(
            {
                "ФИЛИАЛ": _cell_to_str(filial),
                "ФИО": ent.name,
                "ДАТА РОЖДЕНИЯ": ent.dob,
                "ИНН": ent.inn,
                "ТЕЛЕФОН": ent.phone,
                "E-MAIL": "",
                "АДРЕС": "",
                "КАТЕГОРИЯ": ent.category,
                "ИСТОЧНИК (№ СТРОКИ)": source_row,
                "AHANTER_CHECK": ent.ahanter_check,
            }
        )
    return rows


_CATEGORY_RANK = {"физ.лицо": 0, "юр.лицо": 1, "ИП": 2}
_MERGE_FILL_COLS = ("ДАТА РОЖДЕНИЯ", "ИНН", "ТЕЛЕФОН", "E-MAIL", "АДРЕС")


def _nonempty(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def _merge_duplicate_rows(primary: dict, secondary: dict) -> dict:
    """Склеивает дубликаты: дозаполняет пустые поля, ИП важнее физика."""
    out = dict(primary)

    for col in _MERGE_FILL_COLS:
        if not _nonempty(out.get(col)) and _nonempty(secondary.get(col)):
            out[col] = secondary[col]

    # филиалы — уникальный список
    filials: list[str] = []
    for row in (primary, secondary):
        for part in str(row.get("ФИЛИАЛ") or "").split(";"):
            part = part.strip()
            if part and part not in filials:
                filials.append(part)
    if filials:
        out["ФИЛИАЛ"] = "; ".join(filials)

    # категория / ФИО: предпочитаем более «сильный» тип (ИП > юр.лицо > физ.лицо)
    rank_p = _CATEGORY_RANK.get(str(primary.get("КАТЕГОРИЯ")), 0)
    rank_s = _CATEGORY_RANK.get(str(secondary.get("КАТЕГОРИЯ")), 0)
    if rank_s > rank_p:
        out["КАТЕГОРИЯ"] = secondary["КАТЕГОРИЯ"]
        out["ФИО"] = secondary["ФИО"]
        out["AHANTER_CHECK"] = secondary["AHANTER_CHECK"]

    # источники — минимальный номер + перечень через ;
    src_nums: list[int] = []
    for row in (primary, secondary):
        raw = row.get("ИСТОЧНИК (№ СТРОКИ)")
        for part in str(raw).split(";"):
            part = part.strip()
            if part.isdigit():
                n = int(part)
                if n not in src_nums:
                    src_nums.append(n)
    src_nums.sort()
    if src_nums:
        out["ИСТОЧНИК (№ СТРОКИ)"] = (
            src_nums[0] if len(src_nums) == 1 else ";".join(map(str, src_nums))
        )

    return out


def _dedupe_by_key(rows: list[dict], key_fn) -> list[dict]:
    """Схлопывает строки с одинаковым ключом; строки без ключа оставляем как есть."""
    buckets: dict[str, dict] = {}
    order: list[str] = []
    passthrough: list[dict] = []

    for row in rows:
        key = key_fn(row)
        if not key:
            passthrough.append(row)
            continue
        if key in buckets:
            buckets[key] = _merge_duplicate_rows(buckets[key], row)
        else:
            buckets[key] = dict(row)
            order.append(key)

    return [buckets[k] for k in order] + passthrough


def deduplicate_rows(rows: list[dict]) -> list[dict]:
    """Дедуп: ИНН → ФИО+ДР → ФИО (одинаковые имена без ДР/ИНН тоже схлопываем)."""

    def inn_key(row: dict) -> str:
        return str(row.get("ИНН") or "").strip()

    def fio_dr_key(row: dict) -> str:
        fio = _norm_key(str(row.get("ФИО") or ""))
        dr = str(row.get("ДАТА РОЖДЕНИЯ") or "").strip()
        if not fio or not dr:
            return ""
        return f"{fio}|{dr}"

    def fio_key(row: dict) -> str:
        # не схлопываем мусор вроде голого «ООО»
        fio = _norm_key(str(row.get("ФИО") or ""))
        if not fio or fio in {"ООО", "ИП", "АО", "ПАО", "ЗАО"}:
            return ""
        return fio

    after_inn = _dedupe_by_key(rows, inn_key)
    after_fio_dr = _dedupe_by_key(after_inn, fio_dr_key)
    after_fio = _dedupe_by_key(after_fio_dr, fio_key)

    def sort_key(row: dict) -> tuple:
        raw = str(row.get("ИСТОЧНИК (№ СТРОКИ)") or "0").split(";")[0]
        try:
            return (int(raw), str(row.get("ФИО") or ""))
        except ValueError:
            return (0, str(row.get("ФИО") or ""))

    return sorted(after_fio, key=sort_key)


def main() -> None:
    wb = openpyxl.load_workbook(SRC, data_only=False)
    ws = wb.active

    out_rows: list[dict] = []
    for r in range(2, ws.max_row + 1):
        filial = ws.cell(r, 1).value
        rep = ws.cell(r, 2).value
        lawyer = ws.cell(r, 3).value
        ids = ws.cell(r, 4).value
        if rep is None and lawyer is None:
            continue
        out_rows.extend(parse_row(filial, rep, lawyer, ids, source_row=r))

    before = len(out_rows)
    out_rows = deduplicate_rows(out_rows)
    after = len(out_rows)

    out_wb = openpyxl.Workbook()
    out_ws = out_wb.active
    out_ws.title = "Разложенные"
    out_ws.append(OUT_COLS)
    for cell in out_ws[1]:
        cell.font = Font(bold=True)

    for row in out_rows:
        out_ws.append([row[c] for c in OUT_COLS])

    # ширины для читаемости
    widths = {
        "A": 16,
        "B": 40,
        "C": 16,
        "D": 16,
        "E": 18,
        "F": 24,
        "G": 30,
        "H": 12,
        "I": 18,
        "J": 14,
    }
    for col, w in widths.items():
        out_ws.column_dimensions[col].width = w

    saved_paths: list[Path] = []
    for target in (OUT, OUT.with_name(OUT.stem + "_v2.xlsx")):
        try:
            out_wb.save(target)
            saved_paths.append(target)
        except PermissionError:
            print(f"Файл занят (закройте Excel): {target.name}")
    print(f"Источник строк: {ws.max_row - 1}")
    print(f"До дедупа: {before}")
    print(f"После дедупа (ИНН → ФИО+ДР → ФИО): {after}")
    if saved_paths:
        print("Сохранено:")
        for p in saved_paths:
            print(f"  {p}")
    else:
        print("Не удалось сохранить ни один файл — закройте Excel и перезапустите.")


if __name__ == "__main__":
    main()
