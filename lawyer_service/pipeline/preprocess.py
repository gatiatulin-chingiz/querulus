"""Локальная очистка разложенной таблицы (из archive/ahanter.ipynb)."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _to_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, datetime):
        return value.strftime("%d.%m.%Y")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    return str(value).strip()


def clean_fio(fio_text: Any) -> str:
    text = _to_str(fio_text)
    if not text:
        return ""

    ip_prefix = ""
    upper = text.upper()
    if upper.startswith("ИП "):
        ip_prefix = "ИП "
        text = text[3:].strip()
    elif upper.startswith("ИНДИВИДУАЛЬНЫЙ ПРЕДПРИНИМАТЕЛЬ "):
        ip_prefix = "ИП "
        text = re.sub(
            r"^ИНДИВИДУАЛЬНЫЙ\s+ПРЕДПРИНИМАТЕЛЬ\s+",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

    text = re.sub(r"\s*\([^)]*\)", "", text)
    parts = text.split(",")
    if len(parts) > 1:
        first = parts[0].strip()
        if 2 <= len(first.split()) <= 3:
            text = first

    text = re.sub(
        r"\s*,?\s*(автоюрист|адвокат|директор|руководитель|менеджер|владелец)\b.*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = " ".join(text.split())
    return ip_prefix + text if text else _to_str(fio_text)


def clean_inn(inn_text: Any) -> str:
    text = _to_str(inn_text)
    text = re.sub(r"\bИНН\b", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", "", text)


def normalize_email(email: Any) -> str:
    text = _to_str(email).lower()
    return text


def normalize_date(date_value: Any) -> str:
    if date_value is None or (isinstance(date_value, float) and pd.isna(date_value)):
        return ""
    if isinstance(date_value, datetime):
        return date_value.strftime("%d.%m.%Y")
    text = _to_str(date_value)
    m = re.search(r"(\d{2}\.\d{2}\.\d{4})", text)
    if m:
        return m.group(1)
    try:
        dt = pd.to_datetime(text, dayfirst=True, errors="coerce")
        if pd.notna(dt):
            return dt.strftime("%d.%m.%Y")
    except Exception:
        pass
    return text


def normalize_inn(inn_value: Any) -> str:
    return clean_inn(inn_value)


def run_preprocess(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Читает разложенную таблицу, чистит поля, пишет xlsx."""
    df = pd.read_excel(input_path)
    print(f"preprocess: загружено {len(df)} строк из {input_path.name}")

    if "ФИО" in df.columns:
        # Не трогаем юрлица/ИП жёсткой clean_fio — только лёгкая нормализация пробелов
        def _smart_fio(row: pd.Series) -> str:
            cat = str(row.get("КАТЕГОРИЯ") or "")
            raw = row.get("ФИО")
            if cat in {"юр.лицо", "ИП"}:
                return " ".join(_to_str(raw).split())
            return clean_fio(raw)

        df["ФИО"] = df.apply(_smart_fio, axis=1)

    if "ИНН" in df.columns:
        df["ИНН"] = df["ИНН"].apply(normalize_inn)
    if "ДАТА РОЖДЕНИЯ" in df.columns:
        df["ДАТА РОЖДЕНИЯ"] = df["ДАТА РОЖДЕНИЯ"].apply(normalize_date)
    if "E-MAIL" in df.columns:
        df["E-MAIL"] = df["E-MAIL"].apply(normalize_email)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"preprocess: сохранено {len(df)} → {output_path}")
    return df
