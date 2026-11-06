"""Клиент AHunter cleanse/record."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from . import config


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).strip() == ""


def _to_str(value: Any) -> str:
    if _is_blank(value):
        return ""
    if isinstance(value, float) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def standardize_record(items: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
    """POST /site/cleanse/record. None при ошибке сети/ответа."""
    url = f"{config.AHUNTER_BASE.rstrip('/')}/site/cleanse/record"
    payload = {"id": "1", "items": items}
    try:
        resp = requests.get(
            url,
            params={
                "userid": config.AHUNTER_USER_ID,
                "query": json.dumps(payload, ensure_ascii=False),
            },
            timeout=30,
        )
        if resp.status_code != 200:
            return None
        return resp.json()
    except requests.RequestException:
        return None


def _extract_standardized(api_response: Dict[str, Any], column_type: str) -> str:
    """Достаёт стандартизированное значение из ответа AHunter."""
    try:
        items = api_response.get("items") or api_response.get("data", {}).get("items") or []
        for item in items:
            itype = str(item.get("type", "")).lower()
            if column_type == "person" and itype in {"person", "fio", "name"}:
                return _to_str(item.get("quality") or item.get("body") or item.get("value"))
            if column_type == "phone" and itype == "phone":
                return _to_str(item.get("quality") or item.get("body") or item.get("value"))
            if column_type == "address" and itype == "address":
                return _to_str(item.get("quality") or item.get("body") or item.get("value"))
        # fallback: первое quality
        for item in items:
            q = item.get("quality")
            if q:
                return _to_str(q)
    except Exception:
        pass
    return ""


def run_ahunter(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Стандартизация ФИО/телефона/адреса через AHunter."""
    df = pd.read_excel(input_path)
    print(f"ahunter: загружено {len(df)} строк")

    processed = skipped_fio = errors = 0
    for idx, row in df.iterrows():
        items: List[Dict[str, str]] = []
        cat = str(row.get("КАТЕГОРИЯ") or "")
        fio = _to_str(row.get("ФИО"))
        phone = _to_str(row.get("ТЕЛЕФОН"))
        address = _to_str(row.get("АДРЕС"))

        if fio and cat == "физ.лицо":
            items.append({"type": "person", "role": "ФИО", "body": fio})
        elif fio and cat != "физ.лицо":
            skipped_fio += 1

        if phone:
            items.append({"type": "phone", "role": "Телефон", "body": phone})
        if address:
            items.append({"type": "address", "role": "Адрес", "body": address})

        if not items:
            continue

        api_response = standardize_record(items)
        if not api_response:
            errors += 1
            time.sleep(config.AHUNTER_DELAY_SEC)
            continue

        if fio and cat == "физ.лицо":
            std = _extract_standardized(api_response, "person")
            if std:
                df.at[idx, "ФИО"] = std
                processed += 1
        if phone:
            std = _extract_standardized(api_response, "phone")
            if std:
                df.at[idx, "ТЕЛЕФОН"] = std
                processed += 1
        if address:
            std = _extract_standardized(api_response, "address")
            if std:
                df.at[idx, "АДРЕС"] = std
                processed += 1

        time.sleep(config.AHUNTER_DELAY_SEC)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    print(
        f"ahunter: ok_fields={processed}, skip_ul_fio={skipped_fio}, "
        f"errors={errors} → {output_path}"
    )
    return df
