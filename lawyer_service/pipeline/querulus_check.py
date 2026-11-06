"""Проверка: базовый убыток с clf=0 → подмешивание полного контакта → main+second."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from . import config

CONTACT_API_FIELDS = (
    ("APPLICANT", "ФИО"),
    ("APPLICANT_BIRTHDAY", "ДАТА РОЖДЕНИЯ"),
    ("APPLICANT_INN", "ИНН"),
    ("APPLICANT_PHONE", "ТЕЛЕФОН"),
    ("APPLICANT_EMAIL", "E-MAIL"),
    ("APPLICANT_ADDRESS", "АДРЕС"),
    ("PAYMENT_RECIPIENT", "ФИО"),
    ("PAYMENT_RECIPIENT_BIRTHDAY", "ДАТА РОЖДЕНИЯ"),
    ("PAYMENT_RECIPIENT_INN", "ИНН"),
    ("PAYMENT_RECIPIENT_PHONE", "ТЕЛЕФОН"),
    ("PAYMENT_RECIPIENT_EMAIL", "E-MAIL"),
    ("PAYMENT_RECIPIENT_ADDRESS", "АДРЕС"),
)


def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).strip() == ""


def _jsonable(value: Any) -> Any:
    if _blank(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%dT%H:%M:%S")
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    text = str(value).strip()
    # даты ДР из справочника dd.mm.yyyy → ISO date
    if len(text) == 10 and text[2] == "." and text[5] == ".":
        d, m, y = text.split(".")
        return f"{y}-{m}-{d}T00:00:00"
    return text


def _first_num(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and value:
        try:
            return float(value[0])
        except (TypeError, ValueError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_track_metrics(response_json: Dict[str, Any], track: str) -> Dict[str, Any]:
    """Достаёт clf/rg из main_response / second_response (или плоского legacy)."""
    out = {
        "track": track,
        "present": False,
        "http_ok_structure": False,
        "classification": None,
        "regression": None,
        "proba": None,
        "threshold": None,
        "error": None,
    }
    block = response_json.get(f"{track}_response")
    if block is None and track == "main" and "classification_predictions" in response_json:
        # legacy flat
        block = {"result": response_json}
    if not isinstance(block, dict) or not block:
        out["error"] = "missing_or_empty"
        return out

    out["present"] = True
    result = block.get("result") if isinstance(block.get("result"), dict) else block
    out["classification"] = _first_num(result.get("classification_predictions"))
    out["regression"] = _first_num(
        result.get("regression_predictions") or result.get("prediction")
    )
    out["proba"] = _first_num(result.get("classification_proba"))
    thr = result.get("threshold")
    try:
        out["threshold"] = float(thr) if thr is not None else None
    except (TypeError, ValueError):
        out["threshold"] = None
    out["http_ok_structure"] = out["classification"] is not None
    return out


def post_predict(
    vector: Dict[str, Any],
    url: str,
    main_model: str,
    second_model: str,
    max_retries: int = config.MAX_RETRIES,
) -> Tuple[int, Dict[str, Any], str]:
    """POST ServiceRequest. Возвращает (status, json|{}, error_text)."""
    body = {
        "main_model": main_model,
        "main_request": [vector],
        "second_model": second_model,
        "second_request": [deepcopy(vector)],
    }
    last_err = ""
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=body,
                timeout=60,
            )
            try:
                data = resp.json()
            except Exception:
                data = {}
            return resp.status_code, data if isinstance(data, dict) else {}, resp.text[:500]
        except requests.RequestException as exc:
            last_err = str(exc)
            if attempt < max_retries - 1:
                time.sleep(2)
    return 0, {}, last_err


def _load_template(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    req = raw.get("main_request") or raw.get("second_request") or [{}]
    return deepcopy(req[0])


def _row_to_overlays(row: pd.Series) -> Dict[str, Any]:
    """Перекрытия из строки df_for_service на шаблон запроса."""
    out: Dict[str, Any] = {}
    for col, val in row.items():
        if str(col).startswith("preds_") or str(col).startswith("TARGET"):
            continue
        if str(col).startswith("FE_"):
            continue
        if _blank(val):
            continue
        out[str(col)] = _jsonable(val)
    return out


def build_base_vector(template: Dict[str, Any], parquet_row: pd.Series) -> Dict[str, Any]:
    vec = deepcopy(template)
    vec.update(_row_to_overlays(parquet_row))
    vec["IS_LAWYER"] = 0
    # контактные поля обнуляем — baseline без «нашего» юриста
    for api_name, _ in CONTACT_API_FIELDS:
        vec[api_name] = ""
    return vec


def contact_to_fields(row: pd.Series) -> Dict[str, Any]:
    """Полный контактный вектор (все заполненные поля сразу)."""
    fields: Dict[str, Any] = {}
    for api_name, col in CONTACT_API_FIELDS:
        if col not in row.index:
            continue
        val = row.get(col)
        if _blank(val):
            continue
        fields[api_name] = _jsonable(val)
    return fields


def pick_baseline_vector(
    df_service: pd.DataFrame,
    template: Dict[str, Any],
    url: str,
    main_model: str,
    second_model: str,
    scan_limit: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Ищем убыток, у которого API (или parquet) даёт clf=0 на main и second.

    Returns:
        (vector, meta) — meta описывает, как выбран baseline.
    """
    meta: Dict[str, Any] = {
        "method": None,
        "service_reachable": False,
        "parquet_index": None,
        "main": None,
        "second": None,
    }

    # кандидаты: низкий preds_cf
    work = df_service.copy()
    if "preds_cf" in work.columns:
        thr = 0.5
        if "THRESHOLD" in work.columns:
            try:
                thr = float(work["THRESHOLD"].dropna().iloc[0])
            except Exception:
                pass
        work = work.sort_values("preds_cf", ascending=True)
        candidates = work[work["preds_cf"] < thr]
        if candidates.empty:
            candidates = work
    else:
        candidates = work

    candidates = candidates.head(max(1, scan_limit))

    # Пробуем живой API
    for idx, row in candidates.iterrows():
        vec = build_base_vector(template, row)
        status, data, err = post_predict(vec, url, main_model, second_model)
        if status == 0:
            meta["service_reachable"] = False
            meta["error"] = err
            break
        meta["service_reachable"] = True
        main_m = extract_track_metrics(data, "main")
        second_m = extract_track_metrics(data, "second")
        # оба трека ответили структурой и clf == 0
        if (
            main_m["http_ok_structure"]
            and second_m["http_ok_structure"]
            and main_m["classification"] == 0.0
            and second_m["classification"] == 0.0
        ):
            meta.update(
                {
                    "method": "api_confirmed_clf0",
                    "parquet_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                    "main": main_m,
                    "second": second_m,
                }
            )
            return vec, meta
        time.sleep(config.REQUEST_DELAY_SEC)

    # fallback: parquet preds_cf
    row0 = candidates.iloc[0]
    vec = build_base_vector(template, row0)
    meta.update(
        {
            "method": "parquet_preds_cf_assumed_0",
            "parquet_index": int(candidates.index[0])
            if isinstance(candidates.index[0], (int, np.integer))
            else str(candidates.index[0]),
            "note": "Сервис недоступен или не вернул clf=0; взяли минимальный preds_cf",
            "preds_cf": float(row0["preds_cf"]) if "preds_cf" in row0.index else None,
        }
    )
    return vec, meta


def evaluate_pair(
    baseline_main: Dict[str, Any],
    baseline_second: Dict[str, Any],
    after_main: Dict[str, Any],
    after_second: Dict[str, Any],
) -> Dict[str, Any]:
    """Критерии: оба трека живы; желательно clf 0→1 и rg≠0 (если есть числа)."""

    def track_ok(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        present = after.get("present") and after.get("http_ok_structure")
        flip = None
        rg_nonzero = None
        if (
            before.get("classification") is not None
            and after.get("classification") is not None
        ):
            flip = before["classification"] == 0.0 and after["classification"] == 1.0
        if after.get("regression") is not None:
            rg_nonzero = abs(after["regression"]) > 1e-9
        return {
            "structure_ok": bool(present),
            "clf_flip_0_to_1": flip,
            "regression_nonzero": rg_nonzero,
        }

    main_ev = track_ok(baseline_main or {}, after_main)
    second_ev = track_ok(baseline_second or {}, after_second)
    both_structure = main_ev["structure_ok"] and second_ev["structure_ok"]
    flips = [main_ev["clf_flip_0_to_1"], second_ev["clf_flip_0_to_1"]]
    rgs = [main_ev["regression_nonzero"], second_ev["regression_nonzero"]]
    # pass если оба трека ответили; flip/rg — отдельные флаги (могут быть None если сервис без модели)
    return {
        "both_tracks_ok": both_structure,
        "main": main_ev,
        "second": second_ev,
        "pass_strict": both_structure
        and all(x is True for x in flips)
        and all(x is True for x in rgs),
    }


def run_querulus_check(
    contacts_path: Path,
    df_for_service_path: Path,
    output_xlsx: Path,
    summary_json: Path,
    *,
    limit: Optional[int] = None,
    predict_url: Optional[str] = None,
    payload_template: Optional[Path] = None,
) -> Dict[str, Any]:
    """Прогон контактов против /api/predict (main + second)."""
    url = predict_url or config.PREDICT_URL
    template_path = payload_template or config.DEFAULT_PAYLOAD_TEMPLATE
    template = _load_template(template_path)

    df_service = pd.read_parquet(df_for_service_path)
    contacts = pd.read_excel(contacts_path)
    if limit is not None and limit > 0:
        contacts = contacts.head(limit)

    print(f"querulus: contacts={len(contacts)}, url={url}")

    base_vec, baseline_meta = pick_baseline_vector(
        df_service,
        template,
        url,
        config.MAIN_MODEL,
        config.SECOND_MODEL,
        config.BASELINE_SCAN_LIMIT,
    )
    print(f"querulus: baseline method={baseline_meta.get('method')}")

    # Явный baseline-запрос (ещё раз), если сервис жив
    b_status, b_data, b_err = post_predict(
        base_vec, url, config.MAIN_MODEL, config.SECOND_MODEL
    )
    baseline_main = extract_track_metrics(b_data, "main") if b_status == 200 else {}
    baseline_second = extract_track_metrics(b_data, "second") if b_status == 200 else {}
    if b_status != 200:
        baseline_meta["baseline_http"] = b_status
        baseline_meta["baseline_error"] = b_err

    rows_out: List[Dict[str, Any]] = []
    # контрольная строка
    rows_out.append(
        {
            "row_index": -1,
            "contact_label": "BASELINE",
            "http_status": b_status,
            "main_clf": baseline_main.get("classification"),
            "main_rg": baseline_main.get("regression"),
            "second_clf": baseline_second.get("classification"),
            "second_rg": baseline_second.get("regression"),
            "both_tracks_ok": bool(
                baseline_main.get("http_ok_structure")
                and baseline_second.get("http_ok_structure")
            ),
            "pass_strict": False,
            "note": "без контакта справочника",
        }
    )
    time.sleep(config.REQUEST_DELAY_SEC)

    n_pass = n_struct = n_err = 0
    for i, row in contacts.iterrows():
        fields = contact_to_fields(row)
        if not fields:
            rows_out.append(
                {
                    "row_index": i,
                    "contact_label": str(row.get("ФИО") or ""),
                    "http_status": "SKIPPED",
                    "note": "нет контактных полей",
                }
            )
            continue

        vec = deepcopy(base_vec)
        vec.update(fields)
        # шлюз сам выставит IS_LAWYER; в запросе оставляем 0
        vec["IS_LAWYER"] = 0

        status, data, err = post_predict(
            vec, url, config.MAIN_MODEL, config.SECOND_MODEL
        )
        after_main = extract_track_metrics(data, "main") if status == 200 else {}
        after_second = extract_track_metrics(data, "second") if status == 200 else {}
        ev = evaluate_pair(baseline_main, baseline_second, after_main, after_second)

        if status != 200:
            n_err += 1
        if ev["both_tracks_ok"]:
            n_struct += 1
        if ev["pass_strict"]:
            n_pass += 1

        rows_out.append(
            {
                "row_index": i,
                "contact_label": str(row.get("ФИО") or ""),
                "category": row.get("КАТЕГОРИЯ"),
                "http_status": status,
                "main_clf": after_main.get("classification"),
                "main_rg": after_main.get("regression"),
                "second_clf": after_second.get("classification"),
                "second_rg": after_second.get("regression"),
                "both_tracks_ok": ev["both_tracks_ok"],
                "main_flip": ev["main"]["clf_flip_0_to_1"],
                "second_flip": ev["second"]["clf_flip_0_to_1"],
                "pass_strict": ev["pass_strict"],
                "sent_fields": ";".join(sorted(fields.keys())),
                "error": err if status != 200 else "",
            }
        )
        time.sleep(config.REQUEST_DELAY_SEC)

    out_df = pd.DataFrame(rows_out)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(output_xlsx, index=False)

    summary = {
        "predict_url": url,
        "contacts_total": int(len(contacts)),
        "baseline": baseline_meta,
        "baseline_http": b_status,
        "counts": {
            "both_tracks_structure_ok": n_struct,
            "pass_strict_clf_flip_and_rg": n_pass,
            "http_errors": n_err,
        },
        "note": (
            "pass_strict требует clf 0→1 и rg≠0 на main и second. "
            "Если шлюз/модель не подняты — смотрите both_tracks_ok / http_status."
        ),
    }
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"querulus: summary → {summary_json}")
    print(f"querulus: detail → {output_xlsx}")
    return summary
