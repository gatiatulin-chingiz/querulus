"""Сравнение фич df_for_service и Excel-выгрузки из 1С."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .paths import LOSS_NUMBER_COL, PRED_COLS, SERVICE_VECTOR_FEATURES


@dataclass
class CompareReport:
    """Итог сверки векторов."""

    ok: bool
    n_service: int
    n_excel: int
    n_matched_keys: int
    n_only_service: int
    n_only_excel: int
    n_mismatch_cells: int
    features_compared: list[str] = field(default_factory=list)
    features_missing_in_excel: list[str] = field(default_factory=list)
    features_missing_in_service: list[str] = field(default_factory=list)
    only_service_losses: list[str] = field(default_factory=list)
    only_excel_losses: list[str] = field(default_factory=list)
    mismatch_preview: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize_loss_key(series: pd.Series) -> pd.Series:
    return series.map(lambda x: str(x).strip() if pd.notna(x) else "")


def _normalize_value(value: object) -> object:
    """Привести значение к сопоставимому виду для 1:1 сверки."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        return ts.isoformat(sep="T", timespec="seconds")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return int(value)
        return float(value)
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    # Числа из Excel часто приходят строками.
    try:
        num = float(text.replace(",", "."))
        if num.is_integer():
            return int(num)
        return num
    except ValueError:
        return text


def _values_equal(left: object, right: object, *, rtol: float = 0.0) -> bool:
    a = _normalize_value(left)
    b = _normalize_value(right)
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if rtol > 0:
            return bool(np.isclose(float(a), float(b), rtol=rtol, atol=0.0, equal_nan=True))
        return float(a) == float(b)
    return a == b


def resolve_feature_columns(
    service_df: pd.DataFrame,
    excel_df: pd.DataFrame,
    *,
    features: Sequence[str] | None = None,
    all_overlap: bool = False,
) -> tuple[list[str], list[str], list[str]]:
    """Колонки для сверки + списки отсутствующих."""
    service_cols = set(service_df.columns)
    excel_cols = set(excel_df.columns)

    if all_overlap:
        skip = {LOSS_NUMBER_COL, *PRED_COLS}
        candidates = sorted((service_cols & excel_cols) - skip)
    else:
        candidates = list(features or SERVICE_VECTOR_FEATURES)

    compared = [c for c in candidates if c in service_cols and c in excel_cols]
    missing_excel = [c for c in candidates if c in service_cols and c not in excel_cols]
    missing_service = [c for c in candidates if c in excel_cols and c not in service_cols]
    return compared, missing_excel, missing_service


def load_excel_export(path: Path) -> pd.DataFrame:
    """Загрузить выгрузку 1С (xlsx/xls/csv)."""
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, engine="openpyxl" if suffix == ".xlsx" else None)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Неподдерживаемый формат выгрузки: {path.suffix}")


def compare_feature_frames(
    service_df: pd.DataFrame,
    excel_df: pd.DataFrame,
    *,
    features: Sequence[str] | None = None,
    all_overlap: bool = False,
    rtol: float = 0.0,
    preview_limit: int = 50,
) -> tuple[CompareReport, pd.DataFrame]:
    """Сверить фичи по LOSS_NUMBER. Возвращает отчёт и таблицу расхождений."""
    if LOSS_NUMBER_COL not in service_df.columns:
        raise KeyError(f"В df_for_service нет колонки {LOSS_NUMBER_COL}")
    if LOSS_NUMBER_COL not in excel_df.columns:
        raise KeyError(f"В Excel нет колонки {LOSS_NUMBER_COL}")

    compared, missing_excel, missing_service = resolve_feature_columns(
        service_df,
        excel_df,
        features=features,
        all_overlap=all_overlap,
    )
    if not compared:
        raise ValueError(
            "Нет общих колонок для сверки. "
            f"Нет в Excel: {missing_excel[:10]}; нет в service: {missing_service[:10]}"
        )

    left = service_df.copy()
    right = excel_df.copy()
    left["_key"] = _normalize_loss_key(left[LOSS_NUMBER_COL])
    right["_key"] = _normalize_loss_key(right[LOSS_NUMBER_COL])
    left = left[left["_key"] != ""].drop_duplicates("_key", keep="first")
    right = right[right["_key"] != ""].drop_duplicates("_key", keep="first")

    left_keys = set(left["_key"])
    right_keys = set(right["_key"])
    only_service = sorted(left_keys - right_keys)
    only_excel = sorted(right_keys - left_keys)
    common = sorted(left_keys & right_keys)

    left_i = left.set_index("_key").loc[common]
    right_i = right.set_index("_key").loc[common]

    rows: list[dict] = []
    for key in common:
        for col in compared:
            lv = left_i.at[key, col] if col in left_i.columns else None
            rv = right_i.at[key, col] if col in right_i.columns else None
            if not _values_equal(lv, rv, rtol=rtol):
                rows.append(
                    {
                        LOSS_NUMBER_COL: key,
                        "feature": col,
                        "service": _normalize_value(lv),
                        "excel": _normalize_value(rv),
                    }
                )

    mismatches = pd.DataFrame(
        rows,
        columns=[LOSS_NUMBER_COL, "feature", "service", "excel"],
    )
    # missing_* — информативно; --strict-features в CLI может завалить прогон.
    report = CompareReport(
        ok=(
            len(only_service) == 0
            and len(only_excel) == 0
            and len(rows) == 0
        ),
        n_service=len(left),
        n_excel=len(right),
        n_matched_keys=len(common),
        n_only_service=len(only_service),
        n_only_excel=len(only_excel),
        n_mismatch_cells=len(rows),
        features_compared=compared,
        features_missing_in_excel=missing_excel,
        features_missing_in_service=missing_service,
        only_service_losses=only_service[:100],
        only_excel_losses=only_excel[:100],
        mismatch_preview=rows[:preview_limit],
    )
    return report, mismatches


def write_compare_artifacts(
    report: CompareReport,
    mismatches: pd.DataFrame,
    reports_dir: Path,
    stem: str = "vector_compare",
) -> tuple[Path, Path]:
    """Записать JSON-отчёт и CSV с расхождениями."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"{stem}.json"
    csv_path = reports_dir / f"{stem}_mismatches.csv"
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    mismatches.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return json_path, csv_path


def report_to_frame(report: CompareReport) -> pd.DataFrame:
    """Краткая сводка отчёта одной строкой (удобно в ноутбуке)."""
    return pd.DataFrame(
        [
            {
                "ok": report.ok,
                "n_service": report.n_service,
                "n_excel": report.n_excel,
                "n_matched_keys": report.n_matched_keys,
                "n_only_service": report.n_only_service,
                "n_only_excel": report.n_only_excel,
                "n_mismatch_cells": report.n_mismatch_cells,
                "n_features_compared": len(report.features_compared),
                "n_missing_in_excel": len(report.features_missing_in_excel),
                "n_missing_in_service": len(report.features_missing_in_service),
            }
        ]
    )


def mismatch_feature_counts(mismatches: pd.DataFrame) -> pd.DataFrame:
    """Сколько раз каждая фича расходится."""
    if mismatches.empty:
        return pd.DataFrame(columns=["feature", "n_mismatch"])
    return (
        mismatches.groupby("feature", as_index=False)
        .size()
        .rename(columns={"size": "n_mismatch"})
        .sort_values("n_mismatch", ascending=False)
        .reset_index(drop=True)
    )


def side_by_side_for_loss(
    service_df: pd.DataFrame,
    excel_df: pd.DataFrame,
    loss_number: object,
    features: Sequence[str] | None = None,
) -> pd.DataFrame:
    """По одному LOSS_NUMBER: фича | service | excel | match."""
    key = str(loss_number).strip()
    left = service_df.copy()
    right = excel_df.copy()
    left["_key"] = _normalize_loss_key(left[LOSS_NUMBER_COL])
    right["_key"] = _normalize_loss_key(right[LOSS_NUMBER_COL])

    left_row = left.loc[left["_key"] == key]
    right_row = right.loc[right["_key"] == key]
    if left_row.empty and right_row.empty:
        raise KeyError(f"LOSS_NUMBER={key} нет ни в service, ни в excel")

    # Без явного списка — все общие колонки (удобно крутить в ноутбуке).
    cols, _, _ = resolve_feature_columns(
        service_df if not left_row.empty else excel_df,
        excel_df if not right_row.empty else service_df,
        features=features,
        all_overlap=features is None,
    )

    rows: list[dict] = []
    lv = left_row.iloc[0] if not left_row.empty else None
    rv = right_row.iloc[0] if not right_row.empty else None
    for col in cols:
        s_val = None if lv is None or col not in left_row.columns else lv[col]
        e_val = None if rv is None or col not in right_row.columns else rv[col]
        rows.append(
            {
                "feature": col,
                "service": _normalize_value(s_val),
                "excel": _normalize_value(e_val),
                "match": _values_equal(s_val, e_val),
            }
        )
    return pd.DataFrame(rows)


def prepare_keyed_frames(
    service_df: pd.DataFrame,
    excel_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Копии с нормализованным ключом ``_key`` (без дублей)."""
    left = service_df.copy()
    right = excel_df.copy()
    left["_key"] = _normalize_loss_key(left[LOSS_NUMBER_COL])
    right["_key"] = _normalize_loss_key(right[LOSS_NUMBER_COL])
    left = left[left["_key"] != ""].drop_duplicates("_key", keep="first")
    right = right[right["_key"] != ""].drop_duplicates("_key", keep="first")
    return left, right
