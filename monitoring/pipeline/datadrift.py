"""DataDrift: train parquet + Oracle SPR-логи → Motor.agrafana.querulus_DataDrift.

Источники (prod):
  * train — ``TRAIN_PARQUET_PATH`` / ``df_for_service.parquet`` (обучающая выборка)
  * prod  — Oracle ``spr_uu_json_log``, маршрут ОИСУУ → SPR_UU::QUERULUS,
            JSON_FILE разворачивается в фичи (см. ``pipeline.db.expand_json_column``)

Расчёт PSI/KL/JS — OutBoxML ``DataDrift`` (подкласс ``QuerulusDataDrift``
с безопасным кодированием категориальных, если в prod появились новые уровни).

Synthetic — только в ``local/synthetic/run_datadrift.py``.
"""
from __future__ import annotations

import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from outboxml.datadrift import DataDrift as OutboxmlDataDrift

_PIPELINE_ROOT = Path(__file__).resolve().parent
_MONITORING_ROOT = _PIPELINE_ROOT.parent
if str(_MONITORING_ROOT) not in sys.path:
    sys.path.insert(0, str(_MONITORING_ROOT))

from pipeline.config import Settings, load_settings
from pipeline.db import (
    connect_oracle,
    expand_json_column,
    query_querulus_request_logs,
)
from pipeline.write_sql import get_engine, write_append_guarded

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("datadrift")

# Мета-колонки не участвуют в PSI (это ярлыки модели/трека, не фичи)
META_COLS = {"stem", "slot", "track", "run_date", "model_label", "model_name", "model_version"}

# Порядок колонок отчёта как в archive MonitoringReport
ARCHIVE_COLS = [
    "model_name",
    "model_version",
    "date",
    "col",
    "TYPE",
    "PSI",
    "KL",
    "JS",
    "NaN_train",
    "NaN_test",
    "uniq_train",
    "uniq_test",
    "mode_train",
    "mode_test",
    "mean_train",
    "mean_test",
    "dif_train",
    "dif_test",
]


class QuerulusDataDrift(OutboxmlDataDrift):
    """OutBoxML DataDrift + безопасные категориальные коды.

    Базовый outboxml падает/искажает метрики, если в prod появились категории,
    которых не было в train. Здесь train∪test → общий словарь кодов.
    """

    def __init__(self, *args, dif_len_string: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        # длина строк dif_train / dif_test (списки уникальных значений)
        self.dif_len_string = dif_len_string

    def _drift_calc(
        self, base: pd.Series, control: pd.Series, label: str, type: str = "CATEGORICAL"
    ) -> dict:
        """Расширенная статистика по одной колонке (для full_calc)."""
        base_unique = set(base.unique())
        control_unique = set(control.unique())

        def _stringify(values: set) -> str:
            prepared = [(v.item() if hasattr(v, "item") else v) for v in values]
            prepared = sorted(prepared, key=lambda x: str(x))
            return str(prepared)[: self.dif_len_string]

        return {
            "TYPE": type,
            "col": label,
            "date": datetime.now().date(),
            "NaN_train": base.isna().mean(),
            "NaN_test": control.isna().mean(),
            "uniq_train": base.nunique(dropna=False),
            "uniq_test": control.nunique(dropna=False),
            "mode_train": base.mode(dropna=False)[0],
            "mode_test": control.mode(dropna=False)[0],
            "mean_train": np.nan if type == "CATEGORICAL" else base.mean(),
            "mean_test": np.nan if type == "CATEGORICAL" else control.mean(),
            "dif_train": _stringify(base_unique - control_unique),
            "dif_test": _stringify(control_unique - base_unique),
        }

    def report(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        base_data: pd.DataFrame = None,
        control_data: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Поколоночный PSI/KL/JS; при full_calc — доп. описательные поля."""
        result = pd.DataFrame()
        base_data = train_data if base_data is None else base_data
        control_data = test_data if control_data is None else control_data

        for column in train_data.columns:
            if column in self.columns_to_exclude:
                continue
            if column not in test_data.columns:
                logger.warning("Skip %s: not in test", column)
                continue

            x_train = train_data[column].copy()
            x_test = test_data[column].copy()
            # object/string/category → категориальный тип для PSI-бинов
            is_cat = (
                x_train.dtype == "category"
                or x_train.dtype == object
                or str(x_train.dtype) == "string"
                or x_test.dtype == "category"
                or x_test.dtype == object
                or str(x_test.dtype) == "string"
            )
            self.types_dict[column] = "CATEGORICAL" if is_cat else "NUMERICAL"

            try:
                if is_cat:
                    x_train_n, x_test_n = _codes_aligned(x_train, x_test)
                else:
                    x_train_n = pd.to_numeric(x_train, errors="coerce")
                    x_test_n = pd.to_numeric(x_test, errors="coerce")

                psi = self._calculate_psi(x_train_n, x_test_n, self.n_bins)
                kl = self._calculate_kl_divirgence(x_train_n, x_test_n, self.n_bins)
                js = self._calculate_js_divirgence(x_train_n, x_test_n, self.n_bins)
                result[column] = [psi, kl, js]
            except Exception as exc:
                logger.error("No results for %s || %s", column, exc)

        if result.empty:
            return pd.DataFrame(columns=ARCHIVE_COLS)

        result.index = pd.Index(["PSI", "KL", "JS"])
        if self.full_calc:
            full_result = self._full_calculation(base_data, control_data)
            result = pd.concat([result, full_result])
        return result.transpose()


def _codes_aligned(train: pd.Series, test: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Общий словарь категорий train∪test → float-коды для PSI."""
    tr = train.astype("string")
    te = test.astype("string")
    cats = sorted(set(tr.dropna().unique()) | set(te.dropna().unique()))
    mapping = {c: i for i, c in enumerate(cats)}
    return tr.map(mapping).astype(float), te.map(mapping).astype(float)


def _feature_frame(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    cols = [c for c in features if c in df.columns]
    return df[cols].copy()


def _normalize_report(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["col", "PSI", "KL", "JS"])

    out = raw.reset_index()
    if "col" in out.columns and "index" in out.columns:
        out = out.drop(columns=["index"])
    elif "index" in out.columns:
        out = out.rename(columns={"index": "col"})
    elif "col" not in out.columns:
        name_col = out.columns[0]
        if name_col not in {"PSI", "KL", "JS", "TYPE"}:
            out = out.rename(columns={name_col: "col"})
        else:
            out["col"] = out.index.astype(str)
    if out.columns.tolist().count("col") > 1:
        first = out.loc[:, "col"].iloc[:, 0]
        out = out.loc[:, ~out.columns.duplicated(keep="first")]
        out["col"] = first
    return out


def compute_datadrift(
    train: pd.DataFrame,
    prod: pd.DataFrame,
    features: Iterable[str],
    *,
    stem: str = "main",
    track: str = "cf",
    model_name: str = "",
    model_version: str = "",
    model_label: str = "",
    run_date: date | None = None,
    columns_to_exclude: list | None = None,
) -> pd.DataFrame:
    """PSI/KL/JS через outboxml; колонки как archive MonitoringReport."""
    run_date = run_date or datetime.now().date()
    feat_list = [c for c in features if c not in META_COLS]
    x_train = _feature_frame(train, feat_list)
    x_prod = _feature_frame(prod, feat_list)
    if x_train.empty or x_prod.empty:
        return pd.DataFrame()

    drift = QuerulusDataDrift(
        full_calc=True,
        columns_to_exclude=columns_to_exclude or [],
        n_bins=100,
        dif_len_string=100,
    )
    raw = drift.report(x_train, x_prod, x_train, x_prod)
    if raw.empty:
        return pd.DataFrame()

    out = _normalize_report(raw)
    out["date"] = run_date
    out["model_name"] = model_name or stem
    out["model_version"] = model_version or stem
    out["stem"] = stem
    out["track"] = track
    out["model_label"] = model_label or stem
    if "TYPE" not in out.columns:
        out["TYPE"] = "NUMERICAL"

    for c in ("PSI", "KL", "JS", "NaN_train", "NaN_test", "mean_train", "mean_test"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ("uniq_train", "uniq_test"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    def _to_python(v):
        return v.item() if isinstance(v, np.generic) else v

    out = out.apply(lambda s: s.map(_to_python))

    preferred = ARCHIVE_COLS + ["stem", "track", "model_label"]
    cols = [c for c in preferred if c in out.columns]
    return out[cols].sort_values("PSI", ascending=False).reset_index(drop=True)


def compute_datadrift_by_meta(
    train: pd.DataFrame,
    prod: pd.DataFrame,
    features: list[str],
    *,
    run_date: date | None = None,
) -> pd.DataFrame:
    """If stem/track present — per group; else one block."""
    run_date = run_date or datetime.now().date()
    has_meta = {"stem", "track"}.issubset(prod.columns)
    if not has_meta:
        stem = (
            str(train["stem"].iloc[0])
            if "stem" in train.columns and len(train)
            else "main"
        )
        track = (
            str(train["track"].iloc[0])
            if "track" in train.columns and len(train)
            else "cf"
        )
        label = (
            str(train["model_label"].iloc[0])
            if "model_label" in train.columns and len(train)
            else stem
        )
        return compute_datadrift(
            train,
            prod,
            features,
            stem=stem,
            track=track,
            model_name=stem,
            model_version=stem,
            model_label=label,
            run_date=run_date,
        )

    parts = []
    keys = prod[["stem", "track"]].drop_duplicates()
    for _, row in keys.iterrows():
        stem = str(row["stem"])
        track = str(row["track"])
        prod_g = prod[(prod["stem"] == stem) & (prod["track"] == track)]
        if "stem" in train.columns and "track" in train.columns:
            train_g = train[(train["stem"] == stem) & (train["track"] == track)]
            if train_g.empty:
                train_g = train
        else:
            train_g = train
        label = ""
        if "model_label" in prod_g.columns and len(prod_g):
            label = str(prod_g["model_label"].iloc[0])
        parts.append(
            compute_datadrift(
                train_g,
                prod_g,
                features,
                stem=stem,
                track=track,
                model_name=stem,
                model_version=stem,
                model_label=label,
                run_date=run_date,
            )
        )
    if not parts:
        return compute_datadrift(train, prod, features, run_date=run_date)
    return pd.concat(parts, ignore_index=True)


# --- extract (prod) ------------------------------------------------------------


def _load_train(settings: Settings) -> pd.DataFrame:
    """Обучающая выборка сервиса (integration/results/df_for_service.parquet)."""
    path = settings.train_parquet_path
    if not path.is_file():
        raise FileNotFoundError(
            f"Train parquet not found: {path}. "
            "Set TRAIN_PARQUET_PATH (df_for_service.parquet)"
        )
    df = pd.read_parquet(path)
    logger.info("train loaded || path=%s || rows=%s || cols=%s", path, len(df), len(df.columns))
    return df


def _load_prod_oracle_logs(settings: Settings) -> pd.DataFrame:
    """Боевые логи Querulus из Oracle + разворот JSON_FILE в фичи.

    Фильтр как в ноутбуке DataDrift: убрать EVENT_CREATED_BY_GIBDD_FLAG == 'Да'.
    Пустые строки → NA (чтобы не считались отдельной категорией).
    """
    from datetime import timedelta

    cutoff = datetime.now() - timedelta(weeks=settings.drift_window_weeks)
    conn = connect_oracle(settings)
    try:
        raw = query_querulus_request_logs(conn, min_time_stamp=cutoff)
    finally:
        conn.close()

    if raw.empty:
        logger.warning("Oracle querulus logs empty || since=%s", cutoff)
        return raw

    # после query колонки lower; на всякий случай принимаем оба регистра
    json_col = "json_file" if "json_file" in raw.columns else "JSON_FILE"
    prod = expand_json_column(raw, json_col)
    for col in prod.columns:
        prod[col] = prod[col].map(
            lambda v: pd.NA if isinstance(v, str) and v == "" else v
        )
    if "EVENT_CREATED_BY_GIBDD_FLAG" in prod.columns:
        prod = prod.loc[prod["EVENT_CREATED_BY_GIBDD_FLAG"] != "Да"].copy()
    logger.info(
        "prod logs loaded || rows=%s || since=%s || cols=%s",
        len(prod),
        cutoff,
        len(prod.columns),
    )
    return prod


def load_train_and_prod_logs(
    settings: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Вернуть (train, prod, shared_feature_names) для расчёта дрифта."""
    train = _load_train(settings)
    prod = _load_prod_oracle_logs(settings)
    # дрифт только по пересечению колонок (имена фич должны совпасть)
    features = [
        c for c in train.columns if c in prod.columns and c not in META_COLS
    ]
    if not features:
        raise ValueError(
            "No shared feature columns between train and Oracle logs. "
            f"train_cols={list(train.columns)[:15]} "
            f"prod_cols={list(prod.columns)[:15]}"
        )
    logger.info("shared features=%s", len(features))
    return train, prod, features


def main() -> int:
    """Точка входа: extract → PSI-отчёт → append в Motor."""
    settings = load_settings()
    logger.info(
        "datadrift ← train parquet + Oracle SPR → %s.%s",
        settings.schema,
        settings.table_drift,
    )
    train, prod, features = load_train_and_prod_logs(settings)
    report = compute_datadrift_by_meta(train, prod, features)
    if report.empty:
        logger.warning("Empty datadrift report")
        return 0
    engine = get_engine(settings.grafana_sqlalchemy_url)
    n = write_append_guarded(
        report,
        engine=engine,
        table=settings.table_drift,
        schema=settings.schema,
        guard_cols=["date", "model_name", "track"],
    )
    over = int((report["PSI"] > settings.psi_threshold).sum()) if "PSI" in report else 0
    logger.info(
        "datadrift done || rows_written=%s || features=%s || psi_over_threshold=%s",
        n,
        len(features),
        over,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
