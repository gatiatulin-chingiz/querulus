"""DataDrift: Hive reference (main+second) + Oracle SPR → querulus_DataDrift.

Источники (prod):
  * train — Hive на каждую модель из ``DEFAULT_DRIFT_MODELS`` (полный train);
  * prod — Oracle ``spr_uu_json_log`` (+ expand JSON), один снимок на все модели.
  * фичи — ``config.features ∩ train.columns ∩ prod.columns`` (cf∪rg в config);
    отсутствующие фичи → warning в лог.

Kerberos: ``kinit`` в DAG до ``make automl``.
Synthetic — ``local/synthetic/run_datadrift.py``.
"""
from __future__ import annotations

import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_PIPELINE_ROOT = Path(__file__).resolve().parent
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))
import _bootstrap_path  # noqa: F401 — monorepo + standalone GitLab clone

from pipeline.settings import Settings, load_settings
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
# Один расчёт на модель (cf∪rg); колонка track для guard/схемы Motor
DRIFT_TRACK = "all"

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


class QuerulusDataDrift:
    """PSI/KL/JS без зависимости от outboxml.datadrift (на контуре outboxml тянет polars/CUDA).

    Категории: общий словарь train∪test → коды (иначе падает на новых значениях prod).
    """

    def __init__(
        self,
        full_calc: bool = False,
        columns_to_exclude: list | None = None,
        n_bins: int = 100,
        dif_len_string: int = 100,
    ):
        self.full_calc = full_calc
        self.columns_to_exclude = columns_to_exclude or []
        self.n_bins = n_bins
        self.dif_len_string = dif_len_string
        self.types_dict: dict = {}

    def _calculate_psi(self, train_sample, test_sample, n_bins):
        e, p = self.compute_probs(train_sample, n=n_bins)
        _, q = self.compute_probs(test_sample, n=e[:-1])
        res = pd.Series((p - q) * np.log(p / q))
        return res[res != np.inf].sum()

    def _calculate_js_divirgence(self, train_sample, test_sample, n_bins):
        e, p = self.compute_probs(train_sample, n=n_bins)
        _, q = self.compute_probs(test_sample, n=e)
        pairs = self.support_intersection(p, q)
        p, q = self.get_probs(pairs)
        m = 0.5 * (p + q)
        return 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))

    def _calculate_kl_divirgence(self, train_sample, test_sample, n_bins):
        e, p = self.compute_probs(train_sample, n=n_bins)
        _, q = self.compute_probs(test_sample, n=e)
        pairs = self.support_intersection(p, q)
        p, q = self.get_probs(pairs)
        return np.sum(p * np.log(p / q))

    @staticmethod
    def get_probs(list_of_tuples):
        p = np.array([pair[0] for pair in list_of_tuples])
        q = np.array([pair[1] for pair in list_of_tuples])
        return p, q

    @staticmethod
    def support_intersection(p, q):
        return list(filter(lambda x: (x[0] != 0) & (x[1] != 0), zip(p, q)))

    def compute_probs(self, data, n=10):
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
        if data.dtype == object:
            data = pd.to_numeric(data, errors="coerce")
        h, e = np.histogram(data[data.notna()], n)
        p = h / data.shape[0]
        e = np.append(e, np.nan)
        p = np.append(p, data.sum() / data.shape[0])
        return e, p

    def _full_calculation(self, base_data: pd.DataFrame, control_data: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame()
        for column, col_type in self.types_dict.items():
            try:
                res = self._drift_calc(
                    base_data[column],
                    control_data[column],
                    label=column,
                    type=col_type,
                )
                result = pd.concat([result, pd.Series(res, name=column)], axis=1)
            except Exception as exc:
                logger.error("Error while calculating datadrift for %s || %s", column, exc)
        return result

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


def _filter_train_window(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Срез reference по PAYMENT_ORDER_DATE_TIME, если заданы START/END (как BUU)."""
    date_col = "PAYMENT_ORDER_DATE_TIME"
    start = settings.reference_train_start
    end = settings.reference_train_end
    if not start and not end:
        return df
    if date_col not in df.columns:
        logger.warning(
            "reference train window ignored: column %s missing", date_col
        )
        return df
    d = pd.to_datetime(df[date_col], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= d >= start
    if end:
        mask &= d < end
    out = df.loc[mask].copy()
    logger.info(
        "reference train window || start=%s end=%s || rows %s → %s",
        start or "-",
        end or "-",
        len(df),
        len(out),
    )
    return out


def load_reference_dataset(settings: Settings, *, hive_table: str) -> pd.DataFrame:
    """Обучающая выборка одной модели — только Hive (без parquet-fallback)."""
    table = hive_table.strip()
    if not table:
        raise ValueError(
            "Hive table empty for drift model. "
            "Set hive in DEFAULT_DRIFT_MODELS (pipeline/config.py)."
        )

    from pipeline.hadoop_extractor import HadoopExtractor

    # локальный parquet-кэш после hdfs get — уникальное имя на таблицу
    safe = table.replace(".", "_").replace("/", "_")
    logger.info("reference ← Hive %s", table)
    df = HadoopExtractor(
        source=table,
        to_df=f"querulus_hive_reference_{safe}.parquet",
        app_name="querulus_datadrift",
    ).extract_dataset()
    logger.info(
        "Hive reference loaded || table=%s || rows=%s || cols=%s",
        table,
        len(df),
        len(df.columns),
    )
    return _filter_train_window(df, settings)


def _load_prod_oracle_logs(settings: Settings) -> pd.DataFrame:
    """Боевые логи Querulus из Oracle + разворот JSON_FILE в фичи.

    Фильтр как в ноутбуке DataDrift: убрать EVENT_CREATED_BY_GIBDD_FLAG == 'Да'.
    Пустые строки → NA (чтобы не считались отдельной категорией).
    """
    from datetime import timedelta

    cutoff = datetime.now() - timedelta(days=settings.drift_lookback_days)
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


def resolve_drift_features(
    configured: Iterable[str],
    train: pd.DataFrame,
    prod: pd.DataFrame,
    *,
    stem: str,
) -> list[str]:
    """config.features ∩ train ∩ prod; warning, если чего-то нет."""
    wanted = [c for c in configured if c and c not in META_COLS]
    if not wanted:
        raise ValueError(
            f"Empty features list in DEFAULT_DRIFT_MODELS for stem={stem!r}. "
            "Edit pipeline/config.py."
        )

    train_cols = set(train.columns)
    prod_cols = set(prod.columns)
    missing_train = [c for c in wanted if c not in train_cols]
    missing_prod = [c for c in wanted if c not in prod_cols]
    if missing_train:
        logger.warning(
            "ALERT datadrift missing in TRAIN || stem=%s || n=%s || cols=%s",
            stem,
            len(missing_train),
            missing_train,
        )
    if missing_prod:
        logger.warning(
            "ALERT datadrift missing in PROD || stem=%s || n=%s || cols=%s",
            stem,
            len(missing_prod),
            missing_prod,
        )

    features = [c for c in wanted if c in train_cols and c in prod_cols]
    if not features:
        raise ValueError(
            f"No usable features after ∩ train ∩ prod for stem={stem!r}. "
            f"configured={wanted} missing_train={missing_train} "
            f"missing_prod={missing_prod}"
        )
    if missing_train or missing_prod:
        logger.warning(
            "ALERT datadrift features reduced || stem=%s || "
            "configured=%s || used=%s || dropped=%s",
            stem,
            len(wanted),
            len(features),
            len(wanted) - len(features),
        )
    else:
        logger.info(
            "datadrift features ok || stem=%s || n=%s", stem, len(features)
        )
    return features


def main() -> int:
    """Точка входа: для каждой main/second — Hive train + Oracle prod → append."""
    settings = load_settings()
    if not settings.drift_models:
        raise ValueError("DEFAULT_DRIFT_MODELS is empty in pipeline/config.py")

    logger.info(
        "datadrift ← Hive×%s models + Oracle SPR → %s.%s || track=%s",
        len(settings.drift_models),
        settings.schema,
        settings.table_drift,
        DRIFT_TRACK,
    )
    prod = _load_prod_oracle_logs(settings)
    if prod.empty:
        logger.warning("Empty prod logs — skip datadrift write")
        return 0

    parts: list[pd.DataFrame] = []
    for model in settings.drift_models:
        slot = model.get("slot", "")
        stem = str(model.get("stem", ""))
        hive = str(model.get("hive", ""))
        label = str(model.get("label") or stem)
        configured = model.get("features") or ()
        logger.info("datadrift model || slot=%s stem=%s hive=%s", slot, stem, hive)
        train = load_reference_dataset(settings, hive_table=hive)
        features = resolve_drift_features(configured, train, prod, stem=stem)
        part = compute_datadrift(
            train,
            prod,
            features,
            stem=stem,
            track=DRIFT_TRACK,
            model_name=stem,
            model_version=stem,
            model_label=label,
        )
        if not part.empty:
            parts.append(part)
        logger.info(
            "datadrift model done || stem=%s || features=%s || rows=%s",
            stem,
            len(features),
            len(part),
        )

    if not parts:
        logger.warning("Empty datadrift report")
        return 0
    report = pd.concat(parts, ignore_index=True)

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
        "datadrift done || rows_written=%s || models=%s || psi_over_threshold=%s",
        n,
        len(settings.drift_models),
        over,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
