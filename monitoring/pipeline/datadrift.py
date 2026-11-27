"""DataDrift: Hive reference (main+second) + Oracle SPR → Querulus_DataDrift.

Источники:
  * train — Hive raw (без ``prepare_dataset`` / кодирования), затем
    опциональный ``train_filter`` из ``DEFAULT_DRIFT_MODELS``
    (дата + филиалы; только datadrift, не fin_effect);
  * prod — Oracle ``spr_uu_json_log`` (encoded output), категориальные
    раскодируются по ``feature.replace`` из ``{stem}.pickle``;
  * PSI и ``dif_*`` — на одной человекочитаемой плоскости;
  * фичи — из pickle ``model_config`` (или override ``config.features``),
    затем ∩ train ∩ prod; пропуски → ALERT в логе.

Pickle: только ``model_config`` для обратного ``replace``
(``pipeline/models/`` или ``DRIFT_MODELS_DIR``).

Kerberos: ``kinit`` в DAG до ``make automl``.
Synthetic — ``local/synthetic/run_datadrift.py``.
"""
from __future__ import annotations

import logging
import os
import pickle
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

_PIPELINE_ROOT = Path(__file__).resolve().parent
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))
import _bootstrap_path  # noqa: F401 — monorepo + standalone GitLab clone

from pipeline.settings import Settings, load_settings
from pipeline.db import (
    connect_oracle,
    expand_querulus_model_output,
    query_querulus_model_logs,
)
from pipeline.write_sql import get_engine, write_append_guarded

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("datadrift")

# Мета-колонки не участвуют в PSI (это ярлыки модели/трека, не фичи)
META_COLS = {"stem", "slot", "track", "run_date", "model_label", "model_name", "model_version"}
# Один расчёт на модель (cf∪rg); колонка track для guard/схемы Motor
DRIFT_TRACK = "all"
# Pickle stem → [clf, rg]; см. pipeline/models/README.md
_DEFAULT_MODELS_DIR = _PIPELINE_ROOT / "models"

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


def _is_categorical_replace(replace: dict | None) -> bool:
    """Категориальная фича: в replace нет ``_TYPE_: _NUM_``."""
    if not replace:
        return False
    try:
        from outboxml.core.enums import FeatureEngineering

        return replace.get(FeatureEngineering.feature_type) != FeatureEngineering.numerical
    except ImportError:
        return replace.get("_TYPE_") != "_NUM_"


def _prefer_category_label(current: Any, candidate: Any) -> Any:
    """При коллизии code→label предпочитаем человекочитаемую строку, не цифру."""
    cur_s, cand_s = str(current), str(candidate)
    cur_digit = cur_s.isdigit()
    cand_digit = cand_s.isdigit()
    if cur_digit and not cand_digit:
        return candidate
    return current


def _invert_categorical_replace(replace: dict) -> dict[Any, Any]:
    """``{label: code}`` → ``{code: label}`` (+ int/float-ключи для lookup)."""
    inv: dict[Any, Any] = {}
    for label, code in replace.items():
        if isinstance(label, str) and label.startswith("_"):
            continue
        if code is None or (isinstance(code, float) and np.isnan(code)):
            continue
        keys = [code]
        if isinstance(code, bool):
            pass
        elif isinstance(code, (int, np.integer)):
            keys.append(int(code))
            keys.append(float(code))
        elif isinstance(code, (float, np.floating)) and float(code).is_integer():
            keys.append(int(code))
            keys.append(float(code))
        for key in keys:
            if key in inv:
                inv[key] = _prefer_category_label(inv[key], label)
            else:
                inv[key] = label
    return inv


def build_category_label_maps(clf_cfg: Any, rg_cfg: Any) -> dict[str, dict[Any, Any]]:
    """col → (code → label) из clf+rg ``model_config.replace``; shared rg → ``*__rg``."""
    maps: dict[str, dict[Any, Any]] = {}
    clf_names: set[str] = set()
    for feat in getattr(clf_cfg, "features", None) or []:
        name = getattr(feat, "name", None)
        replace = getattr(feat, "replace", None) or {}
        if not name or not _is_categorical_replace(replace):
            continue
        inv = _invert_categorical_replace(replace)
        if inv:
            maps[name] = inv
            clf_names.add(name)
    for feat in getattr(rg_cfg, "features", None) or []:
        name = getattr(feat, "name", None)
        replace = getattr(feat, "replace", None) or {}
        if not name or not _is_categorical_replace(replace):
            continue
        inv = _invert_categorical_replace(replace)
        if not inv:
            continue
        key = f"{name}__rg" if name in clf_names else name
        maps[key] = inv
    return maps


def features_from_model_configs(clf_cfg: Any, rg_cfg: Any) -> list[str]:
    """Имена фич обучения: union clf→rg ``model_config.features`` (порядок как в fit)."""
    names: list[str] = []
    seen: set[str] = set()
    for cfg in (clf_cfg, rg_cfg):
        for feat in getattr(cfg, "features", None) or []:
            name = getattr(feat, "name", None)
            if not name or name in META_COLS or name in seen:
                continue
            seen.add(name)
            names.append(str(name))
    return names


def _decode_category_value(value: Any, inv: dict[Any, Any] | None) -> Any:
    """Код → label по inverse replace; без map / неизвестный код — как есть."""
    if inv is None or value is None:
        return value
    try:
        if pd.isna(value):
            return value
    except (TypeError, ValueError):
        pass
    raw = value.item() if hasattr(value, "item") else value
    if raw in inv:
        return inv[raw]
    if isinstance(raw, (float, np.floating)) and float(raw).is_integer():
        as_int = int(raw)
        if as_int in inv:
            return inv[as_int]
    if isinstance(raw, (int, np.integer)):
        as_float = float(raw)
        if as_float in inv:
            return inv[as_float]
    return raw


def decode_category_columns(
    df: pd.DataFrame,
    label_maps: dict[str, dict[Any, Any]],
) -> pd.DataFrame:
    """Раскодировать категориальные колонки (code→label) → string для PSI/dif."""
    if df.empty or not label_maps:
        return df
    out = df.copy()
    for col, inv in label_maps.items():
        if col not in out.columns or not inv:
            continue
        out[col] = out[col].map(lambda v, m=inv: _decode_category_value(v, m))
        out[col] = out[col].astype("string")
    return out


class QuerulusDataDrift:
    """PSI/KL/JS без зависимости от outboxml.datadrift (на контуре outboxml тянет polars/CUDA).

    Категории на входе уже человекочитаемые (prod раскодирован);
    для PSI — общий словарь train∪test → коды.
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


def _missing_feature_reason(
    col: str,
    train: pd.DataFrame,
    prod: pd.DataFrame,
) -> str:
    in_train = col in train.columns
    in_prod = col in prod.columns
    if not in_train and not in_prod:
        return "absent_train+prod"
    if not in_train:
        return "absent_train"
    if not in_prod:
        return "absent_prod"
    return "absent"


def placeholder_drift_rows(
    missing: Iterable[str],
    *,
    train: pd.DataFrame,
    prod: pd.DataFrame,
    stem: str,
    track: str,
    model_name: str,
    model_version: str,
    model_label: str,
    run_date: date,
) -> pd.DataFrame:
    """Строки в Querulus_DataDrift для фич обучения без PSI (нет в train/prod)."""
    rows: list[dict[str, Any]] = []
    for col in missing:
        if not col or col in META_COLS:
            continue
        reason = _missing_feature_reason(col, train, prod)
        rows.append(
            {
                "model_name": model_name or stem,
                "model_version": model_version or stem,
                "date": run_date,
                "col": col,
                "TYPE": "MISSING",
                "PSI": np.nan,
                "KL": np.nan,
                "JS": np.nan,
                "NaN_train": np.nan,
                "NaN_test": np.nan,
                "uniq_train": np.nan,
                "uniq_test": np.nan,
                "mode_train": None,
                "mode_test": None,
                "mean_train": np.nan,
                "mean_test": np.nan,
                "dif_train": reason,
                "dif_test": reason,
                "stem": stem,
                "track": track,
                "model_label": model_label or stem,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


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
    missing_features: Iterable[str] | None = None,
) -> pd.DataFrame:
    """PSI/KL/JS + ``dif_*``; missing_features → строки с пустым PSI."""
    run_date = run_date or datetime.now().date()
    feat_list = [c for c in features if c not in META_COLS]
    parts: list[pd.DataFrame] = []

    x_train = _feature_frame(train, feat_list)
    x_prod = _feature_frame(prod, feat_list)
    if feat_list and not x_train.empty and not x_prod.empty:
        drift = QuerulusDataDrift(
            full_calc=True,
            columns_to_exclude=columns_to_exclude or [],
            n_bins=100,
            dif_len_string=100,
        )
        raw = drift.report(x_train, x_prod, x_train, x_prod)
        if not raw.empty:
            out = _normalize_report(raw)
            out["date"] = run_date
            out["model_name"] = model_name or stem
            out["model_version"] = model_version or stem
            out["stem"] = stem
            out["track"] = track
            out["model_label"] = model_label or stem
            if "TYPE" not in out.columns:
                out["TYPE"] = "NUMERICAL"

            for c in (
                "PSI",
                "KL",
                "JS",
                "NaN_train",
                "NaN_test",
                "mean_train",
                "mean_test",
            ):
                if c in out.columns:
                    out[c] = pd.to_numeric(out[c], errors="coerce")
            for c in ("uniq_train", "uniq_test"):
                if c in out.columns:
                    out[c] = pd.to_numeric(out[c], errors="coerce")

            def _to_python(v):
                return v.item() if isinstance(v, np.generic) else v

            out = out.apply(lambda s: s.map(_to_python))
            parts.append(out)

    placeholders = placeholder_drift_rows(
        missing_features or (),
        train=train,
        prod=prod,
        stem=stem,
        track=track,
        model_name=model_name or stem,
        model_version=model_version or stem,
        model_label=model_label or stem,
        run_date=run_date,
    )
    if not placeholders.empty:
        parts.append(placeholders)

    if not parts:
        return pd.DataFrame()

    preferred = ARCHIVE_COLS + ["stem", "track", "model_label"]
    out = pd.concat(parts, ignore_index=True)
    cols = [c for c in preferred if c in out.columns]
    out = out[cols]
    if "PSI" in out.columns:
        return out.sort_values(
            "PSI", ascending=False, na_position="last"
        ).reset_index(drop=True)
    return out.reset_index(drop=True)


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


def apply_train_filter(
    df: pd.DataFrame,
    train_filter: dict | None,
    *,
    stem: str = "",
) -> pd.DataFrame:
    """Срез Hive-train для datadrift по ``model['train_filter']``.

    * дата: ``date_col`` (дефолт PAYMENT_ORDER_DATE_TIME), start/end inclusive;
    * филиалы: ``filial_col`` isin, сравнение через ``casefold``.
    Пустой/отсутствующий filter → df без изменений. Не используется в fin_effect.
    """
    if not train_filter:
        return df
    if df.empty:
        return df

    out = df
    n0 = len(out)
    date_col = str(train_filter.get("date_col") or "PAYMENT_ORDER_DATE_TIME")
    date_start = str(train_filter.get("date_start") or "").strip()
    date_end = str(train_filter.get("date_end") or "").strip()
    if date_start or date_end:
        if date_col not in out.columns:
            logger.warning(
                "train_filter date ignored || stem=%s || missing col=%s",
                stem,
                date_col,
            )
        else:
            d = pd.to_datetime(out[date_col], errors="coerce")
            mask = pd.Series(True, index=out.index)
            if date_start:
                mask &= d >= pd.Timestamp(date_start)
            if date_end:
                # inclusive end (как pandas Series.between)
                mask &= d <= pd.Timestamp(date_end)
            out = out.loc[mask].copy()
            logger.info(
                "train_filter date || stem=%s || col=%s || [%s .. %s] || rows %s → %s",
                stem,
                date_col,
                date_start or "-",
                date_end or "-",
                n0,
                len(out),
            )
            n0 = len(out)

    filial_col = str(train_filter.get("filial_col") or "FILIAL")
    filials = tuple(train_filter.get("filials") or ())
    if filials:
        if filial_col not in out.columns:
            logger.warning(
                "train_filter filial ignored || stem=%s || missing col=%s",
                stem,
                filial_col,
            )
        else:
            wanted = {str(f).strip().casefold() for f in filials if str(f).strip()}
            series = out[filial_col].astype("string").fillna("").map(
                lambda v: str(v).strip().casefold()
            )
            out = out.loc[series.isin(wanted)].copy()
            logger.info(
                "train_filter filial || stem=%s || col=%s || n_filials=%s || rows %s → %s",
                stem,
                filial_col,
                len(wanted),
                n0,
                len(out),
            )

    return out


def _ensure_mldataworker_alias() -> None:
    """Prod pickle часто сериализован как mldataworker.*; пакет — outboxml.

    ``outboxml.models`` намеренно не импортируем (patsy/catboost/…) —
    для drift нужен только ``model_config``, см. ``_DriftConfigUnpickler``.
    """
    try:
        import outboxml
        import outboxml.core
        import outboxml.core.pydantic_models
    except ImportError as exc:
        raise ImportError(
            "для prepare_dataset нужны outboxml + pydantic (+ optbinning). "
            "outboxml ставится --no-deps — см. pipeline/requirements.txt"
        ) from exc
    sys.modules.setdefault("mldataworker", outboxml)
    sys.modules.setdefault("mldataworker.core", outboxml.core)
    sys.modules.setdefault(
        "mldataworker.core.pydantic_models", outboxml.core.pydantic_models
    )
    try:
        import outboxml.core.utils as _utils

        sys.modules.setdefault("mldataworker.core.utils", _utils)
    except ImportError:
        pass


class _DriftConfigUnpickler(pickle.Unpickler):
    """Unpickle ``[clf, rg]`` без catboost/xgboost/outboxml.models.

    Веса модели не нужны — только ``model_config`` для prepare_dataset.
    """

    _STUB_PREFIXES = (
        "catboost",
        "xgboost",
        "lightgbm",
        "outboxml.models",
        "mldataworker.models",
        "outboxml.data_subsets",
        "mldataworker.data_subsets",
        "outboxml.datasets_manager",
        "mldataworker.datasets_manager",
        "sklearn",
        "statsmodels",
        "patsy",
    )

    @staticmethod
    def _stub(name: str) -> type:
        """Заглушка без ``__slots__`` — нужен ``__dict__`` для __setstate__."""

        class _Stub:
            def __init__(self, *args, **kwargs):  # noqa: ANN001, ARG002
                pass

            def __setstate__(self, state):  # noqa: ANN001
                if isinstance(state, dict):
                    self.__dict__.update(state)
                else:
                    self._state = state

        _Stub.__name__ = name
        _Stub.__qualname__ = name
        return _Stub

    def find_class(self, module: str, name: str):  # noqa: ANN201
        if any(module == p or module.startswith(p + ".") for p in self._STUB_PREFIXES):
            return self._stub(name)
        return super().find_class(module, name)


def models_dir() -> Path:
    """Каталог `{stem}.pickle`: env DRIFT_MODELS_DIR или pipeline/models/."""
    override = os.environ.get("DRIFT_MODELS_DIR", "").strip()
    if override:
        return Path(override)
    return _DEFAULT_MODELS_DIR


def resolve_pickle_path(stem: str, *, pickle_override: str = "") -> Path:
    """Путь к `{stem}.pickle` (или явный override из config)."""
    if pickle_override.strip():
        path = Path(pickle_override.strip())
        if not path.is_absolute():
            path = models_dir() / path
        return path
    return models_dir() / f"{stem}.pickle"


def _model_config_from_result(item: Any):
    from outboxml.core.pydantic_models import ModelConfig

    if isinstance(item, dict):
        return ModelConfig.model_validate(item["model_config"])
    cfg = getattr(item, "model_config", None)
    if cfg is None:
        raise TypeError(
            f"неизвлечён model_config из {type(item).__name__} "
            "(ожидался dict с model_config или DSManagerResult)"
        )
    return ModelConfig.model_validate(cfg)


def _combine_prepared_matrices(clf_t: pd.DataFrame, reg_t: pd.DataFrame) -> pd.DataFrame:
    """Как integration.main._combine_output_matrices (общий столбец → ``__rg``)."""
    shared = [c for c in reg_t.columns if c in clf_t.columns]
    if not shared:
        return pd.concat([clf_t, reg_t], axis=1)
    reg_part = reg_t.rename(columns={c: f"{c}__rg" for c in shared})
    return pd.concat([clf_t, reg_part], axis=1)


def load_model_pair(stem: str, *, pickle_override: str = "") -> tuple[Any, Any]:
    """Загрузить ``[clf, rg]`` из pickle сервиса (только config, без ML-весов)."""
    _ensure_mldataworker_alias()
    path = resolve_pickle_path(stem, pickle_override=pickle_override)
    if not path.is_file():
        raise FileNotFoundError(
            f"Нет pickle для datadrift: {path}. "
            f"Положите {{stem}}.pickle в {models_dir()} "
            "(см. pipeline/models/README.md) или задайте DRIFT_MODELS_DIR."
        )
    with open(path, "rb") as f:
        group = _DriftConfigUnpickler(f).load()
    if not isinstance(group, (list, tuple)) or len(group) < 2:
        raise ValueError(
            f"{path.name}: ожидается [clf, rg], получено {type(group).__name__}"
        )
    logger.info(
        "pickle loaded (config-only) || stem=%s || path=%s", stem, path
    )
    return group[0], group[1]


def prepare_train_like_service(
    train_raw: pd.DataFrame,
    *,
    stem: str,
    clf: Any,
    rg: Any,
) -> pd.DataFrame:
    """Legacy: Hive → prepare_dataset(clf+rg). Сейчас drift идёт через decode prod."""
    from outboxml.core.data_prepare import prepare_dataset

    clf_cfg = _model_config_from_result(clf)
    rg_cfg = _model_config_from_result(rg)
    base = train_raw.copy()
    logger.info(
        "prepare_dataset train || stem=%s || rows=%s || cols_in=%s",
        stem,
        len(base),
        len(base.columns),
    )
    clf_prep = prepare_dataset(
        group_name=stem,
        data=base,
        train_ind=None,
        test_ind=None,
        model_config=clf_cfg,
        check_prepared=False,
        calc_corr=False,
        save_data=False,
        log=True,
        raise_on_encoding_error=False,
    ).data
    rg_prep = prepare_dataset(
        group_name=stem,
        data=base,
        train_ind=None,
        test_ind=None,
        model_config=rg_cfg,
        check_prepared=False,
        calc_corr=False,
        save_data=False,
        log=True,
        raise_on_encoding_error=False,
    ).data
    out = _combine_prepared_matrices(clf_prep, rg_prep)
    logger.info(
        "prepare_dataset train done || stem=%s || cols_out=%s",
        stem,
        len(out.columns),
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
    return df


def _load_prod_oracle_raw(settings: Settings) -> pd.DataFrame:
    """Сырые ответы модели из Oracle (MODEL::UU_QUERULUS → SPR_UU::QUERULUS)."""
    from datetime import timedelta

    cutoff = datetime.now() - timedelta(days=settings.drift_lookback_days)
    conn = connect_oracle(settings)
    try:
        raw = query_querulus_model_logs(conn, min_time_stamp=cutoff)
    finally:
        conn.close()
    if raw.empty:
        logger.warning("Oracle model→SPR logs empty || since=%s", cutoff)
    else:
        logger.info(
            "Oracle model→SPR raw || rows=%s || since=%s",
            len(raw),
            cutoff,
        )
    return raw


def _prod_features_for_model(
    raw: pd.DataFrame,
    *,
    slot: str,
    stem: str,
) -> pd.DataFrame:
    """Разворот ``*.df.output`` для slot; фильтр GIBDD; пустые строки → NA."""
    if raw.empty:
        return raw
    prod = expand_querulus_model_output(raw, slot=slot, stem=stem)
    if prod.empty:
        logger.warning(
            "prod output empty || slot=%s stem=%s (нет main/second_response.df.output)",
            slot,
            stem,
        )
        return prod
    for col in prod.columns:
        prod[col] = prod[col].map(
            lambda v: pd.NA if isinstance(v, str) and v == "" else v
        )
    if "EVENT_CREATED_BY_GIBDD_FLAG" in prod.columns:
        # в output обычно 0/1; строковый фильтр «Да» — как в старых логах
        gibdd = prod["EVENT_CREATED_BY_GIBDD_FLAG"]
        prod = prod.loc[gibdd.astype(str) != "Да"].copy()
    logger.info(
        "prod output || slot=%s stem=%s || rows=%s || cols=%s",
        slot,
        stem,
        len(prod),
        len(prod.columns),
    )
    return prod


def resolve_drift_features(
    configured: Iterable[str],
    train: pd.DataFrame,
    prod: pd.DataFrame,
    *,
    stem: str,
    source: str = "config",
) -> tuple[list[str], list[str]]:
    """Вернуть ``(usable, missing)``; missing → строки с пустым PSI в таблице."""
    wanted = [c for c in configured if c and c not in META_COLS]
    if not wanted:
        raise ValueError(
            f"Empty features for stem={stem!r} (source={source}). "
            "Проверьте pickle model_config.features или config.features."
        )

    train_cols = set(train.columns)
    prod_cols = set(prod.columns)
    missing_train = [c for c in wanted if c not in train_cols]
    missing_prod = [c for c in wanted if c not in prod_cols]
    if missing_train:
        logger.warning(
            "ALERT datadrift feature NOT IN TRAIN || stem=%s || source=%s || "
            "n=%s || cols=%s",
            stem,
            source,
            len(missing_train),
            missing_train,
        )
    if missing_prod:
        logger.warning(
            "ALERT datadrift feature NOT IN PROD || stem=%s || source=%s || "
            "n=%s || cols=%s",
            stem,
            source,
            len(missing_prod),
            missing_prod,
        )

    features = [c for c in wanted if c in train_cols and c in prod_cols]
    missing = [c for c in wanted if c not in features]
    if missing:
        logger.warning(
            "ALERT datadrift features MISSING → row with empty PSI || "
            "stem=%s || source=%s || wanted=%s || used=%s || missing=%s || cols=%s",
            stem,
            source,
            len(wanted),
            len(features),
            len(missing),
            missing,
        )
    else:
        logger.info(
            "datadrift features ok || stem=%s || source=%s || n=%s",
            stem,
            source,
            len(features),
        )
    return features, missing


def main() -> int:
    """Точка входа: для каждой main/second — Hive train + Oracle prod → append."""
    settings = load_settings()
    if not settings.drift_models:
        raise ValueError("DEFAULT_DRIFT_MODELS is empty in pipeline/config.py")

    logger.info(
        "datadrift ← Hive×%s models + Oracle model→SPR output → %s.%s || track=%s",
        len(settings.drift_models),
        settings.schema,
        settings.table_drift,
        DRIFT_TRACK,
    )
    raw_prod = _load_prod_oracle_raw(settings)
    if raw_prod.empty:
        logger.warning("Empty prod logs — skip datadrift write")
        return 0

    parts: list[pd.DataFrame] = []
    for model in settings.drift_models:
        slot = str(model.get("slot", "main") or "main")
        stem = str(model.get("stem", ""))
        hive = str(model.get("hive", ""))
        label = str(model.get("label") or stem)
        logger.info("datadrift model || slot=%s stem=%s hive=%s", slot, stem, hive)
        prod = _prod_features_for_model(raw_prod, slot=slot, stem=stem)
        if prod.empty:
            logger.warning(
                "skip model || slot=%s stem=%s — no prod output rows",
                slot,
                stem,
            )
            continue
        train_raw = load_reference_dataset(settings, hive_table=hive)
        train_raw = apply_train_filter(
            train_raw,
            model.get("train_filter"),
            stem=stem,
        )
        if train_raw.empty:
            logger.warning(
                "skip model || slot=%s stem=%s — train empty after train_filter",
                slot,
                stem,
            )
            continue
        pickle_override = str(model.get("pickle") or "")
        clf, rg = load_model_pair(stem, pickle_override=pickle_override)
        clf_cfg = _model_config_from_result(clf)
        rg_cfg = _model_config_from_result(rg)
        label_maps = build_category_label_maps(clf_cfg, rg_cfg)
        # train — Hive raw (+ filter); prod — раскодировать cats
        train = decode_category_columns(train_raw, label_maps)
        prod = decode_category_columns(prod, label_maps)
        logger.info(
            "datadrift plane || stem=%s || decoded_cats=%s || train_rows=%s || prod_rows=%s",
            stem,
            sorted(c for c in label_maps if c in train.columns or c in prod.columns),
            len(train),
            len(prod),
        )
        explicit = model.get("features")
        if explicit:
            wanted = list(explicit)
            feat_source = "config"
        else:
            wanted = features_from_model_configs(clf_cfg, rg_cfg)
            feat_source = "pickle.model_config"
            logger.info(
                "datadrift features ← pickle || stem=%s || n=%s || cols=%s",
                stem,
                len(wanted),
                wanted,
            )
        features, missing = resolve_drift_features(
            wanted, train, prod, stem=stem, source=feat_source
        )
        part = compute_datadrift(
            train,
            prod,
            features,
            stem=stem,
            track=DRIFT_TRACK,
            model_name=stem,
            model_version=stem,
            model_label=label,
            missing_features=missing,
        )
        if not part.empty:
            parts.append(part)
        logger.info(
            "datadrift model done || stem=%s || features=%s || missing=%s || rows=%s",
            stem,
            len(features),
            len(missing),
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
