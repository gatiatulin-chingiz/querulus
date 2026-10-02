"""Генерация AllModelsConfig JSON из артефактов train_loop (без ручной правки features[]).

Правила ``features[]`` (см. QUERULUS_WIKI §7.4):

* **categorical** — схлопывание высококардинальных: код получают уровни с долей
  ≥ ``CAT_LEVEL_SHARE_MIN`` (не более ``CAT_MAX_CODES``, но не менее
  ``CAT_MIN_CODES``); все прочие уровни + ``None``/``NaN`` → ``"ПРОЧИЕ": 0``.
  Если NaN-доля ≥ ``CAT_NAN_SHARE_MIN`` — ``NaN`` отдельная категория ``"NAN"``
  (``fillna`` = её код, ``default`` = код моды). Если уровней ≤
  ``CAT_SMALL_CARD_MAX`` — ``NaN`` уходит в моду (``default`` = код моды).
* **numerical** — ``default``: числовая медиана fit-среза (не ``"_MEDIAN_"`` —
  плейсхолдер в OutBoxML не пишется обратно при медиане 0 и ломает predict на
  срезе без train), для фич-годов — последний/больший год fit-среза
  ≤ ``MAX_YEAR_FEATURE_VALUE``; ``clip`` пишем **всем** int/float фичам,
  кроме бинарных: границы из DQ-отчёта сборки датасета, иначе квантили
  ``CLIP_QUANTILES`` на fit-срезе.
* **int-подобные** (``AGE``/``YEAR``/``MONTH``/``COUNT``/…, см.
  ``features.integer_casts``) — ``encoding: to_int`` и целые границы ``clip``
  (``floor``/``ceil``); фичи-годы — верхняя граница
  ``data_quality.MAX_YEAR_FEATURE_VALUE`` (текущий год = 2026).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from querulus import PROJECT_ROOT
from querulus.dataset.schema import DEFAULT_DATASET_SCHEMA
from querulus.features.data_quality import MAX_YEAR_FEATURE_VALUE
from querulus.features.integer_casts import is_integer_like_feature, is_year_feature
from querulus.training.catboost_fit import strip_hpo_meta
from querulus.training.config import TrainingConfig
from querulus.training.feature_selection_io import load_feature_selection_latest
from querulus.training.mvp_types import DEFAULT_MVP_INPUT_TYPES
from querulus.features.date_periods import mask_date_period
from querulus.training.splits import default_inner_periods_from_train, split_by_date_periods
from outboxml.data_subsets import ModelDataSubset
from outboxml.core import utils as outboxml_utils
from outboxml.core.enums import FeatureEngineering
from outboxml.core.pydantic_models import ModelConfig
from outboxml.core import data_prepare as outboxml_data_prepare
from querulus.features.inflation import ensure_legacy_real_column_aliases
from outboxml.core.prepared_datasets import PrepareDataset
from outboxml.core.pydantic_models import AllModelsConfig
from querulus.naming import MODEL_VERSION
from querulus.features.data_quality import clip_bounds_for_outboxml
from querulus.naming import (
        MODEL_CF_NAME,
        MODEL_NAME,
        MODEL_RG_NAME,
        configs_dir_for_version,
    )
from querulus.naming import MODEL_CF_NAME, MODEL_RG_NAME, configs_dir_for_version

DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "data" / "processed" / "train_loop_new"
DEFAULT_CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_HPO_PATH = DEFAULT_ARTIFACTS_DIR / "hpo_best_params_new.json"
PROD_FIT_TEST_FRACTION = 0.70
PROD_TAU_CAL_FRACTION = 0.15
PROD_HOLDOUT_FRACTION = 0.15
PROD_CAL_FRACTION = PROD_HOLDOUT_FRACTION  # alias: доля Test_prod (holdout)

_OTHER_KEY = "ПРОЧИЕ"
# Категория для пропусков: outboxml делает str.upper(), поэтому ключ в UPPER.
_NAN_KEY = "NAN"
# Строковые значения, которые тоже считаем пропуском (иначе «NaN» станет уровнем).
_NA_LIKE_KEYS: frozenset[str] = frozenset({"NAN", "N/A", "NA", "NONE", "NULL", ""})

CAT_LEVEL_SHARE_MIN: float = 0.01  # доля уровня в fit-срезе
CAT_MAX_CODES: int = 20  # максимум «своих» кодов (плюс ПРОЧИЕ/NAN)
CAT_MIN_CODES: int = 5  # минимум уровней, даже если доля ниже порога
CAT_NAN_SHARE_MIN: float = 0.10  # NaN-доля ≥ → отдельная категория
CAT_SMALL_CARD_MAX: int = 10  # уровней ≤ → None/NaN в моду
# Cap на число ключей replace: хвост уровней уходит в default (только для «монстров»).
CAT_MAX_REPLACE_KEYS: int = 5000
# Фолбэк-gраницы числовых фич, которых нет в DQ-отчёте (как в FS depth=0.01).
CLIP_QUANTILES: tuple[float, float] = (0.001, 0.999)
CLIP_MIN_FINITE: int = 20

# Имена, которые обязаны быть категориальными даже если пришли числовым кодом.
KNOWN_CATEGORICAL: frozenset[str] = DEFAULT_DATASET_SCHEMA.known_categorical
# Числовой код с большим числом значений — скорее ID, чем категория.
KNOWN_CATEGORICAL_MAX_NUNIQUE: int = DEFAULT_DATASET_SCHEMA.known_categorical_max_nunique

_load_subset_patched = False
_replace_default_patched = False

logger = logging.getLogger("querulus.training.outboxml_configs")

def _patch_model_data_subset_load_subset() -> None:
    """Согласовать индексы X/y в OutBoxML при ``data_filter_condition`` (RG parity).

    Без патча ``ModelDataSubset.load_subset`` оставляет ``y_train`` длиннее
    ``X_train`` после query-фильтра severity — DSM fit падает или даёт смещение.
    Патч идемпотентен; правки только в querulus, OutBoxML не меняем.
    """
    global _load_subset_patched
    if _load_subset_patched:
        return

    _orig = ModelDataSubset.load_subset

    @classmethod
    def _load_subset_aligned(cls, *args, **kwargs):
        subset = _orig(*args, **kwargs)
        if not subset.X_train.index.equals(subset.y_train.index):
            subset.y_train = subset.y_train.loc[subset.X_train.index]
            if getattr(subset, "exposure_train", None) is not None:
                subset.exposure_train = subset.exposure_train.loc[subset.X_train.index]
        if not subset.X_test.index.equals(subset.y_test.index):
            subset.y_test = subset.y_test.loc[subset.X_test.index]
            if getattr(subset, "exposure_test", None) is not None:
                subset.exposure_test = subset.exposure_test.loc[subset.X_test.index]
        return subset

    ModelDataSubset.load_subset = _load_subset_aligned  # type: ignore[method-assign]
    _load_subset_patched = True

def _patch_update_model_config_replace_keep_default() -> None:
    """OutBoxML: ``default`` (часто ``ПРОЧИЕ=0``) может не быть в train.

    Сток бросает ``NotImplementedError`` → ERROR ``Try to delete default value …``
    и срывает prune остальных unused levels. Патч: default оставляем, остальное чистим.
    Идемпотентен; OutBoxML не меняем.
    """
    global _replace_default_patched
    if _replace_default_patched:
        return

    logger = logging.getLogger("querulus.training.outboxml_patch")

    def _update_model_config_replace_keep_default(model_config, absent_levels):
        dumped = model_config.model_dump()
        for feature_name, levels in absent_levels.items():
            for i in range(len(dumped["features"])):
                if dumped["features"][i]["name"] != feature_name:
                    continue
                default = dumped["features"][i]["default"]
                levels_to_drop = [level for level in levels if level != default]
                if default in levels:
                    logger.warning(
                        "Default value %s of %s absent in train; "
                        "keep in replace, drop other unused levels only",
                        default,
                        feature_name,
                    )
                if not levels_to_drop:
                    break
                for value, level in list(dumped["features"][i]["replace"].items()):
                    if level == FeatureEngineering.not_changed:
                        level = value
                    if level in levels_to_drop:
                        dumped["features"][i]["replace"].pop(value)
                break
        return ModelConfig.model_validate(dumped)

    outboxml_utils.update_model_config_replace = (  # type: ignore[method-assign]
        _update_model_config_replace_keep_default
    )
    # prepare_dataset импортирует имя напрямую — патчим и там.
    try:

        outboxml_data_prepare.update_model_config_replace = (  # type: ignore[attr-defined]
            _update_model_config_replace_keep_default
        )
    except Exception:
        pass
    _replace_default_patched = True

def ensure_outboxml_runtime_patches() -> None:
    """Все runtime-патчи OutBoxML, нужные Querulus (без правок библиотеки)."""
    _patch_model_data_subset_load_subset()
    _patch_update_model_config_replace_keep_default()

def ensure_legacy_inflation_column(df: pd.DataFrame) -> pd.DataFrame:
    """Алиас ``*_REAL_2020`` ← текущий базис. Предпочтительно через ``save_df_final``."""

    return ensure_legacy_real_column_aliases(df)

def prepare_datasets_from_config(
    config_path: str | Path,
    *,
    check_prepared: bool = True,
) -> dict[str, Any]:
    """PrepareDataset'ы OutBoxML; runtime-патчи Querulus до prepare."""
    ensure_outboxml_runtime_patches()

    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    all_cfg = AllModelsConfig.model_validate(raw)
    group_name = f"{all_cfg.project}_{all_cfg.version}"

    return {
        model.name: PrepareDataset(
            model_config=model,
            check_prepared=check_prepared,
            group_name=group_name,
        )
        for model in all_cfg.models_configs
    }

def default_model_version(**_kwargs: Any) -> str:
    """SemVer модели (``2.0.0``). См. ``querulus.naming``."""

    return MODEL_VERSION

def _jsonable(value: Any) -> int | float | str | bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    if isinstance(value, str):
        return value
    return None

def catboost_params_from_hpo(
    hpo: dict[str, Any] | None,
    *,
    classification: bool,
    random_seed: int = 0,
) -> dict[str, int | float | str | bool]:
    """HPO best_params → params_catboost (без ES; iterations = tree_count)."""
    raw = dict(hpo or {})
    iterations = int(raw.get("tree_count") or raw.get("iterations") or (375 if classification else 100))
    merged: dict[str, int | float | str | bool] = {
        "iterations": iterations,
        "random_seed": int(raw.get("random_seed") or random_seed),
        "verbose": False,
        "allow_writing_files": False,
    }
    if classification:
        merged["auto_class_weights"] = "Balanced"
    skip = {
        "iterations",
        "iterations_cap",
        "early_stopping_rounds",
        "verbose",
        "allow_writing_files",
        "objective",
        "loss_function",
        "random_seed",
        "auto_class_weights",
    }
    for key, value in strip_hpo_meta(raw).items():
        if key in skip:
            continue
        converted = _jsonable(value)
        if converted is None:
            continue
        merged[key] = converted
    return merged

def _level_key(value: Any) -> str | None:
    """Ключ уровня для ``replace`` (outboxml делает ``.upper()`` на строках).

    ``None`` — пропуск: NaN/None/пустая строка и строки вида ``"NaN"``/``"N/A"``.
    Пропуски обрабатываются политикой ``fillna``/``default``, а не ключом replace
    (иначе литеральный ``"NaN"`` в данных становится отдельным уровнем).
    """
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return str(number)
    text = str(value).strip()
    if not text or text.upper() in _NA_LIKE_KEYS:
        return None
    return text.upper()

def _categorical_level_stats(series: pd.Series) -> tuple[dict[str, int], int, int, float]:
    """``(частоты по уровням, n_total, n_nan, nan_share)`` на fit-срезе."""
    keyed = series.astype("object").map(_level_key)
    is_na = keyed.isna()
    counts = keyed[~is_na].value_counts()
    freq = {str(key): int(value) for key, value in counts.items()}
    n_total = int(len(series))
    n_nan = int(is_na.sum())
    nan_share = (n_nan / n_total) if n_total else 0.0
    return freq, n_total, n_nan, nan_share

def _ordered_levels(freq: dict[str, int]) -> list[str]:
    """Уровни по убыванию частоты; тай-брейк — по имени (детерминизм конфига)."""
    return sorted(freq, key=lambda key: (-freq[key], key))

def _categorical_feature_spec(
    series: pd.Series,
    name: str,
    *,
    log: bool = True,
) -> dict[str, Any]:
    """``features[]``-спека категориальной фичи: схлопывание + политика NaN.

    * бинарные ``0/1`` — как в ``config_*_3``: свои коды, пропуск в моду;
    * полностью пустая фича — единственная категория ``"NAN"``;
    * ``nan_share ≥ CAT_NAN_SHARE_MIN`` — ``"NAN"`` отдельная категория
      (``fillna`` = её код; ``default`` = код моды, а не NaN-категория);
    * ``уровней ≤ CAT_SMALL_CARD_MAX`` — пропуск уходит в моду (``default``);
    * иначе — пропуск и хвост уровней уходят в ``"ПРОЧИЕ"`` (``default`` = 0).
    """
    freq, n_total, n_nan, nan_share = _categorical_level_stats(series)
    # Литеральный "ПРОЧИЕ" в данных — тот же бакет, что и схлопнутый хвост.
    freq.pop(_OTHER_KEY, None)
    ordered = _ordered_levels(freq)
    n_levels = len(ordered)

    # Всё пусто: единственная «категория» — пропуски.
    if n_levels == 0:
        if log:
            logger.warning("%s || нет ни одного уровня: все значения — NaN", name)
        return {
            "name": name,
            "default": 0,
            "replace": {_OTHER_KEY: 0, _NAN_KEY: 1},
            "encoding": "to_int",
            "fillna": 1,
        }

    # Бинарные 0/1 (флаги): без ПРОЧИЕ, пропуск → мода.
    if n_levels <= 2 and set(ordered) <= {"0", "1"}:
        replace = {key: index for index, key in enumerate(sorted(ordered))}
        mode_code = int(replace[ordered[0]])
        return {
            "name": name,
            "default": mode_code,
            "replace": replace,
            "encoding": "to_int",
            "fillna": mode_code,
        }

    shares = {key: (freq[key] / n_total if n_total else 0.0) for key in ordered}
    selected = [key for key in ordered if shares[key] >= CAT_LEVEL_SHARE_MIN]
    if len(selected) < CAT_MIN_CODES:
        selected = ordered[:CAT_MIN_CODES]
    selected = selected[:CAT_MAX_CODES]
    selected_set = set(selected)

    nan_bucket = nan_share >= CAT_NAN_SHARE_MIN
    small_card = n_levels <= CAT_SMALL_CARD_MAX

    replace = {_OTHER_KEY: 0}
    for code, key in enumerate(selected, start=1):
        replace[key] = code
    nan_code = len(selected) + 1
    if nan_bucket:
        replace[_NAN_KEY] = nan_code

    # Хвост уровней — явно в ПРОЧИЕ: при default=мода они иначе ушли бы в моду.
    leftover = [key for key in ordered if key not in selected_set]
    kept_leftover = 0
    for key in leftover:
        if len(replace) >= CAT_MAX_REPLACE_KEYS:
            break
        replace[key] = 0
        kept_leftover += 1
    if kept_leftover < len(leftover) and log:
        logger.warning(
            "%s || replace обрезан до %s ключей: %s хвостовых уровней уйдут в default",
            name,
            CAT_MAX_REPLACE_KEYS,
            len(leftover) - kept_leftover,
        )

    mode_code = int(replace[ordered[0]])  # мода всегда в selected (макс. доля)
    if nan_bucket:
        default, fillna = mode_code, nan_code
    elif small_card:
        default, fillna = mode_code, None
    else:
        default, fillna = 0, None

    spec: dict[str, Any] = {
        "name": name,
        "default": default,
        "replace": replace,
        "encoding": "to_int",
    }
    if fillna is not None:
        # ВАЖНО: код 0 falsy → outboxml проигнорирует fillna и возьмёт default.
        spec["fillna"] = fillna

    if log:
        logger.info(
            "%s || cat || уровней=%s (кодов=%s, хвост→ПРОЧИЕ=%s) | NaN=%s (%.1f%%) | "
            "default=%s (%s) | fillna=%s",
            name,
            n_levels,
            len(selected),
            kept_leftover,
            n_nan,
            100.0 * nan_share,
            default,
            "мода" if nan_bucket or small_card else "ПРОЧИЕ",
            fillna,
        )
    return spec

def _finite_values(series: pd.Series) -> np.ndarray:
    """Конечные числовые значения серии (без NaN/inf)."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]

def _clip_for_feature(
    series: pd.Series,
    name: str,
    *,
    dq_bounds: dict[str, dict[str, float]] | None = None,
    log: bool = True,
) -> dict[str, float] | None:
    """``clip`` числовой фичи: DQ-границы сборки, иначе квантили ``CLIP_QUANTILES``.

    * int-подобные (``AGE``/``YEAR``/``COUNT``/…) — целые ``floor``/``ceil``;
      ``min`` не ниже 0, если в данных нет отрицательных значений (лаги
      ``FE_DAYS_*`` остаются знаковыми);
    * фичи-годы — верх = ``MAX_YEAR_FEATURE_VALUE`` (2026), ``min`` ≤ верх;
    * ``None`` — границы вырождены (нет конечных/одно значение) → clip не пишем.
    """
    integer_like = is_integer_like_feature(name)
    year = is_year_feature(name)
    finite = _finite_values(series)
    bounds = (dq_bounds or {}).get(name)

    if bounds is not None:
        low = float(bounds["min_value"])
        high = float(bounds["max_value"])
        source = "DQ"
    elif finite.size >= CLIP_MIN_FINITE:
        low, high = (float(q) for q in np.quantile(finite, list(CLIP_QUANTILES)))
        source = f"quantile{CLIP_QUANTILES}"
    elif year and finite.size:
        low = high = float(finite.min())
        source = "observed_min"
    else:
        if log:
            logger.warning(
                "%s || clip не задан: нет DQ-границ и мало конечных значений (%s)",
                name,
                int(finite.size),
            )
        return None

    if integer_like:
        low = float(math.floor(low))
        high = float(math.ceil(high))
        if finite.size and float(finite.min()) >= 0.0:
            low = max(0.0, low)
    if year:
        low = min(max(0.0, low), float(MAX_YEAR_FEATURE_VALUE))
        high = float(MAX_YEAR_FEATURE_VALUE)

    if not (np.isfinite(low) and np.isfinite(high)) or high <= low:
        if log:
            logger.warning(
                "%s || clip не задан: вырожденные границы [%s, %s]", name, low, high
            )
        return None
    if log:
        logger.info("%s || clip [%s, %s] (%s)", name, low, high, source)
    return {"min_value": low, "max_value": high}

def _latest_year_default(series: pd.Series) -> int:
    """Последний (больший) год fit-среза, не выше ``MAX_YEAR_FEATURE_VALUE``.

    Пишем конкретное число, а не ``"_MAX_"``: OutBoxML резолвит плейсхолдеры через
    ``series.max()``, а для **целой** колонки это ``np.int64`` — и проверка
    ``isinstance(default_value, (int, float))`` в ``prepare_numerical_feature_series``
    падает с ``ConfigError`` (``np.int64`` не наследник ``int``). В сервис
    ``EVENT_YEAR`` приходит именно целым, поэтому ``"_MAX_"`` ломал бы инференс.
    """
    finite = _finite_values(series)
    if finite.size == 0:
        return int(MAX_YEAR_FEATURE_VALUE)
    latest = int(math.floor(float(finite.max())))
    return int(min(max(latest, 0), MAX_YEAR_FEATURE_VALUE))

def _median_default(series: pd.Series, *, integer_like: bool = False) -> float | int:
    """Медиана fit-среза как число (не ``"_MEDIAN_"``)."""
    finite = _finite_values(series)
    if finite.size == 0:
        return 0 if integer_like else 0.0
    med = float(np.median(finite))
    if not np.isfinite(med):
        return 0 if integer_like else 0.0
    if integer_like:
        return int(round(med))
    return med

def _numeric_feature_spec(
    series: pd.Series,
    name: str,
    *,
    clip: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Числовая фича.

    ``default``: медиана fit-среза числом (для годов — ``_latest_year_default``).
    ``clip`` — см. ``_clip_for_feature``.
    ``encoding``: ``to_int`` для целочисленных по смыслу, иначе ``to_float``.
    """
    year = is_year_feature(name)
    integer_like = is_integer_like_feature(name)
    spec: dict[str, Any] = {
        "name": name,
        "default": (
            _latest_year_default(series)
            if year
            else _median_default(series, integer_like=integer_like)
        ),
        "replace": {"_TYPE_": "_NUM_"},
        "encoding": "to_int" if integer_like else "to_float",
    }
    if clip is not None:
        spec["clip"] = {
            "min_value": float(clip["min_value"]),
            "max_value": float(clip["max_value"]),
        }
    return spec

def _is_categorical(
    name: str,
    *,
    json_cats: set[str],
    mvp_cats: set[str],
    series: pd.Series | None,
) -> bool:
    if name in json_cats or name in mvp_cats:
        return True
    if series is None:
        return False
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return True
    if pd.api.types.is_bool_dtype(series):
        return True
    numeric = pd.to_numeric(series, errors="coerce")
    nunique = int(numeric.nunique(dropna=True))
    if nunique <= 2 and nunique > 0:
        return True
    # Известные категориальные бизнес-фичи (могут прийти числовым кодом):
    # 0/1 уже отсеяны выше, «монстры» (nunique > 200) — скорее ID, чем категория.
    if str(name).upper() in KNOWN_CATEGORICAL:
        return 0 < nunique <= KNOWN_CATEGORICAL_MAX_NUNIQUE
    return False

def build_features_block(
    df: pd.DataFrame,
    feature_names: list[str],
    *,
    categorical_names: list[str] | None = None,
    mvp_types: dict[str, list[str] | tuple[str, ...]] | None = None,
    fit_index: pd.Index | None = None,
    clip_bounds: dict[str, dict[str, float]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """``features[]`` + ``cat_features_catboost`` по train-срезу.

    ``clip_bounds`` — границы из DQ-отчёта сборки (``clip_bounds_for_outboxml``);
    для фич без записи в отчёте границы считаются квантилями на ``fit_index``.
    """
    types = mvp_types or DEFAULT_MVP_INPUT_TYPES
    mvp_cats = set(types.get("CATEGORIAL") or ()) | set(types.get("BINARY") or ())
    json_cats = set(categorical_names or ())
    clips = clip_bounds or {}
    frame = df.loc[fit_index] if fit_index is not None else df
    features: list[dict[str, Any]] = []
    cat_features: list[str] = []
    missing = [name for name in feature_names if name not in df.columns]
    if missing:
        raise KeyError(f"В df нет фич из latest JSON: {missing[:20]}")
    for name in feature_names:
        series = frame[name] if name in frame.columns else df[name]
        if _is_categorical(name, json_cats=json_cats, mvp_cats=mvp_cats, series=series):
            features.append(_categorical_feature_spec(series, name))
            cat_features.append(name)
        else:
            features.append(
                _numeric_feature_spec(
                    series,
                    name,
                    clip=_clip_for_feature(series, name, dq_bounds=clips),
                )
            )
    return features, cat_features

def load_hpo_best_params(path: Path | str | None = None) -> dict[str, Any]:
    hpo_path = Path(path) if path is not None else DEFAULT_HPO_PATH
    if not hpo_path.exists():
        raise FileNotFoundError(f"Нет HPO JSON: {hpo_path}")
    return json.loads(hpo_path.read_text(encoding="utf-8"))

def load_selected_task(
    task: str,
    *,
    stack: str = "new",
    artifacts_dir: Path | str | None = None,
) -> tuple[list[str], list[str]]:
    payload = load_feature_selection_latest(
        stack, task, directory=artifacts_dir or DEFAULT_ARTIFACTS_DIR
    )
    if not payload:
        raise FileNotFoundError(
            f"Нет {stack}_{task}_latest.json в {artifacts_dir or DEFAULT_ARTIFACTS_DIR}"
        )
    selected = [str(name) for name in payload.get("selected_features") or []]
    cats = [str(name) for name in payload.get("categorical_features") or []]
    if not selected:
        raise ValueError(f"Пустой selected_features в {stack}_{task}_latest.json")
    return selected, cats

def _data_config(
    *,
    parquet_path: str,
    train_period: tuple[str, str],
    test_period: tuple[str, str],
    date_column: str,
    extra_columns: list[str],
) -> dict[str, Any]:
    return {
        "source": "parquet",
        "table_name_source": "",
        "local_name_source": parquet_path,
        "processing": True,
        "showGraphs": False,
        "separation": {
            "kind": "date",
            "random_state": 0,
            "train_period": [train_period[0], train_period[1]],
            "test_period": [test_period[0], test_period[1]],
            "period_column": [date_column],
        },
        "extra_columns": extra_columns,
        "data": {"targetcolumns": [], "targetslices": []},
    }

def build_model_entry(
    *,
    model_name: str,
    column_target: str,
    objective: str,
    features: list[dict[str, Any]],
    cat_features: list[str],
    params_catboost: dict[str, Any],
    data_filter_condition: str | None = None,
) -> dict[str, Any]:
    """Одна модель внутри ``models_configs``."""
    model: dict[str, Any] = {
        "name": model_name,
        "column_target": column_target,
        "objective": objective,
        "wrapper": "catboost",
        "params_catboost": params_catboost,
        "cat_features_catboost": cat_features,
        "relative_features": [],
        "features": features,
    }
    if data_filter_condition:
        model["data_filter_condition"] = data_filter_condition
    return model

def build_all_models_config(
    *,
    project: str,
    version: str,
    group_name: str = "UU",
    parquet_path: str,
    train_period: tuple[str, str],
    test_period: tuple[str, str],
    date_column: str,
    model_name: str,
    column_target: str,
    objective: str,
    features: list[dict[str, Any]],
    cat_features: list[str],
    params_catboost: dict[str, Any],
    extra_columns: list[str] | None = None,
    data_filter_condition: str | None = None,
) -> dict[str, Any]:
    """AllModelsConfig с одной моделью (legacy helper)."""
    extras = list(
        dict.fromkeys(
            [*(extra_columns or []), date_column, column_target]
        )
    )
    model = build_model_entry(
        model_name=model_name,
        column_target=column_target,
        objective=objective,
        features=features,
        cat_features=cat_features,
        params_catboost=params_catboost,
        data_filter_condition=data_filter_condition,
    )
    return {
        "group_name": group_name,
        "project": project,
        "version": version,
        "data_config": _data_config(
            parquet_path=parquet_path,
            train_period=train_period,
            test_period=test_period,
            date_column=date_column,
            extra_columns=extras,
        ),
        "models_configs": [model],
    }

def build_cf_rg_config(
    *,
    project: str,
    version: str,
    group_name: str,
    parquet_path: str,
    train_period: tuple[str, str],
    test_period: tuple[str, str],
    date_column: str,
    cf_name: str,
    rg_name: str,
    freq_block: list[dict[str, Any]],
    freq_cat_out: list[str],
    sev_block: list[dict[str, Any]],
    sev_cat_out: list[str],
    cf_params: dict[str, Any],
    rg_params: dict[str, Any],
) -> dict[str, Any]:
    """Один AllModelsConfig: frequency + severity."""
    extras = list(
        dict.fromkeys(
            [date_column, "TARGET_FREQ", "TARGET_SEV"]
        )
    )
    return {
        "group_name": group_name,
        "project": project,
        "version": version,
        "data_config": _data_config(
            parquet_path=parquet_path,
            train_period=train_period,
            test_period=test_period,
            date_column=date_column,
            extra_columns=extras,
        ),
        "models_configs": [
            build_model_entry(
                model_name=cf_name,
                column_target="TARGET_FREQ",
                objective="binary",
                features=freq_block,
                cat_features=freq_cat_out,
                params_catboost=cf_params,
            ),
            build_model_entry(
                model_name=rg_name,
                column_target="TARGET_SEV",
                objective="regression",
                features=sev_block,
                cat_features=sev_cat_out,
                params_catboost=rg_params,
                data_filter_condition="TARGET_SEV > 0",
            ),
        ],
    }

def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path

def with_periods(
    config: dict[str, Any],
    *,
    train_period: tuple[str, str],
    test_period: tuple[str, str],
) -> dict[str, Any]:
    """Копия конфига с другими date-окнами (prod-refit)."""
    clone = json.loads(json.dumps(config))
    sep = clone["data_config"]["separation"]
    sep["train_period"] = [train_period[0], train_period[1]]
    sep["test_period"] = [test_period[0], test_period[1]]
    return clone

@dataclass(frozen=True)
class PeriodWindow:
    role: str
    start: str
    end: str
    n: int
    n_positive: int

def _fmt(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")

def _pos_count(df: pd.DataFrame, index: pd.Index, target: str) -> int:
    if target not in df.columns or index.empty:
        return 0
    y = pd.to_numeric(df.loc[index, target], errors="coerce").fillna(0)
    return int((y.astype(int) == 1).sum())

def compute_period_windows(
    df: pd.DataFrame,
    *,
    date_column: str = DEFAULT_DATASET_SCHEMA.date_column,
    train_period: tuple[str, str] | None = None,
    test_period: tuple[str, str] | None = None,
    freq_target: str = DEFAULT_DATASET_SCHEMA.frequency_target,
    prod_fit_test_fraction: float = PROD_FIT_TEST_FRACTION,
    prod_tau_cal_fraction: float = PROD_TAU_CAL_FRACTION,
    prod_holdout_fraction: float = PROD_HOLDOUT_FRACTION,
) -> dict[str, Any]:
    """Parity (train_core∪val / cal / test) и prod 70/15/15 внутри holdout Test."""
    cfg = TrainingConfig()
    train_period = train_period or cfg.train_period
    test_period = test_period or cfg.test_period
    train_core, val_period, cal_period = default_inner_periods_from_train(train_period)
    splits = split_by_date_periods(
        df,
        date_column=date_column,
        train_period=train_core,
        val_period=val_period,
        cal_period=cal_period,
        test_period=test_period,
    )
    parity_train = (train_period[0], val_period[1])
    dates = pd.to_datetime(df[date_column], errors="coerce")
    test_idx = pd.Index(splits.test)
    ordered = dates.loc[test_idx].sort_values()
    n_test = int(len(ordered))
    if n_test < 10:
        raise ValueError(f"Слишком мало строк test={n_test}")
    n_fit = max(1, int(round(n_test * float(prod_fit_test_fraction))))
    n_tau = max(1, int(round(n_test * float(prod_tau_cal_fraction))))
    n_hold = n_test - n_fit - n_tau
    if n_hold < 1:
        n_hold = 1
        n_tau = max(1, n_test - n_fit - n_hold)
    if n_fit + n_tau + n_hold != n_test:
        n_hold = n_test - n_fit - n_tau

    ordered_idx = pd.Index(ordered.index)
    fit_test_idx = ordered_idx[:n_fit]
    tau_cal_test_idx = ordered_idx[n_fit : n_fit + n_tau]
    holdout_test_idx = ordered_idx[n_fit + n_tau :]

    tau_start_ts = pd.Timestamp(ordered.iloc[n_fit])
    tau_end_ts = pd.Timestamp(ordered.iloc[n_fit + n_tau - 1])
    holdout_start_ts = pd.Timestamp(ordered.iloc[n_fit + n_tau])
    prod_train_end = tau_start_ts - pd.Timedelta(days=1)
    prod_train_period = (train_period[0], _fmt(prod_train_end))
    prod_tau_cal_period = (_fmt(tau_start_ts), _fmt(tau_end_ts))
    prod_test_period = (_fmt(holdout_start_ts), test_period[1])
    prod_fit_idx = df.index[
        mask_date_period(dates, prod_train_period[0], prod_train_period[1]).fillna(False)
    ]
    prod_tau_cal_idx = tau_cal_test_idx
    prod_holdout_idx = holdout_test_idx
    prod_cal_idx = prod_holdout_idx
    parity_fit_idx = splits.train.union(splits.val)

    def row(role: str, start: str, end: str, index: pd.Index) -> PeriodWindow:
        return PeriodWindow(
            role=role,
            start=start,
            end=end,
            n=int(len(index)),
            n_positive=_pos_count(df, index, freq_target),
        )

    windows = [
        row("parity_train (core∪val)", parity_train[0], parity_train[1], parity_fit_idx),
        row("train_tail (хвост train; в collect — Cal)", cal_period[0], cal_period[1], splits.cal),
        row("holdout Test", test_period[0], test_period[1], splits.test),
        row(
            f"prod_fit (train + {prod_fit_test_fraction:.0%} test)",
            prod_train_period[0],
            prod_train_period[1],
            prod_fit_idx,
        ),
        row(
            f"prod_tau_cal ({prod_tau_cal_fraction:.0%} test; τ в collect)",
            prod_tau_cal_period[0],
            prod_tau_cal_period[1],
            prod_tau_cal_idx,
        ),
        row(
            f"Test_prod ({prod_holdout_fraction:.0%} freshest test)",
            prod_test_period[0],
            prod_test_period[1],
            prod_holdout_idx,
        ),
    ]
    return {
        "date_column": date_column,
        "freq_target": freq_target,
        "base_train_period": train_period,
        "base_test_period": test_period,
        "train_core": train_core,
        "val_period": val_period,
        "cal_period": cal_period,
        "parity_train_period": parity_train,
        "parity_test_period": test_period,
        "prod_train_period": prod_train_period,
        "prod_tau_cal_period": prod_tau_cal_period,
        "prod_test_period": prod_test_period,
        "prod_cutoff": _fmt(holdout_start_ts),
        "prod_tau_cal_cutoff": _fmt(holdout_start_ts),
        "prod_fit_test_fraction": float(prod_fit_test_fraction),
        "prod_tau_cal_fraction": float(prod_tau_cal_fraction),
        "prod_holdout_fraction": float(prod_holdout_fraction),
        "prod_cal_fraction": float(prod_holdout_fraction),
        "prod_fit_idx": prod_fit_idx,
        "prod_tau_cal_idx": prod_tau_cal_idx,
        "prod_holdout_idx": prod_holdout_idx,
        "splits": splits,
        "windows": windows,
        "table": pd.DataFrame([w.__dict__ for w in windows]),
    }

PERIOD_WINDOWS_JSON = "period_windows.json"

_PERIOD_MANIFEST_TUPLE_KEYS: tuple[str, ...] = (
    "base_train_period",
    "base_test_period",
    "train_core",
    "val_period",
    "cal_period",
    "parity_train_period",
    "parity_test_period",
    "prod_train_period",
    "prod_tau_cal_period",
    "prod_test_period",
)

_PERIOD_MANIFEST_SCALAR_KEYS: tuple[str, ...] = (
    "date_column",
    "freq_target",
    "prod_cutoff",
    "prod_tau_cal_cutoff",
    "prod_fit_test_fraction",
    "prod_tau_cal_fraction",
    "prod_holdout_fraction",
    "prod_cal_fraction",
)


def periods_to_manifest(periods: dict[str, Any]) -> dict[str, Any]:
    """Сериализация окон периодов (без Index / DataFrame) для collect → example."""
    payload: dict[str, Any] = {}
    for key in _PERIOD_MANIFEST_TUPLE_KEYS:
        value = periods[key]
        payload[key] = list(value) if isinstance(value, tuple) else value
    for key in _PERIOD_MANIFEST_SCALAR_KEYS:
        if key in periods:
            payload[key] = periods[key]
    return payload


def write_period_windows_manifest(
    periods: dict[str, Any],
    path: Path | str,
) -> Path:
    """Записать ``period_windows.json`` рядом с OutBoxML-конфигами."""
    out = Path(path)
    return write_json(out, periods_to_manifest(periods))


def rebuild_periods_from_manifest(
    df: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Индексы периодов из сохранённых date-окон (без пересчёта 70/15/15)."""
    date_column = str(manifest.get("date_column") or DEFAULT_DATASET_SCHEMA.date_column)
    freq_target = str(manifest.get("freq_target") or DEFAULT_DATASET_SCHEMA.frequency_target)
    if date_column not in df.columns:
        raise ValueError(f"Нет колонки даты: {date_column}")

    def _pair(key: str) -> tuple[str, str]:
        raw = manifest[key]
        return (str(raw[0]), str(raw[1]))

    train_core = _pair("train_core")
    val_period = _pair("val_period")
    cal_period = _pair("cal_period")
    test_period = _pair("parity_test_period")
    parity_train = _pair("parity_train_period")
    prod_train_period = _pair("prod_train_period")
    prod_tau_cal_period = _pair("prod_tau_cal_period")
    prod_test_period = _pair("prod_test_period")
    base_train = _pair("base_train_period")
    base_test = _pair("base_test_period")

    splits = split_by_date_periods(
        df,
        date_column=date_column,
        train_period=train_core,
        val_period=val_period,
        cal_period=cal_period,
        test_period=test_period,
    )
    dates = pd.to_datetime(df[date_column], errors="coerce")
    prod_fit_idx = df.index[
        mask_date_period(dates, *prod_train_period).fillna(False)
    ]
    prod_tau_cal_idx = df.index[
        mask_date_period(dates, *prod_tau_cal_period).fillna(False)
    ]
    prod_holdout_idx = df.index[
        mask_date_period(dates, *prod_test_period).fillna(False)
    ]
    parity_fit_idx = splits.train.union(splits.val)
    prod_fit_frac = float(manifest.get("prod_fit_test_fraction", PROD_FIT_TEST_FRACTION))
    prod_tau_frac = float(manifest.get("prod_tau_cal_fraction", PROD_TAU_CAL_FRACTION))
    prod_hold_frac = float(manifest.get("prod_holdout_fraction", PROD_HOLDOUT_FRACTION))

    def row(role: str, start: str, end: str, index: pd.Index) -> PeriodWindow:
        return PeriodWindow(
            role=role,
            start=start,
            end=end,
            n=int(len(index)),
            n_positive=_pos_count(df, index, freq_target),
        )

    windows = [
        row("parity_train (core∪val)", parity_train[0], parity_train[1], parity_fit_idx),
        row("train_tail (хвост train; в collect — Cal)", cal_period[0], cal_period[1], splits.cal),
        row("holdout Test", test_period[0], test_period[1], splits.test),
        row(
            f"prod_fit (train + {prod_fit_frac:.0%} test)",
            prod_train_period[0],
            prod_train_period[1],
            prod_fit_idx,
        ),
        row(
            f"prod_tau_cal ({prod_tau_frac:.0%} test; τ в example_final)",
            prod_tau_cal_period[0],
            prod_tau_cal_period[1],
            prod_tau_cal_idx,
        ),
        row(
            f"Test_prod ({prod_hold_frac:.0%} freshest test)",
            prod_test_period[0],
            prod_test_period[1],
            prod_holdout_idx,
        ),
    ]
    if len(prod_tau_cal_idx) == 0 or len(prod_holdout_idx) == 0:
        logger.warning(
            "Манифест периодов: пустые индексы τ-cal=%s Test_prod=%s — "
            "df не совпадает с collect (проверьте data_date / parquet).",
            len(prod_tau_cal_idx),
            len(prod_holdout_idx),
        )
    return {
        "date_column": date_column,
        "freq_target": freq_target,
        "base_train_period": base_train,
        "base_test_period": base_test,
        "train_core": train_core,
        "val_period": val_period,
        "cal_period": cal_period,
        "parity_train_period": parity_train,
        "parity_test_period": test_period,
        "prod_train_period": prod_train_period,
        "prod_tau_cal_period": prod_tau_cal_period,
        "prod_test_period": prod_test_period,
        "prod_cutoff": str(manifest.get("prod_cutoff") or prod_test_period[0]),
        "prod_tau_cal_cutoff": str(
            manifest.get("prod_tau_cal_cutoff") or prod_test_period[0]
        ),
        "prod_fit_test_fraction": prod_fit_frac,
        "prod_tau_cal_fraction": prod_tau_frac,
        "prod_holdout_fraction": prod_hold_frac,
        "prod_cal_fraction": float(manifest.get("prod_cal_fraction", prod_hold_frac)),
        "prod_fit_idx": prod_fit_idx,
        "prod_tau_cal_idx": prod_tau_cal_idx,
        "prod_holdout_idx": prod_holdout_idx,
        "splits": splits,
        "windows": windows,
        "table": pd.DataFrame([w.__dict__ for w in windows]),
        "from_manifest": True,
    }


def load_period_windows_manifest(path: Path | str) -> dict[str, Any] | None:
    """Прочитать ``period_windows.json`` или ``None``, если файла нет."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))

def write_outboxml_configs(
    df: pd.DataFrame,
    *,
    version: str | None = None,
    parquet_path: str | None = None,
    artifacts_dir: Path | str | None = None,
    configs_dir: Path | str | None = None,
    hpo_path: Path | str | None = None,
    dq_report_path: Path | str | None = None,
    date_column: str = DEFAULT_DATASET_SCHEMA.date_column,
    train_period: tuple[str, str] | None = None,
    test_period: tuple[str, str] | None = None,
    group_name: str = "UU",
) -> dict[str, Any]:
    """Пишет ``config_parity.json`` / ``config_prod.json`` (CF+RG в каждом).

    Вызывать из collect после ``save_df_final``; example читает через ``load_outboxml_configs``.
    Имена моделей стабильные: ``querulus_cf`` / ``querulus_rg``.
    Numeric ``feature.clip`` — из ``data_quality_report.json`` (обе секции winsorize →
    ``low_raw``/``high_raw``), для фич без записи в отчёте — квантили ``CLIP_QUANTILES``
    на fit-срезе; см. ``build_features_block`` и ``_clip_for_feature``.
    """

    version = version or default_model_version()
    artifacts_dir = Path(artifacts_dir) if artifacts_dir else DEFAULT_ARTIFACTS_DIR
    configs_dir = (
        Path(configs_dir) if configs_dir is not None else configs_dir_for_version(version)
    )
    parquet_path = parquet_path or str(
        (PROJECT_ROOT / "data" / "processed" / "querulus_train_dataset.parquet").as_posix()
    )
    report_path = (
        Path(dq_report_path)
        if dq_report_path is not None
        else PROJECT_ROOT / "data" / "processed" / "data_quality_report.json"
    )
    clip_bounds = clip_bounds_for_outboxml(report_path=report_path)
    periods = compute_period_windows(
        df,
        date_column=date_column,
        train_period=train_period,
        test_period=test_period,
    )
    hpo = load_hpo_best_params(hpo_path)
    freq_feats, freq_cats = load_selected_task("frequency", artifacts_dir=artifacts_dir)
    sev_feats, sev_cats = load_selected_task("severity", artifacts_dir=artifacts_dir)
    fit_index = periods["splits"].train.union(periods["splits"].val)

    freq_block, freq_cat_out = build_features_block(
        df,
        freq_feats,
        categorical_names=freq_cats,
        fit_index=fit_index,
        clip_bounds=clip_bounds,
    )
    sev_block, sev_cat_out = build_features_block(
        df,
        sev_feats,
        categorical_names=sev_cats,
        fit_index=fit_index,
        clip_bounds=clip_bounds,
    )
    cf_name = MODEL_CF_NAME
    rg_name = MODEL_RG_NAME
    cf_params = catboost_params_from_hpo(
        hpo.get("frequency") if isinstance(hpo.get("frequency"), dict) else hpo,
        classification=True,
    )
    rg_params = catboost_params_from_hpo(
        hpo.get("severity") if isinstance(hpo.get("severity"), dict) else {},
        classification=False,
    )
    parity_cfg = build_cf_rg_config(
        project=MODEL_NAME,
        version="1",
        group_name=group_name,
        parquet_path=parquet_path,
        train_period=periods["parity_train_period"],
        test_period=periods["parity_test_period"],
        date_column=date_column,
        cf_name=cf_name,
        rg_name=rg_name,
        freq_block=freq_block,
        freq_cat_out=freq_cat_out,
        sev_block=sev_block,
        sev_cat_out=sev_cat_out,
        cf_params=cf_params,
        rg_params=rg_params,
    )
    prod_cfg = with_periods(
        parity_cfg,
        train_period=periods["prod_train_period"],
        test_period=periods["prod_test_period"],
    )
    parity_path = write_json(configs_dir / "config_parity.json", parity_cfg)
    prod_path = write_json(configs_dir / "config_prod.json", prod_cfg)
    manifest_path = write_period_windows_manifest(
        periods, configs_dir / PERIOD_WINDOWS_JSON
    )
    return {
        "version": version,
        "parity_path": parity_path,
        "prod_path": prod_path,
        # aliases for older call sites
        "cf_path": parity_path,
        "rg_path": parity_path,
        "cf_prod_path": prod_path,
        "rg_prod_path": prod_path,
        "cf_name": cf_name,
        "rg_name": rg_name,
        "periods": periods,
        "period_windows_path": manifest_path,
        "n_frequency_features": len(freq_feats),
        "n_severity_features": len(sev_feats),
        "n_clip_bounds": len(clip_bounds),
        "dq_report_path": str(report_path) if report_path.is_file() else None,
        "configs_dir": str(configs_dir),
    }

_OUTBOXML_CONFIG_FILES: tuple[str, ...] = (
    "config_parity.json",
    "config_prod.json",
)

def load_outboxml_configs(
    df: pd.DataFrame,
    *,
    version: str | None = None,
    configs_dir: Path | str | None = None,
    date_column: str = DEFAULT_DATASET_SCHEMA.date_column,
    train_period: tuple[str, str] | None = None,
    test_period: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Загрузить ``config_parity.json`` / ``config_prod.json`` из collect.

    Окна периодов берутся из ``period_windows.json`` (манифест collect), чтобы
    example / example_final совпадали с collect. Индексы пересобираются по датам
    на текущем ``df``. Если манифеста нет — fallback на ``compute_period_windows``
    с warning (старые прогоны).
    """

    version = version or default_model_version()
    configs_dir = (
        Path(configs_dir) if configs_dir is not None else configs_dir_for_version(version)
    )
    paths = {name: configs_dir / name for name in _OUTBOXML_CONFIG_FILES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Нет OutBoxML-конфигов в {configs_dir}: {', '.join(missing)}. "
            "Сначала запустите collect (ячейка export → write_outboxml_configs)."
        )
    manifest = load_period_windows_manifest(configs_dir / PERIOD_WINDOWS_JSON)
    if manifest is not None:
        if train_period is not None or test_period is not None:
            logger.warning(
                "load_outboxml_configs: train_period/test_period аргументы "
                "игнорируются — периоды из %s",
                PERIOD_WINDOWS_JSON,
            )
        periods = rebuild_periods_from_manifest(df, manifest)
        logger.info(
            "Периоды из манифеста collect (%s): prod_cutoff=%s",
            configs_dir / PERIOD_WINDOWS_JSON,
            periods.get("prod_cutoff"),
        )
    else:
        logger.warning(
            "Нет %s в %s — пересчёт окон из df (периоды могут "
            "разъехаться с collect). Перезапустите write_outboxml_configs.",
            PERIOD_WINDOWS_JSON,
            configs_dir,
        )
        periods = compute_period_windows(
            df,
            date_column=date_column,
            train_period=train_period,
            test_period=test_period,
        )
        periods["from_manifest"] = False
    parity_path = paths["config_parity.json"]
    prod_path = paths["config_prod.json"]
    return {
        "version": version,
        "parity_path": parity_path,
        "prod_path": prod_path,
        "cf_path": parity_path,
        "rg_path": parity_path,
        "cf_prod_path": prod_path,
        "rg_prod_path": prod_path,
        "cf_name": MODEL_CF_NAME,
        "rg_name": MODEL_RG_NAME,
        "periods": periods,
        "period_windows_path": str(configs_dir / PERIOD_WINDOWS_JSON),
        "configs_dir": str(configs_dir),
    }

def unwrap_estimator(model: Any) -> Any:
    """CatBoost из wrapper GLMCatboostCombineModel / CatboostModel."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "predict"):
        return inner
    return model

def ensure_predictable_model(model: Any) -> Any:
    """Вернуть объект с ``predict``; без скрытого ``fit()``.

    Если DSM оставил сырой ``CatboostModel`` без ``.predict``, это ошибка
    пайплайна (нужен результат ``fit()`` → ``GLMCatboostCombineModel``), а не
    повод переобучать модель в метриках/экспорте.
    """
    if hasattr(model, "predict"):
        return model
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "predict"):
        return inner
    raise TypeError(
        f"Модель {type(model).__name__} без predict; нужен результат fit() "
        "(GLMCatboostCombineModel / обёртка с .predict), а не сырой pre-fit wrapper. "
        "Скрытый model.fit() отключён (риск переобучения при экспорте/метриках)."
    )
