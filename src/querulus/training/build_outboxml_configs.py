"""Генерация AllModelsConfig JSON из артефактов train_loop (без ручной правки features[])."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from querulus import PROJECT_ROOT
from querulus.training.catboost_fit import strip_hpo_meta
from querulus.training.config import TrainingConfig
from querulus.training.feature_selection_io import load_feature_selection_latest
from querulus.training.mvp_types import DEFAULT_MVP_INPUT_TYPES
from querulus.training.splits import default_inner_periods_from_train, split_by_date_periods

DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "data" / "processed" / "train_loop_new"
DEFAULT_CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_HPO_PATH = DEFAULT_ARTIFACTS_DIR / "hpo_best_params_new.json"
PROD_FIT_TEST_FRACTION = 0.70
PROD_TAU_CAL_FRACTION = 0.15
PROD_HOLDOUT_FRACTION = 0.15
PROD_CAL_FRACTION = PROD_HOLDOUT_FRACTION  # alias: доля Test_prod (holdout)
_NA_KEY = "N/A"
_OTHER_KEY = "ПРОЧИЕ"
_load_subset_patched = False


def _patch_model_data_subset_load_subset() -> None:
    """Согласовать индексы X/y в OutBoxML при ``data_filter_condition`` (RG parity).

    Без патча ``ModelDataSubset.load_subset`` оставляет ``y_train`` длиннее
    ``X_train`` после query-фильтра severity — DSM fit падает или даёт смещение.
    Патч идемпотентен; правки только в querulus, OutBoxML не меняем.
    """
    global _load_subset_patched
    if _load_subset_patched:
        return
    from outboxml.data_subsets import ModelDataSubset

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


def ensure_legacy_inflation_column(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить алиас ``FE_VALUE_BEFORE_WITHOUT_REAL_2020`` для совместимости с FS-артефактами.

    Отбор severity в train_loop мог быть выполнен до смены базового года CPI;
    без алиаса колонка из ``new_severity_latest.json`` отсутствует в df.
    """
    legacy = "FE_VALUE_BEFORE_WITHOUT_REAL_2020"
    if legacy in df.columns:
        return df
    from querulus.features.inflation import real_feature_name

    current = real_feature_name("VALUE_BEFORE_WITHOUT")
    if current in df.columns:
        out = df.copy()
        out[legacy] = out[current]
        return out
    return df


def dataframe_for_dsm(df: pd.DataFrame) -> pd.DataFrame:
    """Подготовить df для OutBoxML: categorical/string → object.

    OutBoxML ``prepare_categorical`` не допускает присвоение default вне
    ``categories`` pandas.Categorical. DSM дополнительно пишет temp parquet,
    поэтому каст дублируется в ``SafePrepareDataset`` перед prepare.
    """
    out = df.copy()
    for name in out.columns:
        series = out[name]
        dtype = series.dtype
        if isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_categorical_dtype(dtype):
            out[name] = series.astype(object)
        elif pd.api.types.is_string_dtype(dtype) and not pd.api.types.is_object_dtype(dtype):
            # string[pyarrow] / StringDtype — тоже в plain object
            out[name] = series.astype(object)
    return out


def prepare_datasets_from_config(
    config_path: str | Path,
    *,
    check_prepared: bool = True,
) -> dict[str, Any]:
    """PrepareDataset'ы с кастом category→object после temp-parquet DSM."""
    _patch_model_data_subset_load_subset()
    from outboxml.core.prepared_datasets import PrepareDataset
    from outboxml.core.pydantic_models import AllModelsConfig

    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    all_cfg = AllModelsConfig.model_validate(raw)
    group_name = f"{all_cfg.project}_{all_cfg.version}"

    class SafePrepareDataset(PrepareDataset):
        def prepare_dataset(self, data, train_ind, test_ind, target=None):
            if isinstance(data, pd.DataFrame):
                data = dataframe_for_dsm(data)
            return super().prepare_dataset(
                data=data,
                train_ind=train_ind,
                test_ind=test_ind,
                target=target,
            )

    return {
        model.name: SafePrepareDataset(
            model_config=model,
            check_prepared=check_prepared,
            group_name=group_name,
        )
        for model in all_cfg.models_configs
    }


def default_model_version(*, business: str = "2", increment: str = "v1") -> str:
    """Бизнес-версия + дата UTC + инкремент, без слова new."""
    stamp = datetime.now(timezone.utc).strftime("%Y_%m_%d")
    return f"{business}_{stamp}_{increment}"


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


def _level_key(value: Any) -> str:
    """Ключ для replace: outboxml делает .upper() на строках перед матчем."""
    if pd.isna(value):
        return _NA_KEY
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
    if not text:
        return _NA_KEY
    return text.upper()


def _is_binary_level_keys(keys: list[str]) -> bool:
    levels = {key for key in keys if key != _NA_KEY}
    return bool(levels) and levels <= {"0", "1"}


def _categorical_feature_spec(series: pd.Series, name: str) -> dict[str, Any]:
    counts = series.astype("object").where(series.notna(), other=np.nan)
    present = counts.dropna()
    keys: list[str] = []
    seen: set[str] = set()
    for value in present.tolist():
        key = _level_key(value)
        if key not in seen:
            seen.add(key)
            keys.append(key)
    if series.isna().any() and _NA_KEY not in seen:
        keys.append(_NA_KEY)
        seen.add(_NA_KEY)

    if present.empty:
        replace = {_OTHER_KEY: 0, _NA_KEY: 0}
        default = 0
    elif _is_binary_level_keys(keys):
        ordered = sorted(key for key in keys if key != _NA_KEY)
        if _NA_KEY in seen:
            ordered.append(_NA_KEY)
        replace = {key: index for index, key in enumerate(ordered)}
        mode_key = _level_key(present.mode().iloc[0])
        default = int(replace.get(mode_key, 0))
    else:
        # Неизвестный уровень → ПРОЧИЕ (как config_*_3), не мода.
        others = sorted(key for key in keys if key not in {_OTHER_KEY, _NA_KEY})
        replace = {_OTHER_KEY: 0}
        code = 1
        for key in others:
            replace[key] = code
            code += 1
        if _NA_KEY in seen:
            replace[_NA_KEY] = code
        default = 0

    return {
        "name": name,
        "default": default,
        "replace": replace,
        "encoding": "to_int",
        "fillna": default,
    }


def _numeric_feature_spec(series: pd.Series, name: str) -> dict[str, Any]:
    """default=_MEDIAN_: на fit DSM считает медиану по своему train (prod ≠ parity)."""
    _ = series  # уровни NA не запекаем; clip/winsor — DQ-артефакт, не JSON
    return {
        "name": name,
        "default": "_MEDIAN_",
        "replace": {"_TYPE_": "_NUM_"},
        "encoding": "to_float",
    }


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
    return bool(nunique <= 2 and nunique > 0)


def build_features_block(
    df: pd.DataFrame,
    feature_names: list[str],
    *,
    categorical_names: list[str] | None = None,
    mvp_types: dict[str, list[str] | tuple[str, ...]] | None = None,
    fit_index: pd.Index | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """features[] + cat_features_catboost по train-срезу."""
    types = mvp_types or DEFAULT_MVP_INPUT_TYPES
    mvp_cats = set(types.get("CATEGORIAL") or ()) | set(types.get("BINARY") or ())
    json_cats = set(categorical_names or ())
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
            features.append(_numeric_feature_spec(series, name))
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
    extras = list(
        dict.fromkeys(
            [*(extra_columns or []), date_column, column_target]
        )
    )
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
    date_column: str = "PAYMENT_ORDER_DATE_TIME",
    train_period: tuple[str, str] | None = None,
    test_period: tuple[str, str] | None = None,
    freq_target: str = "TARGET_FREQ",
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
        (dates >= pd.Timestamp(prod_train_period[0]))
        & (dates <= pd.Timestamp(prod_train_period[1]))
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


def write_outboxml_configs(
    df: pd.DataFrame,
    *,
    version: str | None = None,
    parquet_path: str | None = None,
    artifacts_dir: Path | str | None = None,
    configs_dir: Path | str | None = None,
    hpo_path: Path | str | None = None,
    date_column: str = "PAYMENT_ORDER_DATE_TIME",
    train_period: tuple[str, str] | None = None,
    test_period: tuple[str, str] | None = None,
    group_name: str = "UU",
) -> dict[str, Any]:
    """Пишет config_cf_{version}.json и config_rg_{version}.json (parity-окна)."""
    version = version or default_model_version()
    artifacts_dir = Path(artifacts_dir) if artifacts_dir else DEFAULT_ARTIFACTS_DIR
    configs_dir = Path(configs_dir) if configs_dir else DEFAULT_CONFIGS_DIR
    parquet_path = parquet_path or str(
        (PROJECT_ROOT / "data" / "processed" / "df_final_3.parquet").as_posix()
    )
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
        df, freq_feats, categorical_names=freq_cats, fit_index=fit_index
    )
    sev_block, sev_cat_out = build_features_block(
        df, sev_feats, categorical_names=sev_cats, fit_index=fit_index
    )
    cf_name = f"querulus_cf_{version}"
    rg_name = f"querulus_rg_{version}"
    cf_cfg = build_all_models_config(
        project=cf_name,
        version="1",
        group_name=group_name,
        parquet_path=parquet_path,
        train_period=periods["parity_train_period"],
        test_period=periods["parity_test_period"],
        date_column=date_column,
        model_name=cf_name,
        column_target="TARGET_FREQ",
        objective="binary",
        features=freq_block,
        cat_features=freq_cat_out,
        params_catboost=catboost_params_from_hpo(
            hpo.get("frequency") if isinstance(hpo.get("frequency"), dict) else hpo,
            classification=True,
        ),
    )
    rg_cfg = build_all_models_config(
        project=rg_name,
        version="1",
        group_name=group_name,
        parquet_path=parquet_path,
        train_period=periods["parity_train_period"],
        test_period=periods["parity_test_period"],
        date_column=date_column,
        model_name=rg_name,
        column_target="TARGET_SEV",
        objective="regression",
        features=sev_block,
        cat_features=sev_cat_out,
        params_catboost=catboost_params_from_hpo(
            hpo.get("severity") if isinstance(hpo.get("severity"), dict) else {},
            classification=False,
        ),
        data_filter_condition="TARGET_SEV > 0",
    )
    cf_path = write_json(configs_dir / f"config_cf_{version}.json", cf_cfg)
    rg_path = write_json(configs_dir / f"config_rg_{version}.json", rg_cfg)
    rg_prod = with_periods(
        rg_cfg,
        train_period=periods["prod_train_period"],
        test_period=periods["prod_test_period"],
    )
    cf_prod = with_periods(
        cf_cfg,
        train_period=periods["prod_train_period"],
        test_period=periods["prod_test_period"],
    )
    cf_prod_path = write_json(configs_dir / f"config_cf_{version}_prod.json", cf_prod)
    rg_prod_path = write_json(configs_dir / f"config_rg_{version}_prod.json", rg_prod)
    return {
        "version": version,
        "cf_path": cf_path,
        "rg_path": rg_path,
        "cf_prod_path": cf_prod_path,
        "rg_prod_path": rg_prod_path,
        "cf_name": cf_name,
        "rg_name": rg_name,
        "periods": periods,
        "n_frequency_features": len(freq_feats),
        "n_severity_features": len(sev_feats),
    }


def unwrap_estimator(model: Any) -> Any:
    """CatBoost из wrapper GLMCatboostCombineModel / CatboostModel."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "predict"):
        return inner
    return model


def ensure_predictable_model(model: Any) -> Any:
    """Если DSM оставил CatboostModel без .predict — добрать возврат fit()."""
    if hasattr(model, "predict"):
        return model
    fit = getattr(model, "fit", None)
    if callable(fit):
        fitted = fit()
        if fitted is not None:
            return fitted
    return model
