"""E2E smoke: синтетика → fit/export 2.0.0 → legacy pickle → FastAPI → сверка preds.

Запуск из корня querulus (или monorepo):

  python examples/querulus/scripts/smoke_integration_service_synthetic.py

Требует: catboost, outboxml, fastapi/uvicorn. mldataworker подменяется shim'ом.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
OUTBOXML_ROOT = PROJECT_ROOT.parent.parent
INTEGRATION = PROJECT_ROOT / "integration"
for p in (SRC, OUTBOXML_ROOT, PROJECT_ROOT, INTEGRATION):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Shim до импорта integration.main / shadow_v2
from integration.mldataworker_shim import install_mldataworker_shim  # noqa: E402

install_mldataworker_shim()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import uvicorn  # noqa: E402
from catboost import CatBoostClassifier, CatBoostRegressor  # noqa: E402
from multiprocessing import Process  # noqa: E402

from configs import config as querulus_outboxml_config  # noqa: E402
from querulus.features.data_quality import (  # noqa: E402
    apply_dataset_data_quality,
    build_service_dq_bounds,
)
from querulus.fin_effect.threshold_policy import (  # noqa: E402
    save_collect_prod_threshold,
    save_collect_val_threshold,
)
from querulus.naming import MODEL_CF_NAME, MODEL_RG_NAME, MODEL_VERSION  # noqa: E402
from querulus.synthetic_dataset import (  # noqa: E402
    DEFAULT_OUTPUT,
    build_synthetic_final_dataset,
)
from querulus.training.automl_fit import fit_automl_bundle  # noqa: E402
from querulus.training.build_outboxml_configs import write_outboxml_configs  # noqa: E402
from querulus.training.feature_selection_io import save_feature_selection  # noqa: E402
from querulus.training.selected_features import (  # noqa: E402
    DEFAULT_FREQUENCY_FEATURES,
    DEFAULT_SEVERITY_FEATURES,
    PROD_FREQUENCY_FEATURES,
    PROD_SEVERITY_FEATURES,
)
from querulus.training.example_pipeline import (  # noqa: E402
    ExampleDatasetBundle,
    ExampleDsmBundle,
    ExamplePaths,
    ExampleThresholds,
    export_prod_service_artifacts,
)
from querulus.training.outboxml_metrics import predict_dsm_series  # noqa: E402

HOST = "127.0.0.1"
PORT = 18080
LEGACY_STEM = "querulus_ansamble_legacy_synthetic"
SHADOW_STEM = "querulus_ansamble"


def _ensure_service_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "EVENT_DATE" not in out.columns:
        out["EVENT_DATE"] = pd.to_datetime(out["PAYMENT_ORDER_DATE_TIME"]) - pd.to_timedelta(
            out["APPLY_DELAY"].clip(lower=0).fillna(0).astype(int), unit="D"
        )
    if "AMOUNT_REPAIR" not in out.columns:
        out["AMOUNT_REPAIR"] = out["VALUE_BEFORE_WITHOUT"]
    if "IS_LAWYER" not in out.columns:
        out["IS_LAWYER"] = 0
    # Сервис ждёт строки UPPER на входе; обучение OutBoxML тоже upper.
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].map(lambda v: v.upper() if isinstance(v, str) else v)
    return out


def _train_legacy_ensemble(df: pd.DataFrame, train_mask: pd.Series) -> list[dict]:
    """Мини-ансамбль под hardcoded CLASSIFICATION/REGRESSION_FEATURES сервиса.

    После prepare_dataset категории уже int — учим CatBoost без cat_features
    (как числовой вход), иначе на инференсе float-коды ломают Pool.
    """
    clf_cols = list(PROD_FREQUENCY_FEATURES)
    rg_cols = list(PROD_SEVERITY_FEATURES)
    train = df.loc[train_mask].copy()

    def _encode_frame(frame: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[dict]]:
        x = pd.DataFrame(index=frame.index)
        specs: list[dict] = []
        for name in cols:
            s = frame[name]
            if s.dtype == object or str(s.dtype) == "string":
                keys = sorted({str(v).upper() for v in s.dropna().unique()})
                replace = {"ПРОЧИЕ": 0}
                for i, k in enumerate(keys, start=1):
                    replace[k] = i
                mapped = s.map(lambda v: replace.get(str(v).upper(), 0) if pd.notna(v) else 0)
                x[name] = mapped.astype(float)
                specs.append(
                    {
                        "name": name,
                        "default": 0,
                        "replace": replace,
                        "encoding": "to_int",
                        "fillna": 0,
                    }
                )
            else:
                num = pd.to_numeric(s, errors="coerce")
                med = float(num.median())
                if np.isnan(med):
                    med = 0.0
                x[name] = num.fillna(med).astype(float)
                specs.append(
                    {
                        "name": name,
                        "default": med,
                        "replace": {"_TYPE_": "_NUM_"},
                        "encoding": "to_float",
                    }
                )
        return x, specs

    x_clf, clf_specs = _encode_frame(train, clf_cols)
    y_clf = train["TARGET_FREQ"].astype(int)
    sev_mask = train["TARGET_SEV"] > 0
    x_rg, rg_specs = _encode_frame(train.loc[sev_mask], rg_cols)
    y_rg = train.loc[sev_mask, "TARGET_SEV"].astype(float)

    clf = CatBoostClassifier(
        iterations=40,
        depth=4,
        learning_rate=0.1,
        verbose=False,
        allow_writing_files=False,
        random_seed=0,
        auto_class_weights="Balanced",
    )
    clf.fit(x_clf, y_clf)

    rg = CatBoostRegressor(
        iterations=40,
        depth=4,
        learning_rate=0.1,
        verbose=False,
        allow_writing_files=False,
        random_seed=0,
    )
    rg.fit(x_rg, y_rg)

    clf_export = {
        "model_config": {
            "name": "querulus_cf_legacy",
            "objective": "binary",
            "column_target": "TARGET_FREQ",
            "features": clf_specs,
            "cat_features_catboost": [],
            "params_catboost": {},
        },
        "model": clf,
    }
    rg_export = {
        "model_config": {
            "name": "querulus_rg_legacy",
            "objective": "regression",
            "column_target": "TARGET_SEV",
            "features": rg_specs,
            "cat_features_catboost": [],
            "params_catboost": {},
            "data_filter_condition": "TARGET_SEV > 0",
        },
        "model": rg,
    }
    return [clf_export, rg_export]


def _write_dq_bounds(df: pd.DataFrame, out_path: Path) -> None:
    _, report = apply_dataset_data_quality(df)
    bounds = build_service_dq_bounds(report, model_version=MODEL_VERSION)
    out_path.write_text(json.dumps(bounds, ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonable_records(df: pd.DataFrame) -> list[dict]:
    """Строки для JSON POST: даты ISO, без REAL_* (сервис посчитает CPI)."""
    out = df.copy()
    drop_real = [c for c in out.columns if "_REAL_" in c]
    out = out.drop(columns=drop_real, errors="ignore")
    # Timestamp / numpy → JSON
    return json.loads(out.to_json(orient="records", force_ascii=False, date_format="iso"))


def _serve() -> None:
    install_mldataworker_shim()
    # cwd = integration for config.base_path / results
    import os

    os.chdir(INTEGRATION)
    from integration.main import app

    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


def main() -> None:
    import httpx

    print("=== [1] synthetic dataset ===")
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df0 = build_synthetic_final_dataset(n_rows=500, seed=42)
    df = _ensure_service_columns(df0)
    df.to_parquet(DEFAULT_OUTPUT, index=False)
    print("wrote", DEFAULT_OUTPUT, "shape", df.shape)

    print("=== [2] thresholds + FS stubs + outboxml configs ===")
    art = PROJECT_ROOT / "data" / "processed" / "train_loop_new"
    art.mkdir(parents=True, exist_ok=True)
    save_collect_val_threshold(0.35, artifacts_dir=art)
    save_collect_prod_threshold(0.37, artifacts_dir=art)
    freq_cats = [
        "EVENT_CREATED_BY_GIBDD_FLAG",
        "FILIAL",
        "VICTIM_VEHICLE_CATEGORY",
        "APPLICANT_FORM",
        "RECIEVE_METHOD",
    ]
    sev_cats = ["LOSS_UNIT_ZONE", "VICTIM_VEHICLE_COUNTRY"]
    save_feature_selection(
        stack="new",
        task="frequency",
        selected_features=list(DEFAULT_FREQUENCY_FEATURES),
        categorical_features=freq_cats,
        directory=art,
    )
    save_feature_selection(
        stack="new",
        task="severity",
        selected_features=list(DEFAULT_SEVERITY_FEATURES),
        categorical_features=sev_cats,
        directory=art,
    )
    # DQ report for clip bounds
    df_dq, _ = apply_dataset_data_quality(
        df, report_path=PROJECT_ROOT / "data" / "processed" / "data_quality_report.json"
    )
    df = _ensure_service_columns(df_dq)
    df.to_parquet(DEFAULT_OUTPUT, index=False)

    built = write_outboxml_configs(
        df,
        version=MODEL_VERSION,
        parquet_path=str(DEFAULT_OUTPUT.as_posix()),
        artifacts_dir=art,
    )
    periods = built["periods"]
    train_mask = df.index.isin(periods["splits"].train.union(periods["splits"].val))

    print("=== [3] prod fit (2.0.0) + export ===")
    dsm_prod, _ = fit_automl_bundle(
        df,
        built["prod_path"],
        external_config=querulus_outboxml_config,
        cf_name=MODEL_CF_NAME,
        threshold=0.37,
        send_mail=False,
        log_mlflow=False,
    )
    results_dir = INTEGRATION / "results" / "querulus" / MODEL_VERSION
    results_dir.mkdir(parents=True, exist_ok=True)
    paths = ExamplePaths(
        project_root=PROJECT_ROOT,
        results_dir=results_dir,
        local_parquet_path=DEFAULT_OUTPUT,
        prefer_hive=False,
        use_synthetic=True,
    )
    bundle = ExampleDatasetBundle(
        df=df,
        dataset_source="synthetic",
        dataset_path=DEFAULT_OUTPUT,
        built=built,
        periods=periods,
        model_version=MODEL_VERSION,
        cf_name=MODEL_CF_NAME,
        rg_name=MODEL_RG_NAME,
    )
    thresholds = ExampleThresholds(parity=0.35, prod=0.37)
    models = ExampleDsmBundle(
        dsm_cf=dsm_prod,
        dsm_rg=dsm_prod,
        dsm_cf_prod=dsm_prod,
        dsm_rg_prod=dsm_prod,
    )
    export = export_prod_service_artifacts(
        models,
        bundle,
        paths,
        thresholds=thresholds,
    )
    _write_dq_bounds(df, results_dir / "dq_bounds.json")
    print("export", export.meta_path)

    # Эталонные preds на holdout test_prod (как example_final)
    test_prod_idx = periods["prod_holdout_idx"]
    sample_idx = pd.Index(test_prod_idx[: min(20, len(test_prod_idx))])
    ref_cf = predict_dsm_series(
        dsm_prod, MODEL_CF_NAME, df.loc[sample_idx], task_type="classification"
    )
    ref_rg = predict_dsm_series(
        dsm_prod,
        MODEL_RG_NAME,
        df.loc[sample_idx],
        task_type="regression",
        ignore_row_filter=True,
    )

    print("=== [4] legacy ensemble for main_ ===")
    legacy_dir = INTEGRATION / "results"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_group = _train_legacy_ensemble(df, train_mask)
    legacy_path = legacy_dir / f"{LEGACY_STEM}.pickle"
    legacy_path.write_bytes(pickle.dumps(legacy_group))
    print("legacy pickle", legacy_path)

    print("=== [5] start FastAPI ===")
    proc = Process(target=_serve, daemon=True)
    proc.start()
    try:
        base = f"http://{HOST}:{PORT}"
        for _ in range(40):
            try:
                r = httpx.get(f"{base}/api/health", timeout=1.0)
                if r.status_code == 200:
                    print("health", r.json())
                    break
            except Exception:
                time.sleep(0.25)
        else:
            raise SystemExit("сервис не поднялся")

        # Вектор для second_ / сверка с ref
        rows = df.loc[sample_idx].copy()
        payload_rows = _jsonable_records(rows)

        body = {
            "main_model": LEGACY_STEM,
            "main_request": payload_rows,
            "second_model": SHADOW_STEM,
            "second_request": payload_rows,
        }
        resp = httpx.post(f"{base}/api/predict", json=body, timeout=120.0)
        if resp.status_code != 200:
            raise SystemExit(f"predict HTTP {resp.status_code}: {resp.text[:2000]}")
        data = resp.json()
        assert data.get("oisuu_responce"), "нет oisuu_responce"
        assert data.get("main_response"), "нет main_response"
        second = data.get("second_response") or {}
        assert second.get("result"), f"пустой second_response: {second}"

        svc_cf = np.asarray(second["result"]["classification_proba"], dtype=float)
        svc_rg = np.asarray(second["result"]["regression_predictions"], dtype=float)
        # ref_rg на sev>0 фильтруется в metrics; здесь ignore_row_filter=True — все строки.
        # Маска сервиса: class=0 → 0 (если не lawyer). Сравниваем proba напрямую;
        # для rg — только где label=1 или сравниваем с masked ref.
        ref_cf_a = np.round(ref_cf.to_numpy(dtype=float), 2)
        svc_cf_a = np.round(svc_cf, 2)
        if len(ref_cf_a) != len(svc_cf_a):
            raise SystemExit(f"len mismatch cf ref={len(ref_cf_a)} svc={len(svc_cf_a)}")
        max_cf = float(np.max(np.abs(ref_cf_a - svc_cf_a)))
        print(f"max |Δ proba CF| = {max_cf:.4f}")

        labels = np.asarray(second["result"]["classification_predictions"], dtype=float)
        ref_rg_masked = np.where(labels >= 1.0, np.round(ref_rg.to_numpy(dtype=float), 2), 0.0)
        svc_rg_a = np.round(svc_rg, 2)
        max_rg = float(np.max(np.abs(ref_rg_masked - svc_rg_a)))
        print(f"max |Δ RG masked| = {max_rg:.4f}")

        # Допуск: prepare_dataset/clip могут чуть сдвинуть; на синтетике ожидаем близко.
        if max_cf > 0.05:
            raise SystemExit(f"CF расхождение слишком большое: {max_cf}")
        if max_rg > 5000:  # руб.; на малом CatBoost/синтетике допустим запас
            raise SystemExit(f"RG расхождение слишком большое: {max_rg}")

        print("=== SMOKE OK: service + second_ ≈ training preds ===")
        print("thr second", second["result"].get("threshold"))
        print("n rows", len(svc_cf_a))
    finally:
        proc.terminate()
        proc.join(timeout=5)


if __name__ == "__main__":
    main()
