"""Собрать синтетические артефакты для integration/tests (без skip).

Пишет в ``integration/results/test_synth/``:
  - querulus_ansamble_legacy.pickle
  - UU_querulus_2026_10_09_16_51_20.pickle
  - cf_calibrator.pickle, rg_calibrator.pickle
  - metadata.json
  - df_main.parquet, df_second.parquet (split=train|test, preds_cf, preds_rg)

Запуск из корня querulus:

  python scripts/build_integration_test_artifacts.py
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
OUTBOXML_ROOT = PROJECT_ROOT.parent.parent
INTEGRATION = PROJECT_ROOT / "integration"
for p in (SRC, OUTBOXML_ROOT, PROJECT_ROOT, INTEGRATION):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from integration.service.mldataworker_shim import install_mldataworker_shim  # noqa: E402

install_mldataworker_shim()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from catboost import CatBoostClassifier, CatBoostRegressor  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

from integration.features.derived import add_second_model_features  # noqa: E402
from integration.scoring.second import (  # noqa: E402
    apply_cf_calibrator,
    apply_rg_calibrator,
    enrich_dates_and_forms,
    prepare_vector_upper,
)
from integration.service.main import (  # noqa: E402
    CLASSIFICATION_FEATURES,
    REGRESSION_FEATURES,
    _classification_proba_and_labels,
    _prepare_matrix,
    _require_columns,
)
from integration.service.masking import masked_predictions  # noqa: E402
from querulus.synthetic_dataset import build_synthetic_final_dataset  # noqa: E402

OUT_DIR = INTEGRATION / "results" / "test_synth"
MAIN_STEM = "querulus_ansamble_legacy"
SECOND_STEM = "UU_querulus_2026_10_09_16_51_20"
THRESHOLD = 0.40

SECOND_CF_FEATURES = [
    "FILIAL",
    "FE_VALUE_BEFORE_WITH_REAL_2022",
    "EVENT_CREATED_BY_GIBDD_FLAG",
    "RECIEVE_METHOD",
    "FL_PHOTO_VIDEO",
    "APPLICANT_FORM",
    "FE_VALUE_BEFORE_WITHOUT_REAL_2022",
    "FE_DAYS_EVENT_TO_LOSS",
    "PAYMENT_RECIPIENT_TYPE",
    "VICTIM_TS_REGION",
    "LOSS_UNIT_TYPE",
    "FE_VALUE_BEFORE_RATIO",
    "VICTIM_VEHICLE_CATEGORY",
    "VICTIM_OBJECT_AGE_MONTHS",
    "VIC_TS_COUNTRY",
    "APPLICANT_AGE",
    "VICTIM_AGE",
    "NOT_NOTIFICATION",
    "FE_SHARE_WORK_TIER",
    "FE_DAYS_TO_PH_CONTRACT_END",
    "IS_TOTAL",
]
SECOND_RG_FEATURES = [
    "FE_VALUE_BEFORE_WITHOUT_REAL_2022",
    "VICTIM_VEHICLE_CATEGORY",
    "FE_VALUE_BEFORE_DIFF",
    "REGION",
    "GUILTY_MAX_WEIGHT",
    "FE_VALUE_BEFORE_RATIO",
    "EVENT_CREATED_BY_GIBDD_FLAG",
    "FE_VALUE_BEFORE_WITH_REAL_2022",
    "APPLICANT_AGE",
    "VICTIM_VEHICLE_COUNTRY",
    "FL_PHOTO_VIDEO",
    "VICTIM_VEHICLE_AGE",
    "FE_VICTIM_POWER_PER_TON",
]


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "EVENT_DATE" not in out.columns:
        out["EVENT_DATE"] = pd.to_datetime(out["PAYMENT_ORDER_DATE_TIME"]) - pd.to_timedelta(
            out.get("APPLY_DELAY", pd.Series(0, index=out.index))
            .clip(lower=0)
            .fillna(0)
            .astype(int),
            unit="D",
        )
    if "AMOUNT_REPAIR" not in out.columns:
        out["AMOUNT_REPAIR"] = out["VALUE_BEFORE_WITHOUT"]
    if "IS_LAWYER" not in out.columns:
        out["IS_LAWYER"] = (np.arange(len(out)) % 7 == 0).astype(int)
    if "WORKS" not in out.columns:
        out["WORKS"] = pd.to_numeric(out["VALUE_BEFORE_WITHOUT"], errors="coerce") * 0.4
    if "LOSS_DATE_TIME" not in out.columns:
        out["LOSS_DATE_TIME"] = out["PAYMENT_ORDER_DATE_TIME"]
    if "POLICYHOLDER_CONTRACT_END_DATE" not in out.columns:
        out["POLICYHOLDER_CONTRACT_END_DATE"] = pd.to_datetime(
            out["EVENT_DATE"]
        ) + pd.Timedelta(days=180)
    defaults = {
        "FL_PHOTO_VIDEO": 0,
        "PAYMENT_RECIPIENT_TYPE": "ИНОЕ",
        "VICTIM_TS_REGION": "МОСКВА",
        "LOSS_UNIT_TYPE": "Филиал",
        "VIC_TS_COUNTRY": "РОССИЯ",
        "VICTIM_AGE": 40,
        "NOT_NOTIFICATION": 0,
        "IS_TOTAL": 0,
        "REGION": "МОСКВА",
        "GUILTY_MAX_WEIGHT": 1500.0,
        "VICTIM_VEHICLE_COUNTRY": "РОССИЯ",
        "VICTIM_CAPACITY_ENGINE": 100.0,
        "VICTIM_MAX_WEIGHT": 1500.0,
        "VICTIM_OBJECT_YEAR": 2015,
        "LOSS_UNIT_ZONE": "ЗОНА",
        "GUILTY_CAPACITY_ENGINE": 100.0,
        "VICTIM_VEHICLE_AGE": 5,
        "EVENT_CREATED_BY_GIBDD_FLAG": 1,
        "RECIEVE_METHOD": "ПОЧТА",
        "APPLICANT_FORM": "ВЫГОДОПРИОБРЕТАТЕЛЬ",
        "FILIAL": "МОСКВА",
        "VICTIM_VEHICLE_CATEGORY": "ЭКОНОМ",
        "APPLICANT_AGE": 40,
    }
    for col, val in defaults.items():
        if col not in out.columns:
            out[col] = val
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].map(lambda v: v.upper() if isinstance(v, str) else v)
    return out


def _encode_frame(
    frame: pd.DataFrame, cols: list[str]
) -> tuple[pd.DataFrame, list[dict]]:
    x = pd.DataFrame(index=frame.index)
    specs: list[dict] = []
    for name in cols:
        if name not in frame.columns:
            frame = frame.copy()
            frame[name] = np.nan
        s = frame[name]
        if s.dtype == object or str(s.dtype) in ("string", "category"):
            keys = sorted({str(v).upper() for v in s.dropna().unique()})
            replace = {"ПРОЧИЕ": 0}
            for i, k in enumerate(keys, start=1):
                replace[k] = i
            mapped = s.map(
                lambda v: replace.get(str(v).upper(), 0) if pd.notna(v) else 0
            )
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
            med = float(num.median()) if num.notna().any() else 0.0
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


def _train_ensemble(
    df: pd.DataFrame,
    train_idx: pd.Index,
    clf_cols: list[str],
    rg_cols: list[str],
    cf_name: str,
    rg_name: str,
) -> list[dict]:
    train = df.loc[train_idx].copy()
    x_clf, clf_specs = _encode_frame(train, clf_cols)
    y_clf = train["TARGET_FREQ"].astype(int)
    sev_mask = train["TARGET_SEV"] > 0
    x_rg, rg_specs = _encode_frame(train.loc[sev_mask], rg_cols)
    y_rg = train.loc[sev_mask, "TARGET_SEV"].astype(float)

    clf = CatBoostClassifier(
        iterations=30,
        depth=3,
        learning_rate=0.15,
        verbose=False,
        allow_writing_files=False,
        random_seed=0,
        auto_class_weights="Balanced",
    )
    clf.fit(x_clf, y_clf)
    rg = CatBoostRegressor(
        iterations=30,
        depth=3,
        learning_rate=0.15,
        verbose=False,
        allow_writing_files=False,
        random_seed=0,
    )
    rg.fit(x_rg, y_rg)
    return [
        {
            "model_config": {
                "name": cf_name,
                "objective": "binary",
                "column_target": "TARGET_FREQ",
                "features": clf_specs,
                "cat_features_catboost": [],
                "params_catboost": {},
            },
            "model": clf,
        },
        {
            "model_config": {
                "name": rg_name,
                "objective": "regression",
                "column_target": "TARGET_SEV",
                "features": rg_specs,
                "cat_features_catboost": [],
                "params_catboost": {},
                "data_filter_condition": "TARGET_SEV > 0",
            },
            "model": rg,
        },
    ]


def _score_legacy(group: list[dict], df_row: pd.DataFrame, thr: float) -> tuple[float, float]:
    clf_df = _require_columns(df_row, CLASSIFICATION_FEATURES, "clf")
    rg_df = _require_columns(df_row, REGRESSION_FEATURES, "rg")
    model_c, t_c = _prepare_matrix(group[0], clf_df)
    proba, labels = _classification_proba_and_labels(model_c, t_c, thr)
    model_r, t_r = _prepare_matrix(group[1], rg_df)
    raw_reg = [float(x) for x in np.asarray(model_r.predict(t_r)).ravel()]
    masked = masked_predictions(labels, raw_reg, df_row)
    return float(proba[0]), float(masked[0])


def _score_second_ref(
    group: list[dict],
    df_fe: pd.DataFrame,
    thr: float,
    cf_cal,
    rg_cal,
) -> tuple[float, float]:
    from integration.scoring.profiles import feature_names_from_model_result
    from integration.scoring.second import _raw_classification_scores

    clf_feats = feature_names_from_model_result(group[0])
    rg_feats = feature_names_from_model_result(group[1])
    clf_df = df_fe.loc[:, clf_feats]
    rg_df = df_fe.loc[:, rg_feats]
    model_c, t_c = _prepare_matrix(group[0], clf_df)
    raw_p = _raw_classification_scores(model_c, t_c)
    proba = apply_cf_calibrator(cf_cal, raw_p, t_c)
    labels = [1.0 if s >= thr else 0.0 for s in proba]
    model_r, t_r = _prepare_matrix(group[1], rg_df)
    raw_reg = [float(x) for x in np.asarray(model_r.predict(t_r)).ravel()]
    cal_reg = apply_rg_calibrator(rg_cal, raw_reg)
    masked = masked_predictions(labels, cal_reg, df_fe)
    return float(proba[0]), float(masked[0])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== synthetic df ===")
    df0 = build_synthetic_final_dataset(n_rows=200, seed=7)
    df = _ensure_columns(df0)
    n = len(df)
    split = np.where(np.arange(n) < int(n * 0.7), "train", "test")
    df["split"] = split
    train_idx = df.index[df["split"] == "train"]

    print("=== legacy main_ ensemble ===")
    # EVENT_YEAR / APPLY_DELAY for legacy
    event_dt = pd.to_datetime(df["EVENT_DATE"], errors="coerce")
    loss_dt = pd.to_datetime(df["PAYMENT_ORDER_DATE_TIME"], errors="coerce")
    df["EVENT_YEAR"] = event_dt.dt.year
    df["APPLY_DELAY"] = (
        loss_dt.dt.normalize() - event_dt.dt.normalize()
    ).dt.days.fillna(-1).astype(int)

    legacy = _train_ensemble(
        df,
        train_idx,
        CLASSIFICATION_FEATURES,
        REGRESSION_FEATURES,
        "querulus_cf_legacy",
        "querulus_rg_legacy",
    )
    (OUT_DIR / f"{MAIN_STEM}.pickle").write_bytes(pickle.dumps(legacy))

    print("=== second FE + ensemble ===")
    records = json.loads(
        df.drop(columns=["split"], errors="ignore").to_json(
            orient="records", force_ascii=False, date_format="iso"
        )
    )
    df_fe = prepare_vector_upper(records)
    df_fe = enrich_dates_and_forms(df_fe)
    df_fe = add_second_model_features(df_fe)
    df_fe["TARGET_FREQ"] = df["TARGET_FREQ"].to_numpy()
    df_fe["TARGET_SEV"] = df["TARGET_SEV"].to_numpy()
    df_fe["IS_LAWYER"] = df["IS_LAWYER"].to_numpy()
    df_fe["split"] = df["split"].to_numpy()

    second = _train_ensemble(
        df_fe,
        train_idx,
        SECOND_CF_FEATURES,
        SECOND_RG_FEATURES,
        "querulus_cf",
        "querulus_rg",
    )
    (OUT_DIR / f"{SECOND_STEM}.pickle").write_bytes(pickle.dumps(second))

    print("=== calibrators (isotonic on train raw scores) ===")
    raw_cf_train: list[float] = []
    y_cf_train: list[int] = []
    raw_rg_train: list[float] = []
    y_rg_train: list[float] = []
    for i in train_idx:
        row = df_fe.loc[[i]]
        clf_df = row.loc[:, SECOND_CF_FEATURES]
        model_c, t_c = _prepare_matrix(second[0], clf_df)
        from integration.scoring.second import _raw_classification_scores

        s = _raw_classification_scores(model_c, t_c)[0]
        raw_cf_train.append(s)
        y_cf_train.append(int(row["TARGET_FREQ"].iloc[0]))
        if float(row["TARGET_SEV"].iloc[0]) > 0:
            rg_df = row.loc[:, SECOND_RG_FEATURES]
            model_r, t_r = _prepare_matrix(second[1], rg_df)
            r = float(np.asarray(model_r.predict(t_r)).ravel()[0])
            raw_rg_train.append(r)
            y_rg_train.append(float(row["TARGET_SEV"].iloc[0]))

    cf_cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    cf_cal.fit(np.asarray(raw_cf_train), np.asarray(y_cf_train))
    rg_cal = IsotonicRegression(y_min=0.0, out_of_bounds="clip")
    rg_cal.fit(np.asarray(raw_rg_train), np.asarray(y_rg_train))
    with open(OUT_DIR / "cf_calibrator.pickle", "wb") as f:
        pickle.dump(cf_cal, f)
    with open(OUT_DIR / "rg_calibrator.pickle", "wb") as f:
        pickle.dump(rg_cal, f)

    # Also copy calibrators next to default results for service path override in tests
    meta = {
        "best_threshold": THRESHOLD,
        "threshold_scale": "calibrated",
        "threshold_source": "synthetic_test",
        "calibration": {
            "cf": str(OUT_DIR / "cf_calibrator.pickle"),
            "rg": str(OUT_DIR / "rg_calibrator.pickle"),
        },
    }
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=== reference preds train+test ===")
    main_rows = []
    second_rows = []
    for i in df.index:
        legacy_row = df.loc[[i]]
        p_m, r_m = _score_legacy(legacy, legacy_row, THRESHOLD)
        main_rows.append(
            {
                **{c: legacy_row.iloc[0][c] for c in legacy_row.columns if c != "split"},
                "split": legacy_row.iloc[0]["split"],
                "preds_cf": round(p_m, 2),
                "preds_rg": round(r_m, 2),
            }
        )
        fe_row = df_fe.loc[[i]]
        p_s, r_s = _score_second_ref(second, fe_row, THRESHOLD, cf_cal, rg_cal)
        second_rows.append(
            {
                **{
                    c: (
                        fe_row.iloc[0][c].isoformat()
                        if hasattr(fe_row.iloc[0][c], "isoformat")
                        else fe_row.iloc[0][c]
                    )
                    for c in fe_row.columns
                    if c not in {"split", "TARGET_FREQ", "TARGET_SEV"}
                    and not str(c).startswith("FE_")
                    and c != "VICTIM_OBJECT_AGE_MONTHS"
                    and c != "SHARE_WORK"
                },
                "EVENT_DATE": pd.Timestamp(fe_row.iloc[0]["EVENT_DATE"]).isoformat(),
                "PAYMENT_ORDER_DATE_TIME": pd.Timestamp(
                    fe_row.iloc[0]["PAYMENT_ORDER_DATE_TIME"]
                ).isoformat(),
                "LOSS_DATE_TIME": pd.Timestamp(fe_row.iloc[0]["LOSS_DATE_TIME"]).isoformat(),
                "POLICYHOLDER_CONTRACT_END_DATE": pd.Timestamp(
                    fe_row.iloc[0]["POLICYHOLDER_CONTRACT_END_DATE"]
                ).isoformat(),
                "WORKS": float(fe_row.iloc[0]["WORKS"]),
                "VALUE_BEFORE_WITH": float(fe_row.iloc[0]["VALUE_BEFORE_WITH"]),
                "VALUE_BEFORE_WITHOUT": float(fe_row.iloc[0]["VALUE_BEFORE_WITHOUT"]),
                "IS_LAWYER": int(fe_row.iloc[0]["IS_LAWYER"]),
                "split": str(fe_row.iloc[0]["split"]),
                "preds_cf": round(p_s, 2),
                "preds_rg": round(r_s, 2),
            }
        )

    pd.DataFrame(main_rows).to_parquet(OUT_DIR / "df_main.parquet", index=False)
    pd.DataFrame(second_rows).to_parquet(OUT_DIR / "df_second.parquet", index=False)
    print("OK ->", OUT_DIR)


if __name__ == "__main__":
    main()
