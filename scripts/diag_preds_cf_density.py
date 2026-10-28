"""Плотность preds_cf: сырой CF vs cal vs эталон в parquet vs score_second.

Запуск из корня querulus (в контуре — на prod results):

  python scripts/diag_preds_cf_density.py
  python scripts/diag_preds_cf_density.py --results-dir integration/results
  python scripts/diag_preds_cf_density.py --idx 45

Сравнивает:
  - parquet ``preds_cf`` (эталон parity);
  - raw CF из pickle (без калибратора);
  - cal CF через ``apply_cf_calibrator``;
  - ответ ``score_second`` (как в тесте).

Drill-down ``--idx``: scores + FE_* parquet vs service-preprocess +
prepared matrix (dtype/значения) + порядок колонок калибратора.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integration.service.mldataworker_shim import install_mldataworker_shim  # noqa: E402

install_mldataworker_shim()

import integration.config as config  # noqa: E402
from integration.scoring.second import (  # noqa: E402
    _align_dtypes_for_catboost,
    _cf_predict_column_order,
    _is_isotonic_1d,
    _prepare_matrix,
    _raw_classification_scores,
    apply_cf_calibrator,
    preprocess_second_vector,
    score_second,
)
from integration.scoring.profiles import (  # noqa: E402
    feature_names_from_model_result,
    load_optional_pickle,
    second_cf_calibrator_path,
    second_pickle_path,
)
from integration.tests.artifacts import load_artifacts  # noqa: E402


_DROP_META = ("preds_cf", "preds_rg", "preds_label", "split")


#new
def _frame_to_request(df: pd.DataFrame) -> list[dict]:
    drop = [c for c in _DROP_META if c in df.columns]
    part = df.drop(columns=drop, errors="ignore")
    return json.loads(
        part.to_json(orient="records", force_ascii=False, date_format="iso")
    )


#new
def _load_second_group(stem: str) -> list[dict]:
    path = second_pickle_path(stem)
    with open(path, "rb") as f:
        return pickle.load(f)


#new
def _raw_and_cal_for_chunk(
    stem: str,
    chunk: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    group = _load_second_group(stem)
    clf_model = group[0]
    feats = feature_names_from_model_result(clf_model)
    req = _frame_to_request(chunk)
    df_common = preprocess_second_vector(req)
    clf_df = df_common.loc[:, feats]
    model_c, t_c = _prepare_matrix(clf_model, clf_df)
    raw = np.asarray(_raw_classification_scores(model_c, t_c), dtype=float)
    cf_cal = load_optional_pickle(second_cf_calibrator_path())
    cal = np.asarray(apply_cf_calibrator(cf_cal, list(raw), t_c), dtype=float)
    return raw, cal


#new
def _density_table(series: pd.Series, name: str, bins: int = 20) -> pd.DataFrame:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return pd.DataFrame({"bin": [], name: []})
    counts, edges = np.histogram(s.to_numpy(dtype=float), bins=bins, range=(0.0, 1.0))
    labels = [f"{edges[i]:.2f}-{edges[i + 1]:.2f}" for i in range(len(counts))]
    return pd.DataFrame({"bin": labels, name: counts})


#new
def _fmt_cell(v: Any) -> str:
    if v is None:
        return "None"
    try:
        if pd.isna(v):
            return "NaN"
    except (ValueError, TypeError):
        pass
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6g}"
    return repr(v) if isinstance(v, str) else str(v)


# Источники производных фич модели (что смотреть, если сама фича разъехалась).
_FEATURE_SOURCES: dict[str, tuple[str, ...]] = {
    "FE_DAYS_EVENT_TO_LOSS": ("LOSS_DATE_TIME", "EVENT_DATE"),
    "FE_DAYS_TO_PH_CONTRACT_END": ("POLICYHOLDER_CONTRACT_END_DATE", "EVENT_DATE"),
    "FE_VALUE_BEFORE_WITH_REAL_2022": ("VALUE_BEFORE_WITH", "PAYMENT_ORDER_DATE_TIME"),
    "FE_VALUE_BEFORE_WITHOUT_REAL_2022": ("VALUE_BEFORE_WITHOUT", "PAYMENT_ORDER_DATE_TIME"),
    "FE_VALUE_BEFORE_RATIO": ("VALUE_BEFORE_WITH", "VALUE_BEFORE_WITHOUT"),
    "FE_VALUE_BEFORE_DIFF": (
        "VALUE_BEFORE_WITH",
        "VALUE_BEFORE_WITHOUT",
        "PAYMENT_ORDER_DATE_TIME",
    ),
    "FE_SHARE_WORK_TIER": ("SHARE_WORK", "WORKS", "WORK", "VALUE_BEFORE_WITHOUT"),
    "FE_VICTIM_POWER_PER_TON": ("VICTIM_CAPACITY_ENGINE", "VICTIM_MAX_WEIGHT"),
    "VICTIM_OBJECT_AGE_MONTHS": ("VICTIM_OBJECT_YEAR", "EVENT_DATE"),
}


#new
def _values_close(a: Any, b: Any) -> bool:
    try:
        if pd.isna(a) and pd.isna(b):
            return True
    except (ValueError, TypeError):
        pass
    if isinstance(a, str) and isinstance(b, str):
        return a.casefold() == b.casefold()
    try:
        fa, fb = float(a), float(b)
        if np.isfinite(fa) and np.isfinite(fb):
            return abs(fa - fb) <= 1e-6 * max(1.0, abs(fa), abs(fb)) or abs(fa - fb) < 1e-9
    except (TypeError, ValueError):
        pass
    return str(a) == str(b)


#new
def _resolve_row(df: pd.DataFrame, idx_arg: int) -> tuple[Any, pd.DataFrame]:
    """Вернуть (index_label, one-row DataFrame) из полного parquet."""
    if idx_arg in df.index:
        return idx_arg, df.loc[[idx_arg]]
    if 0 <= idx_arg < len(df):
        label = df.index[idx_arg]
        return label, df.iloc[[idx_arg]]
    raise KeyError(
        f"idx={idx_arg} нет ни в index, ни как iloc (len={len(df)})"
    )


#new
def _print_drilldown(stem: str, row: pd.DataFrame, idx_label: Any) -> None:
    """Разбор одной строки: scores / фичи модели / prepared matrix / cal order."""
    group = _load_second_group(stem)
    clf_model = group[0]
    rg_model = group[1]
    feats = feature_names_from_model_result(clf_model)
    rg_feats = feature_names_from_model_result(rg_model)
    cf_cal = load_optional_pickle(second_cf_calibrator_path())

    parquet_s = row.iloc[0]
    thr = float(parquet_s["THRESHOLD"]) if "THRESHOLD" in row.columns else float("nan")
    exp_cf = float(parquet_s["preds_cf"])
    exp_label = parquet_s.get("preds_label")
    exp_rg = parquet_s.get("preds_rg")

    req = _frame_to_request(row)
    df_pre = preprocess_second_vector(req)
    pre_s = df_pre.iloc[0]

    clf_df = df_pre.loc[:, feats]
    model_c, t_c = _prepare_matrix(clf_model, clf_df)
    raw = float(_raw_classification_scores(model_c, t_c)[0])
    cal = float(apply_cf_calibrator(cf_cal, [raw], t_c)[0])
    one = score_second(stem, req)
    svc_cf = float(one["result"]["classification_proba"][0])
    svc_label = float(one["result"]["classification_predictions"][0])
    svc_rg = float(one["result"]["regression_predictions"][0])
    svc_thr = float(one["result"]["threshold"])

    print(f"\n{'=' * 72}")
    print(f"=== DRILL-DOWN idx={idx_label!r} stem={stem} ===")
    print(f"{'=' * 72}")

    print("\n--- scores ---")
    print(
        f"{'source':<18} {'cf':>10} {'cf@2dp':>8} {'label':>6} {'rg':>14} {'thr':>6}"
    )
    print(
        f"{'parquet':<18} {exp_cf:10.6f} {round(exp_cf, 2):8.2f} "
        f"{_fmt_cell(exp_label):>6} {_fmt_cell(exp_rg):>14} {thr:6.2f}"
    )
    print(
        f"{'service raw':<18} {raw:10.6f} {round(raw, 2):8.2f} "
        f"{'—':>6} {'—':>14} {'—':>6}"
    )
    print(
        f"{'service cal':<18} {cal:10.6f} {round(cal, 2):8.2f} "
        f"{'—':>6} {'—':>14} {'—':>6}"
    )
    print(
        f"{'score_second':<18} {svc_cf:10.6f} {round(svc_cf, 2):8.2f} "
        f"{svc_label:6.0f} {svc_rg:14.4f} {svc_thr:6.2f}"
    )
    print(
        f"Δ parquet−service(cf) = {exp_cf - svc_cf:+.6f} "
        f"(rounded {round(exp_cf, 2) - round(svc_cf, 2):+.2f})"
    )
    exp_lab_from_thr = 1.0 if exp_cf >= thr else 0.0
    svc_lab_from_thr = 1.0 if svc_cf >= thr else 0.0
    print(
        f"label @ thr={thr:.2f}: parquet_cf→{exp_lab_from_thr:.0f} "
        f"service_cf→{svc_lab_from_thr:.0f} "
        f"(preds_label={_fmt_cell(exp_label)}, service_label={svc_label:.0f})"
    )

    print("\n--- calibrator ---")
    if cf_cal is None:
        print("cf_calibrator: MISSING")
    else:
        print(f"type: {type(cf_cal).__name__}")
        print(f"isotonic_1d: {_is_isotonic_1d(cf_cal)}")
        print(f"has predict_proba: {hasattr(cf_cal, 'predict_proba')}")
        order = None
        try:
            order = _cf_predict_column_order(cf_cal, t_c)
        except Exception as exc:  # noqa: BLE001
            print(f"column order error: {exc}")
        sklearn_in = getattr(cf_cal, "feature_names_in_", None)
        if sklearn_in is not None:
            print(f"feature_names_in_ ({len(sklearn_in)}): {list(sklearn_in)[:20]}...")
        if order is not None:
            print(f"cal predict order ({len(order)}): {order}")
            if list(t_c.columns) != order:
                print(
                    "WARN: prepared matrix column order != cal order "
                    f"(prepared={list(t_c.columns)[:8]}...)"
                )

    id_cols = [
        c
        for c in (
            "LOSS_NUMBER",
            "CLAIM_ID",
            "IS_LAWYER",
            "EVENT_DATE",
            "LOSS_DATE_TIME",
            "PAYMENT_ORDER_DATE_TIME",
        )
        if c in row.columns
    ]
    if id_cols:
        print("\n--- identity / key fields (parquet) ---")
        for c in id_cols:
            print(f"  {c}: {_fmt_cell(parquet_s[c])}")

    def _clip_note(model_result: dict, name: str) -> str:
        cfg = model_result.get("model_config") if isinstance(model_result, dict) else None
        if hasattr(cfg, "model_dump"):
            cfg = cfg.model_dump()
        if not isinstance(cfg, dict):
            return ""
        for feat in cfg.get("features") or []:
            if not isinstance(feat, dict):
                if hasattr(feat, "model_dump"):
                    feat = feat.model_dump()
                else:
                    continue
            if feat.get("name") != name:
                continue
            clip = feat.get("clip") or {}
            if not clip:
                return ""
            return f" clip[{clip.get('min_value')}, {clip.get('max_value')}]"
        return ""

    model_feats: list[tuple[str, str, dict]] = (
        [("CF", name, clf_model) for name in feats]
        + [("RG", name, rg_model) for name in rg_feats if name not in feats]
    )
    print("\n--- model features: parquet vs service preprocess ---")
    print("строки UPPER/lower считаются равными")
    print(
        f"{'lane':<4} {'column':<40} {'parquet':<22} {'service_pre':<22} {'same':<5}"
    )
    n_diff = 0
    for lane, name, model_result in model_feats:
        in_p = name in row.columns
        pv = parquet_s[name] if in_p else "«нет»"
        sv = pre_s[name] if name in df_pre.columns else "«нет»"
        same = "yes" if in_p and name in df_pre.columns and _values_close(pv, sv) else "NO"
        if same == "NO":
            n_diff += 1
        mark = "" if same == "yes" else _clip_note(model_result, name)
        print(
            f"{lane:<4} {name:<40} {_fmt_cell(pv):<22} {_fmt_cell(sv):<22} {same:<5}{mark}"
        )
        for src in _FEATURE_SOURCES.get(name, ()):
            if src not in row.columns and src not in df_pre.columns:
                continue
            sp = parquet_s[src] if src in row.columns else "«нет»"
            ss = pre_s[src] if src in df_pre.columns else "«нет»"
            src_same = "yes" if _values_close(sp, ss) else "NO"
            print(
                f"{'':4} {'  ↳ ' + src:<40} {_fmt_cell(sp):<22} {_fmt_cell(ss):<22} {src_same:<5}"
            )
    print(f"summary: model features={len(model_feats)}, diff={n_diff}")

    print("\n--- prepared CF matrix (service prepare_dataset) ---")
    print(f"shape: {t_c.shape}, columns ({len(t_c.columns)}): {list(t_c.columns)}")
    prep = t_c.iloc[0]
    # matrix after cal dtype align (what CCCV actually sees)
    t_aligned = t_c
    if cf_cal is not None and hasattr(cf_cal, "predict_proba") and not _is_isotonic_1d(cf_cal):
        order = _cf_predict_column_order(cf_cal, t_c)
        if order is not None:
            t_aligned = t_c.loc[:, order]
        t_aligned = _align_dtypes_for_catboost(t_aligned, cf_cal)
    aligned = t_aligned.iloc[0]
    print(
        f"{'feature':<42} {'prepared':<18} {'dtype_prep':<12} "
        f"{'aligned':<18} {'dtype_aln':<12}"
    )
    for c in t_aligned.columns:
        print(
            f"{c:<42} {_fmt_cell(prep[c] if c in prep.index else '—'):<18} "
            f"{str(t_c[c].dtype if c in t_c.columns else '—'):<12} "
            f"{_fmt_cell(aligned[c]):<18} {str(t_aligned[c].dtype):<12}"
        )

    # parquet raw model-feature values vs prepared (if present)
    print("\n--- parquet raw feature vs prepared (если колонка была во входе) ---")
    print(f"{'feature':<42} {'parquet_raw':<22} {'prepared':<22} {'same':<5}")
    n_mat_diff = 0
    for c in feats:
        if c not in row.columns:
            continue
        pv = parquet_s[c]
        mv = prep[c] if c in prep.index else "—"
        same = "yes" if _values_close(pv, mv) else "NO"
        if same == "NO":
            n_mat_diff += 1
            print(
                f"{c:<42} {_fmt_cell(pv):<22} {_fmt_cell(mv):<22} {same:<5}"
            )
    if n_mat_diff == 0:
        print("  (нет расхождений среди фич, которые были в parquet до prepare)")
    else:
        print(f"summary: parquet_raw≠prepared for {n_mat_diff} features")

    print("\n--- hypothesis ---")
    if abs(exp_cf - svc_cf) < 0.015:
        print("OK: parquet ≈ service на этой строке.")
    elif abs(exp_cf - raw) < abs(exp_cf - cal):
        print(
            "Похоже parquet ближе к RAW, а сервис отдаёт CAL — "
            "эталон мог быть без калибратора / другой cal."
        )
    elif n_diff > 0:
        print(
            f"Похоже расходятся FE_*/вход до модели ({n_diff} полей): "
            "сервис пересчитывает FE_*, эталон писал preds по другому кадру."
        )
    elif n_mat_diff > 0:
        print(
            "Входные фичи есть, но prepare_dataset меняет значения "
            "(clip/fillna/category collapse) иначе, чем при экспорте эталона."
        )
    else:
        print(
            "FE_* и сырой вход похожи, а cal всё равно другой — "
            "смотри порядок/dtype колонок CCCV (prepared vs aligned) "
            "или эталон считался через DSM prepare_dsm_features."
        )
    print(f"{'=' * 72}\n")


#new
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Каталог артефактов (default: ART.results_dir / INTEGRATION_TEST_*)",
    )
    parser.add_argument("--idx", type=int, default=45, help="Индекс строки для drill-down")
    parser.add_argument("--max-rows", type=int, default=2000, help="Сколько строк для density")
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument(
        "--skip-density",
        action="store_true",
        help="Только drill-down по --idx (быстрее)",
    )
    args = parser.parse_args()

    art = load_artifacts()
    if args.results_dir is not None:
        import os

        os.environ["INTEGRATION_TEST_RESULTS_DIR"] = str(args.results_dir.resolve())
        art = load_artifacts()

    config.prod_models_path = str(art.results_dir.resolve())
    config.shadow_models_subdir = ""
    config.second_cf_calibrator_filename = art.cf_calibrator.name
    config.second_rg_calibrator_filename = art.rg_calibrator.name

    print("results_dir:", art.results_dir.resolve())
    print("second_stem:", art.second_stem)
    print("second_pickle:", art.second_pickle, "exists=", art.second_pickle.is_file())
    print("cf_cal:", art.cf_calibrator, "exists=", art.cf_calibrator.is_file())
    print("df_second:", art.df_for_service_second, "exists=", art.df_for_service_second.is_file())

    df = pd.read_parquet(art.df_for_service_second)
    print("df_second shape:", df.shape)
    print(
        "THRESHOLD unique:",
        df["THRESHOLD"].dropna().unique()[:10] if "THRESHOLD" in df.columns else None,
    )

    if not args.skip_density:
        work = df.head(min(args.max_rows, len(df))).copy()
        print("preds_cf describe (sample):\n", work["preds_cf"].describe())

        raw, cal = _raw_and_cal_for_chunk(art.second_stem, work)
        out = score_second(art.second_stem, _frame_to_request(work))
        svc = np.asarray(out["result"]["classification_proba"], dtype=float)

        exp = np.round(work["preds_cf"].to_numpy(dtype=float), 2)
        raw_r = np.round(raw, 2)
        cal_r = np.round(cal, 2)
        svc_r = np.round(svc, 2)

        print("\n=== match rates на sample (до 2 знаков) ===")
        print(f"parquet vs raw:     {(exp == raw_r).mean():.2%}")
        print(f"parquet vs cal:     {(exp == cal_r).mean():.2%}")
        print(f"parquet vs service: {(exp == svc_r).mean():.2%}")
        print(f"cal vs service:     {(cal_r == svc_r).mean():.2%}")
        print(f"raw vs service:     {(raw_r == svc_r).mean():.2%}")

        dens = _density_table(pd.Series(exp), "parquet_preds_cf", bins=args.bins)
        dens = dens.merge(
            _density_table(pd.Series(raw), "raw_cf", bins=args.bins),
            on="bin",
            how="outer",
        )
        dens = dens.merge(
            _density_table(pd.Series(cal), "cal_cf", bins=args.bins),
            on="bin",
            how="outer",
        )
        dens = dens.merge(
            _density_table(pd.Series(svc), "service_cf", bins=args.bins),
            on="bin",
            how="outer",
        )
        dens = dens.fillna(0).astype(
            {
                "parquet_preds_cf": int,
                "raw_cf": int,
                "cal_cf": int,
                "service_cf": int,
            }
        )
        print("\n=== density (0..1 bins) ===")
        print(dens.to_string(index=False))

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(exp, bins=args.bins, range=(0, 1), alpha=0.45, label="parquet preds_cf")
            ax.hist(raw, bins=args.bins, range=(0, 1), alpha=0.45, label="raw CF")
            ax.hist(cal, bins=args.bins, range=(0, 1), alpha=0.45, label="cal CF")
            ax.hist(svc, bins=args.bins, range=(0, 1), alpha=0.45, label="score_second")
            ax.set_xlabel("P(class=1)")
            ax.set_ylabel("count")
            ax.legend()
            ax.set_title("preds_cf: parquet vs raw vs cal vs service")
            out_png = art.results_dir / "diag_preds_cf_density.png"
            fig.tight_layout()
            fig.savefig(out_png, dpi=120)
            print("plot ->", out_png)
        except Exception as exc:  # noqa: BLE001
            print("plot skipped:", exc)

    try:
        idx_label, row = _resolve_row(df, args.idx)
    except KeyError as exc:
        print(exc)
        return
    _print_drilldown(art.second_stem, row, idx_label)


if __name__ == "__main__":
    main()
