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

Если parquet ≈ cal и score_second ≈ cal → эталон ок, ищем FE/матрицу.
Если parquet ≈ raw, а service ≈ cal → эталон выгружен без калибратора.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integration.service.mldataworker_shim import install_mldataworker_shim  # noqa: E402

install_mldataworker_shim()

import integration.config as config  # noqa: E402
from integration.scoring.second import (  # noqa: E402
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
from integration.tests.artifacts import ART, load_artifacts  # noqa: E402


#new
def _frame_to_request(df: pd.DataFrame) -> list[dict]:
    drop = [
        c
        for c in ("preds_cf", "preds_rg", "preds_label", "split")
        if c in df.columns
    ]
    part = df.drop(columns=drop, errors="ignore")
    return json.loads(
        part.to_json(orient="records", force_ascii=False, date_format="iso")
    )


#new
def _raw_and_cal_for_chunk(
    stem: str,
    chunk: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    group_path = second_pickle_path(stem)
    import pickle

    with open(group_path, "rb") as f:
        group = pickle.load(f)
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
    print("second_pickle:", art.second_pickle, "exists=", art.second_pickle.is_file())
    print("cf_cal:", art.cf_calibrator, "exists=", art.cf_calibrator.is_file())
    print("df_second:", art.df_for_service_second, "exists=", art.df_for_service_second.is_file())

    df = pd.read_parquet(art.df_for_service_second)
    work = df.head(min(args.max_rows, len(df))).copy()
    print("THRESHOLD unique:", work["THRESHOLD"].dropna().unique()[:10] if "THRESHOLD" in work.columns else None)
    print("preds_cf describe:\n", work["preds_cf"].describe())

    raw, cal = _raw_and_cal_for_chunk(art.second_stem, work)
    out = score_second(art.second_stem, _frame_to_request(work))
    svc = np.asarray(out["result"]["classification_proba"], dtype=float)

    exp = np.round(work["preds_cf"].to_numpy(dtype=float), 2)
    raw_r = np.round(raw, 2)
    cal_r = np.round(cal, 2)
    svc_r = np.round(svc, 2)

    print("\n=== match rates (до 2 знаков) ===")
    print(f"parquet vs raw:     {(exp == raw_r).mean():.2%}")
    print(f"parquet vs cal:     {(exp == cal_r).mean():.2%}")
    print(f"parquet vs service: {(exp == svc_r).mean():.2%}")
    print(f"cal vs service:     {(cal_r == svc_r).mean():.2%}")
    print(f"raw vs service:     {(raw_r == svc_r).mean():.2%}")

    dens = _density_table(pd.Series(exp), "parquet_preds_cf", bins=args.bins)
    dens = dens.merge(
        _density_table(pd.Series(raw), "raw_cf", bins=args.bins), on="bin", how="outer"
    )
    dens = dens.merge(
        _density_table(pd.Series(cal), "cal_cf", bins=args.bins), on="bin", how="outer"
    )
    dens = dens.merge(
        _density_table(pd.Series(svc), "service_cf", bins=args.bins), on="bin", how="outer"
    )
    dens = dens.fillna(0).astype({"parquet_preds_cf": int, "raw_cf": int, "cal_cf": int, "service_cf": int})
    print("\n=== density (0..1 bins) ===")
    print(dens.to_string(index=False))

    # optional plot
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

    # drill-down one index
    idx = args.idx
    if idx in work.index:
        row = work.loc[[idx]]
        pos = list(work.index).index(idx)
    elif 0 <= idx < len(work):
        row = work.iloc[[idx]]
        pos = idx
        idx = work.index[idx]
    else:
        print(f"idx={args.idx} outside sample; skip drill-down")
        return

    print(f"\n=== drill-down idx={idx} (pos={pos}) ===")
    print(
        f"parquet preds_cf={work.loc[idx, 'preds_cf']} "
        f"label={work.loc[idx].get('preds_label')} "
        f"rg={work.loc[idx, 'preds_rg']} "
        f"thr={work.loc[idx].get('THRESHOLD')}"
    )
    print(f"raw={raw[pos]:.4f} cal={cal[pos]:.4f} service={svc[pos]:.4f}")
    one = score_second(art.second_stem, _frame_to_request(row))
    print("score_second one-row:", one["result"]["classification_proba"])


if __name__ == "__main__":
    main()
