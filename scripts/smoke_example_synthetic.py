"""Smoke-run ключевых шагов example на синтетике (без Hive/IPython)."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
OUTBOXML_ROOT = PROJECT_ROOT.parent.parent
for p in (SRC, OUTBOXML_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402

from configs import config as querulus_outboxml_config  # noqa: E402
from querulus.training.build_outboxml_configs import write_outboxml_configs  # noqa: E402
from querulus.naming import MODEL_CF_NAME, MODEL_RG_NAME, MODEL_VERSION  # noqa: E402
from querulus.training.automl_fit import fit_automl_bundle  # noqa: E402
from querulus.training.outboxml_metrics import (  # noqa: E402
    display_dsm_collect_metrics_cross_test,
    predict_dsm_series,
)
from querulus.fin_effect import (  # noqa: E402
    export_business_html,
    load_collect_val_threshold,
    resolve_fin_effect_config,
    run_fin_effect_pipeline,
)


def main() -> None:
    synthetic_path = PROJECT_ROOT / "data" / "processed" / "df_final_3_synthetic.parquet"
    if not synthetic_path.is_file():
        raise SystemExit(f"Нет {synthetic_path}; запустите: python -m querulus.synthetic_dataset")

    df = pd.read_parquet(synthetic_path)
    model_version = MODEL_VERSION
    built = write_outboxml_configs(
        df,
        version=model_version,
        parquet_path=str(synthetic_path.as_posix()),
    )
    periods = built["periods"]
    cf_name = built.get("cf_name") or MODEL_CF_NAME
    rg_name = built.get("rg_name") or MODEL_RG_NAME

    print("[1/6] configs OK", built["parity_path"].name, built["prod_path"].name, "n=", len(df))

    thr_val = load_collect_val_threshold(PROJECT_ROOT)
    print(f"[2/6] τ (collect) = {thr_val:.2f}")

    dsm, _ = fit_automl_bundle(
        df,
        built["parity_path"],
        external_config=querulus_outboxml_config,
        cf_name=cf_name,
        threshold=thr_val,
        send_mail=False,
        log_mlflow=False,
    )
    print("[3/6] parity fit OK", list(dsm.get_result()))

    test_idx = periods["splits"].test
    test_prod_idx = periods["prod_holdout_idx"]
    test_slices = {"test": test_idx, "test_prod": test_prod_idx}

    for name, task, thr, ignore_filter in (
        (cf_name, "classification", thr_val, False),
        (rg_name, "regression", None, False),
    ):
        table = display_dsm_collect_metrics_cross_test(
            dsm,
            name,
            df,
            task_type=task,
            val_threshold=thr,
            test_slices=test_slices,
            title=f"{name}: train / test / test_prod",
            ignore_row_filter=ignore_filter,
        )
        print(table)

    print("[4/6] metrics OK")

    proba = predict_dsm_series(
        dsm, cf_name, df.loc[test_idx], task_type="classification"
    )
    sev = predict_dsm_series(
        dsm,
        rg_name,
        df.loc[test_idx],
        task_type="regression",
        ignore_row_filter=True,
    )
    fe_cfg = resolve_fin_effect_config()
    fe = run_fin_effect_pipeline(
        df.loc[test_idx],
        proba,
        sev,
        threshold=thr_val,
        config=fe_cfg,
    )
    print("[5/6] fin-effect OK", "best_thr≈", round(float(fe.best_threshold), 2))

    dsm_prod, _ = fit_automl_bundle(
        df,
        built["prod_path"],
        external_config=querulus_outboxml_config,
        cf_name=cf_name,
        threshold=thr_val,
        send_mail=False,
        log_mlflow=False,
    )
    print("[6/6] prod fit OK", list(dsm_prod.get_result()))
    _ = export_business_html
    print("smoke OK")


if __name__ == "__main__":
    main()
