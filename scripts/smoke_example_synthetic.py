"""Smoke-run ключевых шагов example.ipynb на синтетике (без Hive/IPython)."""
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
from outboxml.datasets_manager import DataSetsManager  # noqa: E402
from querulus.training.build_outboxml_configs import (  # noqa: E402
    dataframe_for_dsm,
    default_model_version,
    ensure_legacy_inflation_column,
    ensure_predictable_model,
    prepare_datasets_from_config,
    write_outboxml_configs,
)
from querulus.training.outboxml_metrics import (  # noqa: E402
    display_dsm_collect_metrics_cross_test,
    predict_dsm_series,
)
from querulus.training.dsm_fit import fit_dsm_classification  # noqa: E402
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

    df_raw = pd.read_parquet(synthetic_path)
    df_raw = ensure_legacy_inflation_column(df_raw)
    if "FE_VALUE_BEFORE_WITHOUT_REAL_2020" in df_raw.columns:
        df_raw.to_parquet(synthetic_path, index=False)

    df = dataframe_for_dsm(df_raw)
    model_version = default_model_version(business="2", increment="v1")
    built = write_outboxml_configs(
        df,
        version=model_version,
        parquet_path=str(synthetic_path.as_posix()),
    )
    periods = built["periods"]
    cf_name = f"querulus_cf_{model_version}"
    rg_name = f"querulus_rg_{model_version}"

    def patch_dsm(dsm):
        for res in dsm.get_result().values():
            res.model = ensure_predictable_model(res.model)

    print("[1/6] configs OK", built["cf_path"].name, "n=", len(df))

    thr_val = load_collect_val_threshold(PROJECT_ROOT)
    print(f"[2/6] τ (collect) = {thr_val:.2f}")

    dsm_cf = DataSetsManager(
        config_name=str(built["cf_path"]),
        external_config=querulus_outboxml_config,
        prepared_datasets=prepare_datasets_from_config(built["cf_path"]),
    )
    dsm_cf.load_dataset(data=df)
    fit_dsm_classification(dsm_cf, cf_name, threshold=thr_val)
    patch_dsm(dsm_cf)

    dsm_rg = DataSetsManager(
        config_name=str(built["rg_path"]),
        external_config=querulus_outboxml_config,
        prepared_datasets=prepare_datasets_from_config(built["rg_path"]),
    )
    dsm_rg.load_dataset(data=df)
    dsm_rg.fit_models()
    patch_dsm(dsm_rg)
    print("[3/6] parity fit OK")

    test_idx = periods["splits"].test
    test_prod_idx = periods["prod_holdout_idx"]
    test_slices = {"test": test_idx, "test_prod": test_prod_idx}

    for dsm, name, task, thr, ignore_filter in (
        (dsm_cf, cf_name, "classification", thr_val, False),
        (dsm_rg, rg_name, "regression", None, False),
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
        for col in ("train", "test", "test_prod"):
            if col not in table.columns:
                raise SystemExit(f"FAIL {name}: нет колонки {col}")
            non_null = table[col].notna().sum()
            if non_null == 0:
                raise SystemExit(f"FAIL {name}: колонка {col} пустая")
            print(f"  {name} {col}: {non_null}/{len(table)} метрик заполнено")
    print("[4/6] cross-test metrics OK")

    proba_test = predict_dsm_series(
        dsm_cf, cf_name, df.loc[test_idx], task_type="classification"
    )
    sev_test = predict_dsm_series(
        dsm_rg, rg_name, df.loc[test_idx], task_type="regression", ignore_row_filter=True
    )
    cfg = resolve_fin_effect_config(
        df, frequency_target="TARGET_FREQ", severity_target="TARGET_SEV"
    )
    common = (
        test_idx.intersection(proba_test.dropna().index)
        .intersection(sev_test.dropna().index)
        .intersection(df.index)
    )
    fe = run_fin_effect_pipeline(
        df.loc[common],
        proba_test.reindex(common),
        sev_test.reindex(common),
        df.loc[common, "TARGET_FREQ"],
        threshold=thr_val,
        config=cfg,
    )
    print(f"[5/6] fin-effect Test: net={fe.net_effect:,.0f}, n={len(common)}")

    dsm_cf_prod = DataSetsManager(
        config_name=str(built["cf_prod_path"]),
        external_config=querulus_outboxml_config,
        prepared_datasets=prepare_datasets_from_config(built["cf_prod_path"]),
    )
    dsm_cf_prod.load_dataset(data=df)
    fit_dsm_classification(dsm_cf_prod, cf_name, threshold=thr_val)
    patch_dsm(dsm_cf_prod)

    dsm_rg_prod = DataSetsManager(
        config_name=str(built["rg_prod_path"]),
        external_config=querulus_outboxml_config,
        prepared_datasets=prepare_datasets_from_config(built["rg_prod_path"]),
    )
    dsm_rg_prod.load_dataset(data=df)
    dsm_rg_prod.fit_models()
    patch_dsm(dsm_rg_prod)
    print("[6/6] prod-refit fit OK")

    html_path = export_business_html(
        fe,
        cfg,
        path=PROJECT_ROOT / "notebooks" / "fin_effect_detailed.html",
        subtitle="smoke synthetic, Test",
    )
    print(f"HTML: {html_path}")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
