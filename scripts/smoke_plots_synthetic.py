"""Smoke: prod-fit на синтетике → FactorsPlot/cohort zip (для ручного просмотра)."""
from __future__ import annotations

import sys
import warnings
import zipfile
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
from querulus.naming import (  # noqa: E402
    MODEL_CF_NAME,
    MODEL_RG_NAME,
    MODEL_VERSION,
    artifacts_dir_for_version,
)
from querulus.synthetic_dataset import write_synthetic_final_dataset  # noqa: E402
from querulus.training.automl_fit import fit_automl_bundle  # noqa: E402
from querulus.training.build_outboxml_configs import write_outboxml_configs  # noqa: E402
from querulus.training.calibration_compare import compare_cf_calibrations  # noqa: E402
from querulus.training.example_pipeline import (  # noqa: E402
    ExampleDatasetBundle,
    ExampleDsmBundle,
    run_prod_plots_and_email,
)
from querulus.training.factors_export import dsm_feature_lists  # noqa: E402


def main() -> None:
    # Больше строк → prod_tau_cal ≥30 для Querulus CF-калибровки.
    synthetic_path = write_synthetic_final_dataset(n_rows=1500, seed=42)
    df = pd.read_parquet(synthetic_path)
    print("synthetic", synthetic_path, "shape", df.shape)

    built = write_outboxml_configs(
        df,
        version=MODEL_VERSION,
        parquet_path=str(synthetic_path.as_posix()),
    )
    periods = built["periods"]
    cf_name = built.get("cf_name") or MODEL_CF_NAME
    rg_name = built.get("rg_name") or MODEL_RG_NAME

    dsm_prod, _ = fit_automl_bundle(
        df,
        built["prod_path"],
        external_config=querulus_outboxml_config,
        cf_name=cf_name,
        threshold=0.5,
        send_mail=False,
        log_mlflow=False,
    )
    print("prod fit OK", list(dsm_prod.get_result()))

    bundle = ExampleDatasetBundle(
        df=df,
        dataset_source="synthetic",
        dataset_path=synthetic_path,
        built=built,
        periods=periods,
        model_version=MODEL_VERSION,
        cf_name=cf_name,
        rg_name=rg_name,
    )
    models = ExampleDsmBundle(
        dsm_cf=None,
        dsm_rg=None,
        dsm_cf_prod=dsm_prod,
        dsm_rg_prod=dsm_prod,
    )

    cal_compare = compare_cf_calibrations(
        models.dsm_cf_prod,
        model_name=cf_name,
        df=df,
        periods=periods,
        prefer_prod_tau_cal=True,
        method="isotonic",
        balance_ours=True,
    )
    print("cf cal n=", cal_compare.cal_n)

    results_dir = artifacts_dir_for_version(
        MODEL_VERSION, results_root=PROJECT_ROOT / "integration" / "results"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    out = run_prod_plots_and_email(
        models,
        bundle,
        external_config=querulus_outboxml_config,
        send_email=False,
        results_dir=results_dir,
        save_plots_zip=True,
        cf_calibrator=cal_compare.ours_calibrator,
        cf_calibration_label="querulus_cal",
        fit_rg_isotonic=True,
        rg_isotonic_min_samples=20,
        show_figures=False,
        plots_tag="synthetic_smoke",
    )
    print("SMOKE PLOTS OK")
    print("zip:", out.zip_path)
    nums, cats = dsm_feature_lists(dsm_prod, cf_name)
    print("cf features", len(nums) + len(cats), "num", len(nums), "cat", len(cats))
    if out.zip_path is not None:
        extract_dir = results_dir / "synthetic_smoke_plots"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out.zip_path, "r") as zf:
            zf.extractall(extract_dir)
        print("extracted to", extract_dir)
        for cohort in sorted(extract_dir.glob("*cohort*.html")):
            print(" cohort:", cohort.name, "bytes", cohort.stat().st_size)


if __name__ == "__main__":
    main()
