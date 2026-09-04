"""Блок B collect: train_loop_new + fit → val_threshold_latest.json."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from querulus.dataset.steps.targets import ensure_claims_targets
from querulus.features.data_quality import apply_dataset_data_quality
from querulus.features.integer_casts import cast_integer_like_columns
from querulus.training import TrainLoopFlags, run_train_loop_new
from querulus.fin_effect.threshold_policy import collect_val_threshold_path
from querulus.training.config import TrainingConfig


def main() -> None:
    parquet = PROJECT_ROOT / "data" / "processed" / "df_final_3_synthetic.parquet"
    if not parquet.is_file():
        raise SystemExit(f"Нет {parquet}; сначала: python -m querulus.synthetic_dataset")

    df = pd.read_parquet(parquet)
    df = ensure_claims_targets(df)
    df = cast_integer_like_columns(df)
    df, _ = apply_dataset_data_quality(
        df,
        report_path=PROJECT_ROOT / "data" / "processed" / "data_quality_report.json",
    )
    print(f"df.shape={df.shape}")

    flags = TrainLoopFlags(
        use_fe_features=True,
        run_psi_filter=False,
        run_shap_select=False,
        run_noise_cut=False,
        run_backward_elim=False,
        run_fit=True,
        run_hpo=False,
        run_calibration=False,
        use_mlflow=False,
    )
    loop_cfg = TrainingConfig(
        features_source="selected",
        frequency_select_features=False,
        severity_select_features=False,
        use_fe_features=True,
        use_train_val_test_split=True,
    )
    print("[B] >>> train_loop_new START (HPO off, cal off)")
    result = run_train_loop_new(df, loop_cfg, flags)
    thr = result.training.val_threshold
    out = collect_val_threshold_path(result.artifacts_dir)
    print(f"[B] >>> DONE val_threshold={thr}")
    print(f"[B] artifact: {out}")


if __name__ == "__main__":
    main()
