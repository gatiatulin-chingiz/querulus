"""Оркестратор блока B: train-loop кандидата ``new`` с флагами этапов."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd

from querulus import PROJECT_ROOT
from querulus.training.calibration import expected_calibration_error, fit_probability_calibrator
from querulus.training.config import TrainingConfig, resolve_features_config
from querulus.training.corr_filter import correlation_filter_features, slice_mvp_types
from querulus.training.drift import filter_features_by_drift
from querulus.training.feature_selection_io import save_feature_selection
from querulus.training.hpo import HpoResult, run_hpo
from querulus.training.pipeline import TrainingArtifacts, frequency_predict_proba, train_models
from querulus.training.splits import (
    DateSplitParts,
    default_inner_periods_from_train,
    split_by_date_periods,
)
from querulus.training.stage_log import stage_done, stage_skipped, stage_start


@dataclass(frozen=True)
class TrainLoopFlags:
    """Флаги этапов блока B (выключенный этап = skip)."""

    use_fe_features: bool = True
    run_corr_filter: bool = True
    run_psi_filter: bool = True
    psi_threshold: float = 0.5
    run_hpo: bool = False
    run_shap_select: bool = True
    run_hpo_retune: bool = False
    run_calibration: bool = True
    hpo_n_trials: int = 10
    hpo_cv: int = 3
    use_mlflow: bool = True


@dataclass
class TrainLoopResult:
    """Артефакты прогона train-loop new."""

    training: TrainingArtifacts
    flags: TrainLoopFlags
    splits: DateSplitParts
    frequency_features: list[str]
    severity_features: list[str]
    frequency_mvp_types: dict[str, tuple[str, ...]]
    severity_mvp_types: dict[str, tuple[str, ...]]
    frequency_hpo: HpoResult | None = None
    severity_hpo: HpoResult | None = None
    ece_before: float | None = None
    ece_after: float | None = None
    artifacts_dir: Path | None = None
    psi_dropped: list[str] = field(default_factory=list)
    psi_report: pd.DataFrame | None = None


def _drop_fe_columns(features: list[str] | tuple[str, ...]) -> list[str]:
    return [name for name in features if not str(name).startswith("FE_")]


def _merge_hpo_into_catboost(
    best_params: dict[str, Any],
    base_params: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Слить best_params Optuna в kwargs CatBoost; iterations отдельно."""
    merged = dict(base_params)
    iterations = int(best_params.get("iterations", merged.get("iterations", 375)))
    skip = {"iterations", "early_stopping_rounds"}
    for key, value in best_params.items():
        if key in skip or value is None:
            continue
        merged[key] = value
    merged["verbose"] = base_params.get("verbose", 250)
    return merged, iterations


def print_flags_table(flags: TrainLoopFlags) -> None:
    """Сводка флагов блока B."""
    rows = [
        ("USE_FE_FEATURES", flags.use_fe_features, "derived/incident FE_* в пуле"),
        ("RUN_CORR_FILTER", flags.run_corr_filter, "Pearson-filter числовых, раздельно freq/sev"),
        ("RUN_PSI_FILTER", flags.run_psi_filter, f"дроп фич с PSI/L1 > {flags.psi_threshold} (train vs Val)"),
        ("RUN_HPO", flags.run_hpo, "Optuna+MLflow TimeSeriesSplit"),
        ("RUN_SHAP_SELECT", flags.run_shap_select, "RecursiveByShapValues"),
        ("RUN_HPO_RETUNE", flags.run_hpo_retune, "короткий ре-тюнинг после SHAP"),
        ("RUN_CALIBRATION", flags.run_calibration, "калибровка freq на Cal + ECE"),
    ]
    print("[B] === Train-loop flags ===")
    for name, enabled, sense in rows:
        mark = "ON " if enabled else "OFF"
        print(f"[B]   {mark}  {name:18} — {sense}")


def run_train_loop_new(
    df: pd.DataFrame,
    config: TrainingConfig | None = None,
    flags: TrainLoopFlags | None = None,
    *,
    artifacts_dir: Path | str | None = None,
) -> TrainLoopResult:
    """Пайплайн блока B только для стека new (таргеты из config).

    Порядок: FE-флаг → corr → PSI(train vs Val) → HPO → SHAP(select) → (retune) → fit → cal.
    Early-stop / SHAP eval — на Val; Test из config.test_period в HPO не идёт.
    """
    flags = flags or TrainLoopFlags()
    base = resolve_features_config(config or TrainingConfig())
    base = replace(base, use_fe_features=flags.use_fe_features)

    print_flags_table(flags)

    if base.val_period is None or base.cal_period is None:
        train_core, val_period, cal_period = default_inner_periods_from_train(base.train_period)
    else:
        train_core = base.train_period
        val_period = base.val_period
        cal_period = base.cal_period

    splits = split_by_date_periods(
        df,
        date_column=base.date_column,
        train_period=train_core,
        val_period=val_period,
        cal_period=cal_period,
        test_period=base.test_period,
    )
    print(
        f"[B] splits: train={len(splits.train)} val={len(splits.val)} "
        f"cal={len(splits.cal)} test={len(splits.test)} "
        f"(train_core={train_core}, val={val_period}, cal={cal_period})"
    )

    out_dir = Path(artifacts_dir) if artifacts_dir else (
        PROJECT_ROOT / "data" / "processed" / "train_loop_new"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    freq_features = list(base.frequency_features or ())
    sev_features = list(base.severity_features or ())
    if not freq_features or not sev_features:
        # mvp: стартовый пул = все колонки из mvp_input_types, пересечение с df
        from querulus.training.mvp_types import DEFAULT_OTHER_COLS

        pool = []
        for cols in base.mvp_input_types.values():
            pool.extend(cols)
        pool = [
            c
            for c in dict.fromkeys(pool)
            if c in df.columns and c not in DEFAULT_OTHER_COLS and c not in base.drop_columns
        ]
        if not freq_features:
            freq_features = list(pool)
        if not sev_features:
            sev_features = list(pool)
        print(f"[B] mvp starter pools: freq={len(freq_features)} sev={len(sev_features)}")

    if not flags.use_fe_features:
        stage_start("use_fe_features", detail="drop FE_* from pools")
        if freq_features:
            freq_features = _drop_fe_columns(freq_features)
        if sev_features:
            sev_features = _drop_fe_columns(sev_features)
        stage_done("use_fe_features", detail=f"freq={len(freq_features)} sev={len(sev_features)}")
    else:
        print("[B] >>> STAGE use_fe_features ON (FE_* allowed in pools)")

    train_df = df.loc[splits.train]
    freq_mvp = dict(base.mvp_input_types)
    sev_mvp = dict(base.mvp_input_types)

    if flags.run_corr_filter and freq_features and sev_features:
        stage_start("corr_filter", detail="freq + sev on train")
        freq_corr = correlation_filter_features(
            train_df,
            freq_features,
            base.frequency_target,
            threshold=base.corr_filter_threshold,
        )
        sev_corr = correlation_filter_features(
            train_df,
            sev_features,
            base.severity_target,
            threshold=base.corr_filter_threshold,
        )
        freq_features = list(freq_corr.kept_features)
        sev_features = list(sev_corr.kept_features)
        freq_mvp = slice_mvp_types(freq_mvp, freq_features)
        sev_mvp = slice_mvp_types(sev_mvp, sev_features)
        (out_dir / "corr_filter_new.json").write_text(
            json.dumps(
                {
                    "frequency": {
                        "kept": freq_features,
                        "eliminated": list(freq_corr.eliminated_features),
                    },
                    "severity": {
                        "kept": sev_features,
                        "eliminated": list(sev_corr.eliminated_features),
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        stage_done(
            "corr_filter",
            detail=(
                f"freq kept={len(freq_features)} drop={len(freq_corr.eliminated_features)}; "
                f"sev kept={len(sev_features)} drop={len(sev_corr.eliminated_features)}"
            ),
        )
    else:
        stage_skipped("corr_filter", "RUN_CORR_FILTER")

    psi_dropped: list[str] = []
    psi_report: pd.DataFrame | None = None
    if flags.run_psi_filter and (freq_features or sev_features):
        stage_start(
            "psi_filter",
            detail=(
                f"threshold={flags.psi_threshold} "
                f"ref=train_core vs Val ({val_period})"
            ),
        )
        # Один union-пул: drift по времени, не по таргету.
        union = list(dict.fromkeys([*freq_features, *sev_features]))
        cat_names = set(freq_mvp.get("CATEGORIAL", ())) | set(freq_mvp.get("BINARY", ()))
        cat_names |= set(sev_mvp.get("CATEGORIAL", ())) | set(sev_mvp.get("BINARY", ()))
        kept_union, psi_report = filter_features_by_drift(
            df,
            union,
            date_column=base.date_column,
            reference_period=train_core,
            compare_period=val_period,
            threshold=flags.psi_threshold,
            categorical_features=cat_names,
        )
        kept_set = set(kept_union)
        psi_dropped = [
            name
            for name in union
            if name not in kept_set
        ]
        freq_features = [f for f in freq_features if f in kept_set]
        sev_features = [f for f in sev_features if f in kept_set]
        freq_mvp = slice_mvp_types(freq_mvp, freq_features)
        sev_mvp = slice_mvp_types(sev_mvp, sev_features)
        print(f"[B] PSI-filter dropped ({len(psi_dropped)}):")
        if psi_dropped:
            drop_view = psi_report.loc[psi_report["dropped"], ["feature", "kind", "drift_score"]]
            print(drop_view.to_string(index=False))
        else:
            print("[B]   (none)")
        stage_done(
            "psi_filter",
            detail=(
                f"drop={len(psi_dropped)} "
                f"freq={len(freq_features)} sev={len(sev_features)}"
            ),
        )
    else:
        stage_skipped("psi_filter", "RUN_PSI_FILTER")

    freq_hpo: HpoResult | None = None
    sev_hpo: HpoResult | None = None
    freq_params: dict[str, Any] = dict(base.frequency_classifier_params)
    sev_params: dict[str, Any] = dict(base.severity_regressor_params)
    freq_iters = base.frequency_iterations
    sev_iters = base.severity_iterations

    hpo_frame = df.loc[splits.train.union(splits.val)]
    if flags.run_hpo and freq_features and sev_features:
        stage_start("hpo_frequency", detail=f"trials={flags.hpo_n_trials}")
        freq_hpo = run_hpo(
            hpo_frame,
            features=freq_features,
            target_column=base.frequency_target,
            date_column=base.date_column,
            task_type="classification",
            optimize_metric="roc_auc",
            direction="maximize",
            experiment_name="querulus_hpo_frequency_new",
            n_trials=flags.hpo_n_trials,
            cv=flags.hpo_cv,
            mvp_types=freq_mvp,
            use_mlflow=flags.use_mlflow,
        )
        freq_params, freq_iters = _merge_hpo_into_catboost(freq_hpo.best_params, freq_params)
        stage_done("hpo_frequency", detail=f"best={freq_hpo.best_value:.4f}")

        stage_start("hpo_severity", detail=f"trials={flags.hpo_n_trials}")
        sev_hpo = run_hpo(
            hpo_frame,
            features=sev_features,
            target_column=base.severity_target,
            date_column=base.date_column,
            task_type="regression",
            optimize_metric="mae",
            direction="minimize",
            experiment_name="querulus_hpo_severity_new",
            n_trials=flags.hpo_n_trials,
            cv=flags.hpo_cv,
            mvp_types=sev_mvp,
            use_mlflow=flags.use_mlflow,
        )
        sev_params, sev_iters = _merge_hpo_into_catboost(sev_hpo.best_params, sev_params)
        stage_done("hpo_severity", detail=f"best={sev_hpo.best_value:.4f}")
        (out_dir / "hpo_best_params_new.json").write_text(
            json.dumps(
                {"frequency": freq_hpo.best_params, "severity": sev_hpo.best_params},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        stage_skipped("hpo", "RUN_HPO")

    fe_extra_drop: tuple[str, ...] = ()
    if not flags.use_fe_features:
        fe_extra_drop = tuple(c for c in df.columns if str(c).startswith("FE_"))

    stage_start("fit_new", detail=f"SHAP_SELECT={flags.run_shap_select}; eval=val")
    fit_config = replace(
        base,
        train_period=train_core,
        test_period=val_period,
        frequency_features=tuple(freq_features) if freq_features else None,
        severity_features=tuple(sev_features) if sev_features else None,
        frequency_select_features=flags.run_shap_select and base.features_source == "mvp",
        severity_select_features=flags.run_shap_select and base.features_source == "mvp",
        frequency_iterations=freq_iters,
        severity_iterations=sev_iters,
        frequency_classifier_params=freq_params,
        severity_regressor_params=sev_params,
        frequency_calibration_enabled=False,
        mvp_input_types=freq_mvp,
        extra_drop_columns=base.extra_drop_columns + fe_extra_drop,
    )
    training = train_models(df, fit_config)
    freq_features = list(training.frequency_features)
    sev_features = list(training.severity_features)
    save_feature_selection(
        stack="new",
        task="frequency",
        selected_features=freq_features,
        summary=training.frequency_feature_selection_summary,
        directory=out_dir,
    )
    save_feature_selection(
        stack="new",
        task="severity",
        selected_features=sev_features,
        summary=training.severity_feature_selection_summary,
        directory=out_dir,
    )
    stage_done(
        "fit_new",
        detail=f"freq_features={len(freq_features)} sev_features={len(sev_features)}",
    )

    if flags.run_hpo_retune and flags.run_hpo and freq_features and sev_features:
        stage_start("hpo_retune", detail="short HPO on selected features")
        retune_trials = max(3, flags.hpo_n_trials // 3)
        freq_hpo = run_hpo(
            hpo_frame,
            features=freq_features,
            target_column=base.frequency_target,
            date_column=base.date_column,
            task_type="classification",
            optimize_metric="roc_auc",
            direction="maximize",
            experiment_name="querulus_hpo_frequency_new_retune",
            n_trials=retune_trials,
            cv=flags.hpo_cv,
            mvp_types=slice_mvp_types(freq_mvp, freq_features),
            use_mlflow=flags.use_mlflow,
        )
        sev_hpo = run_hpo(
            hpo_frame,
            features=sev_features,
            target_column=base.severity_target,
            date_column=base.date_column,
            task_type="regression",
            optimize_metric="mae",
            direction="minimize",
            experiment_name="querulus_hpo_severity_new_retune",
            n_trials=retune_trials,
            cv=flags.hpo_cv,
            mvp_types=slice_mvp_types(sev_mvp, sev_features),
            use_mlflow=flags.use_mlflow,
        )
        freq_params, freq_iters = _merge_hpo_into_catboost(freq_hpo.best_params, freq_params)
        sev_params, sev_iters = _merge_hpo_into_catboost(sev_hpo.best_params, sev_params)
        fit_config = replace(
            fit_config,
            frequency_iterations=freq_iters,
            severity_iterations=sev_iters,
            frequency_classifier_params=freq_params,
            severity_regressor_params=sev_params,
            frequency_features=tuple(freq_features),
            severity_features=tuple(sev_features),
            frequency_select_features=False,
            severity_select_features=False,
        )
        training = train_models(df, fit_config)
        stage_done("hpo_retune", detail="refit done")
    else:
        stage_skipped("hpo_retune", "RUN_HPO_RETUNE")

    ece_before: float | None = None
    ece_after: float | None = None
    if flags.run_calibration and len(splits.cal) > 0:
        stage_start("calibration", detail=f"cal_n={len(splits.cal)}")
        cal_frame = df.loc[splits.cal]
        x_cal = cal_frame[training.frequency_features].copy()
        for col in training.frequency_categorical_features:
            if col in x_cal.columns:
                x_cal[col] = x_cal[col].astype(str)
        y_cal = cal_frame[base.frequency_target]
        proba_before = frequency_predict_proba(training, x_cal)
        ece_before = expected_calibration_error(y_cal, proba_before)
        calibrator = fit_probability_calibrator(
            training.frequency_model,
            x_cal,
            y_cal,
            method=base.frequency_calibration_method,
        )
        training = replace(training, frequency_calibrator=calibrator)
        proba_after = frequency_predict_proba(training, x_cal)
        ece_after = expected_calibration_error(y_cal, proba_after)
        stage_done(
            "calibration",
            detail=f"ECE before={ece_before:.4f} after={ece_after:.4f}",
        )
    else:
        stage_skipped("calibration", "RUN_CALIBRATION")

    print(f"[B] artifacts → {out_dir}")
    return TrainLoopResult(
        training=training,
        flags=flags,
        splits=splits,
        frequency_features=freq_features,
        severity_features=sev_features,
        frequency_mvp_types=freq_mvp,
        severity_mvp_types=sev_mvp,
        frequency_hpo=freq_hpo,
        severity_hpo=sev_hpo,
        ece_before=ece_before,
        ece_after=ece_after,
        artifacts_dir=out_dir,
        psi_dropped=psi_dropped,
        psi_report=psi_report,
    )
