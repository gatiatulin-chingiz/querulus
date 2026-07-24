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
from querulus.training.drift_thresholds import DEFAULT_L1_THRESHOLD, DEFAULT_PSI_THRESHOLD
from querulus.training.drift import filter_features_by_drift, format_psi_filter_report
from querulus.training.feature_selection_io import save_feature_selection
from querulus.training.hpo import HpoResult, run_hpo
from querulus.training.pipeline import (
    TrainingArtifacts,
    frequency_predict_proba,
    resolve_mvp_types,
    train_models,
)
from querulus.training.severity_training import fit_severity_model
from querulus.training.severity_variant import resolve_severity_variant
from querulus.training.severity_zoo import _segment_indices
from querulus.training.splits import (
    DateSplitParts,
    default_inner_periods_from_train,
    split_by_date_periods,
)
from querulus.training.stage_log import stage_done, stage_skipped, stage_start


@dataclass(frozen=True)
class TrainLoopFlags:
    """Флаги этапов блока B (выключенный этап = skip).

    Только отбор фич: ``run_shap_select=True``, ``run_fit=False``
    (HPO/cal тогда тоже пропускаются).
    """

    use_fe_features: bool = True
    run_corr_filter: bool = True
    run_psi_filter: bool = True
    psi_threshold: float = DEFAULT_PSI_THRESHOLD
    l1_threshold: float = DEFAULT_L1_THRESHOLD
    run_shap_select: bool = True
    shap_n_features: int = 30
    run_fit: bool = True
    run_hpo: bool = False
    run_calibration: bool = True
    hpo_n_trials: int = 10
    hpo_cv: int = 3
    use_mlflow: bool = True
    severity_variant: str = "raw"
    severity_value_column: str = "VALUE_BEFORE_WITH"
    severity_value_threshold: float = 50_000.0


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


def _types_as_tuples(types: dict[str, list[str]]) -> dict[str, tuple[str, ...]]:
    return {key: tuple(cols) for key, cols in types.items()}


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
        (
            "RUN_PSI_FILTER",
            flags.run_psi_filter,
            f"PSI>{flags.psi_threshold} / L1>{flags.l1_threshold} (train vs Val)",
        ),
        ("RUN_FEATURE_SELECT", flags.run_shap_select, f"SHAP RecursiveByShapValues → {flags.shap_n_features}"),
        ("RUN_FIT", flags.run_fit, "финальный fit (+ HPO/cal если включены)"),
        ("RUN_HPO", flags.run_hpo and flags.run_fit, "Optuna+MLflow (только при RUN_FIT)"),
        ("RUN_CALIBRATION", flags.run_calibration and flags.run_fit, "калибровка freq на Cal"),
        ("SEVERITY_VARIANT", True, flags.severity_variant),
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

    Порядок: FE → corr → PSI(train vs Val) → SHAP(n) → HPO → fit → cal.
    Пул и cat_features — из ``value_type`` + ``correct_types`` (resolve_mvp_types).
    Early-stop / SHAP eval — на Val; Test в HPO не идёт.
    """
    flags = flags or TrainLoopFlags()
    base = resolve_features_config(config or TrainingConfig())
    base = replace(base, use_fe_features=flags.use_fe_features)
    sev_spec = resolve_severity_variant(flags.severity_variant)
    base = replace(
        base,
        severity_target_transform=sev_spec.transform,
        severity_sample_weight=sev_spec.sample_weight,
    )

    print_flags_table(flags)
    print(
        f"[B] severity_variant={sev_spec.name} "
        f"(transform={sev_spec.transform}, weight={sev_spec.sample_weight}, "
        f"segment={sev_spec.segment})"
    )

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
        f"(train_core={train_core}, val={val_period}, cal={cal_period}, "
        f"test={base.test_period})"
    )

    out_dir = Path(artifacts_dir) if artifacts_dir else (
        PROJECT_ROOT / "data" / "processed" / "train_loop_new"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    fe_extra_drop: tuple[str, ...] = ()
    if not flags.use_fe_features:
        fe_extra_drop = tuple(c for c in df.columns if str(c).startswith("FE_"))
        stage_start("use_fe_features", detail="drop FE_* from pools")
    else:
        print("[B] >>> STAGE use_fe_features ON (FE_* allowed in pools)")

    # value_type + correct_types(TO_DROP) — единый источник пула и катов
    resolve_cfg = replace(base, extra_drop_columns=base.extra_drop_columns + fe_extra_drop)
    resolved_types = resolve_mvp_types(df, resolve_cfg)
    resolved_mvp = _types_as_tuples(resolved_types)
    auto_pool = [
        c
        for c in dict.fromkeys(
            [
                *resolved_types.get("BINARY", []),
                *resolved_types.get("CATEGORIAL", []),
                *resolved_types.get("NUMERIC", []),
            ]
        )
        if c in df.columns and c not in resolve_cfg.drop_columns
    ]
    if not flags.use_fe_features:
        auto_pool = _drop_fe_columns(auto_pool)
        stage_done("use_fe_features", detail=f"pool={len(auto_pool)}")

    freq_features = list(base.frequency_features or auto_pool)
    sev_features = list(base.severity_features or auto_pool)
    if not flags.use_fe_features:
        freq_features = _drop_fe_columns(freq_features)
        sev_features = _drop_fe_columns(sev_features)
    freq_features = [f for f in freq_features if f in df.columns]
    sev_features = [f for f in sev_features if f in df.columns]
    freq_mvp = slice_mvp_types(resolved_mvp, freq_features)
    sev_mvp = slice_mvp_types(resolved_mvp, sev_features)
    print(
        f"[B] mvp pools (value_type): freq={len(freq_features)} sev={len(sev_features)} "
        f"cats={len(freq_mvp.get('CATEGORIAL', ())) + len(freq_mvp.get('BINARY', ()))}"
    )

    train_df = df.loc[splits.train]

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
        freq_mvp = slice_mvp_types(resolved_mvp, freq_features)
        sev_mvp = slice_mvp_types(resolved_mvp, sev_features)
        print(
            f"[B] corr-filter freq dropped ({len(freq_corr.eliminated_features)}): "
            f"{list(freq_corr.eliminated_features) or '(none)'}"
        )
        print(
            f"[B] corr-filter sev dropped ({len(sev_corr.eliminated_features)}): "
            f"{list(sev_corr.eliminated_features) or '(none)'}"
        )
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
                f"PSI>{flags.psi_threshold} L1>{flags.l1_threshold} "
                f"ref=train_core vs Val ({val_period})"
            ),
        )
        union = list(dict.fromkeys([*freq_features, *sev_features]))
        cat_names = set(freq_mvp.get("CATEGORIAL", ())) | set(freq_mvp.get("BINARY", ()))
        cat_names |= set(sev_mvp.get("CATEGORIAL", ())) | set(sev_mvp.get("BINARY", ()))
        kept_union, psi_report = filter_features_by_drift(
            df,
            union,
            date_column=base.date_column,
            reference_period=train_core,
            compare_period=val_period,
            psi_threshold=flags.psi_threshold,
            l1_threshold=flags.l1_threshold,
            categorical_features=cat_names,
        )
        kept_set = set(kept_union)
        psi_dropped = [name for name in union if name not in kept_set]
        freq_features = [f for f in freq_features if f in kept_set]
        sev_features = [f for f in sev_features if f in kept_set]
        freq_mvp = slice_mvp_types(resolved_mvp, freq_features)
        sev_mvp = slice_mvp_types(resolved_mvp, sev_features)
        print(f"[B] PSI-filter dropped ({len(psi_dropped)}):")
        if psi_dropped and psi_report is not None:
            print(format_psi_filter_report(psi_report.loc[psi_report["dropped"]]))
        else:
            print("[B]   (none)")
        print("[B] PSI-filter report (metric=PSI|L1, score, note):")
        print(format_psi_filter_report(psi_report))
        stage_done(
            "psi_filter",
            detail=(
                f"drop={len(psi_dropped)} "
                f"freq={len(freq_features)} sev={len(sev_features)}"
            ),
        )
    else:
        stage_skipped("psi_filter", "RUN_PSI_FILTER")

    # SHAP select → фиксируем пул до HPO/fit
    shap_training: TrainingArtifacts | None = None
    if flags.run_shap_select and freq_features and sev_features:
        stage_start(
            "feature_select",
            detail=f"n={flags.shap_n_features}; eval=val",
        )
        shap_config = replace(
            base,
            train_period=train_core,
            test_period=val_period,
            frequency_features=tuple(freq_features),
            severity_features=tuple(sev_features),
            frequency_select_features=True,
            severity_select_features=True,
            frequency_num_features_to_select=flags.shap_n_features,
            severity_num_features_to_select=flags.shap_n_features,
            frequency_calibration_enabled=False,
            mvp_input_types=base.mvp_input_types,
            extra_drop_columns=base.extra_drop_columns + fe_extra_drop,
        )
        shap_training = train_models(df, shap_config)
        freq_features = list(shap_training.frequency_features)
        sev_features = list(shap_training.severity_features)
        freq_mvp = slice_mvp_types(resolved_mvp, freq_features)
        sev_mvp = slice_mvp_types(resolved_mvp, sev_features)
        save_feature_selection(
            stack="new",
            task="frequency",
            selected_features=freq_features,
            summary=shap_training.frequency_feature_selection_summary,
            directory=out_dir,
        )
        save_feature_selection(
            stack="new",
            task="severity",
            selected_features=sev_features,
            summary=shap_training.severity_feature_selection_summary,
            directory=out_dir,
        )
        stage_done(
            "feature_select",
            detail=f"freq={len(freq_features)} sev={len(sev_features)} → {out_dir}",
        )
    else:
        stage_skipped("feature_select", "RUN_FEATURE_SELECT")

    # Только отбор фич: без HPO / финального fit / cal
    if not flags.run_fit:
        if shap_training is None:
            raise ValueError(
                "RUN_FIT=False требует RUN_FEATURE_SELECT=True "
                "(иначе нет модели/списка отобранных фич)"
            )
        stage_skipped("hpo", "RUN_FIT=False")
        stage_skipped("fit_new", "RUN_FIT=False")
        stage_skipped("calibration", "RUN_FIT=False")
        print(f"[B] artifacts → {out_dir} (feature select only)")
        return TrainLoopResult(
            training=shap_training,
            flags=flags,
            splits=splits,
            frequency_features=freq_features,
            severity_features=sev_features,
            frequency_mvp_types=freq_mvp,
            severity_mvp_types=sev_mvp,
            frequency_hpo=None,
            severity_hpo=None,
            ece_before=None,
            ece_after=None,
            artifacts_dir=out_dir,
            psi_dropped=psi_dropped,
            psi_report=psi_report,
        )

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

    stage_start("fit_new", detail="fixed features after SHAP; eval=val")
    fit_config = replace(
        base,
        train_period=train_core,
        test_period=val_period,
        frequency_features=tuple(freq_features) if freq_features else None,
        severity_features=tuple(sev_features) if sev_features else None,
        frequency_select_features=False,
        severity_select_features=False,
        frequency_iterations=freq_iters,
        severity_iterations=sev_iters,
        frequency_classifier_params=freq_params,
        severity_regressor_params=sev_params,
        frequency_calibration_enabled=False,
        mvp_input_types=base.mvp_input_types,
        extra_drop_columns=base.extra_drop_columns + fe_extra_drop,
    )
    training = train_models(df, fit_config)
    # Сегментные варианты (*_le50 / *_gt50): переобучить severity на срезе VALUE_BEFORE_*.
    if sev_spec.segment != "all" and training.severity_split is not None:
        stage_start(
            "severity_segment",
            detail=(
                f"{sev_spec.segment} {flags.severity_value_column} "
                f"thr={flags.severity_value_threshold:g}"
            ),
        )
        train_idx = _segment_indices(
            df,
            training.severity_split.x_train.index,
            flags.severity_value_column,
            flags.severity_value_threshold,
            sev_spec.segment,
        )
        eval_idx = _segment_indices(
            df,
            training.severity_split.x_test.index,
            flags.severity_value_column,
            flags.severity_value_threshold,
            sev_spec.segment,
        )
        if len(train_idx) == 0:
            stage_done("severity_segment", detail="skip: empty train segment")
        else:
            sev_model = fit_severity_model(
                training,
                fit_config,
                transform=sev_spec.transform,
                sample_weight=sev_spec.sample_weight,
                train_index=train_idx,
                eval_index=eval_idx if len(eval_idx) else None,
            )
            training = replace(
                training,
                severity_model=sev_model,
                severity_target_transform=sev_spec.transform,
            )
            stage_done("severity_segment", detail=f"train_n={len(train_idx)}")
    freq_features = list(training.frequency_features)
    sev_features = list(training.severity_features)
    stage_done(
        "fit_new",
        detail=f"freq_features={len(freq_features)} sev_features={len(sev_features)}",
    )

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
