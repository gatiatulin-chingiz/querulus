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
from querulus.features.data_quality import apply_data_quality
from querulus.training.feature_selection_io import (
    drop_zero_importance_features,
    save_feature_selection,
)
from querulus.training.feature_selection_report import save_feature_selection_report
from querulus.training.hpo import HpoResult, run_hpo
from querulus.training.backward_elim import (
    BackwardElimResult,
    backward_eliminate_by_metric,
)
from querulus.training.noise_cut import NoiseCutResult, filter_features_by_noise
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
    (HPO/cal тогда тоже пропускаются). После SHAP — опциональный
    ``run_noise_cut`` (отсев слабее ``FE_NOISE_UNIFORM``), затем
    ``run_backward_elim`` (срез с конца по PR-AUC / MAE до 1 фичи).
    """

    use_fe_features: bool = True
    run_corr_filter: bool = True
    corr_filter_threshold: float = 0.95
    run_psi_filter: bool = True
    psi_threshold: float = DEFAULT_PSI_THRESHOLD
    l1_threshold: float = DEFAULT_L1_THRESHOLD
    run_shap_select: bool = True
    shap_n_features: int = 30
    run_noise_cut: bool = True
    run_backward_elim: bool = True
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
    frequency_noise_cut: NoiseCutResult | None = None
    severity_noise_cut: NoiseCutResult | None = None
    frequency_backward_elim: BackwardElimResult | None = None
    severity_backward_elim: BackwardElimResult | None = None


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
        (
            "RUN_CORR_FILTER",
            flags.run_corr_filter,
            f"Pearson |r|>{flags.corr_filter_threshold:g}, раздельно freq/sev",
        ),
        (
            "RUN_PSI_FILTER",
            flags.run_psi_filter,
            f"PSI>{flags.psi_threshold} / L1>{flags.l1_threshold} (train vs Val)",
        ),
        ("RUN_FEATURE_SELECT", flags.run_shap_select, f"SHAP RecursiveByShapValues → {flags.shap_n_features}"),
        (
            "RUN_NOISE_CUT",
            flags.run_noise_cut and flags.run_shap_select,
            "после FS: отсев фич слабее FE_NOISE_UNIFORM",
        ),
        (
            "RUN_BACKWARD_ELIM",
            flags.run_backward_elim and flags.run_shap_select,
            "после noise-cut: срез с конца, best PR-AUC / MAE",
        ),
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

    # DQ: clip ≥0 (money/DIFF) + winsorize IQR(log1p)→expm1; границы с train
    stage_start("data_quality", detail="clip≥0 + winsorize log1p-IQR on train")
    numeric_for_dq = [
        c
        for c in resolved_types.get("NUMERIC", [])
        if c in df.columns
        and c not in resolve_cfg.drop_columns
        and c
        not in {
            base.frequency_target,
            base.severity_target,
            base.date_column,
        }
    ]
    df, dq_report = apply_data_quality(
        df,
        train_index=splits.train,
        numeric_columns=numeric_for_dq,
    )
    dq_payload = dq_report.to_dict()
    dq_payload["pipeline_context"] = {
        "when": "after MVP types, before corr/PSI/FS",
        "fit_split": "train",
        "train_period": [str(train_core[0]), str(train_core[1])],
        "numeric_columns_considered": numeric_for_dq,
        "n_numeric_considered": len(numeric_for_dq),
        "downstream_stages": [
            "corr_filter (may drop features, not rows)",
            "psi_filter (may drop features, not rows)",
            "feature_select / noise_cut / backward_elim / zero_importance",
        ],
        "not_used": ["row_drop", "nan_impute", "percentile_winsor"],
    }
    (out_dir / "data_quality_report.json").write_text(
        json.dumps(dq_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = dq_payload["summary"]
    print(
        f"[B] data_quality: hard_clip_cells={summary['n_hard_clip_cells']} "
        f"winsor_cols={summary['n_winsorize_columns']} "
        f"winsor_cells={summary['n_winsorize_cells']} "
        f"skipped={summary['n_skipped_columns']}"
    )
    stage_done(
        "data_quality",
        detail=(
            f"winsor={summary['n_winsorize_columns']} cols, "
            f"clip_cells={summary['n_hard_clip_cells']}"
        ),
    )

    train_df = df.loc[splits.train]

    if flags.run_corr_filter and freq_features and sev_features:
        thr = float(flags.corr_filter_threshold)
        stage_start("corr_filter", detail=f"|r|>{thr:g}; freq + sev on train")
        freq_corr = correlation_filter_features(
            train_df,
            freq_features,
            base.frequency_target,
            threshold=thr,
        )
        sev_corr = correlation_filter_features(
            train_df,
            sev_features,
            base.severity_target,
            threshold=thr,
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
                    "threshold": thr,
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
    freq_noise: NoiseCutResult | None = None
    sev_noise: NoiseCutResult | None = None
    freq_back: BackwardElimResult | None = None
    sev_back: BackwardElimResult | None = None
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
        stage_done(
            "feature_select",
            detail=f"freq={len(freq_features)} sev={len(sev_features)}",
        )

        # После FS: шум + отсев всего не сильнее шума (шум в итог не входит).
        if flags.run_noise_cut and freq_features and sev_features:
            stage_start("noise_cut", detail="FE_NOISE_UNIFORM vs selected")
            freq_noise = filter_features_by_noise(
                df,
                features=freq_features,
                target_column=base.frequency_target,
                train_index=splits.train,
                eval_index=splits.val,
                task_type="classification",
                mvp_types=freq_mvp,
                random_state=base.frequency_random_state,
                iterations=base.frequency_select_iterations,
                early_stopping_rounds=base.frequency_select_early_stopping_rounds,
            )
            sev_noise = filter_features_by_noise(
                df,
                features=sev_features,
                target_column=base.severity_target,
                train_index=splits.train,
                eval_index=splits.val,
                task_type="regression",
                mvp_types=sev_mvp,
                positive_target=base.severity_range is None,
                random_state=base.severity_random_state,
                iterations=base.severity_select_iterations,
                early_stopping_rounds=base.severity_select_early_stopping_rounds,
            )
            print(
                f"[B] noise-cut freq: rank={freq_noise.noise_rank}/"
                f"{len(freq_noise.importances)} "
                f"drop={list(freq_noise.dropped_below_noise) or '(none)'} "
                f"kept={len(freq_noise.kept_features)}"
            )
            print(
                f"[B] noise-cut sev: rank={sev_noise.noise_rank}/"
                f"{len(sev_noise.importances)} "
                f"drop={list(sev_noise.dropped_below_noise) or '(none)'} "
                f"kept={len(sev_noise.kept_features)}"
            )
            freq_features = list(freq_noise.kept_features)
            sev_features = list(sev_noise.kept_features)
            if not freq_features or not sev_features:
                raise ValueError(
                    "Noise-cut обнулил пул (шум важнее всех отобранных). "
                    f"freq kept={len(freq_features)} sev kept={len(sev_features)}"
                )
            freq_mvp = slice_mvp_types(resolved_mvp, freq_features)
            sev_mvp = slice_mvp_types(resolved_mvp, sev_features)
            shap_training = replace(
                shap_training,
                frequency_features=freq_features,
                severity_features=sev_features,
                frequency_categorical_features=[
                    c
                    for c in shap_training.frequency_categorical_features
                    if c in freq_features
                ],
                severity_categorical_features=[
                    c
                    for c in shap_training.severity_categorical_features
                    if c in sev_features
                ],
            )
            stage_done(
                "noise_cut",
                detail=(
                    f"freq={len(freq_features)} sev={len(sev_features)}"
                ),
            )
        else:
            stage_skipped("noise_cut", "RUN_NOISE_CUT / пустой пул")

        # После noise-cut: снимаем с конца по одной до 1, берём best PR-AUC / MAE.
        if flags.run_backward_elim and freq_features and sev_features:
            stage_start(
                "backward_elim",
                detail="drop from end → best pr_auc / mae",
            )

            def _noise_order(noise: NoiseCutResult | None, pool: list[str]) -> list[str] | None:
                if noise is None:
                    return None
                ranked = (
                    noise.importances[
                        noise.importances["feature"] != noise.noise_feature
                    ]
                    .sort_values(["importance", "feature"], ascending=[False, True])
                )
                ordered = [f for f in ranked["feature"].tolist() if f in pool]
                return ordered or None

            freq_back = backward_eliminate_by_metric(
                df,
                features=freq_features,
                target_column=base.frequency_target,
                train_index=splits.train,
                eval_index=splits.val,
                task_type="classification",
                mvp_types=freq_mvp,
                random_state=base.frequency_random_state,
                iterations=base.frequency_select_iterations,
                early_stopping_rounds=base.frequency_select_early_stopping_rounds,
                importance_order=_noise_order(freq_noise, freq_features),
            )
            sev_back = backward_eliminate_by_metric(
                df,
                features=sev_features,
                target_column=base.severity_target,
                train_index=splits.train,
                eval_index=splits.val,
                task_type="regression",
                mvp_types=sev_mvp,
                positive_target=base.severity_range is None,
                random_state=base.severity_random_state,
                iterations=base.severity_select_iterations,
                early_stopping_rounds=base.severity_select_early_stopping_rounds,
                importance_order=_noise_order(sev_noise, sev_features),
            )
            print(
                f"[B] backward-elim freq: best {freq_back.metric_name}="
                f"{freq_back.best_metric:.4f} "
                f"n={len(freq_back.selected_features)}/{len(freq_features)} "
                f"(steps={len(freq_back.history)})"
            )
            print(
                f"[B] backward-elim sev: best {sev_back.metric_name}="
                f"{sev_back.best_metric:.4f} "
                f"n={len(sev_back.selected_features)}/{len(sev_features)} "
                f"(steps={len(sev_back.history)})"
            )
            freq_features = list(freq_back.selected_features)
            sev_features = list(sev_back.selected_features)
            if not freq_features or not sev_features:
                raise ValueError(
                    "Backward-elim обнулил пул. "
                    f"freq={len(freq_features)} sev={len(sev_features)}"
                )
            freq_mvp = slice_mvp_types(resolved_mvp, freq_features)
            sev_mvp = slice_mvp_types(resolved_mvp, sev_features)
            shap_training = replace(
                shap_training,
                frequency_features=freq_features,
                severity_features=sev_features,
                frequency_categorical_features=[
                    c
                    for c in shap_training.frequency_categorical_features
                    if c in freq_features
                ],
                severity_categorical_features=[
                    c
                    for c in shap_training.severity_categorical_features
                    if c in sev_features
                ],
            )
            stage_done(
                "backward_elim",
                detail=f"freq={len(freq_features)} sev={len(sev_features)}",
            )
        else:
            stage_skipped("backward_elim", "RUN_BACKWARD_ELIM / пустой пул")

        # CatBoost может оставить фичи с importance == 0 — выкидываем их.
        stage_start("zero_importance", detail="drop importance ≤ 0")
        freq_features, freq_zero_drop, freq_imp = drop_zero_importance_features(
            freq_features,
            shap_training.frequency_importance,
        )
        sev_features, sev_zero_drop, sev_imp = drop_zero_importance_features(
            sev_features,
            shap_training.severity_importance,
        )
        print(
            f"[B] zero-importance freq drop={list(freq_zero_drop) or '(none)'} "
            f"kept={len(freq_features)}"
        )
        print(
            f"[B] zero-importance sev drop={list(sev_zero_drop) or '(none)'} "
            f"kept={len(sev_features)}"
        )
        if not freq_features or not sev_features:
            raise ValueError(
                "После drop importance≤0 пустой пул. "
                f"freq={len(freq_features)} sev={len(sev_features)}"
            )
        freq_mvp = slice_mvp_types(resolved_mvp, freq_features)
        sev_mvp = slice_mvp_types(resolved_mvp, sev_features)
        shap_training = replace(
            shap_training,
            frequency_features=freq_features,
            severity_features=sev_features,
            frequency_importance=(
                freq_imp if freq_imp is not None else shap_training.frequency_importance
            ),
            severity_importance=(
                sev_imp if sev_imp is not None else shap_training.severity_importance
            ),
            frequency_categorical_features=[
                c
                for c in shap_training.frequency_categorical_features
                if c in freq_features
            ],
            severity_categorical_features=[
                c
                for c in shap_training.severity_categorical_features
                if c in sev_features
            ],
        )
        stage_done(
            "zero_importance",
            detail=(
                f"freq drop={len(freq_zero_drop)} kept={len(freq_features)}; "
                f"sev drop={len(sev_zero_drop)} kept={len(sev_features)}"
            ),
        )

        freq_summary = dict(shap_training.frequency_feature_selection_summary or {})
        sev_summary = dict(shap_training.severity_feature_selection_summary or {})
        if freq_noise is not None:
            freq_summary["noise_cut"] = {
                "noise_feature": freq_noise.noise_feature,
                "noise_rank": freq_noise.noise_rank,
                "noise_was_last": freq_noise.noise_was_last,
                "dropped_below_noise": list(freq_noise.dropped_below_noise),
            }
        if sev_noise is not None:
            sev_summary["noise_cut"] = {
                "noise_feature": sev_noise.noise_feature,
                "noise_rank": sev_noise.noise_rank,
                "noise_was_last": sev_noise.noise_was_last,
                "dropped_below_noise": list(sev_noise.dropped_below_noise),
            }
        if freq_back is not None:
            freq_summary["backward_elim"] = {
                "metric_name": freq_back.metric_name,
                "best_metric": freq_back.best_metric,
                "n_selected": len(freq_back.selected_features),
                "ordered_features": list(freq_back.ordered_features),
                "history": [
                    {
                        "n_features": step.n_features,
                        "metric": step.metric,
                        "dropped_feature": step.dropped_feature,
                    }
                    for step in freq_back.history
                ],
            }
        if sev_back is not None:
            sev_summary["backward_elim"] = {
                "metric_name": sev_back.metric_name,
                "best_metric": sev_back.best_metric,
                "n_selected": len(sev_back.selected_features),
                "ordered_features": list(sev_back.ordered_features),
                "history": [
                    {
                        "n_features": step.n_features,
                        "metric": step.metric,
                        "dropped_feature": step.dropped_feature,
                    }
                    for step in sev_back.history
                ],
            }
        if freq_zero_drop:
            freq_summary["zero_importance_dropped"] = list(freq_zero_drop)
        if sev_zero_drop:
            sev_summary["zero_importance_dropped"] = list(sev_zero_drop)
        shap_training = replace(
            shap_training,
            frequency_feature_selection_summary=freq_summary or None,
            severity_feature_selection_summary=sev_summary or None,
        )
        save_feature_selection(
            stack="new",
            task="frequency",
            selected_features=freq_features,
            summary=freq_summary,
            directory=out_dir,
            importance=shap_training.frequency_importance,
            categorical_features=list(shap_training.frequency_categorical_features),
        )
        save_feature_selection(
            stack="new",
            task="severity",
            selected_features=sev_features,
            summary=sev_summary,
            directory=out_dir,
            importance=shap_training.severity_importance,
            categorical_features=list(shap_training.severity_categorical_features),
        )
        # HTML для бизнеса: RU-нейминг + EDA-графики + severity-only
        cat_feats = list(
            dict.fromkeys(
                [
                    *shap_training.frequency_categorical_features,
                    *shap_training.severity_categorical_features,
                ]
            )
        )
        report_path = save_feature_selection_report(
            df,
            frequency_features=freq_features,
            severity_features=sev_features,
            categorical_features=cat_feats,
            frequency_importance=shap_training.frequency_importance,
            severity_importance=shap_training.severity_importance,
            frequency_target=base.frequency_target,
            severity_target=base.severity_target,
            stack="new",
            directory=out_dir,
        )
        print(f"[B] feature select artifacts → {out_dir}")
        print(f"[B] feature select report → {report_path}")
    else:
        stage_skipped("feature_select", "RUN_FEATURE_SELECT")
        stage_skipped("noise_cut", "нет feature_select")
        stage_skipped("backward_elim", "нет feature_select")

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
            frequency_noise_cut=freq_noise,
            severity_noise_cut=sev_noise,
            frequency_backward_elim=freq_back,
            severity_backward_elim=sev_back,
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
        frequency_noise_cut=freq_noise,
        severity_noise_cut=sev_noise,
        frequency_backward_elim=freq_back,
        severity_backward_elim=sev_back,
    )
