"""Фин. эффект: штрафы модели, варианты severity, сегмент VALUE_BEFORE_WITH."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from querulus.fin_effect.calculator import (
    FinEffectResult,
    apply_model_predictions,
    prepare_effect_frame,
)
from querulus.fin_effect.compare_report import plain_floats
from querulus.fin_effect.config import FinEffectConfig
from querulus.training.config import TrainingConfig
from querulus.training.pipeline import TrainingArtifacts
from querulus.training.severity_training import (
    SeveritySampleWeight,
    SeverityTargetTransform,
    fit_severity_model,
    severity_predict,
)


def fin_effect_penalty_table(
    result: FinEffectResult,
    config: FinEffectConfig,
) -> pd.DataFrame:
    """Где модель «штрафуется» относительно факта (pred < fact).

    - ``freq_miss``: pred_freq=0, fact=1 (пропуск);
    - ``freq_false_positive``: pred_freq=1, fact=0;
    - ``sev_under``: pred=1, fact=1, pred_sev < true_sev (недооценка → полная база);
    - ``sev_over``: pred=1, fact=1, pred_sev ≥ true_sev (для сравнения).
    """
    frame = result.frame
    freq_col = config.frequency_target_column
    sev_col = config.severity_target_column
    pred = frame["pred_freq"].astype(int)
    fact = pd.to_numeric(frame[freq_col], errors="coerce").fillna(0).astype(int)
    y_pred_sev = pd.to_numeric(frame["pred_sev"], errors="coerce").fillna(0)
    y_true_sev = pd.to_numeric(frame[sev_col], errors="coerce").fillna(0)
    fact_signed = pd.to_numeric(frame["fin_effect_fact"], errors="coerce").fillna(0)
    base_sum = -fact_signed if config.negate_fact_for_report else fact_signed
    model = pd.to_numeric(frame["fin_effect_model"], errors="coerce").fillna(0)

    specs = [
        ("freq_miss", (pred == 0) & (fact == 1), "pred=0, fact=1"),
        ("freq_false_positive", (pred == 1) & (fact == 0), "pred=1, fact=0"),
        (
            "sev_under",
            (pred == 1) & (fact == 1) & (y_pred_sev < y_true_sev),
            "pred=1, fact=1, pred_sev<true_sev",
        ),
        (
            "sev_over",
            (pred == 1) & (fact == 1) & (y_pred_sev >= y_true_sev),
            "pred=1, fact=1, pred_sev≥true_sev",
        ),
    ]
    rows: list[dict[str, object]] = []
    for penalty_type, mask, label in specs:
        m = mask.fillna(False)
        extra_under = (
            float((base_sum.loc[m] - y_pred_sev.loc[m]).sum())
            if penalty_type == "sev_under"
            else 0.0
        )
        rows.append(
            {
                "penalty_type": penalty_type,
                "case": label,
                "n": int(m.sum()),
                "sum_base": float(base_sum.loc[m].sum()),
                "sum_pred_sev": float(y_pred_sev.loc[m].sum()),
                "sum_true_sev": float(y_true_sev.loc[m].sum()),
                "sum_fin_effect_model": float(model.loc[m].sum()),
                "sum_fin_effect_fact": float(fact_signed.loc[m].sum()),
                "delta_model_minus_fact": float((model.loc[m] - fact_signed.loc[m]).sum()),
                "extra_under_sev_rub": extra_under,
            }
        )
    return plain_floats(pd.DataFrame(rows))


def _effect_index_and_freq(
    training: TrainingArtifacts,
    split: str,
) -> tuple[pd.Index, pd.Series]:
    frequency_split = training.frequency_split
    if frequency_split is None:
        raise ValueError("training.frequency_split должен быть заполнен")
    if split == "train":
        return frequency_split.x_train.index, frequency_split.y_train
    if split == "test":
        return frequency_split.x_test.index, frequency_split.y_test
    idx = frequency_split.x_train.index.union(frequency_split.x_test.index)
    y = pd.concat([frequency_split.y_train, frequency_split.y_test])
    return idx, y


def _fin_on_index(
    df: pd.DataFrame,
    training: TrainingArtifacts,
    idx: pd.Index,
    sev: np.ndarray,
    fin_config: FinEffectConfig,
    *,
    threshold: float | None,
) -> FinEffectResult:
    """Фин. эффект на произвольном индексе строк (один порог)."""
    from querulus.fin_effect.calculator import (
        _feature_rows_for_predict,
        _frequency_proba_from_training,
    )

    frame = df.loc[idx]
    predict_frame = _feature_rows_for_predict(training, idx, frame)
    freq_proba = _frequency_proba_from_training(
        training, predict_frame[training.frequency_features]
    )
    y_true = frame[fin_config.frequency_target_column]
    prepared = prepare_effect_frame(frame, fin_config)
    return apply_model_predictions(
        prepared,
        freq_proba,
        sev,
        y_true,
        threshold=threshold,
        config=fin_config,
    )


def run_fin_effect_with_severity_predictions(
    df: pd.DataFrame,
    training: TrainingArtifacts,
    y_pred_sev: np.ndarray | pd.Series,
    *,
    split: str = "test",
    threshold: float | None = None,
    config: FinEffectConfig | None = None,
    frequency_target_column: str | None = None,
) -> FinEffectResult:
    """Фин. эффект с подменой severity-предикта (freq — из ``training``)."""
    from querulus.fin_effect.calculator import _frequency_proba_from_training
    from querulus.fin_effect.calculator import _feature_rows_for_predict

    config = config or FinEffectConfig()
    effect_index, y_true_freq = _effect_index_and_freq(training, split)
    effect_frame = df.loc[effect_index]
    predict_frame = _feature_rows_for_predict(training, effect_index, effect_frame)
    freq_proba = _frequency_proba_from_training(
        training, predict_frame[training.frequency_features]
    )
    if frequency_target_column:
        y_true = effect_frame[frequency_target_column]
    else:
        y_true = y_true_freq
    prepared = prepare_effect_frame(effect_frame, config)
    return apply_model_predictions(
        prepared,
        freq_proba,
        np.asarray(y_pred_sev, dtype=float),
        y_true,
        threshold=threshold,
        config=config,
    )


def _value_mask(
    df: pd.DataFrame,
    index: pd.Index,
    value_column: str,
    value_threshold: float,
) -> pd.Series:
    values = pd.to_numeric(df.loc[index, value_column], errors="coerce")
    return values <= value_threshold


@dataclass(frozen=True)
class SeverityVariantResult:
    """Одна обученная severity + фин. эффект на полном test."""

    name: str
    transform: str
    sample_weight: str
    model: object
    fin_effect: FinEffectResult
    mae: float
    bias: float
    pred_sum: float
    true_sum: float


@dataclass(frozen=True)
class SeverityVariantsCompare:
    """Сравнительная таблица обученных severity-вариантов."""

    summary: pd.DataFrame
    variants: dict[str, SeverityVariantResult]


def compare_severity_fin_effect_variants(
    df: pd.DataFrame,
    training: TrainingArtifacts,
    fin_config: FinEffectConfig,
    training_config: TrainingConfig | None = None,
    *,
    split: str = "test",
    threshold: float | None = None,
) -> SeverityVariantsCompare:
    """Обучить raw / log1p / weighted severity и сравнить fin-effect + MAE."""
    training_config = training_config or TrainingConfig()
    effect_index, _ = _effect_index_and_freq(training, split)
    from querulus.fin_effect.calculator import _feature_rows_for_predict

    predict_frame = _feature_rows_for_predict(training, effect_index, df.loc[effect_index])
    features = predict_frame[training.severity_features]
    cat = training.severity_categorical_features
    y_true_sev = pd.to_numeric(
        df.loc[effect_index, fin_config.severity_target_column], errors="coerce"
    ).fillna(0.0)

    specs: list[tuple[str, SeverityTargetTransform, SeveritySampleWeight, object | None]] = [
        ("raw", "raw", "none", training.severity_model),
        ("log1p", "log1p", "none", None),
        ("weighted_sqrt", "raw", "sqrt", None),
        ("weighted_linear", "raw", "linear", None),
    ]
    variants: dict[str, SeverityVariantResult] = {}
    rows: list[dict[str, object]] = []
    for name, transform, weight_mode, preset_model in specs:
        model = preset_model or fit_severity_model(
            training,
            training_config,
            transform=transform,
            sample_weight=weight_mode,
        )
        sev_pred = severity_predict(model, features, cat, transform=transform)
        result = run_fin_effect_with_severity_predictions(
            df,
            training,
            sev_pred,
            split=split,
            threshold=threshold,
            config=fin_config,
        )
        err = sev_pred - y_true_sev.to_numpy(dtype=float)
        mae = float(np.nanmean(np.abs(err)))
        bias = float(np.nanmean(err))
        pred_sum = float(np.nansum(sev_pred))
        true_sum = float(np.nansum(y_true_sev))
        variants[name] = SeverityVariantResult(
            name=name,
            transform=transform,
            sample_weight=weight_mode,
            model=model,
            fin_effect=result,
            mae=mae,
            bias=bias,
            pred_sum=pred_sum,
            true_sum=true_sum,
        )
        rows.append(
            {
                "variant": name,
                "transform": transform,
                "sample_weight": weight_mode,
                "best_threshold": result.best_threshold,
                "net_effect": result.net_effect,
                "model_effect": result.model_effect_total,
                "fact_effect": result.fact_effect_total,
                "mae": mae,
                "bias": bias,
                "pred_sum": pred_sum,
                "true_sum": true_sum,
                "sum_bias": pred_sum - true_sum,
            }
        )
    return SeverityVariantsCompare(
        summary=plain_floats(pd.DataFrame(rows)),
        variants=variants,
    )


@dataclass(frozen=True)
class SegmentFinEffectCompare:
    """1:1 на сегменте ≤ threshold: общая severity vs severity, обученная на сегменте."""

    segment_mask_n: int
    value_column: str
    value_threshold: float
    full_on_segment: FinEffectResult
    segment_model_on_segment: FinEffectResult
    summary: pd.DataFrame


def compare_value_before_segment_strategies(
    df: pd.DataFrame,
    training: TrainingArtifacts,
    fin_config: FinEffectConfig,
    training_config: TrainingConfig | None = None,
    *,
    split: str = "test",
    value_column: str = "VALUE_BEFORE_WITH",
    value_threshold: float = 50_000.0,
    threshold: float | None = None,
    segment_transform: SeverityTargetTransform = "raw",
) -> SegmentFinEffectCompare:
    """Сравнить на одних и тех же строках ``value <= threshold``:

    - **full**: severity общей модели;
    - **segment_model**: severity, обученная только на train с ``value <= threshold``.

    Один и тот же freq + один порог → сравнение 1:1.
    """
    training_config = training_config or TrainingConfig()
    if training.severity_split is None:
        raise ValueError("severity_split отсутствует")

    effect_index, _ = _effect_index_and_freq(training, split)
    from querulus.fin_effect.calculator import _feature_rows_for_predict

    train_sev_idx = training.severity_split.x_train.index
    test_sev_idx = training.severity_split.x_test.index
    train_small = train_sev_idx[_value_mask(df, train_sev_idx, value_column, value_threshold)]
    test_small = test_sev_idx[_value_mask(df, test_sev_idx, value_column, value_threshold)]

    segment_model = fit_severity_model(
        training,
        training_config,
        transform=segment_transform,
        sample_weight="none",
        train_index=train_small,
        eval_index=test_small,
    )

    effect_small_mask = _value_mask(df, effect_index, value_column, value_threshold)
    seg_idx = effect_index[effect_small_mask.to_numpy()]
    segment_n = int(len(seg_idx))

    predict_frame = _feature_rows_for_predict(training, seg_idx, df.loc[seg_idx])
    feats = predict_frame[training.severity_features]
    cat = training.severity_categorical_features

    full_sev = severity_predict(
        training.severity_model,
        feats,
        cat,
        transform=getattr(training, "severity_target_transform", "raw"),
    )
    segment_sev = severity_predict(
        segment_model, feats, cat, transform=segment_transform
    )

    # Общий порог: от полной модели на полном test (если не задан явно).
    if threshold is None:
        from querulus.fin_effect.calculator import _feature_rows_for_predict as _fr

        all_frame = _fr(training, effect_index, df.loc[effect_index])
        full_all_sev = severity_predict(
            training.severity_model,
            all_frame[training.severity_features],
            cat,
            transform=getattr(training, "severity_target_transform", "raw"),
        )
        full_all = run_fin_effect_with_severity_predictions(
            df, training, full_all_sev, split=split, config=fin_config
        )
        use_threshold = full_all.best_threshold
    else:
        use_threshold = threshold

    full_on_seg = _fin_on_index(
        df, training, seg_idx, full_sev, fin_config, threshold=use_threshold
    )
    seg_on_seg = _fin_on_index(
        df, training, seg_idx, segment_sev, fin_config, threshold=use_threshold
    )

    err_full = full_sev - pd.to_numeric(
        df.loc[seg_idx, fin_config.severity_target_column], errors="coerce"
    ).fillna(0).to_numpy(dtype=float)
    err_seg = segment_sev - pd.to_numeric(
        df.loc[seg_idx, fin_config.severity_target_column], errors="coerce"
    ).fillna(0).to_numpy(dtype=float)

    scope = f"{value_column}<={value_threshold}"
    summary = plain_floats(
        pd.DataFrame(
            [
                {
                    "scope": scope,
                    "strategy": "full_model",
                    "n": segment_n,
                    "threshold": use_threshold,
                    "net_effect": full_on_seg.net_effect,
                    "model_effect": full_on_seg.model_effect_total,
                    "fact_effect": full_on_seg.fact_effect_total,
                    "mae": float(np.nanmean(np.abs(err_full))),
                    "bias": float(np.nanmean(err_full)),
                },
                {
                    "scope": scope,
                    "strategy": "segment_trained_model",
                    "n": segment_n,
                    "threshold": use_threshold,
                    "net_effect": seg_on_seg.net_effect,
                    "model_effect": seg_on_seg.model_effect_total,
                    "fact_effect": seg_on_seg.fact_effect_total,
                    "mae": float(np.nanmean(np.abs(err_seg))),
                    "bias": float(np.nanmean(err_seg)),
                },
            ]
        )
    )
    return SegmentFinEffectCompare(
        segment_mask_n=segment_n,
        value_column=value_column,
        value_threshold=value_threshold,
        full_on_segment=full_on_seg,
        segment_model_on_segment=seg_on_seg,
        summary=summary,
    )
