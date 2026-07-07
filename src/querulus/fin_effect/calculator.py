"""Расчёт финансового эффекта по факту и модели (Litigant fin_effect.py)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from querulus.fin_effect.config import FinEffectConfig

SplitName = Literal["train", "test", "all"]


@dataclass
class ThresholdMetrics:
    """Метрики одного порога классификации."""

    threshold: float
    net_effect: float
    total_model: float
    total_fact: float
    n_positive_preds: int
    n_actual_positive: int
    precision: float
    recall: float


@dataclass
class ThresholdStrategyResult:
    """Результат подбора порога по одной стратегии."""

    strategy: str
    threshold: float
    net_effect: float
    average_precision: float
    f1: float


@dataclass
class FinEffectResult:
    """Результат полного расчёта фин. эффекта."""

    frame: pd.DataFrame
    best_threshold: float
    threshold_metrics: dict[float, ThresholdMetrics]
    net_effect: float
    baseline_fact_total: float
    net_effect_vs_baseline: float
    threshold_strategies: dict[str, ThresholdStrategyResult] | None = None


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Числовая колонка или нули."""
    if column not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def payments_fee(row: pd.Series, config: FinEffectConfig) -> float:
    """Судебные взносы по строке (как payments_fee в Litigant)."""
    payments = 0.0
    if row.get(config.fu_recovery_column, 0) > 0:
        payments += config.fu_fee_amount
    if config.apply_court_fee and row.get(config.court_recovery_column, 0) > 0:
        payments += config.court_fee_amount
    return payments


def add_premiums_column(df: pd.DataFrame, config: FinEffectConfig | None = None) -> pd.Series:
    """Рассчитать колонку Взносы."""
    config = config or FinEffectConfig()
    return df.apply(lambda row: payments_fee(row, config), axis=1)


def compute_fin_effect_fact(df: pd.DataFrame, config: FinEffectConfig | None = None) -> pd.Series:
    """Фактический фин. эффект до финального negate."""
    config = config or FinEffectConfig()
    total = (
        _numeric_series(df, config.pretension_payments_column)
        + _numeric_series(df, config.fu_recovery_column)
        + _numeric_series(df, config.court_recovery_column)
        + _numeric_series(df, config.premiums_column)
    )
    if config.include_surcharge_in_fact:
        total = total + _numeric_series(df, config.surcharge_column) + _numeric_series(
            df, config.uts_surcharge_column
        )
    return total


def prepare_effect_frame(df: pd.DataFrame, config: FinEffectConfig | None = None) -> pd.DataFrame:
    """Подготовить _df_effect: fillna, взносы, fin_effect_fact."""
    config = config or FinEffectConfig()
    result = df.copy()

    for column in (
        config.pretension_payments_column,
        config.fu_recovery_column,
        config.court_recovery_column,
        config.surcharge_column,
        config.uts_surcharge_column,
    ):
        if column in result.columns:
            result[column] = _numeric_series(result, column)

    result[config.premiums_column] = add_premiums_column(result, config)
    result["fin_effect_fact"] = compute_fin_effect_fact(result, config)
    return result


def compute_fin_effect_model(
    pred_freq: np.ndarray,
    y_true_freq: np.ndarray,
    y_pred_sev: np.ndarray,
    y_true_sev: np.ndarray,
    base_sum: np.ndarray,
) -> np.ndarray:
    """Модельный фин. эффект по квадрантам confusion matrix."""
    pred_freq = np.asarray(pred_freq, dtype=int)
    y_true_freq = np.asarray(y_true_freq, dtype=int)
    y_pred_sev = np.asarray(y_pred_sev, dtype=float)
    y_true_sev = np.asarray(y_true_sev, dtype=float)
    base_sum = np.asarray(base_sum, dtype=float)

    fin_effect_model = np.zeros(len(base_sum), dtype=float)

    mask_00 = (pred_freq == 0) & (y_true_freq == 0)
    mask_01 = (pred_freq == 0) & (y_true_freq == 1)
    mask_10 = (pred_freq == 1) & (y_true_freq == 0)
    mask_11 = (pred_freq == 1) & (y_true_freq == 1)

    fin_effect_model[mask_00] = -base_sum[mask_00]
    fin_effect_model[mask_01] = -base_sum[mask_01]
    fin_effect_model[mask_10] = -y_pred_sev[mask_10] - base_sum[mask_10]

    mask_11_over = mask_11 & (y_pred_sev >= y_true_sev)
    mask_11_under = mask_11 & (y_pred_sev < y_true_sev)
    fin_effect_model[mask_11_over] = -y_pred_sev[mask_11_over]
    fin_effect_model[mask_11_under] = -base_sum[mask_11_under]
    return fin_effect_model


def _threshold_grid(config: FinEffectConfig) -> np.ndarray:
    """Сетка порогов для подбора."""
    return np.arange(config.threshold_start, config.threshold_stop, config.threshold_step)


def evaluate_threshold(
    threshold: float,
    y_proba_freq: np.ndarray,
    y_true_freq: np.ndarray,
    y_pred_sev: np.ndarray,
    y_true_sev: np.ndarray,
    base_sum: np.ndarray,
) -> ThresholdMetrics:
    """Метрики фин. эффекта для одного порога."""
    pred_freq = (np.asarray(y_proba_freq) >= threshold).astype(int)
    y_true_freq = np.asarray(y_true_freq, dtype=int)
    fin_effect_model = compute_fin_effect_model(
        pred_freq, y_true_freq, y_pred_sev, y_true_sev, base_sum
    )
    total_effect_model = float(fin_effect_model.sum())
    total_effect_fact = float(np.asarray(base_sum, dtype=float).sum())
    net_effect = total_effect_model - (-total_effect_fact)

    tp = int(np.sum((pred_freq == 1) & (y_true_freq == 1)))
    n_pred = int(pred_freq.sum())
    n_actual = int(y_true_freq.sum())
    precision = tp / max(n_pred, 1)
    recall = tp / max(n_actual, 1)

    return ThresholdMetrics(
        threshold=round(float(threshold), 2),
        net_effect=net_effect,
        total_model=total_effect_model,
        total_fact=total_effect_fact,
        n_positive_preds=n_pred,
        n_actual_positive=n_actual,
        precision=precision,
        recall=recall,
    )


def search_best_threshold(
    y_proba_freq: np.ndarray,
    y_true_freq: np.ndarray,
    y_pred_sev: np.ndarray,
    y_true_sev: np.ndarray,
    base_sum: np.ndarray,
    config: FinEffectConfig | None = None,
) -> tuple[float, dict[float, ThresholdMetrics]]:
    """Подбор порога по максимальному чистому фин. эффекту."""
    config = config or FinEffectConfig()
    results: dict[float, ThresholdMetrics] = {}
    for threshold in _threshold_grid(config):
        metrics = evaluate_threshold(
            threshold, y_proba_freq, y_true_freq, y_pred_sev, y_true_sev, base_sum
        )
        results[metrics.threshold] = metrics
    best_threshold = max(results, key=lambda key: results[key].net_effect)
    return best_threshold, results


def _baseline_fact_total(base_sum: np.ndarray) -> float:
    """Эталон: суммарный fin_effect_fact (текущий бизнес-процесс)."""
    return float(np.asarray(base_sum, dtype=float).sum())


def _f1_score(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def search_best_threshold_by_f1(
    y_proba_freq: np.ndarray,
    y_true_freq: np.ndarray,
    y_pred_sev: np.ndarray,
    y_true_sev: np.ndarray,
    base_sum: np.ndarray,
    config: FinEffectConfig | None = None,
) -> tuple[float, dict[float, ThresholdMetrics]]:
    """Подбор порога по максимальному F1 на сетке."""
    config = config or FinEffectConfig()
    best_threshold = float(config.threshold_start)
    best_f1 = -1.0
    results: dict[float, ThresholdMetrics] = {}
    for threshold in _threshold_grid(config):
        metrics = evaluate_threshold(
            threshold, y_proba_freq, y_true_freq, y_pred_sev, y_true_sev, base_sum
        )
        results[metrics.threshold] = metrics
        f1 = _f1_score(metrics.precision, metrics.recall)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = metrics.threshold
    return best_threshold, results


def search_threshold_strategies(
    y_proba_freq: np.ndarray,
    y_true_freq: np.ndarray,
    y_pred_sev: np.ndarray,
    y_true_sev: np.ndarray,
    base_sum: np.ndarray,
    config: FinEffectConfig | None = None,
) -> dict[str, ThresholdStrategyResult]:
    """Сравнить пороги: net_effect (primary), F1, average precision."""
    from sklearn.metrics import average_precision_score

    config = config or FinEffectConfig()
    ap = float(average_precision_score(y_true_freq, y_proba_freq))

    best_net, net_metrics = search_best_threshold(
        y_proba_freq, y_true_freq, y_pred_sev, y_true_sev, base_sum, config
    )
    best_f1, f1_metrics = search_best_threshold_by_f1(
        y_proba_freq, y_true_freq, y_pred_sev, y_true_sev, base_sum, config
    )

    strategies: dict[str, ThresholdStrategyResult] = {}
    for name, threshold, metrics_map in (
        ("best_net_effect", best_net, net_metrics),
        ("best_f1", best_f1, f1_metrics),
    ):
        metrics = metrics_map[round(float(threshold), 2)]
        strategies[name] = ThresholdStrategyResult(
            strategy=name,
            threshold=float(threshold),
            net_effect=metrics.net_effect,
            average_precision=ap,
            f1=_f1_score(metrics.precision, metrics.recall),
        )
    strategies["average_precision"] = ThresholdStrategyResult(
        strategy="average_precision",
        threshold=float(best_f1),
        net_effect=f1_metrics[round(float(best_f1), 2)].net_effect,
        average_precision=ap,
        f1=strategies["best_f1"].f1,
    )
    return strategies


def apply_model_predictions(
    effect_df: pd.DataFrame,
    y_proba_freq: np.ndarray | pd.Series,
    y_pred_sev: np.ndarray | pd.Series,
    y_true_freq: np.ndarray | pd.Series,
    *,
    threshold: float | None = None,
    config: FinEffectConfig | None = None,
) -> FinEffectResult:
    """Добавить pred_freq, pred_sev, fin_effect_model; подобрать порог при необходимости."""
    config = config or FinEffectConfig()
    frame = effect_df.copy()
    y_true_sev = _numeric_series(frame, config.severity_target_column).to_numpy()
    base_sum = frame["fin_effect_fact"].to_numpy()
    y_true_freq_arr = np.asarray(y_true_freq, dtype=int)
    y_proba_arr = np.asarray(y_proba_freq, dtype=float)
    y_pred_sev_arr = np.asarray(y_pred_sev, dtype=float)
    baseline_total = _baseline_fact_total(base_sum)
    threshold_strategies = search_threshold_strategies(
        y_proba_arr,
        y_true_freq_arr,
        y_pred_sev_arr,
        y_true_sev,
        base_sum,
        config,
    )

    if threshold is None:
        best_threshold, threshold_metrics = search_best_threshold(
            y_proba_arr,
            y_true_freq_arr,
            y_pred_sev_arr,
            y_true_sev,
            base_sum,
            config,
        )
    else:
        best_threshold = float(threshold)
        threshold_metrics = {
            round(best_threshold, 2): evaluate_threshold(
                best_threshold,
                y_proba_arr,
                y_true_freq_arr,
                y_pred_sev_arr,
                y_true_sev,
                base_sum,
            )
        }

    pred_freq = (y_proba_arr >= best_threshold).astype(int)
    fin_effect_model = compute_fin_effect_model(
        pred_freq, y_true_freq_arr, y_pred_sev_arr, y_true_sev, base_sum
    )

    frame["pred_freq"] = pred_freq
    frame["pred_sev"] = y_pred_sev_arr
    frame["fin_effect_model"] = fin_effect_model
    # negate — только для отчёта; net_effect считаем до инверсии (как в Litigant)
    net_effect = float(frame["fin_effect_model"].sum() - (-frame["fin_effect_fact"].sum()))
    if config.negate_fact_for_report:
        frame["fin_effect_fact"] = -frame["fin_effect_fact"]
    return FinEffectResult(
        frame=frame,
        best_threshold=best_threshold,
        threshold_metrics=threshold_metrics,
        net_effect=net_effect,
        baseline_fact_total=baseline_total,
        net_effect_vs_baseline=net_effect - baseline_total,
        threshold_strategies=threshold_strategies,
    )


def _feature_rows_for_predict(
    training: object,
    effect_index: pd.Index,
    effect_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Строки с признаками после AutoMVP (как при обучении CatBoost)."""
    feature_frame = getattr(training, "feature_frame", None)
    if feature_frame is not None:
        return feature_frame.loc[effect_index]
    from querulus.training.pipeline import _stringify_categorical_columns

    cat_features = getattr(training, "severity_categorical_features", []) + getattr(
        training, "frequency_categorical_features", []
    )
    cat_features = list(dict.fromkeys(cat_features))
    return _stringify_categorical_columns(effect_frame, cat_features)


def _frequency_proba_from_training(training: object, features: pd.DataFrame) -> np.ndarray:
    """Вероятность ПСР с учётом калибратора, если он есть."""
    calibrator = getattr(training, "frequency_calibrator", None)
    if calibrator is not None:
        return np.asarray(calibrator.predict_proba(features)[:, 1], dtype=float)
    return _catboost_predict_proba(
        training.frequency_model,
        features,
        getattr(training, "frequency_categorical_features", []),
    )


def _catboost_predict(
    model: object,
    features: pd.DataFrame,
    cat_features: list[str],
) -> np.ndarray:
    """predict с явным Pool для категориальных признаков."""
    cat_features = [column for column in cat_features if column in features.columns]
    if cat_features:
        from catboost import Pool

        pool = Pool(features, cat_features=cat_features)
        return np.asarray(model.predict(pool), dtype=float)
    return np.asarray(model.predict(features), dtype=float)


def _catboost_predict_proba(
    model: object,
    features: pd.DataFrame,
    cat_features: list[str],
) -> np.ndarray:
    """predict_proba[:, 1] с явным Pool для категориальных признаков."""
    cat_features = [column for column in cat_features if column in features.columns]
    if cat_features:
        from catboost import Pool

        pool = Pool(features, cat_features=cat_features)
        return np.asarray(model.predict_proba(pool)[:, 1], dtype=float)
    return np.asarray(model.predict_proba(features)[:, 1], dtype=float)


def run_fin_effect_pipeline(
    df: pd.DataFrame,
    frequency_proba: np.ndarray | pd.Series,
    severity_prediction: np.ndarray | pd.Series,
    y_true_freq: np.ndarray | pd.Series,
    *,
    threshold: float | None = None,
    config: FinEffectConfig | None = None,
) -> FinEffectResult:
    """Полный пайплайн: prepare_effect_frame → подбор порога → fin_effect_model."""
    config = config or FinEffectConfig()
    prepared = prepare_effect_frame(df, config)
    return apply_model_predictions(
        prepared,
        frequency_proba,
        severity_prediction,
        y_true_freq,
        threshold=threshold,
        config=config,
    )


def run_fin_effect_from_training(
    df: pd.DataFrame,
    training: object,
    *,
    split: SplitName = "test",
    frequency_target_column: str | None = None,
    threshold: float | None = None,
    config: FinEffectConfig | None = None,
) -> FinEffectResult:
    """Расчёт на сплите из TrainingArtifacts (модели frequency + severity).

    Как в Litigant: база — все строки frequency test (``X_test_freq``),
    severity предсказывается на тех же строках, а не на severity_split.
    """
    config = config or FinEffectConfig()
    frequency_split = getattr(training, "frequency_split", None)
    if frequency_split is None:
        raise ValueError("training.frequency_split должен быть заполнен")

    freq_features = training.frequency_features
    sev_features = training.severity_features

    if split == "train":
        effect_index = frequency_split.x_train.index
        y_true_freq = frequency_split.y_train
    elif split == "test":
        effect_index = frequency_split.x_test.index
        y_true_freq = frequency_split.y_test
    else:
        effect_index = frequency_split.x_train.index.union(frequency_split.x_test.index)
        y_true_freq = pd.concat([frequency_split.y_train, frequency_split.y_test])

    effect_frame = df.loc[effect_index]
    predict_frame = _feature_rows_for_predict(training, effect_index, effect_frame)

    freq_proba = pd.Series(
        _frequency_proba_from_training(training, predict_frame[freq_features]),
        index=effect_index,
    )
    sev_pred = pd.Series(
        _catboost_predict(
            training.severity_model,
            predict_frame[sev_features],
            getattr(training, "severity_categorical_features", []),
        ),
        index=effect_index,
    )

    if frequency_target_column:
        y_true = effect_frame[frequency_target_column]
    else:
        y_true = y_true_freq

    return run_fin_effect_pipeline(
        effect_frame,
        freq_proba,
        sev_pred,
        y_true,
        threshold=threshold,
        config=config,
    )


def print_best_threshold_report(result: FinEffectResult) -> None:
    """Вывод оптимального порога и чистого эффекта (как в Litigant fin_effect.py)."""
    print("\n" + "=" * 70)
    print("ОПТИМАЛЬНЫЙ ПОРОГ КЛАССИФИКАЦИИ")
    print("=" * 70)
    print(f"Порог вероятности (net_effect): {result.best_threshold:.2f}")
    print(f"Чистый финансовый эффект   : {result.net_effect:,.2f} ₽")
    print(f"Baseline fin_effect_fact   : {result.baseline_fact_total:,.2f} ₽")
    print(f"Δ net_effect vs baseline   : {result.net_effect_vs_baseline:,.2f} ₽")
    if result.threshold_strategies:
        print("\nСравнение стратегий порога:")
        for strategy in result.threshold_strategies.values():
            print(
                f"  {strategy.strategy:20s} threshold={strategy.threshold:.2f} "
                f"net_effect={strategy.net_effect:,.0f} F1={strategy.f1:.3f} AP={strategy.average_precision:.3f}"
            )


def prepare_analytics_export(
    df: pd.DataFrame,
    config: FinEffectConfig | None = None,
    *,
    rename: bool = True,
) -> pd.DataFrame:
    """Таблица для Excel с человекочитаемыми заголовками."""
    from querulus.fin_effect.config import ANALYTICS_RENAME_DICT

    config = config or FinEffectConfig()
    columns = [column for column in config.export_columns if column in df.columns]
    export_df = df[columns].copy()
    if rename:
        export_df = export_df.rename(columns=ANALYTICS_RENAME_DICT)
    return export_df
