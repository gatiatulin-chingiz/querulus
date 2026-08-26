"""Расчёт финансового эффекта по факту и модели (Litigant fin_effect.py).

Контракт (единственная точка правды)
------------------------------------
1. ``prepare_effect_frame`` → колонка ``fin_effect_fact`` (пока **положительная**
   величина расхода / базы).
2. ``apply_model_predictions`` / ``run_fin_effect_pipeline``:
   - выравнивает ``proba`` / ``sev`` / ``y_true`` на ``frame.index``
     (Series с чужим index → reindex; строки без pred отбрасываются);
   - пишет ``pred_freq``, ``pred_sev``, ``fin_effect_model``;
   - при ``negate_fact_for_report`` переводит ``fin_effect_fact`` в знак «расход < 0»;
   - пишет ``fin_effect_economy`` универсально:
     ``(−fact) − (−model)`` = расход_факт − расход_модель
     (эквивалент ``model − fact`` при эффектах ≤ 0; ≡ ``net_effect``).
3. ``create_summary_table`` / ``FinEffectResult.summary_table`` **только суммируют**
   колонки кадра по квадрантам fact×pred. Никакого второго расчёта факта.

Итоги: ``sum(model)``, ``sum(fact)``, ``sum(economy)`` ≡ поля ``FinEffectResult``.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Literal

import numpy as np
import pandas as pd

from querulus.fin_effect.config import FinEffectConfig
from querulus.training.severity_training import severity_predict

SplitName = Literal["train", "test", "all"]
logger = logging.getLogger("querulus.fin_effect")


def economy_from_signed_effects(
    fin_effect_fact: np.ndarray | pd.Series | float,
    fin_effect_model: np.ndarray | pd.Series | float,
) -> np.ndarray:
    """Универсальная экономия при знаковой конвенции «фин. эффект ≤ 0 = расход».

    Считаем в шкале положительных расходов::

        расход_факт   = −fin_effect_fact
        расход_модель = −fin_effect_model
        экономия      = расход_факт − расход_модель

    Примеры (одинаковая формула):
    - 1–1: fact=−1000, model=−400 → economy = 1000 − 400 = +600 (сэкономили);
    - 0–1: fact=0, model=−200 → economy = 0 − 200 = −200 (ложный штраф).

    Алгебраически то же, что ``model − fact``, без «минус на минус» в интерпретации.
    """
    fact = np.asarray(fin_effect_fact, dtype=float)
    model = np.asarray(fin_effect_model, dtype=float)
    cost_fact = -fact
    cost_model = -model
    return cost_fact - cost_model


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
    model_effect_total: float
    fact_effect_total: float
    threshold_strategies: dict[str, ThresholdStrategyResult] | None = None

    def summary_table(self, config: FinEffectConfig | None = None) -> pd.DataFrame:
        """Сводка по квадрантам: только агрегация колонок ``frame``."""
        from querulus.fin_effect.summary import create_summary_table

        return create_summary_table(self.frame, config)

def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Числовая колонка или нули."""
    if column not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def align_effect_inputs(
    frame: pd.DataFrame,
    frequency_proba: np.ndarray | pd.Series,
    severity_prediction: np.ndarray | pd.Series,
    y_true_freq: np.ndarray | pd.Series,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Выровнять pred/y_true на ``frame.index``; отбросить строки без предсказаний.

    Единый контракт для pipeline / training / OutBoxML parity: дальше все формулы
    работают только с массивами той же длины и в том же порядке, что ``frame``.
    """
    index = frame.index

    def _to_series(values: np.ndarray | pd.Series, name: str) -> pd.Series:
        if isinstance(values, pd.Series):
            return pd.to_numeric(values, errors="coerce")
        arr = np.asarray(values, dtype=float)
        if len(arr) != len(index):
            raise ValueError(
                f"{name}: len={len(arr)} != frame len={len(index)}. "
                "Передайте Series с index строк df или ndarray той же длины."
            )
        return pd.Series(arr, index=index, dtype=float)

    proba = _to_series(frequency_proba, "frequency_proba").reindex(index)
    sev = _to_series(severity_prediction, "severity_prediction").reindex(index)
    if isinstance(y_true_freq, pd.Series):
        y_true = pd.to_numeric(y_true_freq, errors="coerce").reindex(index)
    else:
        y_arr = np.asarray(y_true_freq)
        if len(y_arr) != len(index):
            raise ValueError(
                f"y_true_freq: len={len(y_arr)} != frame len={len(index)}"
            )
        y_true = pd.Series(y_arr, index=index)

    valid = proba.notna() & sev.notna() & y_true.notna()
    if not bool(valid.any()):
        raise ValueError(
            "После выравнивания на frame.index нет строк с proba/sev/y_true. "
            "Индекс предсказаний должен совпадать с индексом df "
            "(частая ошибка: X после preprocessor с другим index)."
        )
    n_drop = int((~valid).sum())
    if n_drop:
        logger.warning(
            "fin_effect: отброшено %s/%s строк без выровненных proba/sev/y_true",
            n_drop,
            len(index),
        )
    out = frame.loc[valid].copy()
    return (
        out,
        proba.loc[valid].to_numpy(dtype=float),
        sev.loc[valid].to_numpy(dtype=float),
        y_true.loc[valid].to_numpy(dtype=float).astype(int),
    )


def _fee_base_amount(row: pd.Series, config: FinEffectConfig) -> float:
    """Денежная база факта, к которой привязывается взнос ФУ."""
    if config.uses_legacy_psr_fact:
        return float(
            float(row.get(config.pretension_payments_column, 0) or 0)
            + float(row.get(config.fu_recovery_column, 0) or 0)
            + float(row.get(config.court_recovery_column, 0) or 0)
        )
    return float(row.get(config.fact_amount_column, 0) or 0)


def payments_fee(row: pd.Series, config: FinEffectConfig) -> float:
    """Судебные взносы: ФУ — boolean-триггер из ПСР; суд — по исковой сумме icnl.

    Взнос ФУ только при ненулевой базе факта. Иначе в icnl (TARGET_FREQ без ФУ
    в amount) взносы «сироты» попадают в квадранты fact=0.
    """
    payments = 0.0
    fu_trigger = float(row.get(config.fu_fee_trigger_column, 0) or 0) > 0
    if fu_trigger and _fee_base_amount(row, config) > 0:
        payments += config.fu_fee_amount
    claims_amount = float(row.get(config.freq_claims_amount_column, 0) or 0)
    if config.apply_court_fee and claims_amount > 0:
        payments += config.court_fee_amount
    return payments


def add_premiums_column(df: pd.DataFrame, config: FinEffectConfig | None = None) -> pd.Series:
    """Рассчитать колонку Взносы."""
    config = config or FinEffectConfig()
    return df.apply(lambda row: payments_fee(row, config), axis=1)


def compute_fin_effect_fact(df: pd.DataFrame, config: FinEffectConfig | None = None) -> pd.Series:
    """Фактический фин. эффект: icnl (TARGET_FREQ_AMOUNT) или legacy ПСР."""
    config = config or FinEffectConfig()
    premiums = _numeric_series(df, config.premiums_column)
    if config.uses_legacy_psr_fact:
        return (
            _numeric_series(df, config.pretension_payments_column)
            + _numeric_series(df, config.fu_recovery_column)
            + _numeric_series(df, config.court_recovery_column)
            + premiums
        )
    return _numeric_series(df, config.fact_amount_column) + premiums


def prepare_effect_frame(df: pd.DataFrame, config: FinEffectConfig | None = None) -> pd.DataFrame:
    """Подготовить _df_effect: fillna, взносы, fin_effect_fact."""
    config = config or FinEffectConfig()
    result = df.copy()

    for column in config.fill_zero_columns:
        if column in result.columns:
            result[column] = _numeric_series(result, column)

    result[config.premiums_column] = add_premiums_column(result, config)
    result["fin_effect_fact"] = compute_fin_effect_fact(result, config)
    return result


def compute_fin_effect_model_legacy(
    pred_freq: np.ndarray,
    y_true_freq: np.ndarray,
    y_pred_sev: np.ndarray,
    y_true_sev: np.ndarray,
    base_sum: np.ndarray,
) -> np.ndarray:
    """Старые квадранты Litigant (режим legacy_psr и сравнение формул)."""
    pred_freq = np.asarray(pred_freq, dtype=int)
    y_true_freq = np.asarray(y_true_freq, dtype=int)
    y_pred_sev = np.nan_to_num(np.asarray(y_pred_sev, dtype=float), nan=0.0)
    y_true_sev = np.nan_to_num(np.asarray(y_true_sev, dtype=float), nan=0.0)
    base_sum = np.nan_to_num(np.asarray(base_sum, dtype=float), nan=0.0)

    fin_effect_model = np.zeros(len(base_sum), dtype=float)
    # Имена mask_XY: X=pred, Y=fact (как в Litigant).
    mask_00 = (pred_freq == 0) & (y_true_freq == 0)
    mask_01 = (pred_freq == 0) & (y_true_freq == 1)  # пропуск
    mask_10 = (pred_freq == 1) & (y_true_freq == 0)  # ложная тревога
    mask_11 = (pred_freq == 1) & (y_true_freq == 1)
    fin_effect_model[mask_00] = -base_sum[mask_00]
    fin_effect_model[mask_01] = -base_sum[mask_01]
    fin_effect_model[mask_10] = -y_pred_sev[mask_10] - base_sum[mask_10]
    mask_11_over = mask_11 & (y_pred_sev >= y_true_sev)
    mask_11_under = mask_11 & (y_pred_sev < y_true_sev)
    fin_effect_model[mask_11_over] = -y_pred_sev[mask_11_over]
    fin_effect_model[mask_11_under] = -base_sum[mask_11_under]
    return fin_effect_model


def compute_fin_effect_model_coverage(
    pred_freq: np.ndarray,
    y_true_freq: np.ndarray,
    y_pred_sev: np.ndarray,
    y_true_sev: np.ndarray,
    psr: np.ndarray,
    premiums: np.ndarray,
) -> np.ndarray:
    """Новые квадранты (расход отрицательный).

    Порядок в комментарии — fact, pred (как маски ниже):
    fact0 pred0 → 0;
    fact0 pred1 (ложная тревога) → −pred_sev;
    fact1 pred0 (пропуск) → −(ПСР+взносы);
    fact1 pred1 хватило → −pred_sev;
    fact1 pred1 не хватило → −(ПСР×(1−pred_sev/T)+взносы); при T=0 — как пропуск.
    """
    pred_freq = np.asarray(pred_freq, dtype=int)
    y_true_freq = np.asarray(y_true_freq, dtype=int)
    y_pred_sev = np.nan_to_num(np.asarray(y_pred_sev, dtype=float), nan=0.0)
    y_true_sev = np.nan_to_num(np.asarray(y_true_sev, dtype=float), nan=0.0)
    psr = np.maximum(np.nan_to_num(np.asarray(psr, dtype=float), nan=0.0), 0.0)
    premiums = np.maximum(np.nan_to_num(np.asarray(premiums, dtype=float), nan=0.0), 0.0)
    fact = psr + premiums

    out = np.zeros(len(fact), dtype=float)
    # Маски: fact & pred (не путать с legacy mask_XY, где X=pred, Y=fact).
    m00 = (y_true_freq == 0) & (pred_freq == 0)
    m01 = (y_true_freq == 0) & (pred_freq == 1)  # ложная тревога
    m10 = (y_true_freq == 1) & (pred_freq == 0)  # пропуск
    m11 = (y_true_freq == 1) & (pred_freq == 1)
    covered = m11 & (y_pred_sev >= y_true_sev)
    short = m11 & ~covered

    out[m00] = 0.0
    out[m01] = -y_pred_sev[m01]
    out[m10] = -fact[m10]
    out[covered] = -y_pred_sev[covered]
    share = np.zeros(len(fact), dtype=float)
    need = short & (y_true_sev > 0)
    share[need] = np.clip(y_pred_sev[need] / y_true_sev[need], 0.0, 1.0)
    out[short] = -(psr[short] * (1.0 - share[short]) + premiums[short])
    return out


def compute_fin_effect_model(
    pred_freq: np.ndarray,
    y_true_freq: np.ndarray,
    y_pred_sev: np.ndarray,
    y_true_sev: np.ndarray,
    base_sum: np.ndarray,
    *,
    formula: str = "coverage",
    psr: np.ndarray | None = None,
    premiums: np.ndarray | None = None,
) -> np.ndarray:
    """Модельный фин. эффект. ``legacy`` — старые квадранты; иначе покрытие."""
    if formula == "legacy":
        return compute_fin_effect_model_legacy(
            pred_freq, y_true_freq, y_pred_sev, y_true_sev, base_sum
        )
    psr_arr = base_sum if psr is None else psr
    prem_arr = np.zeros(len(np.asarray(base_sum)), dtype=float) if premiums is None else premiums
    return compute_fin_effect_model_coverage(
        pred_freq, y_true_freq, y_pred_sev, y_true_sev, psr_arr, prem_arr
    )


def _formula_from_config(config: FinEffectConfig | None) -> str:
    if config is not None and config.uses_legacy_psr_fact:
        return "legacy"
    return "coverage"


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
    *,
    formula: str = "coverage",
    psr: np.ndarray | None = None,
    premiums: np.ndarray | None = None,
) -> ThresholdMetrics:
    """Метрики фин. эффекта для одного порога."""
    pred_freq = (np.asarray(y_proba_freq) >= threshold).astype(int)
    y_true_freq = np.asarray(y_true_freq, dtype=int)
    fin_effect_model = compute_fin_effect_model(
        pred_freq,
        y_true_freq,
        y_pred_sev,
        y_true_sev,
        base_sum,
        formula=formula,
        psr=psr,
        premiums=premiums,
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
    *,
    formula: str | None = None,
    psr: np.ndarray | None = None,
    premiums: np.ndarray | None = None,
) -> tuple[float, dict[float, ThresholdMetrics]]:
    """Подбор порога по максимальному чистому фин. эффекту."""
    config = config or FinEffectConfig()
    formula = formula or _formula_from_config(config)
    results: dict[float, ThresholdMetrics] = {}
    for threshold in _threshold_grid(config):
        metrics = evaluate_threshold(
            threshold,
            y_proba_freq,
            y_true_freq,
            y_pred_sev,
            y_true_sev,
            base_sum,
            formula=formula,
            psr=psr,
            premiums=premiums,
        )
        results[metrics.threshold] = metrics
    best_threshold = max(results, key=lambda key: results[key].net_effect)
    return best_threshold, results


def _signed_effect_totals(
    fin_effect_model: np.ndarray,
    fact_before_negate: np.ndarray,
) -> tuple[float, float, float]:
    """Итоги в знаковой конвенции Litigant: факт и модель отрицательные, net — экономия."""
    model_total = float(np.asarray(fin_effect_model, dtype=float).sum())
    fact_positive = float(np.asarray(fact_before_negate, dtype=float).sum())
    fact_signed = -fact_positive
    net_effect = model_total - fact_signed
    return model_total, fact_signed, net_effect


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
    *,
    formula: str | None = None,
    psr: np.ndarray | None = None,
    premiums: np.ndarray | None = None,
) -> tuple[float, dict[float, ThresholdMetrics]]:
    """Подбор порога по максимальному F1 на сетке."""
    config = config or FinEffectConfig()
    formula = formula or _formula_from_config(config)
    best_threshold = float(config.threshold_start)
    best_f1 = -1.0
    results: dict[float, ThresholdMetrics] = {}
    for threshold in _threshold_grid(config):
        metrics = evaluate_threshold(
            threshold,
            y_proba_freq,
            y_true_freq,
            y_pred_sev,
            y_true_sev,
            base_sum,
            formula=formula,
            psr=psr,
            premiums=premiums,
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
    *,
    formula: str | None = None,
    psr: np.ndarray | None = None,
    premiums: np.ndarray | None = None,
) -> dict[str, ThresholdStrategyResult]:
    """Сравнить пороги: ``best_net_effect`` и ``pr_auc`` (порог по F1 на PR)."""
    from sklearn.metrics import average_precision_score

    config = config or FinEffectConfig()
    formula = formula or _formula_from_config(config)
    ap = float(average_precision_score(y_true_freq, y_proba_freq))

    best_net, net_metrics = search_best_threshold(
        y_proba_freq,
        y_true_freq,
        y_pred_sev,
        y_true_sev,
        base_sum,
        config,
        formula=formula,
        psr=psr,
        premiums=premiums,
    )
    best_pr, pr_metrics = search_best_threshold_by_f1(
        y_proba_freq,
        y_true_freq,
        y_pred_sev,
        y_true_sev,
        base_sum,
        config,
        formula=formula,
        psr=psr,
        premiums=premiums,
    )

    strategies: dict[str, ThresholdStrategyResult] = {}
    for name, threshold, metrics_map in (
        ("best_net_effect", best_net, net_metrics),
        ("pr_auc", best_pr, pr_metrics),
    ):
        metrics = metrics_map[round(float(threshold), 2)]
        strategies[name] = ThresholdStrategyResult(
            strategy=name,
            threshold=float(threshold),
            net_effect=metrics.net_effect,
            average_precision=ap,
            f1=_f1_score(metrics.precision, metrics.recall),
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
    """Добавить pred_freq, pred_sev, fin_effect_*; подобрать порог при необходимости.

    Pred/y_true выравниваются на индекс кадра через ``align_effect_inputs``.
    """
    config = config or FinEffectConfig()
    frame, y_proba_arr, y_pred_sev_arr, y_true_freq_arr = align_effect_inputs(
        effect_df,
        y_proba_freq,
        y_pred_sev,
        y_true_freq,
    )
    y_true_sev = _numeric_series(frame, config.severity_target_column).to_numpy()
    base_sum = frame["fin_effect_fact"].to_numpy()
    formula = _formula_from_config(config)
    psr = _numeric_series(frame, config.fact_amount_column).to_numpy()
    premiums = _numeric_series(frame, config.premiums_column).to_numpy()
    threshold_strategies = search_threshold_strategies(
        y_proba_arr,
        y_true_freq_arr,
        y_pred_sev_arr,
        y_true_sev,
        base_sum,
        config,
        formula=formula,
        psr=psr,
        premiums=premiums,
    )

    if threshold is None:
        best_threshold, threshold_metrics = search_best_threshold(
            y_proba_arr,
            y_true_freq_arr,
            y_pred_sev_arr,
            y_true_sev,
            base_sum,
            config,
            formula=formula,
            psr=psr,
            premiums=premiums,
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
                formula=formula,
                psr=psr,
                premiums=premiums,
            )
        }

    pred_freq = (y_proba_arr >= best_threshold).astype(int)
    fin_effect_model = compute_fin_effect_model(
        pred_freq,
        y_true_freq_arr,
        y_pred_sev_arr,
        y_true_sev,
        base_sum,
        formula=formula,
        psr=psr,
        premiums=premiums,
    )

    frame["pred_freq"] = pred_freq
    frame["pred_sev"] = y_pred_sev_arr
    frame["fin_effect_model"] = fin_effect_model
    # Явный y_true, на котором считались формулы (после align) — для сводки.
    frame["fin_effect_y_true"] = y_true_freq_arr
    model_total, fact_signed, net_effect = _signed_effect_totals(
        fin_effect_model,
        frame["fin_effect_fact"].to_numpy(),
    )
    if config.negate_fact_for_report:
        frame["fin_effect_fact"] = -frame["fin_effect_fact"]
    frame["fin_effect_economy"] = economy_from_signed_effects(
        frame["fin_effect_fact"],
        frame["fin_effect_model"],
    )
    return FinEffectResult(
        frame=frame,
        best_threshold=best_threshold,
        threshold_metrics=threshold_metrics,
        net_effect=net_effect,
        model_effect_total=model_total,
        fact_effect_total=fact_signed,
        threshold_strategies=threshold_strategies,
    )


def recompute_fin_effect_model(
    frame: pd.DataFrame,
    config: FinEffectConfig | None = None,
    *,
    formula: str,
) -> np.ndarray:
    """Пересчитать ``fin_effect_model`` по уже посчитанным pred_* (сравнение формул)."""
    config = config or FinEffectConfig()
    pred_freq = pd.to_numeric(frame["pred_freq"], errors="coerce").fillna(0).to_numpy()
    y_true = pd.to_numeric(
        frame[config.frequency_target_column], errors="coerce"
    ).fillna(0).to_numpy()
    pred_sev = pd.to_numeric(frame["pred_sev"], errors="coerce").fillna(0).to_numpy()
    y_sev = _numeric_series(frame, config.severity_target_column).to_numpy()
    psr = _numeric_series(frame, config.fact_amount_column).to_numpy()
    premiums = _numeric_series(frame, config.premiums_column).to_numpy()
    base = psr + premiums
    return compute_fin_effect_model(
        pred_freq,
        y_true,
        pred_sev,
        y_sev,
        base,
        formula=formula,
        psr=psr,
        premiums=premiums,
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
    """Вероятность frequency с учётом калибратора, если он есть."""
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
    effect_index: pd.Index | None = None,
    frequency_target_column: str | None = None,
    threshold: float | None = None,
    config: FinEffectConfig | None = None,
) -> FinEffectResult:
    """Расчёт на сплите из TrainingArtifacts (модели frequency + severity).

    Как в Litigant: база — все строки frequency test (``X_test_freq``),
    severity предсказывается на тех же строках, а не на severity_split.

    ``effect_index`` — явный набор строк (holdout Test блока B и т.п.).
    Если задан, ``split`` игнорируется; ``y_true`` — из ``frequency_target_column``
    или ``config.frequency_target_column`` / колонки в ``df``.
    """
    config = config or FinEffectConfig()
    frequency_split = getattr(training, "frequency_split", None)
    if effect_index is None and frequency_split is None:
        raise ValueError("training.frequency_split должен быть заполнен")

    freq_features = training.frequency_features
    sev_features = training.severity_features

    y_true_freq: pd.Series | None = None
    if effect_index is None:
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
    sev_raw = severity_predict(
        training.severity_model,
        predict_frame[sev_features],
        getattr(training, "severity_categorical_features", []),
        transform=getattr(training, "severity_target_transform", "raw"),
    )
    sev_calibrator = getattr(training, "severity_calibrator", None)
    if sev_calibrator is not None:
        from querulus.training.calibration import apply_severity_calibrator

        sev_raw = apply_severity_calibrator(sev_calibrator, sev_raw)
    sev_pred = pd.Series(np.asarray(sev_raw, dtype=float), index=effect_index)

    if frequency_target_column:
        y_true = effect_frame[frequency_target_column]
    elif y_true_freq is not None:
        y_true = y_true_freq
    else:
        freq_col = getattr(config, "frequency_target_column", None)
        if not freq_col or freq_col not in effect_frame.columns:
            for candidate in ("TARGET_FREQ", "TARGET_2", "TARGET_FREQ_CLAIMS"):
                if candidate in effect_frame.columns:
                    freq_col = candidate
                    break
        if not freq_col or freq_col not in effect_frame.columns:
            raise ValueError(
                "Не удалось определить frequency target для effect_index; "
                "передайте frequency_target_column="
            )
        y_true = effect_frame[freq_col]

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
    print(f"Порог вероятности              : {result.best_threshold:.2f}")
    print(f"Фин. эффект факт (baseline)    : {result.fact_effect_total:,.2f} ₽")
    print(f"Фин. эффект модели             : {result.model_effect_total:,.2f} ₽")
    print(f"Чистый финансовый эффект (Δ)   : {result.net_effect:,.2f} ₽")
    if result.threshold_strategies:
        print("\nСравнение стратегий порога:")
        for strategy in result.threshold_strategies.values():
            print(
                f"  {strategy.strategy:20s} threshold={strategy.threshold:.2f} "
                f"net_effect={strategy.net_effect:,.0f} F1={strategy.f1:.3f} "
                f"PR-AUC={strategy.average_precision:.3f}"
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
