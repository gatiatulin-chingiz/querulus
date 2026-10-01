"""Знаковые эффекты и формулы модели — без зависимости от summary/calculator cycle."""
from __future__ import annotations

import numpy as np
import pandas as pd

from querulus.fin_effect.config import FinEffectConfig


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


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Числовая колонка или нули."""
    if column not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


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
