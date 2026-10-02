"""Bootstrap-оценка финэффекта: N фолдов с возвращением → медиана.

Зачем: точечный финэффект на Test_prod неустойчив — выборка маленькая, редкий
класс ``TARGET_FREQ=1`` и единичные крупные убытки двигают ``net_effect``.
Считаем эффект не один раз на всей выборке, а на ``n_folds`` bootstrap-выборках
(каждая — ``n`` строк **с возвращением** из Test_prod), затем берём медиану
метрик; разброс по фолдам (min/max/std) отдаём для отчёта.

τ по умолчанию фиксирован: фолды отличаются только составом
выборки, а не порогом. При ``threshold=None`` порог ищется внутри каждого фолда.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from querulus.fin_effect.calculator import (
    FinEffectResult,
    apply_model_predictions,
    align_effect_inputs,
    prepare_effect_frame,
)
from querulus.fin_effect.config import FinEffectConfig

DEFAULT_BOOTSTRAP_FOLDS = 200
DEFAULT_BOOTSTRAP_SEED = 42
MEDIAN_ROW_LABEL = "median"

EFFECT_COLUMNS: tuple[str, ...] = ("net_effect", "model_effect", "fact_effect")


@dataclass
class BootstrapFinEffect:
    """Финэффект на bootstrap-фолдах: метрики по фолдам + медиана.

    ``n_rows`` — число строк Test_prod с предсказаниями (= размер каждого
    bootstrap-фолда). Итоговая цифра отчёта — ``median_net_effect``.
    ``threshold`` — фиксированный τ (None, если порог подбирался в каждом фолде,
    тогда смотреть колонку ``thr`` таблицы фолдов).
    """

    folds: pd.DataFrame
    n_folds: int
    seed: int
    threshold: float | None
    n_rows: int
    median_net_effect: float
    median_model_effect: float
    median_fact_effect: float
    min_net_effect: float
    max_net_effect: float
    std_net_effect: float

    @property
    def net_effects(self) -> np.ndarray:
        """net_effect по фолдам (для гистограмм/доверительных интервалов)."""
        return self.folds["net_effect"].to_numpy(dtype=float)

    def summary_table(self) -> pd.DataFrame:
        """Фолды + строка ``median`` (основная цифра отчёта)."""
        median_row: dict[str, float | str] = {
            "fold": MEDIAN_ROW_LABEL,
            "n": self.n_rows,
            "n_fact_1": int(round(self.folds["n_fact_1"].median())),
            "n_pred_1": int(round(self.folds["n_pred_1"].median())),
            "thr": (
                self.threshold
                if self.threshold is not None
                else float(self.folds["thr"].median())
            ),
        }
        for column in EFFECT_COLUMNS:
            median_row[column] = float(self.folds[column].median())
        return pd.concat(
            [self.folds, pd.DataFrame([median_row])], ignore_index=True
        )


def _bootstrap_positions(
    n_rows: int,
    n_folds: int,
    seed: int,
) -> list[np.ndarray]:
    """Позиции фолдов: каждый — ``n_rows`` индексов с возвращением."""
    rng = np.random.default_rng(int(seed))
    return [rng.integers(0, n_rows, size=n_rows) for _ in range(int(n_folds))]


def _fold_frame(aligned: pd.DataFrame, positions: np.ndarray) -> pd.DataFrame:
    """Строки по позициям с новым RangeIndex: дубли index ломают reindex."""
    fold = aligned.iloc[positions].copy()
    fold.index = pd.RangeIndex(len(fold))
    return fold


def _fold_row(
    fold_number: int,
    result: FinEffectResult,
) -> dict[str, float | int]:
    frame = result.frame
    return {
        "fold": int(fold_number),
        "n": int(len(frame)),
        "n_fact_1": int(pd.to_numeric(frame["fin_effect_y_true"]).sum()),
        "n_pred_1": int(pd.to_numeric(frame["pred_freq"]).sum()),
        "thr": float(result.best_threshold),
        "net_effect": float(result.net_effect),
        "model_effect": float(result.model_effect_total),
        "fact_effect": float(result.fact_effect_total),
    }


def bootstrap_fin_effect(
    frame: pd.DataFrame,
    frequency_proba: np.ndarray | pd.Series,
    severity_prediction: np.ndarray | pd.Series,
    y_true_freq: np.ndarray | pd.Series | None = None,
    *,
    threshold: float | None = None,
    config: FinEffectConfig | None = None,
    n_folds: int = DEFAULT_BOOTSTRAP_FOLDS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> BootstrapFinEffect:
    """Финэффект по ``n_folds`` bootstrap-выборкам строк ``frame`` → медиана.

    ``frame`` — строки выборки (Test_prod) с колонками финэффекта; ``proba`` /
    ``sev`` / ``y_true`` выравниваются на её индекс, строки без предсказаний
    отбрасываются (как в ``run_fin_effect_pipeline``).
    """
    if int(n_folds) < 1:
        raise ValueError("n_folds должен быть >= 1")
    config = config or FinEffectConfig()
    prepared = prepare_effect_frame(frame, config)
    if y_true_freq is None:
        y_true_freq = prepared[config.frequency_target_column]
    aligned, proba_arr, sev_arr, y_true_arr = align_effect_inputs(
        prepared, frequency_proba, severity_prediction, y_true_freq
    )
    n_rows = len(aligned)
    if n_rows == 0:
        raise ValueError("Нет строк с выровненными proba/sev/y_true для bootstrap.")

    rows: list[dict[str, float | int]] = []
    for fold_number, positions in enumerate(
        _bootstrap_positions(n_rows, int(n_folds), seed), start=1
    ):
        result = apply_model_predictions(
            _fold_frame(aligned, positions),
            proba_arr[positions],
            sev_arr[positions],
            y_true_arr[positions],
            threshold=threshold,
            config=config,
        )
        rows.append(_fold_row(fold_number, result))

    folds = pd.DataFrame(rows)
    return BootstrapFinEffect(
        folds=folds,
        n_folds=int(n_folds),
        seed=int(seed),
        # None → порог подбирался внутри каждого фолда (см. колонку thr).
        threshold=None if threshold is None else float(threshold),
        n_rows=n_rows,
        median_net_effect=float(folds["net_effect"].median()),
        median_model_effect=float(folds["model_effect"].median()),
        median_fact_effect=float(folds["fact_effect"].median()),
        min_net_effect=float(folds["net_effect"].min()),
        max_net_effect=float(folds["net_effect"].max()),
        std_net_effect=(
            float(folds["net_effect"].std(ddof=1)) if len(folds) > 1 else 0.0
        ),
    )
