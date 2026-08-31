"""Обучение моделей OutBoxML (DSM) с порогом frequency из collect.

OutBoxML ``BaseMetrics`` для classification жёстко использует ``cut_off = 0.5``
(см. ``outboxml.metrics.base_metrics``). Библиотеку не меняем; на время
``fit_models`` подменяем порог на τ из collect и затем пересчитываем метрики
в формате collect (``enrich_dsm_model_metrics``).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from querulus.training.build_outboxml_configs import ensure_predictable_model
from querulus.training.outboxml_metrics import enrich_dsm_model_metrics


@contextmanager
def outboxml_classification_cutoff(cutoff: float) -> Iterator[None]:
    """Временно задать порог бинаризации для OutBoxML ``BaseMetrics`` (classification).

    В логе будет ``Metrics for classification||cut_off = <τ>`` вместо 0.5.
    Не изменяет код OutBoxML; действует только внутри контекста.
    """
    from loguru import logger
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    from outboxml.core.enums import ModelTypes
    from outboxml.metrics.base_metrics import BaseMetrics

    original = BaseMetrics.calculate_metric
    threshold = float(cutoff)

    def _calculate_metric(self, model_type: str = "regression") -> dict:
        if model_type != ModelTypes.classification:
            return original(self, model_type)

        y_pred_exp = (
            self._y_pred * self._exposure
            if self._exposure is not None
            else self._y_pred
        )
        labels = (y_pred_exp > threshold).astype(int)
        logger.info(f"Metrics for classification||cut_off = {threshold}")
        return {
            "f1_score": round(
                f1_score(self._y_true, labels, sample_weight=self._exposure),
                4,
            ),
            "precision_score": round(
                precision_score(self._y_true, labels, sample_weight=self._exposure),
                4,
            ),
            "recall_score": round(
                recall_score(self._y_true, labels, sample_weight=self._exposure),
                4,
            ),
            "gini": round(
                2 * roc_auc_score(self._y_true, labels, sample_weight=self._exposure) - 1,
                4,
            ),
        }

    BaseMetrics.calculate_metric = _calculate_metric  # type: ignore[method-assign]
    try:
        yield
    finally:
        BaseMetrics.calculate_metric = original  # type: ignore[method-assign]


def fit_dsm_classification(
    dsm: Any,
    model_name: str,
    *,
    threshold: float,
    enrich_collect_metrics: bool = True,
) -> None:
    """``fit_models`` для frequency (classification) с τ collect и полным набором метрик.

    1. Обучение и встроенные метрики OutBoxML — при пороге ``threshold``.
    2. Опционально: ``metrics[*]['full']`` заменяются на collect-style (PR-AUC, MCC, …).
    """
    with outboxml_classification_cutoff(threshold):
        dsm.fit_models()
    for res in dsm.get_result().values():
        res.model = ensure_predictable_model(res.model)
    if enrich_collect_metrics:
        enrich_dsm_model_metrics(
            dsm,
            model_name,
            task_type="classification",
            val_threshold=threshold,
        )
