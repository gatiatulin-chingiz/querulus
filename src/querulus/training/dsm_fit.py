"""Обучение frequency (classification) в OutBoxML DSM.

Встроенные метрики ядра OutBoxML (F1 @ 0.5) не используем как источник
истины. После ``fit_models`` пересчитываем collect-style метрики при τ
из collect (``enrich_dsm_model_metrics``).
"""
from __future__ import annotations

from typing import Any

from querulus.training.build_outboxml_configs import ensure_predictable_model
from querulus.training.outboxml_metrics import enrich_dsm_model_metrics


def fit_dsm_classification(
    dsm: Any,
    model_name: str,
    *,
    threshold: float,
    enrich_collect_metrics: bool = True,
) -> None:
    """``fit_models`` для frequency + опционально collect-style метрики @ τ.

    ``threshold`` — τ из collect; нужен для ``enrich_dsm_model_metrics``,
    не для патча ядра OutBoxML.
    """
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
