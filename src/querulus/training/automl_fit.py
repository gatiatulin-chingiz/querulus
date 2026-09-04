"""Fit CF+RG через AutoMLManager (без FS/HPO — уже в collect JSON)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from querulus import PROJECT_ROOT
from querulus.naming import MODEL_CF_NAME
from querulus.training.build_outboxml_configs import (
    ensure_predictable_model,
    prepare_datasets_from_config,
)
from querulus.training.outboxml_metrics import enrich_dsm_model_metrics

logger = logging.getLogger(__name__)

DEFAULT_AUTOML_CONFIG = PROJECT_ROOT / "configs" / "automl_querulus.json"


def _make_extractor(data: pd.DataFrame) -> Any:
    from outboxml.extractors import Extractor

    class _FrameExtractor(Extractor):
        def __init__(self, frame: pd.DataFrame, *params: Any) -> None:
            super().__init__(*params)
            self._frame = frame

        def extract_dataset(self) -> pd.DataFrame:
            return self._frame

    return _FrameExtractor(data)


def create_querulus_automl(
    data: pd.DataFrame,
    models_config: str | Path,
    *,
    external_config: Any,
    auto_ml_config: str | Path | dict[str, Any] | None = None,
    retro: bool = False,
    hp_tune: bool = False,
) -> Any:
    """AutoMLManager с extractor=df; severity X/y patch через prepare_datasets_from_config."""
    from outboxml.automl_manager import AutoMLManager

    config_path = str(models_config)
    # side-effect: патч ModelDataSubset.load_subset (severity filter)
    prepare_datasets_from_config(config_path)

    automl_cfg = auto_ml_config or DEFAULT_AUTOML_CONFIG
    if isinstance(automl_cfg, Path) and not automl_cfg.is_file():
        raise FileNotFoundError(f"Нет AutoML-конфига: {automl_cfg}")

    return AutoMLManager(
        auto_ml_config=str(automl_cfg) if not isinstance(automl_cfg, dict) else automl_cfg,
        models_config=config_path,
        external_config=external_config,
        extractor=_make_extractor(data),
        retro=retro,
        hp_tune=hp_tune,
    )


def fit_automl_bundle(
    data: pd.DataFrame,
    models_config: str | Path,
    *,
    external_config: Any,
    auto_ml_config: str | Path | None = None,
    cf_name: str = MODEL_CF_NAME,
    threshold: float | None = None,
    send_mail: bool = False,
    log_mlflow: bool = True,
    enrich_metrics: bool = True,
) -> tuple[Any, Any]:
    """Обучить CF+RG через AutoMLManager; опционально письмо и MLflow.

    ``retro=False``, ``hp_tune=False`` — фичи/HP уже в JSON из collect.
    Returns ``(automl_manager, automl_results | None)``.
    """
    automl = create_querulus_automl(
        data,
        models_config,
        external_config=external_config,
        auto_ml_config=auto_ml_config,
        retro=False,
        hp_tune=False,
    )

    results = None
    if log_mlflow or send_mail:
        results = automl.update_models(send_mail=send_mail)
    else:
        automl.load_dataset()
        automl.fit_models()

    for res in automl.get_result().values():
        res.model = ensure_predictable_model(res.model)

    if enrich_metrics:
        if threshold is not None and cf_name in automl.get_result():
            enrich_dsm_model_metrics(
                automl,
                cf_name,
                task_type="classification",
                val_threshold=threshold,
            )
        for name in automl.get_result():
            if name == cf_name:
                continue
            try:
                enrich_dsm_model_metrics(
                    automl,
                    name,
                    task_type="regression",
                    val_threshold=None,
                )
            except Exception as exc:
                logger.warning("enrich metrics %s skip: %s", name, exc)

    logger.info(
        "AutoML fit done: models=%s send_mail=%s mlflow=%s",
        list(automl.get_result()),
        send_mail,
        log_mlflow,
    )
    return automl, results
