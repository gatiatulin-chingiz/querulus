"""Fit CF+RG через AutoMLManager.update_models (без FS/HPO — уже в collect JSON)."""
from __future__ import annotations

import logging
import os
import shutil
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
from outboxml.extractors import Extractor
import mlflow
from querulus.training.hpo import (
    _configure_mlflow,
    _is_mlflow_auth_error,
    _looks_like_login_html,
)
from outboxml.automl_manager import AutoMLManager

logger = logging.getLogger(__name__)

DEFAULT_AUTOML_CONFIG = PROJECT_ROOT / "configs" / "automl_querulus.json"
_LOCAL_MLRUNS = PROJECT_ROOT / "data" / "processed" / "mlruns"
SERVICE_ENSEMBLE_PICKLE = "querulus_ansamble.pickle"


def _make_extractor(data: pd.DataFrame) -> Any:

    class _FrameExtractor(Extractor):
        def __init__(self, frame: pd.DataFrame, *params: Any) -> None:
            super().__init__(*params)
            self._frame = frame

        def extract_dataset(self) -> pd.DataFrame:
            return self._frame

    return _FrameExtractor(data)


def _set_tracking_uri(external_config: Any, uri: str) -> None:
    if hasattr(external_config, "mlflow_tracking_uri"):
        external_config.mlflow_tracking_uri = uri
    os.environ["MLFLOW_TRACKING_URI"] = uri


def _use_local_mlflow(external_config: Any, *, reason: str) -> None:
    """File store: AutoMLManager.__init__ всегда зовёт set_experiment."""
    _LOCAL_MLRUNS.mkdir(parents=True, exist_ok=True)
    uri = _LOCAL_MLRUNS.resolve().as_uri()
    _set_tracking_uri(external_config, uri)
    logger.warning("MLflow → local %s (%s)", uri, reason)


def _prepare_mlflow_for_automl(external_config: Any, *, log_mlflow: bool) -> None:
    """До ``AutoMLManager(...)``: без Keycloak remote URI падает на HTML login.

    ``log_mlflow=False`` → сразу локальный mlruns.
    ``log_mlflow=True`` → Keycloak/token; при SSO-ошибке → local + warning.
    """
    if not log_mlflow:
        _use_local_mlflow(external_config, reason="log_mlflow=False")
        return

    try:

        _configure_mlflow(mlflow)
        # Проба API до AutoMLManager (тот же set_experiment упадёт так же).
        exp = getattr(external_config, "mlflow_experiment", None) or "Querulus"
        mlflow.set_experiment(str(exp))
        if hasattr(external_config, "mlflow_tracking_uri"):
            external_config.mlflow_tracking_uri = mlflow.get_tracking_uri()
        logger.info("MLflow remote OK: %s exp=%s", mlflow.get_tracking_uri(), exp)
    except Exception as exc:  # noqa: BLE001

        msg = str(exc)
        if (
            _is_mlflow_auth_error(exc)
            or _looks_like_login_html(msg)
            or "Keycloak" in msg
            or "страницу логина" in msg
            or "not in a valid JSON format" in msg
        ):
            _use_local_mlflow(
                external_config,
                reason=f"remote auth failed: {type(exc).__name__}",
            )
        else:
            raise


def create_querulus_automl(
    data: pd.DataFrame,
    models_config: str | Path,
    *,
    external_config: Any,
    auto_ml_config: str | Path | dict[str, Any] | None = None,
    retro: bool = False,
    hp_tune: bool = False,
    log_mlflow: bool = False,
) -> Any:
    """AutoMLManager с extractor=df; runtime-патчи Querulus через prepare_datasets_from_config."""

    config_path = str(models_config)
    # side-effect: ensure_outboxml_runtime_patches (parity X/y + default replace)
    prepare_datasets_from_config(config_path)

    automl_cfg = auto_ml_config or DEFAULT_AUTOML_CONFIG
    if isinstance(automl_cfg, Path) and not automl_cfg.is_file():
        raise FileNotFoundError(f"Нет AutoML-конфига: {automl_cfg}")

    _prepare_mlflow_for_automl(external_config, log_mlflow=log_mlflow)

    return AutoMLManager(
        auto_ml_config=str(automl_cfg) if not isinstance(automl_cfg, dict) else automl_cfg,
        models_config=config_path,
        external_config=external_config,
        extractor=_make_extractor(data),
        retro=retro,
        hp_tune=hp_tune,
    )


def _copy_stable_ensemble_pickle(
    automl: Any,
    *,
    stable_name: str = SERVICE_ENSEMBLE_PICKLE,
) -> Path | None:
    """Копия timestamped AutoML pickle → ``querulus_ansamble.pickle`` для shadow."""
    stamped = getattr(getattr(automl, "automl_results", None), "result_pickle_name", None)
    if not stamped or stamped == "No pickle":
        logger.warning("Нет result_pickle_name после update_models — stable copy skip")
        return None
    results_path = Path(automl._external_config.results_path)
    src = results_path / stamped
    if not src.is_file():
        logger.warning("AutoML pickle не найден: %s", src)
        return None
    dst = results_path / stable_name
    shutil.copy2(src, dst)
    logger.info("AutoML pickle: %s → stable %s", src.name, dst.name)
    print(f"=== AutoML models saved ===\n  {src}\n  stable: {dst}")
    return dst


def fit_automl_bundle(
    data: pd.DataFrame,
    models_config: str | Path,
    *,
    external_config: Any,
    auto_ml_config: str | Path | None = None,
    cf_name: str = MODEL_CF_NAME,
    threshold: float | None = None,
    send_mail: bool = False,
    log_mlflow: bool = False,
    enrich_metrics: bool = True,
    results_dir: Path | str | None = None,
) -> tuple[Any, Any]:
    """Обучить CF+RG через ``AutoMLManager.update_models`` (fit + pickle + review).

    ``retro=False``, ``hp_tune=False`` — FS/HPO уже в collect JSON.
    ``load_dataset`` внутри ``update_models`` — отдельно не вызываем.
    Pickle → ``results_dir`` или ``integration/results``; плюс
    ``querulus_ansamble.pickle`` для shadow.
    Returns ``(automl_manager, None)``.
    """
    out = Path(results_dir) if results_dir is not None else (
        PROJECT_ROOT / "integration" / "results"
    )
    out.mkdir(parents=True, exist_ok=True)
    external_config.results_path = out

    automl = create_querulus_automl(
        data,
        models_config,
        external_config=external_config,
        auto_ml_config=auto_ml_config,
        retro=False,
        hp_tune=False,
        log_mlflow=log_mlflow,
    )

    automl.update_models(send_mail=send_mail)

    results = automl.get_result()
    if not results:
        raise RuntimeError(
            "update_models не заполнил get_result() — смотри log / email error"
        )

    for res in results.values():
        res.model = ensure_predictable_model(res.model)

    if enrich_metrics:
        if threshold is not None and cf_name in results:
            enrich_dsm_model_metrics(
                automl,
                cf_name,
                task_type="classification",
                val_threshold=threshold,
            )
        for name in results:
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

    _copy_stable_ensemble_pickle(automl)

    logger.info(
        "AutoML update_models done: models=%s",
        list(results),
    )
    return automl, None
