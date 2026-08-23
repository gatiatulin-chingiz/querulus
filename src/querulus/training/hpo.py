"""HPO: Optuna + MLflow + TimeSeriesSplit (перенос логики modeldiagnostics.tuning).

Не импортирует modeldiagnostics.tuning. Test holdout в objective не участвует.
"""
from __future__ import annotations

import json
import logging
import math
import os
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

TaskType = Literal["classification", "regression"]
OptimizeDirection = Literal["maximize", "minimize"]

from querulus.training.catboost_fit import (
    apply_gap_penalty,
    catboost_fit_stats,
    train_val_gap,
)

logger = logging.getLogger("querulus.training.hpo")
# Параллельные trial-логи через MlflowClient (без fluent active run).
_MLFLOW_TRIAL_LOCK = threading.Lock()
_MLFLOW_TOKEN_LOCK = threading.Lock()
# epoch seconds; access_token Keycloak обычно ~300s — без refresh children обрываются.
_TOKEN_EXPIRES_AT: float | None = None
_TOKEN_REFRESH_SKEW_SEC = 60.0
_DEFAULT_GAP_LAMBDA = 0.3

# Дефолты под корпоративный MLflow (oauth2-proxy → Keycloak), см. login HTML.
_DEFAULT_KEYCLOAK_SERVER = "https://auth.vsk.ru/auth"
_DEFAULT_KEYCLOAK_REALM = "users_auth"
_DEFAULT_KEYCLOAK_CLIENT_ID = "prod-keycloak_users_auth_lanmlflow-1"


def _env_flag(name: str) -> str:
    return os.getenv(name, "").strip().lower()


def _tls_verify_context() -> ssl.SSLContext | None:
    """SSL context для urllib: CA-bundle / insecure / None (= default verify)."""
    cert_path = (os.getenv("MLFLOW_TRACKING_SERVER_CERT_PATH") or "").strip()
    if cert_path:
        return ssl.create_default_context(cafile=cert_path)
    if _env_flag("MLFLOW_TRACKING_INSECURE_TLS") in {"1", "true", "yes"}:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    return None


def _keycloak_token_url() -> str:
    explicit = (os.getenv("KEYCLOAK_TOKEN_URL") or "").strip()
    if explicit:
        return explicit
    server = (os.getenv("KEYCLOAK_SERVER_URL") or _DEFAULT_KEYCLOAK_SERVER).rstrip("/")
    realm = (os.getenv("KEYCLOAK_REALM") or _DEFAULT_KEYCLOAK_REALM).strip()
    return f"{server}/realms/{realm}/protocol/openid-connect/token"


def _keycloak_client_id() -> str:
    return (os.getenv("KEYCLOAK_CLIENT_ID") or _DEFAULT_KEYCLOAK_CLIENT_ID).strip()


def _keycloak_token_request(form: dict[str, str]) -> dict[str, Any]:
    """POST к Keycloak token endpoint → JSON payload."""
    token_url = _keycloak_token_url()
    data = urllib.parse.urlencode(form).encode("utf-8")
    request = urllib.request.Request(
        token_url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    context = _tls_verify_context()
    open_kwargs: dict[str, Any] = {"timeout": 60}
    if context is not None:
        open_kwargs["context"] = context
    try:
        with urllib.request.urlopen(request, **open_kwargs) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(
            f"Keycloak token error HTTP {exc.code} at {token_url}: {body}. "
            "Проверьте логин/пароль, KEYCLOAK_CLIENT_ID/SECRET и что у client "
            "включён Direct Access Grants (Resource Owner Password Credentials)."
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Keycloak token request failed ({token_url}): {exc}") from exc


def _apply_keycloak_token_payload(payload: dict[str, Any]) -> str:
    """Выставить MLFLOW_TRACKING_TOKEN (+ refresh, если есть) и срок жизни."""
    global _TOKEN_EXPIRES_AT
    token = payload.get("access_token")
    if not token:
        raise RuntimeError(
            f"Keycloak не вернул access_token (keys={list(payload.keys())}). "
            "Возможен запрет password grant на client."
        )
    os.environ["MLFLOW_TRACKING_TOKEN"] = str(token)
    refresh = payload.get("refresh_token")
    if refresh:
        os.environ["MLFLOW_KEYCLOAK_REFRESH_TOKEN"] = str(refresh)
    expires_in = payload.get("expires_in")
    try:
        ttl = float(expires_in) if expires_in is not None else 300.0
    except (TypeError, ValueError):
        ttl = 300.0
    _TOKEN_EXPIRES_AT = time.time() + max(30.0, ttl)
    return str(token)


def _fetch_keycloak_access_token() -> str:
    """Password grant → access_token для Bearer (oauth2-proxy / MLflow).

    Env:
    - ``KEYCLOAK_USERNAME`` / ``KEYCLOAK_PASSWORD`` (обязательны)
    - ``KEYCLOAK_CLIENT_ID`` (default: prod-keycloak_users_auth_lanmlflow-1)
    - ``KEYCLOAK_CLIENT_SECRET`` (если confidential client)
    - ``KEYCLOAK_TOKEN_URL`` или ``KEYCLOAK_SERVER_URL`` + ``KEYCLOAK_REALM``
    - ``KEYCLOAK_SCOPE`` (default: openid)
    """
    username = (os.getenv("KEYCLOAK_USERNAME") or "").strip()
    password = os.getenv("KEYCLOAK_PASSWORD") or ""
    if not username or not password:
        raise RuntimeError(
            "MLflow за Keycloak: задайте KEYCLOAK_USERNAME и KEYCLOAK_PASSWORD в .env "
            "(Direct Access Grants / password grant)."
        )

    client_id = _keycloak_client_id()
    client_secret = (os.getenv("KEYCLOAK_CLIENT_SECRET") or "").strip()
    scope = (os.getenv("KEYCLOAK_SCOPE") or "openid").strip()

    form: dict[str, str] = {
        "grant_type": "password",
        "client_id": client_id,
        "username": username,
        "password": password,
        "scope": scope,
    }
    if client_secret:
        form["client_secret"] = client_secret

    payload = _keycloak_token_request(form)
    token = _apply_keycloak_token_payload(payload)
    logger.info(
        "MLflow: получен Keycloak access_token (client=%s, grant=password)",
        client_id,
    )
    return token


def _refresh_mlflow_bearer_token() -> None:
    """Обновить Bearer: refresh_token, иначе повторный password grant.

    Access token Keycloak часто живёт ~5 мин; HPO с nested trials дольше —
    без refresh ``end_run`` ловит HTML login.
    """
    username = (os.getenv("KEYCLOAK_USERNAME") or "").strip()
    password = os.getenv("KEYCLOAK_PASSWORD") or ""
    refresh = (os.getenv("MLFLOW_KEYCLOAK_REFRESH_TOKEN") or "").strip()
    client_id = _keycloak_client_id()
    client_secret = (os.getenv("KEYCLOAK_CLIENT_SECRET") or "").strip()

    if refresh:
        form: dict[str, str] = {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "refresh_token": refresh,
        }
        if client_secret:
            form["client_secret"] = client_secret
        try:
            payload = _keycloak_token_request(form)
            _apply_keycloak_token_payload(payload)
            logger.info("MLflow: обновлён Keycloak access_token (grant=refresh_token)")
            return
        except RuntimeError as exc:
            logger.warning(
                "MLflow: refresh_token не удался (%s) — пробую password grant",
                exc,
            )
            os.environ.pop("MLFLOW_KEYCLOAK_REFRESH_TOKEN", None)

    if username and password:
        _fetch_keycloak_access_token()
        return

    logger.warning(
        "MLflow: не удалось обновить token (нет KEYCLOAK_* и refresh) — "
        "дальнейшие вызовы могут получить HTML SSO"
    )


def _token_needs_refresh() -> bool:
    """True, если токена нет или до expiry меньше skew (дефолт 60с)."""
    global _TOKEN_EXPIRES_AT
    existing = (os.getenv("MLFLOW_TRACKING_TOKEN") or "").strip()
    if not existing:
        return True
    if _TOKEN_EXPIRES_AT is None:
        # Ручной MLFLOW_TRACKING_TOKEN без TTL — не держим вечно: refresh через ~4 мин.
        _TOKEN_EXPIRES_AT = time.time() + 240.0
        return False
    return time.time() >= (_TOKEN_EXPIRES_AT - _TOKEN_REFRESH_SKEW_SEC)


def _ensure_mlflow_bearer_token(*, force_refresh: bool = False) -> None:
    """Выставить / обновить MLFLOW_TRACKING_TOKEN через Keycloak.

    При длинном HPO access_token (~5 мин) истекает; без refresh
    ``MlflowClient.create_run`` для children молча падает → в UI только первые trials.
    """
    with _MLFLOW_TOKEN_LOCK:
        existing = (os.getenv("MLFLOW_TRACKING_TOKEN") or "").strip()
        username = (os.getenv("KEYCLOAK_USERNAME") or "").strip()
        password = os.getenv("KEYCLOAK_PASSWORD") or ""
        need = force_refresh or _token_needs_refresh()
        if need:
            if existing or (username and password):
                _refresh_mlflow_bearer_token()
            return
        if existing:
            return
        if not username or not password:
            return
        _fetch_keycloak_access_token()


def _is_mlflow_auth_error(exc: BaseException) -> bool:
    message = str(exc)
    return (
        _looks_like_login_html(message)
        or "not in a valid JSON format" in message
        or "UNAUTHORIZED" in message.upper()
        or "401" in message
        or "403" in message
    )


def _looks_like_login_html(text: str) -> bool:
    low = text.lower()
    return "<!doctype html" in low or "<html" in low


def _raise_mlflow_auth_hint(exc: BaseException) -> None:
    """Перепаковать типичный SSO HTML в понятную ошибку."""
    message = str(exc)
    if not _looks_like_login_html(message) and "not in a valid JSON format" not in message:
        return
    raise RuntimeError(
        "MLflow вернул страницу логина Keycloak вместо JSON API. "
        "Нужна неинтерактивная авторизация: KEYCLOAK_USERNAME + KEYCLOAK_PASSWORD "
        "(или MLFLOW_TRACKING_TOKEN). Если HPO уже шёл и упал на end_run — "
        "скорее истёк access token (обновляем через refresh/password автоматически). "
        f"Исходная ошибка: {type(exc).__name__}"
    ) from exc


def _configure_mlflow(mlflow: Any) -> None:
    """Настроить tracking URI / TLS / Keycloak Bearer так, чтобы HPO писал в MLflow.

    Переменные:
    - ``MLFLOW_TRACKING_URI`` — куда писать (fallback: https://mlflow.vsk.ru/)
    - ``MLFLOW_TRACKING_SERVER_CERT_PATH`` — CA-bundle (предпочтительно)
    - ``MLFLOW_TRACKING_INSECURE_TLS`` — ``true``/``false``; если CA нет и
      значение не ``false``, для корпоративного HTTPS включается insecure TLS
    - ``MLFLOW_TRACKING_TOKEN`` — готовый Bearer; иначе Keycloak password grant
    - ``KEYCLOAK_USERNAME`` / ``KEYCLOAK_PASSWORD`` — логин Keycloak

    MLflow запрещает одновременно ``INSECURE_TLS=true`` и заданный
    ``SERVER_CERT_PATH`` (даже пустая строка в env считается «заданным»).
    Пустой/пробельный CERT_PATH снимаем из окружения.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip() or "https://mlflow.vsk.ru/"
    mlflow.set_tracking_uri(tracking_uri)

    cert_raw = os.getenv("MLFLOW_TRACKING_SERVER_CERT_PATH")
    cert_path = (cert_raw or "").strip()
    if cert_raw is not None and not cert_path:
        # Пустая строка в .env → MLflow всё равно видит «path is set».
        os.environ.pop("MLFLOW_TRACKING_SERVER_CERT_PATH", None)

    insecure_raw = _env_flag("MLFLOW_TRACKING_INSECURE_TLS")
    if cert_path:
        # CA задан — insecure не должен конфликтовать.
        if insecure_raw in {"1", "true", "yes"}:
            os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "false"
            logger.warning(
                "MLFLOW_TRACKING_SERVER_CERT_PATH задан — "
                "сбрасываю MLFLOW_TRACKING_INSECURE_TLS=false (взаимоисключающие)."
            )
    elif insecure_raw not in {"0", "false", "no"}:
        # Без CA на корпоративном HTTPS self-signed ломает запись — включаем insecure.
        os.environ.pop("MLFLOW_TRACKING_SERVER_CERT_PATH", None)
        os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
        logger.warning(
            "MLFLOW_TRACKING_INSECURE_TLS=true (нет MLFLOW_TRACKING_SERVER_CERT_PATH). "
            "HPO пишет в %s без проверки TLS. Задайте CA-bundle, когда будет доступен.",
            tracking_uri,
        )

    _ensure_mlflow_bearer_token()
    if not (os.getenv("MLFLOW_TRACKING_TOKEN") or "").strip():
        logger.warning(
            "MLFLOW_TRACKING_TOKEN не задан и Keycloak login/password не заданы "
            "(KEYCLOAK_USERNAME/PASSWORD). Запросы к %s могут получить HTML SSO.",
            tracking_uri,
        )


def _connect_mlflow_experiment(mlflow: Any, experiment_name: str) -> str:
    """get/create experiment; set по experiment_id (иначе nested children уезжают в другой exp)."""
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        # Важно: set по id, не по имени — иначе UI/клиент может подменить experiment.
        mlflow.set_experiment(experiment_id=experiment_id)
        return str(experiment_id)
    except Exception as exc:  # noqa: BLE001 — перепаковка любой MLflow/HTTP ошибки
        _raise_mlflow_auth_hint(exc)
        raise


@dataclass(frozen=True)
class HpoResult:
    """Результат Optuna-поиска."""

    best_params: dict[str, Any]
    best_value: float
    optimize_metric: str
    direction: OptimizeDirection
    n_trials: int
    experiment_name: str
    run_name: str
    parent_run_id: str | None = None
    best_value_raw: float | None = None
    iterations_cap: int | None = None
    tree_count: int | None = None
    best_iteration: int | None = None
    gap_lambda: float = _DEFAULT_GAP_LAMBDA
    mean_train_val_gap: float | None = None


# Фиксированные (не ищем в Optuna).
_EARLY_STOPPING_ROUNDS = 50
_AUTO_CLASS_WEIGHTS = "Balanced"
_DEFAULT_THREAD_COUNT = 3
_DEFAULT_N_JOBS = 4
_PRUNER_STARTUP_TRIALS = 5

# Описание сетки для артефакта search_space.json (актуально после правок).
_SEARCH_SPACE_DOC: dict[str, Any] = {
    "iterations": "int 100..800 step 100 (cap; фактическое tree_count после ES)",
    "learning_rate": "float 0.01..0.25 log",
    "depth": "int 3..6",
    "l2_leaf_reg": "float 3.0..15.0 step 0.5",
    "min_data_in_leaf": "int 5..50",
    "bootstrap_type": ["Bayesian", "Bernoulli", "MVS", "No"],
    "grow_policy": ["SymmetricTree", "Depthwise", "Lossguide"],
    "random_strength": "float 0.1..10.0 log",
    "rsm": "float 0.5..0.9",
    "leaf_estimation_method": ["Newton", "Gradient"],
    "one_hot_max_size": "int 2..32",
    "bagging_temperature": "float 0..1 if Bayesian",
    "subsample": "float 0.5..1 if Bernoulli|MVS",
    "max_leaves": "int 2..64 if Lossguide",
    "auto_class_weights": f"fixed {_AUTO_CLASS_WEIGHTS!r} (classification only)",
    "early_stopping_rounds": f"fixed {_EARLY_STOPPING_ROUNDS}",
    "thread_count": f"fixed (default {_DEFAULT_THREAD_COUNT}; n_jobs×thread_count ≈ cores)",
    "n_jobs": f"Optuna parallel trials (default {_DEFAULT_N_JOBS})",
    "pruner": f"MedianPruner n_startup_trials={_PRUNER_STARTUP_TRIALS}",
    "gap_lambda": f"fixed objective penalty λ (default {_DEFAULT_GAP_LAMBDA})",
    "objective": "val_metric ∓ λ·train_val_gap (см. catboost_fit.apply_gap_penalty)",
}


def _suggest_catboost_params(
    trial,
    *,
    task_type: TaskType,
    random_seed: int,
    thread_count: int = _DEFAULT_THREAD_COUNT,
) -> dict[str, Any]:
    """Пространство поиска CatBoost (общее для clf/reg)."""
    bootstrap = trial.suggest_categorical(
        "bootstrap_type", ["Bayesian", "Bernoulli", "MVS", "No"]
    )
    grow = trial.suggest_categorical(
        "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
    )
    params: dict[str, Any] = {
        "iterations": trial.suggest_int("iterations", 100, 800, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
        "depth": trial.suggest_int("depth", 3, 6),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
        "grow_policy": grow,
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 3.0, 15.0, step=0.5),
        "bootstrap_type": bootstrap,
        "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
        "rsm": trial.suggest_float("rsm", 0.5, 0.9),
        "leaf_estimation_method": trial.suggest_categorical(
            "leaf_estimation_method", ["Newton", "Gradient"]
        ),
        "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 32),
        "random_seed": random_seed,
        "thread_count": int(thread_count),
        "verbose": False,
        "allow_writing_files": False,
        "early_stopping_rounds": _EARLY_STOPPING_ROUNDS,
    }
    if task_type == "classification":
        params["auto_class_weights"] = _AUTO_CLASS_WEIGHTS
    if bootstrap == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 1.0)
    if bootstrap in ("Bernoulli", "MVS"):
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
    if grow == "Lossguide":
        params["max_leaves"] = trial.suggest_int("max_leaves", 2, 64)
    return {key: value for key, value in params.items() if value is not None}


def _cat_feature_names(
    features: list[str],
    mvp_types: dict[str, tuple[str, ...]] | None,
) -> list[str]:
    """CatBoost cat_features = features ∩ (CATEGORIAL ∪ BINARY) из resolved MVP."""
    if not mvp_types:
        return []
    cats = set(mvp_types.get("CATEGORIAL", ())) | set(mvp_types.get("BINARY", ()))
    return [name for name in features if name in cats]


_MD_CLS: Any | None = None
_MD_STUBS: dict[str, Any] = {}


def _load_model_diagnostics_class() -> Any:
    """ModelDiagnostics из внешнего modeldiagnostics (как в pipeline)."""
    global _MD_CLS
    if _MD_CLS is None:
        from querulus.training.config import TrainingConfig
        from querulus.training.pipeline import _require_model_diagnostics

        _MD_CLS = _require_model_diagnostics(TrainingConfig())
    return _MD_CLS


def _model_diagnostics_stub(task_type: TaskType) -> Any:
    """Лёгкий инстанс только для compute_*_metrics (без реального fit)."""
    cached = _MD_STUBS.get(task_type)
    if cached is not None:
        return cached
    ModelDiagnostics = _load_model_diagnostics_class()
    dummy_x = pd.DataFrame({"_stub": [0.0, 1.0]})
    dummy_y = pd.Series([0, 1])
    stub = ModelDiagnostics(
        X_train=dummy_x,
        y_train=dummy_y,
        X_test=dummy_x,
        y_test=dummy_y,
        model=object(),
        features=["_stub"],
        cat_features=[],
        task_type=task_type,
    )
    _MD_STUBS[task_type] = stub
    return stub


def _fold_metric(
    y_true: pd.Series,
    y_pred: np.ndarray,
    *,
    task_type: TaskType,
    optimize_metric: str,
) -> float:
    """Одна метрика (для Optuna objective) из набора ModelDiagnostics."""
    bundle = _fold_metrics_bundle(y_true, y_pred, task_type=task_type)
    key = "pr_auc" if optimize_metric == "average_precision" else optimize_metric
    if key not in bundle:
        raise ValueError(f"Неизвестная метрика для {task_type}: {optimize_metric}")
    return float(bundle[key])


def _classification_metrics_at_threshold(
    diagnostics: Any,
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Набор метрик как в ModelDiagnostics, пороговые — только на ``threshold``.

    Совместимо со старым MD без аргумента ``thresholds`` (там иначе шёл
    полный перебор 0..1).
    """
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    pred_labels = (proba >= threshold).astype(int)

    metrics: dict[str, Any] = {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "best_threshold": float(threshold),
    }
    try:
        metrics["ece"] = float(diagnostics._calculate_ece(y_true, proba))
    except Exception:  # noqa: BLE001
        metrics["ece"] = float("nan")
    try:
        from modeldiagnostics.src.gini import Gini

        metrics["gini"] = float(Gini(y_true, proba))
    except Exception:  # noqa: BLE001
        metrics["gini"] = float("nan")

    tp = int(((y_true == 1) & (pred_labels == 1)).sum())
    tn = int(((y_true == 0) & (pred_labels == 0)).sum())
    fp = int(((y_true == 0) & (pred_labels == 1)).sum())
    fn = int(((y_true == 1) & (pred_labels == 0)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    metrics.update(
        {
            "f1_score": float(f1_score(y_true, pred_labels, zero_division=0)),
            "precision_score": float(
                precision_score(y_true, pred_labels, zero_division=0)
            ),
            "recall_score": float(recall_score(y_true, pred_labels, zero_division=0)),
            "sensitivity": float(sensitivity) if sensitivity == sensitivity else float("nan"),
            "specificity": float(specificity) if specificity == specificity else float("nan"),
            "youden_index": (
                float(sensitivity + specificity - 1)
                if sensitivity == sensitivity and specificity == specificity
                else float("nan")
            ),
            "mcc": float(matthews_corrcoef(y_true, pred_labels)),
            "shift": (
                float(pred_labels.sum() / y_true.sum())
                if float(y_true.sum()) > 0
                else float("nan")
            ),
        }
    )
    return metrics


def _fold_metrics_bundle(
    y_true: pd.Series,
    y_pred: np.ndarray,
    *,
    task_type: TaskType,
) -> dict[str, float]:
    """Метрики fold в духе ModelDiagnostics.

    Classification: порогозависимые метрики только на 0.5 (без перебора 0..1).
    """
    diagnostics = _model_diagnostics_stub(task_type)
    y_np = np.asarray(y_true)
    if task_type == "classification":
        proba = np.asarray(y_pred, dtype=float)
        # Новый MD: thresholds=[0.5]; старый — без kwargs, тогда считаем сами @0.5.
        try:
            raw = diagnostics.compute_classification_metrics(
                y_np, proba, thresholds=[0.5]
            )
        except TypeError:
            raw = _classification_metrics_at_threshold(
                diagnostics, y_np, proba, threshold=0.5
            )
    else:
        pred = np.asarray(y_pred, dtype=float)
        raw = diagnostics.compute_regression_metrics(y_np, pred)

    out: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, (bool, str)):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(number) or math.isinf(number):
            continue
        out[str(key)] = number
    return out


def _mean_metrics(fold_bundles: list[dict[str, float]]) -> dict[str, float]:
    """Среднее по фолдам; NaN-фолды пропускаем per-key."""
    if not fold_bundles:
        return {}
    keys = sorted({key for bundle in fold_bundles for key in bundle})
    out: dict[str, float] = {}
    for key in keys:
        values = [
            float(bundle[key])
            for bundle in fold_bundles
            if key in bundle and not math.isnan(float(bundle[key]))
        ]
        if values:
            out[key] = float(np.mean(values))
    return out


def _mlflow_safe_params(params: dict[str, Any]) -> dict[str, Any]:
    """Параметры для mlflow.log_params: Python-скаляры, без None/numpy."""
    out: dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        # Optuna/numpy часто отдают np.float64 / np.int64
        if isinstance(value, (np.integer,)):
            out[key] = int(value)
        elif isinstance(value, (np.floating,)):
            out[key] = float(value)
        elif isinstance(value, (np.bool_,)):
            out[key] = bool(value)
        elif isinstance(value, (bool, int, float, str)):
            out[key] = value
        else:
            out[key] = str(value)
    return out


def _mlflow_log_params(mlflow: Any, params: dict[str, Any]) -> None:
    """log_params; при 500/log-batch — по одному; ошибки не пробрасываем."""
    safe = _mlflow_safe_params(params)
    if not safe:
        return
    try:
        try:
            mlflow.log_params(safe, synchronous=True)
        except TypeError:
            mlflow.log_params(safe)
        return
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "MLflow log_params(batch) failed (%s) — пробую по одному",
            type(exc).__name__,
        )
    ok = 0
    for key, value in safe.items():
        try:
            mlflow.log_param(key, value)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow log_param(%s) failed: %s", key, type(exc).__name__)
    if ok == 0:
        logger.warning("MLflow: ни один param не записан")


def _mlflow_log_metrics(mlflow: Any, metrics: dict[str, float]) -> None:
    """log_metrics; при сбое batch — по одной; ошибки не пробрасываем."""
    if not metrics:
        return
    clean = {key: float(value) for key, value in metrics.items()}
    try:
        try:
            mlflow.log_metrics(clean, synchronous=True)
        except TypeError:
            mlflow.log_metrics(clean)
        return
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "MLflow log_metrics(batch) failed (%s) — пробую по одной",
            type(exc).__name__,
        )
    ok = 0
    for key, value in clean.items():
        try:
            mlflow.log_metric(key, value)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow log_metric(%s) failed: %s", key, type(exc).__name__)
    if ok == 0:
        logger.warning("MLflow: ни одна metric не записана")


def _mlflow_set_tags(mlflow: Any, tags: dict[str, str]) -> None:
    """set_tags; при сбое — set_tag по одному; ошибки не пробрасываем."""
    if not tags:
        return
    try:
        mlflow.set_tags(tags)
        return
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "MLflow set_tags(batch) failed (%s) — пробую по одному",
            type(exc).__name__,
        )
    for key, value in tags.items():
        try:
            mlflow.set_tag(key, str(value))
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow set_tag(%s) failed: %s", key, type(exc).__name__)


def _client_log_trial_run(
    *,
    parent_run_id: str,
    experiment_id: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, str],
    status: str = "FINISHED",
) -> str | None:
    """Создать child run и залогировать params/metrics через MlflowClient.

    Thread-safe: не трогает fluent ``active_run``. Подходит для Optuna ``n_jobs>1``.
    ``status``: FINISHED | FAILED | KILLED (pruned).
    При auth/HTML SSO — один retry с ``force_refresh``.
    """
    from mlflow.entities import RunStatus
    from mlflow.tracking import MlflowClient

    try:
        from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME
    except ImportError:  # pragma: no cover
        MLFLOW_PARENT_RUN_ID = "mlflow.parentRunId"
        MLFLOW_RUN_NAME = "mlflow.runName"

    safe_params = _mlflow_safe_params(params)
    clean_metrics = {
        str(key): float(value)
        for key, value in metrics.items()
        if value is not None
        and not (isinstance(value, float) and (math.isnan(value) or math.isinf(value)))
    }
    tag_payload = {
        str(MLFLOW_PARENT_RUN_ID): parent_run_id,
        str(MLFLOW_RUN_NAME): run_name,
        **{str(key): str(value) for key, value in tags.items()},
    }

    status_map = {
        "FINISHED": RunStatus.FINISHED,
        "FAILED": RunStatus.FAILED,
        "KILLED": RunStatus.KILLED,
    }
    run_status = status_map.get(status.upper(), RunStatus.FINISHED)

    def _once(*, force_refresh: bool) -> str:
        _ensure_mlflow_bearer_token(force_refresh=force_refresh)
        client = MlflowClient()
        try:
            created = client.create_run(
                experiment_id=str(experiment_id),
                tags=tag_payload,
                run_name=run_name,
            )
        except TypeError:
            created = client.create_run(
                experiment_id=str(experiment_id),
                tags=tag_payload,
            )
        child_id = created.info.run_id
        try:
            client.set_tag(child_id, "mlflow.parentRunId", parent_run_id)
        except Exception:  # noqa: BLE001
            pass
        for key, value in safe_params.items():
            try:
                client.log_param(child_id, key, value)
            except Exception:  # noqa: BLE001
                pass
        for key, value in clean_metrics.items():
            try:
                client.log_metric(child_id, key, value)
            except Exception:  # noqa: BLE001
                pass
        try:
            client.set_terminated(child_id, status=RunStatus.to_string(run_status))
        except TypeError:
            client.set_terminated(child_id, status=status.upper())
        except Exception:  # noqa: BLE001
            try:
                client.set_terminated(child_id)
            except Exception:  # noqa: BLE001
                pass
        return child_id

    with _MLFLOW_TRIAL_LOCK:
        try:
            child_id = _once(force_refresh=False)
        except Exception as exc:  # noqa: BLE001
            if _is_mlflow_auth_error(exc):
                logger.warning(
                    "MLflow: auth при log %s (%s) — refresh и retry",
                    run_name,
                    type(exc).__name__,
                )
                try:
                    child_id = _once(force_refresh=True)
                except Exception as retry_exc:  # noqa: BLE001
                    logger.warning(
                        "MLflow: client log trial %s не удался после refresh (%s)",
                        run_name,
                        type(retry_exc).__name__,
                    )
                    return None
            else:
                logger.warning(
                    "MLflow: client log trial %s не удался (%s: %s)",
                    run_name,
                    type(exc).__name__,
                    str(exc)[:200],
                )
                return None
        logger.info(
            "MLflow: child trial %s → parent %s (%s)",
            child_id,
            parent_run_id,
            status.upper(),
        )
        return child_id


def _suppress_insecure_request_warning() -> None:
    """Не засорять ноутбук urllib3 InsecureRequestWarning при insecure MLflow TLS."""
    try:
        import urllib3

        warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
    except Exception:  # noqa: BLE001
        pass


def _log_study_artifacts(
    mlflow: Any,
    *,
    feature_list: list[str],
    cat_features: list[str],
    study: Any,
    optimize_metric: str,
    best_params: dict[str, Any],
    best_value: float,
) -> None:
    """features.json / best_params.json / trials.csv / search_space.json.

    На корпоративном MLflow ``artifact_uri`` часто указывает на локальный
    ``/mlflow`` у клиента → PermissionError. Тогда только warning: params/metrics
    уже записаны, HPO не валим.
    """
    import tempfile
    from pathlib import Path

    try:
        with tempfile.TemporaryDirectory(prefix="querulus_hpo_") as tmp:
            root = Path(tmp)
            (root / "features.json").write_text(
                json.dumps(
                    {
                        "features": feature_list,
                        "categorical_features": cat_features,
                        "n_features": len(feature_list),
                        "n_categorical": len(cat_features),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (root / "best_params.json").write_text(
                json.dumps(
                    {
                        "optimize_metric": optimize_metric,
                        "best_value": best_value,
                        "best_params": best_params,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (root / "search_space.json").write_text(
                json.dumps(_SEARCH_SPACE_DOC, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            study.trials_dataframe().to_csv(root / "trials.csv", index=False)
            mlflow.log_artifacts(str(root))
            mlflow.set_tag("artifacts_logged", "true")
    except (PermissionError, OSError) as exc:
        logger.warning(
            "MLflow: не удалось записать artifacts (%s). "
            "Часто artifact_uri указывает на локальный /mlflow без прав на Jupyter. "
            "Params/metrics run уже в tracking; локально см. hpo_best_params_new.json. "
            "Попросите админов MLflow настроить remote artifact store (S3/MinIO).",
            exc,
        )
        try:
            mlflow.set_tag("artifacts_logged", "false")
            mlflow.set_tag("artifacts_error", type(exc).__name__)
        except Exception:  # noqa: BLE001
            pass


def run_hpo(
    df: pd.DataFrame,
    *,
    features: list[str] | tuple[str, ...],
    target_column: str,
    date_column: str,
    task_type: TaskType,
    optimize_metric: str,
    direction: OptimizeDirection,
    experiment_name: str,
    run_name: str = "catboost_hpo",
    n_trials: int = 20,
    cv: int = 3,
    random_seed: int = 42,
    mvp_types: dict[str, tuple[str, ...]] | None = None,
    use_mlflow: bool = True,
    positive_target: bool = False,
    gap_lambda: float = _DEFAULT_GAP_LAMBDA,
    n_jobs: int = _DEFAULT_N_JOBS,
    thread_count: int = _DEFAULT_THREAD_COUNT,
    use_pruner: bool = True,
) -> HpoResult:
    """Optuna HPO на TimeSeriesSplit по ``date_column`` (без holdout Test).

    Objective: val-метрика с штрафом ``λ·train_val_gap`` (см. ``apply_gap_penalty``).
    ``positive_target`` — только ``target > 0`` (severity).
    ``n_jobs`` — параллельные Optuna trials; ``thread_count`` — потоки CatBoost
    (держать ``n_jobs × thread_count ≈ CPU cores``).
    ``use_pruner`` — MedianPruner между CV-фолдами.
    Child trial-runs всегда через ``MlflowClient`` (parent + children в UI),
    в т.ч. при ``n_jobs>1`` / freq∥sev parallel.
    """
    import optuna
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    from sklearn.model_selection import TimeSeriesSplit

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("optuna").setLevel(logging.WARNING)

    feature_list = [f for f in features if f in df.columns]
    if not feature_list:
        raise ValueError("Пустой список признаков для HPO")
    if target_column not in df.columns:
        raise ValueError(f"Нет таргета {target_column}")

    n_jobs = max(1, int(n_jobs))
    thread_count = max(1, int(thread_count))

    sorted_df = df.sort_values(date_column).reset_index(drop=True)
    if positive_target:
        sorted_df = sorted_df[
            pd.to_numeric(sorted_df[target_column], errors="coerce") > 0
        ].reset_index(drop=True)
    x_all = sorted_df[feature_list].copy()
    y_all = sorted_df[target_column]
    cat_features = _cat_feature_names(feature_list, mvp_types)
    for col in cat_features:
        x_all[col] = x_all[col].astype(str)

    model_cls = CatBoostClassifier if task_type == "classification" else CatBoostRegressor
    splitter = TimeSeriesSplit(n_splits=cv)

    mlflow = None
    resolved_experiment_id: str | None = None
    if use_mlflow:
        import mlflow as _mlflow

        _configure_mlflow(_mlflow)
        _suppress_insecure_request_warning()
        # Autolog (если кто-то включил в ноутбуке) плодит sibling-runs на каждый fit.
        try:
            _mlflow.autolog(disable=True)
        except Exception:  # noqa: BLE001
            pass
        resolved_experiment_id = _connect_mlflow_experiment(_mlflow, experiment_name)
        mlflow = _mlflow
        logger.info(
            "MLflow tracking URI=%s experiment=%s id=%s (child trials via MlflowClient)",
            _mlflow.get_tracking_uri(),
            experiment_name,
            resolved_experiment_id,
        )

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_catboost_params(
            trial,
            task_type=task_type,
            random_seed=random_seed,
            thread_count=thread_count,
        )
        early = int(params.pop("early_stopping_rounds", _EARLY_STOPPING_ROUNDS))
        fit_params = dict(params)

        def _run_folds() -> tuple[float, float, dict[str, float], dict[str, float]]:
            """CV → (penalized score, raw val mean, mean metrics, fit stats)."""
            bundles: list[dict[str, float]] = []
            train_bundles: list[dict[str, float]] = []
            tree_counts: list[int] = []
            best_iters: list[int] = []
            opt_key = (
                "pr_auc" if optimize_metric == "average_precision" else optimize_metric
            )
            for fold_i, (tr_idx, va_idx) in enumerate(splitter.split(x_all)):
                x_tr, y_tr = x_all.iloc[tr_idx], y_all.iloc[tr_idx]
                x_va, y_va = x_all.iloc[va_idx], y_all.iloc[va_idx]
                pool = Pool(x_tr, y_tr, cat_features=cat_features, feature_names=feature_list)
                eval_pool = Pool(
                    x_va, y_va, cat_features=cat_features, feature_names=feature_list
                )
                iterations_cap = int(fit_params.get("iterations", 800))
                model = model_cls(**fit_params)
                model.fit(
                    pool,
                    eval_set=eval_pool,
                    early_stopping_rounds=early,
                    verbose=False,
                )
                stats = catboost_fit_stats(model, iterations_cap=iterations_cap)
                tree_counts.append(stats["tree_count"])
                best_iters.append(stats["best_iteration"])
                if task_type == "classification":
                    pred_va = model.predict_proba(x_va)[:, 1]
                    pred_tr = model.predict_proba(x_tr)[:, 1]
                else:
                    pred_va = np.asarray(model.predict(x_va), dtype=float)
                    pred_tr = np.asarray(model.predict(x_tr), dtype=float)
                bundles.append(
                    _fold_metrics_bundle(y_va, pred_va, task_type=task_type)
                )
                train_bundles.append(
                    _fold_metrics_bundle(y_tr, pred_tr, task_type=task_type)
                )
                # Intermediate score for MedianPruner (после каждого CV-фолда).
                if use_pruner:
                    part_val = _mean_metrics(bundles)
                    part_train = _mean_metrics(train_bundles)
                    if opt_key in part_val:
                        raw_v = float(part_val[opt_key])
                        raw_t = float(part_train.get(opt_key, raw_v))
                        gap_i = train_val_gap(raw_t, raw_v, task_type=task_type)
                        score_i = apply_gap_penalty(
                            raw_v, gap_i, task_type=task_type, gap_lambda=gap_lambda
                        )
                        trial.report(score_i, fold_i)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
            mean_metrics = _mean_metrics(bundles)
            mean_train = _mean_metrics(train_bundles)
            if opt_key not in mean_metrics:
                raise ValueError(f"Нет {opt_key} в fold-метриках")
            raw_val = float(mean_metrics[opt_key])
            raw_train = float(mean_train.get(opt_key, raw_val))
            gap = train_val_gap(raw_train, raw_val, task_type=task_type)
            penalized = apply_gap_penalty(
                raw_val, gap, task_type=task_type, gap_lambda=gap_lambda
            )
            fit_stats = {
                "tree_count_mean": float(np.mean(tree_counts)),
                "tree_count_median": float(np.median(tree_counts)),
                "best_iteration_mean": float(np.mean(best_iters)),
                "train_val_gap": gap,
                f"val_{opt_key}": raw_val,
                f"train_{opt_key}": raw_train,
                "objective_penalized": penalized,
                "gap_lambda": gap_lambda,
            }
            mean_metrics.update(
                {
                    "train_val_gap": gap,
                    "objective_penalized": penalized,
                    "gap_lambda": gap_lambda,
                    "tree_count_mean": fit_stats["tree_count_mean"],
                }
            )
            return penalized, raw_val, mean_metrics, fit_stats

        # Мета study + гиперпараметры trial (одинаковый набор ключей у children).
        log_params = _mlflow_safe_params(
            {
                **fit_params,
                "early_stopping_rounds": early,
                "n_trials": n_trials,
                "cv": cv,
                "random_seed": random_seed,
                "n_features": len(feature_list),
                "n_rows": len(x_all),
                "n_cat_features": len(cat_features),
                "target_column": target_column,
                "date_column": date_column,
                "optimize_metric": optimize_metric,
                "direction": direction,
                "task_type": task_type,
                "trial_number": trial.number,
                "gap_lambda": gap_lambda,
                "positive_target": positive_target,
                "n_jobs": n_jobs,
                "thread_count": thread_count,
            }
        )
        child_tags = {
            "project": "querulus",
            "optimizer": "optuna",
            "model_family": "catboost",
            "task_type": task_type,
            "optimize_metric": optimize_metric,
            "direction": direction,
            "stage": "hpo_trial",
            "trial_number": str(trial.number),
        }

        def _finish_trial_attrs(
            *,
            mean_score: float,
            raw_val: float,
            mean_metrics: dict[str, float],
            fit_stats: dict[str, float],
            child_id: str | None,
        ) -> float:
            trial.set_user_attr("tree_count", int(round(fit_stats["tree_count_mean"])))
            trial.set_user_attr("best_value_raw", raw_val)
            trial.set_user_attr("mean_train_val_gap", fit_stats["train_val_gap"])
            trial.set_user_attr("mlflow_metrics", mean_metrics)
            trial.set_user_attr("mlflow_params", log_params)
            if child_id:
                trial.set_user_attr("mlflow_run_id", child_id)
            logger.info(
                "HPO trial %s: val=%.4f gap=%.4f → objective=%.4f tree_count≈%s",
                trial.number,
                raw_val,
                fit_stats["train_val_gap"],
                mean_score,
                int(round(fit_stats["tree_count_mean"])),
            )
            return mean_score

        def _log_child(
            *,
            mean_metrics: dict[str, float],
            fit_stats: dict[str, float],
            status: str,
            extra_tags: dict[str, str] | None = None,
        ) -> str | None:
            if mlflow is None or not parent_run_id or not parent_experiment_id:
                return None
            metrics_payload = dict(mean_metrics)
            metrics_payload.update(
                {
                    "tree_count_mean": fit_stats.get("tree_count_mean", float("nan")),
                    "best_iteration_mean": fit_stats.get(
                        "best_iteration_mean", float("nan")
                    ),
                    "objective_penalized": fit_stats.get(
                        "objective_penalized", float("nan")
                    ),
                    "train_val_gap": fit_stats.get("train_val_gap", float("nan")),
                }
            )
            return _client_log_trial_run(
                parent_run_id=parent_run_id,
                experiment_id=parent_experiment_id,
                run_name=f"trial_{trial.number}",
                params=log_params,
                metrics=metrics_payload,
                tags={**child_tags, **(extra_tags or {})},
                status=status,
            )

        try:
            mean_score, raw_val, mean_metrics, fit_stats = _run_folds()
        except optuna.TrialPruned:
            _log_child(
                mean_metrics={},
                fit_stats={
                    "tree_count_mean": 0.0,
                    "best_iteration_mean": 0.0,
                    "objective_penalized": float("nan"),
                    "train_val_gap": float("nan"),
                },
                status="KILLED",
                extra_tags={"optuna_state": "PRUNED", "status": "pruned"},
            )
            raise
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            logger.exception("HPO trial %s failed: %s", trial.number, err_msg)
            _log_child(
                mean_metrics={},
                fit_stats={
                    "tree_count_mean": 0.0,
                    "best_iteration_mean": 0.0,
                    "objective_penalized": float("nan"),
                    "train_val_gap": float("nan"),
                },
                status="FAILED",
                extra_tags={
                    "optuna_state": "FAIL",
                    "status": "failed",
                    "error": err_msg[:250],
                    "error_type": type(exc).__name__,
                },
            )
            raise

        child_id = _log_child(
            mean_metrics=mean_metrics,
            fit_stats=fit_stats,
            status="FINISHED",
            extra_tags={"optuna_state": "COMPLETE"},
        )
        return _finish_trial_attrs(
            mean_score=mean_score,
            raw_val=raw_val,
            mean_metrics=mean_metrics,
            fit_stats=fit_stats,
            child_id=child_id,
        )

    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=_PRUNER_STARTUP_TRIALS)
        if use_pruner
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(direction=direction, pruner=pruner)
    parent_run_id: str | None = None
    parent_experiment_id: str | None = None

    def _enrich_best_params(raw: dict[str, Any], *, champion: optuna.Trial | None) -> dict[str, Any]:
        best = dict(raw)
        best["early_stopping_rounds"] = _EARLY_STOPPING_ROUNDS
        best["gap_lambda"] = gap_lambda
        best["thread_count"] = thread_count
        cap = int(best.get("iterations", 800))
        best["iterations_cap"] = cap
        if champion is not None:
            tc = champion.user_attrs.get("tree_count")
            if tc is not None:
                best["tree_count"] = int(tc)
                best["iterations"] = int(tc)
            raw_val = champion.user_attrs.get("best_value_raw")
            if raw_val is not None:
                best["best_value_raw"] = float(raw_val)
            gap = champion.user_attrs.get("mean_train_val_gap")
            if gap is not None:
                best["mean_train_val_gap"] = float(gap)
        if task_type == "classification":
            best["auto_class_weights"] = _AUTO_CLASS_WEIGHTS
        return best

    logger.info(
        "HPO %s/%s: objective=val_%s ∓ λ·gap (λ=%.2f, positive_target=%s, "
        "n_jobs=%s, thread_count=%s, pruner=%s)",
        experiment_name,
        target_column,
        optimize_metric,
        gap_lambda,
        positive_target,
        n_jobs,
        thread_count,
        "MedianPruner" if use_pruner else "Nop",
    )

    def _optimize_study() -> None:
        """Optuna trials в threads: CatBoost отпускает GIL; без loky/pickle на Windows."""
        from joblib import parallel_backend

        with parallel_backend("threading", n_jobs=n_jobs):
            study.optimize(
                objective,
                n_trials=n_trials,
                n_jobs=n_jobs,
                show_progress_bar=True,
            )

    if mlflow is None:
        _optimize_study()
        best_trial = study.best_trial
        best_params = _enrich_best_params(dict(study.best_params), champion=best_trial)
        best_value = (
            float(study.best_value) if study.best_value is not None else float("nan")
        )
    else:
        with mlflow.start_run(
            run_name=run_name,
            experiment_id=resolved_experiment_id,
        ) as parent_run:
            parent_run_id = parent_run.info.run_id
            parent_experiment_id = str(parent_run.info.experiment_id)
            if (
                resolved_experiment_id
                and parent_experiment_id != str(resolved_experiment_id)
            ):
                logger.warning(
                    "MLflow: parent exp_id=%s ≠ set_experiment %s — "
                    "children могут не сгруппироваться в UI",
                    parent_experiment_id,
                    resolved_experiment_id,
                )
            try:
                _mlflow_set_tags(
                    mlflow,
                    {
                        "project": "querulus",
                        "optimizer": "optuna",
                        "model_family": "catboost",
                        "task_type": task_type,
                        "optimize_metric": optimize_metric,
                        "direction": direction,
                        "stage": "hpo",
                    },
                )
                _mlflow_log_params(
                    mlflow,
                    {
                        "n_trials": n_trials,
                        "cv": cv,
                        "random_seed": random_seed,
                        "n_features": len(feature_list),
                        "n_rows": len(x_all),
                        "n_cat_features": len(cat_features),
                        "target_column": target_column,
                        "date_column": date_column,
                        "optimize_metric": optimize_metric,
                        "direction": direction,
                        "early_stopping_rounds": _EARLY_STOPPING_ROUNDS,
                        "gap_lambda": gap_lambda,
                        "positive_target": positive_target,
                        "n_jobs": n_jobs,
                        "thread_count": thread_count,
                        "use_pruner": use_pruner,
                    },
                )
                if task_type == "classification":
                    _mlflow_log_params(
                        mlflow, {"auto_class_weights": _AUTO_CLASS_WEIGHTS}
                    )

                _optimize_study()
                best_trial = study.best_trial
                best_params = _enrich_best_params(
                    dict(study.best_params), champion=best_trial
                )
                best_value = (
                    float(study.best_value)
                    if study.best_value is not None
                    else float("nan")
                )
                n_complete = sum(
                    1
                    for t in study.trials
                    if t.state.name == "COMPLETE"
                )
                n_fail = sum(1 for t in study.trials if t.state.name == "FAIL")
                n_prune = sum(1 for t in study.trials if t.state.name == "PRUNED")
                n_logged = sum(
                    1
                    for t in study.trials
                    if t.user_attrs.get("mlflow_run_id")
                )
                logger.info(
                    "HPO study done: trials=%s complete=%s fail=%s pruned=%s "
                    "mlflow_children=%s",
                    len(study.trials),
                    n_complete,
                    n_fail,
                    n_prune,
                    n_logged,
                )
                _mlflow_log_metrics(
                    mlflow,
                    {
                        "n_trials_complete": float(n_complete),
                        "n_trials_fail": float(n_fail),
                        "n_trials_pruned": float(n_prune),
                        "n_mlflow_children": float(n_logged),
                    },
                )
            except Exception as exc:
                _raise_mlflow_auth_hint(exc)
                raise
            finally:
                # Критично: end_run parent после длинного HPO.
                _ensure_mlflow_bearer_token(force_refresh=True)

            # Champion metrics/params/artifacts — soft-fail (500 на log-batch не сжигает HPO).
            try:
                _ensure_mlflow_bearer_token(force_refresh=True)
                best_trial = study.best_trial

                trial_metrics = best_trial.user_attrs.get("mlflow_metrics")
                if not isinstance(trial_metrics, dict) or not trial_metrics:
                    trial_metrics = {}
                    if not (
                        isinstance(best_value, float) and math.isnan(best_value)
                    ):
                        trial_metrics[optimize_metric] = best_value
                _mlflow_log_metrics(mlflow, trial_metrics)

                parent_meta_keys = {
                    "n_trials",
                    "cv",
                    "random_seed",
                    "n_features",
                    "n_rows",
                    "n_cat_features",
                    "target_column",
                    "date_column",
                    "optimize_metric",
                    "direction",
                    "early_stopping_rounds",
                    "auto_class_weights",
                    "champion_trial_number",
                    "champion_child_run_id",
                    "task_type",
                    "trial_number",
                    "verbose",
                    "allow_writing_files",
                }
                trial_params = best_trial.user_attrs.get("mlflow_params")
                if not isinstance(trial_params, dict) or not trial_params:
                    trial_params = best_params
                hyper_params = {
                    key: value
                    for key, value in trial_params.items()
                    if key not in parent_meta_keys
                }
                _mlflow_log_params(mlflow, hyper_params)

                _mlflow_log_params(
                    mlflow,
                    {"champion_trial_number": int(best_trial.number)},
                )
                best_child = best_trial.user_attrs.get("mlflow_run_id")
                if best_child:
                    _mlflow_log_params(
                        mlflow, {"champion_child_run_id": str(best_child)}
                    )
                _mlflow_set_tags(
                    mlflow,
                    {
                        "champion_trial_number": str(best_trial.number),
                        "has_champion_metrics": "true",
                    },
                )

                _mlflow_log_metrics(
                    mlflow,
                    {
                        "champion_tree_count": float(
                            best_trial.user_attrs.get("tree_count", 0)
                        ),
                        "champion_train_val_gap": float(
                            best_trial.user_attrs.get("mean_train_val_gap", 0)
                        ),
                        "champion_best_value_raw": float(
                            best_trial.user_attrs.get("best_value_raw", best_value)
                        ),
                        "objective_penalized": float(best_value),
                    },
                )

                _log_study_artifacts(
                    mlflow,
                    feature_list=feature_list,
                    cat_features=cat_features,
                    study=study,
                    optimize_metric=optimize_metric,
                    best_params=best_params,
                    best_value=best_value,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "MLflow: логирование champion/artifacts не удалось (%s) — "
                    "best_params всё равно вернутся в pipeline",
                    type(exc).__name__,
                )
            finally:
                _ensure_mlflow_bearer_token(force_refresh=True)

    champion = study.best_trial
    best_value_raw = float(champion.user_attrs.get("best_value_raw", best_value))
    tree_count_attr = champion.user_attrs.get("tree_count")
    tree_count = int(tree_count_attr) if tree_count_attr is not None else None
    mean_gap = champion.user_attrs.get("mean_train_val_gap")
    iterations_cap = int(
        best_params.get("iterations_cap", best_params.get("iterations", 800))
    )
    logger.info(
        "HPO champion: objective=%.4f val_raw=%.4f gap=%.4f tree_count=%s cap=%s",
        best_value,
        best_value_raw,
        float(mean_gap) if mean_gap is not None else float("nan"),
        tree_count,
        iterations_cap,
    )

    return HpoResult(
        best_params=best_params,
        best_value=best_value,
        optimize_metric=optimize_metric,
        direction=direction,
        n_trials=n_trials,
        experiment_name=experiment_name,
        run_name=run_name,
        parent_run_id=parent_run_id,
        best_value_raw=best_value_raw,
        iterations_cap=iterations_cap,
        tree_count=tree_count,
        best_iteration=(tree_count - 1) if tree_count else None,
        gap_lambda=gap_lambda,
        mean_train_val_gap=float(mean_gap) if mean_gap is not None else None,
    )
