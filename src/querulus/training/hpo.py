"""HPO: Optuna + MLflow + TimeSeriesSplit (перенос логики modeldiagnostics.tuning).

Не импортирует modeldiagnostics.tuning. Test holdout в objective не участвует.
"""
from __future__ import annotations

import json
import logging
import math
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

TaskType = Literal["classification", "regression"]
OptimizeDirection = Literal["maximize", "minimize"]

logger = logging.getLogger("querulus.training.hpo")

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

    client_id = (
        os.getenv("KEYCLOAK_CLIENT_ID") or _DEFAULT_KEYCLOAK_CLIENT_ID
    ).strip()
    client_secret = (os.getenv("KEYCLOAK_CLIENT_SECRET") or "").strip()
    scope = (os.getenv("KEYCLOAK_SCOPE") or "openid").strip()
    token_url = _keycloak_token_url()

    form: dict[str, str] = {
        "grant_type": "password",
        "client_id": client_id,
        "username": username,
        "password": password,
        "scope": scope,
    }
    if client_secret:
        form["client_secret"] = client_secret

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
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(
            f"Keycloak token error HTTP {exc.code} at {token_url}: {body}. "
            "Проверьте логин/пароль, KEYCLOAK_CLIENT_ID/SECRET и что у client "
            "включён Direct Access Grants (Resource Owner Password Credentials)."
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Keycloak token request failed ({token_url}): {exc}") from exc

    token = payload.get("access_token")
    if not token:
        raise RuntimeError(
            f"Keycloak не вернул access_token (keys={list(payload.keys())}). "
            "Возможен запрет password grant на client."
        )
    return str(token)


def _ensure_mlflow_bearer_token() -> None:
    """Если нет MLFLOW_TRACKING_TOKEN — получить его через Keycloak login/password."""
    existing = (os.getenv("MLFLOW_TRACKING_TOKEN") or "").strip()
    if existing:
        return
    username = (os.getenv("KEYCLOAK_USERNAME") or "").strip()
    password = os.getenv("KEYCLOAK_PASSWORD") or ""
    if not username or not password:
        return
    token = _fetch_keycloak_access_token()
    os.environ["MLFLOW_TRACKING_TOKEN"] = token
    logger.info(
        "MLflow: получен Keycloak access_token (client=%s), "
        "выставлен MLFLOW_TRACKING_TOKEN",
        (os.getenv("KEYCLOAK_CLIENT_ID") or _DEFAULT_KEYCLOAK_CLIENT_ID).strip(),
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
        "Нужна неинтерактивная авторизация: задайте в .env "
        "KEYCLOAK_USERNAME + KEYCLOAK_PASSWORD "
        "(опционально KEYCLOAK_CLIENT_ID / KEYCLOAK_CLIENT_SECRET), "
        "либо готовый MLFLOW_TRACKING_TOKEN. "
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


def _connect_mlflow_experiment(mlflow: Any, experiment_name: str) -> None:
    """get/create experiment с понятной ошибкой при SSO HTML."""
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
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


def _suggest_catboost_params(trial, *, task_type: TaskType, random_seed: int) -> dict[str, Any]:
    """Пространство поиска CatBoost (общее для clf/reg)."""
    bootstrap = trial.suggest_categorical(
        "bootstrap_type", ["Bayesian", "Bernoulli", "MVS", "No"]
    )
    grow = trial.suggest_categorical(
        "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
    )
    params: dict[str, Any] = {
        "iterations": trial.suggest_int("iterations", 100, 2000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "depth": trial.suggest_int("depth", 3, 12),
        "grow_policy": grow,
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, step=0.5),
        "bootstrap_type": bootstrap,
        "random_strength": trial.suggest_float("random_strength", 1e-9, 10.0, log=True),
        "rsm": trial.suggest_float("rsm", 0.5, 1.0),
        "leaf_estimation_method": trial.suggest_categorical(
            "leaf_estimation_method", ["Newton", "Gradient"]
        ),
        "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 255),
        "random_seed": random_seed,
        "verbose": False,
        "allow_writing_files": False,
    }
    if task_type == "classification":
        params["auto_class_weights"] = trial.suggest_categorical(
            "auto_class_weights", ["Balanced", "SqrtBalanced", None]
        )
    if bootstrap == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 1.0)
    if bootstrap in ("Bernoulli", "MVS"):
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
    if grow == "Lossguide":
        params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 1, 20)
        params["max_leaves"] = trial.suggest_int("max_leaves", 2, 64)
    params["early_stopping_rounds"] = trial.suggest_int("early_stopping_rounds", 10, 100)
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


def _fold_metric(
    y_true: pd.Series,
    y_pred: np.ndarray,
    *,
    task_type: TaskType,
    optimize_metric: str,
) -> float:
    """Метрика без ModelDiagnostics (fallback)."""
    from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score

    if task_type == "classification":
        if optimize_metric == "roc_auc":
            return float(roc_auc_score(y_true, y_pred))
        raise ValueError(f"Неизвестная classification-метрика: {optimize_metric}")
    if optimize_metric == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if optimize_metric == "r2":
        return float(r2_score(y_true, y_pred))
    raise ValueError(f"Неизвестная regression-метрика: {optimize_metric}")


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
) -> HpoResult:
    """Optuna HPO на TimeSeriesSplit по ``date_column`` (без holdout Test).

    ``mvp_types`` — types_dict для cat_features (CATEGORIAL/BINARY).
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

    sorted_df = df.sort_values(date_column).reset_index(drop=True)
    x_all = sorted_df[feature_list].copy()
    y_all = sorted_df[target_column]
    cat_features = _cat_feature_names(feature_list, mvp_types)
    for col in cat_features:
        x_all[col] = x_all[col].astype(str)

    model_cls = CatBoostClassifier if task_type == "classification" else CatBoostRegressor
    splitter = TimeSeriesSplit(n_splits=cv)

    mlflow = None
    if use_mlflow:
        import mlflow as _mlflow

        _configure_mlflow(_mlflow)
        _connect_mlflow_experiment(_mlflow, experiment_name)
        mlflow = _mlflow
        logger.info(
            "MLflow tracking URI=%s experiment=%s",
            _mlflow.get_tracking_uri(),
            experiment_name,
        )

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_catboost_params(trial, task_type=task_type, random_seed=random_seed)
        early = int(params.pop("early_stopping_rounds"))
        fold_scores: list[float] = []
        # Без nested MLflow-run на каждый trial — меньше шума в логах.
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(x_all)):
            x_tr, y_tr = x_all.iloc[tr_idx], y_all.iloc[tr_idx]
            x_va, y_va = x_all.iloc[va_idx], y_all.iloc[va_idx]
            pool = Pool(x_tr, y_tr, cat_features=cat_features, feature_names=feature_list)
            eval_pool = Pool(
                x_va, y_va, cat_features=cat_features, feature_names=feature_list
            )
            model = model_cls(**params)
            model.fit(pool, eval_set=eval_pool, early_stopping_rounds=early, verbose=False)
            if task_type == "classification":
                pred = model.predict_proba(x_va)[:, 1]
            else:
                pred = np.asarray(model.predict(x_va), dtype=float)
            fold_scores.append(
                _fold_metric(
                    y_va, pred, task_type=task_type, optimize_metric=optimize_metric
                )
            )
        return float(np.mean(fold_scores))

    study = optuna.create_study(direction=direction)
    parent = mlflow.start_run(run_name=run_name) if mlflow else None
    try:
        if parent is not None:
            parent.__enter__()
            mlflow.log_param("n_features", len(feature_list))
            mlflow.log_param("cv", cv)
            mlflow.log_param("optimize_metric", optimize_metric)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        if mlflow is not None:
            mlflow.log_params(study.best_params)
            if study.best_value is not None and not (
                isinstance(study.best_value, float) and math.isnan(study.best_value)
            ):
                mlflow.log_metric(f"best_{optimize_metric}", float(study.best_value))
    finally:
        if parent is not None:
            parent.__exit__(None, None, None)

    return HpoResult(
        best_params=dict(study.best_params),
        best_value=float(study.best_value) if study.best_value is not None else float("nan"),
        optimize_metric=optimize_metric,
        direction=direction,
        n_trials=n_trials,
        experiment_name=experiment_name,
        run_name=run_name,
    )
