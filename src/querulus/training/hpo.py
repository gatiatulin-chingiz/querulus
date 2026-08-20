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
    """Выставить MLFLOW_TRACKING_TOKEN (+ refresh, если есть)."""
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


def _ensure_mlflow_bearer_token(*, force_refresh: bool = False) -> None:
    """Выставить / обновить MLFLOW_TRACKING_TOKEN через Keycloak."""
    existing = (os.getenv("MLFLOW_TRACKING_TOKEN") or "").strip()
    username = (os.getenv("KEYCLOAK_USERNAME") or "").strip()
    password = os.getenv("KEYCLOAK_PASSWORD") or ""
    if force_refresh:
        if existing or (username and password):
            _refresh_mlflow_bearer_token()
        return
    if existing:
        return
    if not username or not password:
        return
    _fetch_keycloak_access_token()


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
    parent_run_id: str | None = None


# Фиксированные (не ищем в Optuna).
_EARLY_STOPPING_ROUNDS = 50
_AUTO_CLASS_WEIGHTS = "Balanced"

# Описание сетки для артефакта search_space.json (актуально после правок).
_SEARCH_SPACE_DOC: dict[str, Any] = {
    "iterations": "int 100..2000 step 100",
    "learning_rate": "float 0.001..0.3 log",
    "depth": "int 3..8",
    "l2_leaf_reg": "float 1.0..10.0 step 0.5",
    "bootstrap_type": ["Bayesian", "Bernoulli", "MVS", "No"],
    "grow_policy": ["SymmetricTree", "Depthwise", "Lossguide"],
    "random_strength": "float 0.1..10.0 log",
    "rsm": "float 0.5..1.0",
    "leaf_estimation_method": ["Newton", "Gradient"],
    "one_hot_max_size": "int 2..32",
    "bagging_temperature": "float 0..1 if Bayesian",
    "subsample": "float 0.5..1 if Bernoulli|MVS",
    "min_data_in_leaf": "int 1..20 if Lossguide",
    "max_leaves": "int 2..64 if Lossguide",
    "auto_class_weights": f"fixed {_AUTO_CLASS_WEIGHTS!r} (classification only)",
    "early_stopping_rounds": f"fixed {_EARLY_STOPPING_ROUNDS}",
}


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
        "depth": trial.suggest_int("depth", 3, 8),
        "grow_policy": grow,
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, step=0.5),
        "bootstrap_type": bootstrap,
        "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
        "rsm": trial.suggest_float("rsm", 0.5, 1.0),
        "leaf_estimation_method": trial.suggest_categorical(
            "leaf_estimation_method", ["Newton", "Gradient"]
        ),
        "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 32),
        "random_seed": random_seed,
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
        params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 1, 20)
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


def _fold_metric(
    y_true: pd.Series,
    y_pred: np.ndarray,
    *,
    task_type: TaskType,
    optimize_metric: str,
) -> float:
    """Метрика без ModelDiagnostics (fallback)."""
    from sklearn.metrics import (
        average_precision_score,
        mean_absolute_error,
        r2_score,
        roc_auc_score,
    )

    if task_type == "classification":
        if optimize_metric in {"pr_auc", "average_precision"}:
            return float(average_precision_score(y_true, y_pred))
        if optimize_metric == "roc_auc":
            return float(roc_auc_score(y_true, y_pred))
        raise ValueError(f"Неизвестная classification-метрика: {optimize_metric}")
    if optimize_metric == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if optimize_metric == "r2":
        return float(r2_score(y_true, y_pred))
    raise ValueError(f"Неизвестная regression-метрика: {optimize_metric}")


def _mlflow_safe_params(params: dict[str, Any]) -> dict[str, Any]:
    """Параметры, пригодные для mlflow.log_params (без None)."""
    out: dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (bool, int, float, str)):
            out[key] = value
        else:
            out[key] = str(value)
    return out


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
) -> HpoResult:
    """Optuna HPO на TimeSeriesSplit по ``date_column`` (без holdout Test).

    ``mvp_types`` — types_dict для cat_features (CATEGORIAL/BINARY).
    При ``use_mlflow=True``: parent study-run + nested trial runs (канон MLflow).
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
        early = int(params.pop("early_stopping_rounds", _EARLY_STOPPING_ROUNDS))
        fit_params = dict(params)
        fold_scores: list[float] = []

        def _run_folds() -> list[float]:
            scores: list[float] = []
            for fold, (tr_idx, va_idx) in enumerate(splitter.split(x_all)):
                x_tr, y_tr = x_all.iloc[tr_idx], y_all.iloc[tr_idx]
                x_va, y_va = x_all.iloc[va_idx], y_all.iloc[va_idx]
                pool = Pool(x_tr, y_tr, cat_features=cat_features, feature_names=feature_list)
                eval_pool = Pool(
                    x_va, y_va, cat_features=cat_features, feature_names=feature_list
                )
                model = model_cls(**fit_params)
                model.fit(
                    pool,
                    eval_set=eval_pool,
                    early_stopping_rounds=early,
                    verbose=False,
                )
                if task_type == "classification":
                    pred = model.predict_proba(x_va)[:, 1]
                else:
                    pred = np.asarray(model.predict(x_va), dtype=float)
                score = _fold_metric(
                    y_va, pred, task_type=task_type, optimize_metric=optimize_metric
                )
                scores.append(score)
            return scores

        if mlflow is None:
            fold_scores = _run_folds()
            return float(np.mean(fold_scores))

        # Access token часто короче одного trial (CV) — обновляем перед nested run.
        _ensure_mlflow_bearer_token(force_refresh=True)
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}") as child_run:
            log_params = _mlflow_safe_params(
                {**fit_params, "early_stopping_rounds": early}
            )
            mlflow.log_params(log_params)
            mlflow.set_tags(
                {
                    "trial_number": str(trial.number),
                    "stage": "hpo_trial",
                }
            )
            try:
                fold_scores = _run_folds()
                mean_score = float(np.mean(fold_scores))
                std_score = float(np.std(fold_scores)) if len(fold_scores) > 1 else 0.0
                metrics = {
                    optimize_metric: mean_score,
                    f"{optimize_metric}_std": std_score,
                }
                for fold_i, score in enumerate(fold_scores):
                    metrics[f"{optimize_metric}_fold_{fold_i}"] = float(score)
                _ensure_mlflow_bearer_token(force_refresh=True)
                mlflow.log_metrics(metrics)
                mlflow.set_tag("optuna_state", "COMPLETE")
                trial.set_user_attr("mlflow_run_id", child_run.info.run_id)
                return mean_score
            except Exception:
                try:
                    _ensure_mlflow_bearer_token(force_refresh=True)
                    mlflow.set_tag("optuna_state", "FAIL")
                    mlflow.set_tag("status", "failed")
                except Exception:  # noqa: BLE001 — не маскируем исходную ошибку trial
                    logger.warning("MLflow: не удалось проставить FAIL-tags после ошибки trial")
                raise
            finally:
                # Свежий token на end_run nested child.
                _ensure_mlflow_bearer_token(force_refresh=True)

    study = optuna.create_study(direction=direction)
    parent_run_id: str | None = None

    def _enrich_best_params(raw: dict[str, Any]) -> dict[str, Any]:
        best = dict(raw)
        best["early_stopping_rounds"] = _EARLY_STOPPING_ROUNDS
        if task_type == "classification":
            best["auto_class_weights"] = _AUTO_CLASS_WEIGHTS
        return best

    if mlflow is None:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = _enrich_best_params(dict(study.best_params))
        best_value = (
            float(study.best_value) if study.best_value is not None else float("nan")
        )
    else:
        with mlflow.start_run(run_name=run_name) as parent_run:
            parent_run_id = parent_run.info.run_id
            try:
                mlflow.set_tags(
                    {
                        "project": "querulus",
                        "optimizer": "optuna",
                        "model_family": "catboost",
                        "task_type": task_type,
                        "optimize_metric": optimize_metric,
                        "direction": direction,
                        "stage": "hpo",
                    }
                )
                mlflow.log_params(
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
                    }
                )
                if task_type == "classification":
                    mlflow.log_param("auto_class_weights", _AUTO_CLASS_WEIGHTS)

                study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
                best_params = _enrich_best_params(dict(study.best_params))
                best_value = (
                    float(study.best_value)
                    if study.best_value is not None
                    else float("nan")
                )

                _ensure_mlflow_bearer_token(force_refresh=True)
                mlflow.log_params(
                    _mlflow_safe_params({f"best_{k}": v for k, v in best_params.items()})
                )
                if not (isinstance(best_value, float) and math.isnan(best_value)):
                    mlflow.log_metric(f"best_{optimize_metric}", best_value)

                best_trial = study.best_trial
                mlflow.log_param("best_trial_number", best_trial.number)
                best_child = best_trial.user_attrs.get("mlflow_run_id")
                if best_child:
                    mlflow.log_param("best_child_run_id", best_child)

                _log_study_artifacts(
                    mlflow,
                    feature_list=feature_list,
                    cat_features=cat_features,
                    study=study,
                    optimize_metric=optimize_metric,
                    best_params=best_params,
                    best_value=best_value,
                )
            except Exception as exc:
                _raise_mlflow_auth_hint(exc)
                raise
            finally:
                # Критично: end_run parent после длинного HPO.
                _ensure_mlflow_bearer_token(force_refresh=True)

    return HpoResult(
        best_params=best_params,
        best_value=best_value,
        optimize_metric=optimize_metric,
        direction=direction,
        n_trials=n_trials,
        experiment_name=experiment_name,
        run_name=run_name,
        parent_run_id=parent_run_id,
    )
