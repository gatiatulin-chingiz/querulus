"""Пайплайн example.ipynb: загрузка данных, DSM, финэффект, экспорт prod."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from querulus.dataset.hadoop import load_df_final
from querulus.dataset.schema import DEFAULT_DATASET_SCHEMA
from querulus.features.derived import VICTIM_OBJECT_AGE_MONTHS_COL
from querulus.fin_effect import (
    DEFAULT_BOOTSTRAP_FOLDS,
    DEFAULT_BOOTSTRAP_SEED,
    BootstrapFinEffect,
    bootstrap_fin_effect,
    create_summary_table,
    export_business_html,
    print_best_threshold_report,
    resolve_fin_effect_config,
    run_fin_effect_pipeline,
)
from querulus.fin_effect.calculator import FinEffectResult
from querulus.fin_effect.threshold_policy import (
    load_collect_prod_threshold,
    load_collect_val_threshold,
)
from querulus.naming import (
    DEFAULT_HIVE_TABLE,
    MODEL_CF_NAME,
    MODEL_RG_NAME,
    default_model_version,
)
from querulus.training.automl_fit import fit_automl_bundle
from querulus.training.build_outboxml_configs import (
    ensure_predictable_model,
    load_outboxml_configs,
    prepare_datasets_from_config,
)
from querulus.training.dsm_fit import fit_dsm_classification
from querulus.training.email_report import QuerulusEMailDSResult
from querulus.training.calibration import SeverityCalibrator
from querulus.training.factors_export import (
    build_factors_figures,
    fit_rg_severity_calibrator_isotonic,
)
from querulus.training.outboxml_metrics import (
    display_dsm_collect_metrics,
    display_dsm_collect_metrics_cross_test,
    predict_dsm_series,
)
from outboxml.datasets_manager import DataSetsManager

try:
    from IPython.display import Markdown, display as ipy_display
except ImportError:  # pragma: no cover
    Markdown = None
    ipy_display = None

logger = logging.getLogger(__name__)


def _display(obj: Any) -> None:
    if ipy_display is None:
        print(obj)
        return
    ipy_display(obj)


def _markdown(title: str) -> None:
    if ipy_display is None or Markdown is None:
        print(title)
        return
    ipy_display(Markdown(title))


@dataclass(frozen=True)
class ExamplePaths:
    """Пути и флаги загрузки датасета для example."""

    project_root: Path
    results_dir: Path
    local_parquet_path: Path
    prefer_hive: bool
    use_synthetic: bool


@dataclass
class ExampleDatasetBundle:
    """Датасет и OutBoxML-конфиги (конфиги пишет collect, example только читает)."""

    df: pd.DataFrame
    dataset_source: str
    dataset_path: Path
    built: dict[str, Any]
    periods: dict[str, Any]
    model_version: str
    cf_name: str
    rg_name: str


@dataclass
class ExampleThresholds:
    parity: float
    prod: float


@dataclass
class ExampleDsmBundle:
    dsm_cf: Any
    dsm_rg: Any
    dsm_cf_prod: Any | None = None
    dsm_rg_prod: Any | None = None


@dataclass
class FinEffectTableResult:
    fin_effect: FinEffectResult
    summary: pd.DataFrame


@dataclass
class TestProdCompareResult:
    """Финэффект Test_prod по bootstrap-фолдам (медиана + разброс).

    ``parity`` / ``prod`` — ``BootstrapFinEffect`` (метрики по фолдам и медиана),
    ``compare_table`` — сводка «parity train vs prod train» по медианам.
    """

    parity: BootstrapFinEffect | None
    prod: BootstrapFinEffect
    compare_table: pd.DataFrame
    test_prod_idx: pd.Index


# Стабильное имя ensemble-pickle для shadow (копия AutoML timestamped dump).
SERVICE_ENSEMBLE_PICKLE = "querulus_ansamble.pickle"


@dataclass
class ProdExportResult:
    """Sidecar для сервиса (без записи моделей — их пишет AutoML)."""

    service_df_path: Path
    meta_path: Path
    ensemble_pkl: Path | None
    dq_bounds_path: Path | None
    meta: dict[str, Any]


@dataclass
class ProdPlotsResult:
    """Результат ``run_prod_plots_and_email``."""

    zip_path: Path | None
    rg_calibrator: SeverityCalibrator | None = None


def resolve_example_paths(
    project_root: Path | str,
    *,
    use_synthetic: bool = False,
    results_dir: Path | str | None = None,
) -> ExamplePaths:
    """Пути parquet/Hive и плоский каталог артефактов ``integration/results``."""
    root = Path(project_root)
    local_default = root / "data" / "processed" / "querulus_train_dataset.parquet"
    local_legacy = root / "data" / "processed" / "df_final_3.parquet"
    local_synthetic = root / "data" / "processed" / "df_final_3_synthetic.parquet"
    if use_synthetic:
        local_path = local_synthetic
        prefer_hive = False
    elif local_default.is_file():
        local_path = local_default
        prefer_hive = True
    elif local_legacy.is_file():
        local_path = local_legacy
        prefer_hive = True
    else:
        local_path = local_default
        prefer_hive = True
        logger.info(
            "querulus_train_dataset.parquet нет — попробуем Hive / legacy parquet"
        )
    out_dir = (
        Path(results_dir)
        if results_dir is not None
        else root / "integration" / "results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return ExamplePaths(
        project_root=root,
        results_dir=out_dir,
        local_parquet_path=local_path,
        prefer_hive=prefer_hive,
        use_synthetic=use_synthetic,
    )


def load_example_dataset(
    paths: ExamplePaths,
    *,
    hive_table: str = DEFAULT_HIVE_TABLE,
    model_version: str | None = None,
    data_date: str | None = None,
    dataset_version: str | None = None,
) -> ExampleDatasetBundle:
    """Hive/parquet → df + загрузка OutBoxML JSON из collect."""
    version = model_version or default_model_version()
    df_raw, dataset_source = load_df_final(
        hive_table=hive_table,
        parquet_path=paths.local_parquet_path,
        prefer_hive=paths.prefer_hive,
        generate_synthetic_if_missing=paths.use_synthetic,
        model_version=version,
        data_date=data_date,
        dataset_version=dataset_version,
        fallback_parquet_path=paths.project_root / "data" / "processed" / "df_final_3.parquet",
    )
    if dataset_source.startswith("hive:"):
        if len(df_raw) == 0:
            raise ValueError(
                "Hive вернул пустой DataFrame после load_df_final — "
                "кэш parquet не перезаписываем (внутренняя ошибка fallback)."
            )
        paths.local_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_parquet(paths.local_parquet_path, index=False)
        logger.info("кэш после Hive записан в parquet: %s", paths.local_parquet_path)
    else:
        logger.info("Hive не использован; данные из файла: %s", paths.local_parquet_path)

    if paths.use_synthetic:
        paths.local_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_parquet(paths.local_parquet_path, index=False)

    # Итоговый df из collect (features уже в parquet/Hive). FE здесь не собираем.
    df = df_raw
    if VICTIM_OBJECT_AGE_MONTHS_COL not in df.columns:
        raise ValueError(
            f"В датасете нет {VICTIM_OBJECT_AGE_MONTHS_COL}. "
            "Перегоните collect (run_features / export): возраст ТС в месяцах "
            "собирается только там, не в example/example_final."
        )
    schema_issues = DEFAULT_DATASET_SCHEMA.validate(df, raise_on_error=False)
    critical = [
        p
        for p in schema_issues
        if p.startswith("нет обязательных") or p.startswith("len(df)") or "NaT" in p
    ]
    if critical:
        from querulus.dataset.schema import DatasetSchemaError

        raise DatasetSchemaError(critical)
    for warning in schema_issues:
        if warning not in critical:
            logger.warning("DatasetSchema: %s", warning)

    built = load_outboxml_configs(df, version=version)
    cf_name = built.get("cf_name") or MODEL_CF_NAME
    rg_name = built.get("rg_name") or MODEL_RG_NAME
    logger.info(
        "df.shape=%s dataset_source=%s configs=%s %s model_version=%s",
        df.shape,
        dataset_source,
        built["parity_path"].name,
        built["prod_path"].name,
        version,
    )
    return ExampleDatasetBundle(
        df=df,
        dataset_source=dataset_source,
        dataset_path=paths.local_parquet_path,
        built=built,
        periods=built["periods"],
        model_version=version,
        cf_name=cf_name,
        rg_name=rg_name,
    )


def patch_dsm_models(dsm: Any) -> None:
    """CatBoost из DSM — через ensure_predictable_model."""
    for res in dsm.get_result().values():
        res.model = ensure_predictable_model(res.model)


def predict_cf(dsm: Any, model_name: str, data: pd.DataFrame) -> pd.Series:
    return predict_dsm_series(
        dsm,
        model_name,
        data,
        task_type="classification",
        ignore_row_filter=False,
    )


def predict_rg(dsm: Any, model_name: str, data: pd.DataFrame) -> pd.Series:
    """Severity на всех строках (без фильтра TARGET_SEV > 0)."""
    return predict_dsm_series(
        dsm,
        model_name,
        data,
        task_type="regression",
        ignore_row_filter=True,
    )


def load_example_thresholds(
    project_root: Path | str,
    *,
    collect_training: object | None = None,
) -> ExampleThresholds:
    """Опциональные τ из collect (или placeholder 0.5).

    Сервисный порог — ``pick_prod_threshold_on_dsm`` после prod DSM fit.
    Collect JSON больше не обязателен для запуска example / example_final.
    """
    parity = load_collect_val_threshold(project_root, training=collect_training)
    prod = load_collect_prod_threshold(project_root)
    logger.info(
        "τ fit-placeholder/collect parity=%.2f; prod=%.2f "
        "(сервисный τ — после pick_prod_threshold_on_dsm)",
        parity,
        prod,
    )
    return ExampleThresholds(parity=parity, prod=prod)


def pick_prod_threshold_on_dsm(
    models: ExampleDsmBundle,
    bundle: ExampleDatasetBundle,
) -> float:
    """Вариант B: подобрать τ на τ-cal по **raw** proba prod DSM.

    Калибратор в сервис не едет — порог только на сырой шкале модели, которая
    уходит в прод (example_final), а не collect CatBoost.
    """
    if models.dsm_cf_prod is None or models.dsm_rg_prod is None:
        raise ValueError("Нужны обученные dsm_cf_prod / dsm_rg_prod")
    tau_idx = bundle.periods["prod_tau_cal_idx"]
    if len(tau_idx) == 0:
        raise ValueError(
            "prod_tau_cal_idx пуст — нельзя подобрать τ; проверьте period_windows.json / df"
        )
    df = bundle.df
    proba = predict_cf(models.dsm_cf_prod, bundle.cf_name, df.loc[tau_idx])
    sev = predict_rg(models.dsm_rg_prod, bundle.rg_name, df.loc[tau_idx])
    cfg = resolve_fin_effect_config(
        df,
        frequency_target="TARGET_FREQ",
        severity_target="TARGET_SEV",
    )
    y_true = df.loc[tau_idx, "TARGET_FREQ"]
    fe = run_fin_effect_pipeline(
        df.loc[tau_idx],
        proba,
        sev,
        y_true,
        threshold=None,
        config=cfg,
    )
    thr = float(fe.best_threshold)
    logger.info(
        "τ prod (DSM raw, τ-cal n=%s, period %s…%s) = %.2f; net_effect=%.0f",
        len(tau_idx),
        bundle.periods["prod_tau_cal_period"][0],
        bundle.periods["prod_tau_cal_period"][1],
        thr,
        fe.net_effect,
    )
    print(
        f"τ prod подобран на DSM (raw) на τ-cal "
        f"({bundle.periods['prod_tau_cal_period'][0]} … "
        f"{bundle.periods['prod_tau_cal_period'][1]}, n={len(tau_idx)}): "
        f"{thr:.2f}"
    )
    return thr


def _create_dsm(
    built: dict[str, Any],
    config_key: str,
    external_config: Any,
) -> Any:
    config_path = built[config_key]
    dsm = DataSetsManager(
        config_name=str(config_path),
        external_config=external_config,
        prepared_datasets=prepare_datasets_from_config(config_path),
    )
    return dsm


def fit_parity_models(
    bundle: ExampleDatasetBundle,
    *,
    external_config: Any,
    threshold: float,
    use_automl: bool = True,
    send_mail: bool = False,
    log_mlflow: bool = False,
) -> ExampleDsmBundle:
    """Parity: CF+RG из ``config_parity.json`` (по умолчанию через AutoMLManager)."""
    if use_automl:
        dsm, _ = fit_automl_bundle(
            bundle.df,
            bundle.built["parity_path"],
            external_config=external_config,
            cf_name=bundle.cf_name,
            threshold=threshold,
            send_mail=send_mail,
            log_mlflow=log_mlflow,
        )
        return ExampleDsmBundle(dsm_cf=dsm, dsm_rg=dsm)

    dsm = _create_dsm(bundle.built, "parity_path", external_config)
    dsm.load_dataset(data=bundle.df)
    fit_dsm_classification(dsm, bundle.cf_name, threshold=threshold)
    patch_dsm_models(dsm)
    return ExampleDsmBundle(dsm_cf=dsm, dsm_rg=dsm)


def fit_prod_models(
    bundle: ExampleDatasetBundle,
    *,
    external_config: Any,
    threshold: float,
    parity: ExampleDsmBundle | None = None,
    use_automl: bool = True,
    send_mail: bool = False,
    log_mlflow: bool = False,
) -> ExampleDsmBundle:
    """Prod-refit: CF+RG из ``config_prod.json`` (AutoMLManager; MLflow опционально)."""
    base = parity or ExampleDsmBundle(dsm_cf=None, dsm_rg=None)
    if use_automl:
        dsm_prod, _ = fit_automl_bundle(
            bundle.df,
            bundle.built["prod_path"],
            external_config=external_config,
            cf_name=bundle.cf_name,
            threshold=threshold,
            send_mail=send_mail,
            log_mlflow=log_mlflow,
        )
        return ExampleDsmBundle(
            dsm_cf=base.dsm_cf,
            dsm_rg=base.dsm_rg,
            dsm_cf_prod=dsm_prod,
            dsm_rg_prod=dsm_prod,
        )

    dsm_prod = _create_dsm(bundle.built, "prod_path", external_config)
    dsm_prod.load_dataset(data=bundle.df)
    fit_dsm_classification(dsm_prod, bundle.cf_name, threshold=threshold)
    patch_dsm_models(dsm_prod)
    return ExampleDsmBundle(
        dsm_cf=base.dsm_cf,
        dsm_rg=base.dsm_rg,
        dsm_cf_prod=dsm_prod,
        dsm_rg_prod=dsm_prod,
    )


def run_parity_cross_test_metrics(
    models: ExampleDsmBundle,
    bundle: ExampleDatasetBundle,
    *,
    threshold: float,
) -> None:
    test_idx = bundle.periods["splits"].test
    test_prod_idx = bundle.periods["prod_holdout_idx"]
    for dsm, name, task, thr, ignore_filter in (
        (models.dsm_cf, bundle.cf_name, "classification", threshold, False),
        (models.dsm_rg, bundle.rg_name, "regression", None, False),
    ):
        display_dsm_collect_metrics_cross_test(
            dsm,
            name,
            bundle.df,
            task_type=task,
            val_threshold=thr,
            test_slices={"test": test_idx, "test_prod": test_prod_idx},
            title=f"{name}: train / test / test_prod",
            ignore_row_filter=ignore_filter,
        )


def _fin_effect_common_index(
    df: pd.DataFrame,
    index: pd.Index,
    proba: pd.Series,
    sev: pd.Series,
) -> pd.Index:
    """Строки выборки, на которых есть и факт, и оба предсказания.

    Единая проверка покрытия для точечного и bootstrap-расчёта: severity
    предсказывается на всех строках (без ``data_filter_condition``), иначе
    покрытие < 95% и расчёт не имеет смысла.
    """
    common = (
        pd.Index(index)
        .intersection(proba.dropna().index)
        .intersection(sev.dropna().index)
        .intersection(df.index)
    )
    if len(common) == 0:
        raise ValueError("Нет пересечения index с proba/sev: проверьте index предсказаний.")
    coverage = len(common) / max(len(pd.Index(index)), 1)
    if coverage < 0.95:
        raise ValueError(
            f"pred покрывает только {len(common)}/{len(index)} строк ({coverage:.1%}). "
            "Для severity нужен predict без data_filter_condition."
        )
    if len(common) < len(index):
        logger.info(
            "fin_effect: строк с pred %s/%s (без proba/sev: %s)",
            len(common),
            len(index),
            len(index) - len(common),
        )
    return common


def fin_effect_table(
    df: pd.DataFrame,
    index: pd.Index,
    proba: pd.Series,
    sev: pd.Series,
    *,
    threshold: float | None = None,
    title: str = "",
    render: bool = True,
) -> FinEffectTableResult:
    """Финэффект на index с фиксированным τ; опционально печать и display."""
    cfg = resolve_fin_effect_config(
        df,
        frequency_target="TARGET_FREQ",
        severity_target="TARGET_SEV",
    )
    common = _fin_effect_common_index(df, index, proba, sev)
    aligned = df.loc[common]
    fe = run_fin_effect_pipeline(
        aligned,
        proba.reindex(common),
        sev.reindex(common),
        aligned["TARGET_FREQ"],
        threshold=threshold,
        config=cfg,
    )
    summary = fe.summary_table(cfg)
    if render:
        if title:
            _markdown(f"### {title}")
        print_best_threshold_report(fe)
        _display(summary.style.format("{:,.0f}", subset=summary.columns[3:], na_rep="—"))
        print(
            f"проверка Σ: model={summary['ФИН. ЭФФЕКТ МОДЕЛЬ'].sum():,.0f} "
            f"(отчёт {fe.model_effect_total:,.0f}), "
            f"fact={summary['ФИН. ЭФФЕКТ ФАКТ'].sum():,.0f} "
            f"(отчёт {fe.fact_effect_total:,.0f}), "
            f"экон.={summary['Экономия'].sum():,.0f} "
            f"(отчёт net {fe.net_effect:,.0f})"
        )
        n_pos = int((aligned["TARGET_FREQ"] == 1).sum())
        n_neg = int((aligned["TARGET_FREQ"] == 0).sum())
        print(f"выборка: n = {len(aligned)}, TARGET_FREQ=1: {n_pos}, TARGET_FREQ=0: {n_neg}")
    return FinEffectTableResult(fin_effect=fe, summary=summary)


def render_bootstrap_fin_effect(
    result: BootstrapFinEffect,
    *,
    title: str = "",
) -> None:
    """Display: таблица фолдов + строка медианы и печать итоговой цифры."""
    if title:
        _markdown(f"### {title}")
    table = result.summary_table()
    _display(
        table.style.format(
            {
                "fold": "{:}",
                "n": "{:,.0f}",
                "n_fact_1": "{:,.0f}",
                "n_pred_1": "{:,.0f}",
                "thr": "{:.2f}",
                "net_effect": "{:,.0f}",
                "model_effect": "{:,.0f}",
                "fact_effect": "{:,.0f}",
            },
            na_rep="—",
        )
    )
    thr_text = "—" if result.threshold is None else f"{result.threshold:.2f}"
    print(
        f"Медиана по {result.n_folds} bootstrap-фолдам "
        f"(n = {result.n_rows}, seed = {result.seed}, τ = {thr_text}): "
        f"model={result.median_model_effect:,.0f} ₽, "
        f"fact={result.median_fact_effect:,.0f} ₽, "
        f"net={result.median_net_effect:,.0f} ₽ "
        f"[min {result.min_net_effect:,.0f}; max {result.max_net_effect:,.0f}; "
        f"std {result.std_net_effect:,.0f}]"
    )


def bootstrap_fin_effect_table(
    df: pd.DataFrame,
    index: pd.Index,
    proba: pd.Series,
    sev: pd.Series,
    *,
    threshold: float | None = None,
    n_folds: int = DEFAULT_BOOTSTRAP_FOLDS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    title: str = "",
    render: bool = True,
) -> BootstrapFinEffect:
    """Bootstrap-финэффект на index: ``n_folds`` выборок с возвращением → медиана.

    В отличие от ``fin_effect_table`` считает эффект не на всей выборке, а на
    ``n_folds`` bootstrap-фолдах; итог — медиана метрик. τ обычно фиксирован
    (prod DSM / parity).
    """
    cfg = resolve_fin_effect_config(
        df,
        frequency_target="TARGET_FREQ",
        severity_target="TARGET_SEV",
    )
    common = _fin_effect_common_index(df, index, proba, sev)
    aligned = df.loc[common]
    result = bootstrap_fin_effect(
        aligned,
        proba.reindex(common),
        sev.reindex(common),
        aligned["TARGET_FREQ"],
        threshold=threshold,
        config=cfg,
        n_folds=n_folds,
        seed=seed,
    )
    if render:
        render_bootstrap_fin_effect(result, title=title)
    return result


def run_test_fin_effect(
    models: ExampleDsmBundle,
    bundle: ExampleDatasetBundle,
    paths: ExamplePaths,
    *,
    threshold: float,
    fin_effect_collect: FinEffectResult | None = None,
    fin_effect_config_collect: Any | None = None,
) -> tuple[FinEffectTableResult, Path]:
    """Финэффект parity на holdout Test + HTML отчёт."""
    if fin_effect_collect is not None:
        _markdown("### Сравнение: collect C3 (Test)")
        print_best_threshold_report(fin_effect_collect)
        if fin_effect_config_collect is not None:
            sum_c = create_summary_table(fin_effect_collect.frame, fin_effect_config_collect)
            _display(sum_c.style.format("{:,.0f}", subset=sum_c.columns[3:], na_rep="—"))

    test_idx = bundle.periods["splits"].test
    proba_test = predict_cf(models.dsm_cf, bundle.cf_name, bundle.df.loc[test_idx])
    sev_test = predict_rg(models.dsm_rg, bundle.rg_name, bundle.df.loc[test_idx])
    result = fin_effect_table(
        bundle.df,
        test_idx,
        proba_test,
        sev_test,
        threshold=threshold,
        title=f"Финэффект на Test (τ = {threshold:.2f})",
    )

    fe_html = fin_effect_collect if fin_effect_collect is not None else result.fin_effect
    cfg_html = fin_effect_config_collect
    if cfg_html is None:
        cfg_html = resolve_fin_effect_config(
            bundle.df, frequency_target="TARGET_FREQ", severity_target="TARGET_SEV"
        )
    html_path = export_business_html(
        fe_html,
        cfg_html,
        path=paths.project_root / "notebooks" / "fin_effect_detailed.html",
        subtitle="Collect C3, Test" if fin_effect_collect is not None else "OutBoxML parity, Test",
    )
    logger.info("HTML для бизнеса: %s", html_path)
    return result, html_path


def run_prod_metrics(
    models: ExampleDsmBundle,
    bundle: ExampleDatasetBundle,
    *,
    threshold: float,
) -> None:
    display_dsm_collect_metrics(
        models.dsm_cf_prod,
        bundle.cf_name,
        task_type="classification",
        val_threshold=threshold,
        title=f"prod {bundle.cf_name}",
    )
    display_dsm_collect_metrics(
        models.dsm_rg_prod,
        bundle.rg_name,
        task_type="regression",
        title=f"prod {bundle.rg_name}",
    )


def _bootstrap_compare_row(
    train: str,
    train_period: tuple[str, str],
    *,
    n_test_prod: int,
    threshold: float,
    bootstrap: BootstrapFinEffect,
) -> dict[str, Any]:
    """Строка сводки: медиана по фолдам + разброс net_effect."""
    return {
        "train": train,
        "train_period": f"{train_period[0]} … {train_period[1]}",
        "n_test_prod": n_test_prod,
        "n_folds": bootstrap.n_folds,
        "seed": bootstrap.seed,
        "thr": threshold,
        "net_effect": bootstrap.median_net_effect,
        "net_effect_min": bootstrap.min_net_effect,
        "net_effect_max": bootstrap.max_net_effect,
        "net_effect_std": bootstrap.std_net_effect,
        "model_effect": bootstrap.median_model_effect,
        "fact_effect": bootstrap.median_fact_effect,
    }


def run_test_prod_fin_effect(
    models: ExampleDsmBundle,
    bundle: ExampleDatasetBundle,
    *,
    thresholds: ExampleThresholds,
    n_folds: int = DEFAULT_BOOTSTRAP_FOLDS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> TestProdCompareResult:
    """Bootstrap-финэффект на Test_prod (15% holdout, без τ-cal).

    Считаем не один эффект на всей выборке, а ``n_folds`` bootstrap-фолдов
    (выборки того же размера **с возвращением**); в результатах — **медиана**
    по фолдам и её разброс (min/max/std). τ фиксирован (prod DSM после подбора
    в example_final / parity из collect).

    Если есть parity DSM — сравнивает parity vs prod; иначе только prod
    (для ``example_final``).
    """
    test_prod_idx = bundle.periods["prod_holdout_idx"]
    logger.info(
        "Test_prod: n=%s, период %s … %s; bootstrap %s фолдов (seed=%s)",
        len(test_prod_idx),
        bundle.periods["prod_test_period"][0],
        bundle.periods["prod_test_period"][1],
        n_folds,
        seed,
    )

    rows: list[dict[str, Any]] = []
    fe_parity: BootstrapFinEffect | None = None
    if models.dsm_cf is not None and models.dsm_rg is not None:
        fe_parity = bootstrap_fin_effect_table(
            bundle.df,
            test_prod_idx,
            predict_cf(models.dsm_cf, bundle.cf_name, bundle.df.loc[test_prod_idx]),
            predict_rg(models.dsm_rg, bundle.rg_name, bundle.df.loc[test_prod_idx]),
            threshold=thresholds.parity,
            n_folds=n_folds,
            seed=seed,
            title=(
                f"Bootstrap-финэффект Test_prod — parity train "
                f"(τ = {thresholds.parity:.2f}, {n_folds} фолдов)"
            ),
        )
        rows.append(
            _bootstrap_compare_row(
                "parity",
                bundle.periods["parity_train_period"],
                n_test_prod=len(test_prod_idx),
                threshold=thresholds.parity,
                bootstrap=fe_parity,
            )
        )

    fe_prod = bootstrap_fin_effect_table(
        bundle.df,
        test_prod_idx,
        predict_cf(models.dsm_cf_prod, bundle.cf_name, bundle.df.loc[test_prod_idx]),
        predict_rg(models.dsm_rg_prod, bundle.rg_name, bundle.df.loc[test_prod_idx]),
        threshold=thresholds.prod,
        n_folds=n_folds,
        seed=seed,
        title=(
            f"Bootstrap-финэффект Test_prod — prod train "
            f"(τ = {thresholds.prod:.2f}, {n_folds} фолдов)"
        ),
    )
    rows.append(
        _bootstrap_compare_row(
            "prod",
            bundle.periods["prod_train_period"],
            n_test_prod=len(test_prod_idx),
            threshold=thresholds.prod,
            bootstrap=fe_prod,
        )
    )

    compare = pd.DataFrame(rows)
    title = (
        "Сводка: медиана финэффекта на Test_prod по bootstrap-фолдам "
        "(parity train vs prod train)"
        if fe_parity is not None
        else "Сводка: медиана финэффекта на Test_prod по bootstrap-фолдам (prod)"
    )
    _markdown(f"### {title}")
    _display(
        compare.style.format(
            {
                "n_test_prod": "{:,.0f}",
                "n_folds": "{:,.0f}",
                "seed": "{:,.0f}",
                "thr": "{:.2f}",
                "net_effect": "{:,.0f}",
                "net_effect_min": "{:,.0f}",
                "net_effect_max": "{:,.0f}",
                "net_effect_std": "{:,.0f}",
                "model_effect": "{:,.0f}",
                "fact_effect": "{:,.0f}",
            },
            na_rep="—",
        )
    )
    if fe_parity is not None:
        print(
            f"τ parity (meta val_threshold): {thresholds.parity:.2f}; "
            f"τ prod (meta best_threshold): {thresholds.prod:.2f}"
        )
        print(
            f"Итог (медиана {n_folds} фолдов, seed={seed}): "
            f"parity net={fe_parity.median_net_effect:,.0f} ₽; "
            f"prod net={fe_prod.median_net_effect:,.0f} ₽"
        )
    else:
        print(f"τ prod (meta best_threshold): {thresholds.prod:.2f}")
        print(
            f"Итог (медиана {n_folds} фолдов, seed={seed}): "
            f"prod net={fe_prod.median_net_effect:,.0f} ₽"
        )
    return TestProdCompareResult(
        parity=fe_parity,
        prod=fe_prod,
        compare_table=compare,
        test_prod_idx=test_prod_idx,
    )


def _show_figure(fig: Any, title: str) -> None:
    if fig is None:
        logger.warning("%s: figure is None", title)
        return
    show = getattr(fig, "show", None)
    if callable(show):
        show()
    else:
        _display(fig)
    print(title, type(fig))


def _figure_to_html_bytes(fig: Any) -> bytes | None:
    """Plotly → self-contained HTML bytes; None если не figure.

    Inline plotly.js; ``default_height=600`` — иначе height:100% → пустой canvas.
    """
    if fig is None:
        return None
    to_html = getattr(fig, "to_html", None)
    if not callable(to_html):
        return None
    return to_html(
        include_plotlyjs=True,
        full_html=True,
        default_width="100%",
        default_height=600,
    ).encode("utf-8")


def _save_figures_zip(
    figures: dict[str, Any],
    zip_path: Path,
) -> Path:
    """Упаковать Plotly-фигуры (FactorsPlot / cohort) в zip HTML."""
    import zipfile

    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, fig in figures.items():
            payload = _figure_to_html_bytes(fig)
            if payload is None:
                logger.warning("zip plots: skip %s (нет to_html)", name)
                continue
            safe = str(name).replace("/", "_").replace("\\", "_")
            zf.writestr(f"{safe}.html", payload)
            written += 1
    logger.info("FactorsPlot zip: %s html → %s", written, zip_path)
    print(
        f"FactorsPlot / cohort сохранены в zip: {zip_path} "
        f"({written} self-contained html)"
    )
    return zip_path


DEFAULT_SEGMENT_FIN_COLS: tuple[str, ...] = (
    "IS_TOTAL",
    "FL_PHOTO_VIDEO",
    "NOT_NOTIFICATION",
    "FILIAL",
    "FE_SHARE_WORK_TIER",
)


def run_segment_fin_effect(
    models: ExampleDsmBundle,
    bundle: ExampleDatasetBundle,
    *,
    threshold: float,
    segment_cols: tuple[str, ...] | list[str] = DEFAULT_SEGMENT_FIN_COLS,
    min_n: int = 30,
    rg_calibrator: SeverityCalibrator | None = None,
) -> pd.DataFrame:
    """Точечный финэффект prod на Test_prod по сегментам (фиксированный τ).

    Severity: raw DSM, либо ``rg_calibrator`` (isotonic), если передан.
    Мелкие сегменты (n < min_n) помечаются, но остаются в таблице.
    """
    from querulus.fin_effect.calculator import (
        apply_model_predictions,
        prepare_effect_frame,
    )
    from querulus.training.calibration import apply_severity_calibrator

    if models.dsm_cf_prod is None or models.dsm_rg_prod is None:
        raise ValueError("Нужны dsm_cf_prod / dsm_rg_prod")
    idx = bundle.periods["prod_holdout_idx"]
    df = bundle.df
    proba = predict_cf(models.dsm_cf_prod, bundle.cf_name, df.loc[idx])
    sev = predict_rg(models.dsm_rg_prod, bundle.rg_name, df.loc[idx])
    if rg_calibrator is not None:
        sev = apply_severity_calibrator(rg_calibrator, sev)
    cfg = resolve_fin_effect_config(
        df,
        frequency_target="TARGET_FREQ",
        severity_target="TARGET_SEV",
    )
    common = proba.index.intersection(sev.index).intersection(df.index)
    prepared = prepare_effect_frame(df.loc[common], cfg)
    proba = proba.reindex(prepared.index)
    sev_s = (
        sev.reindex(prepared.index)
        if isinstance(sev, pd.Series)
        else pd.Series(sev, index=prepared.index)
    )
    rows: list[dict[str, Any]] = []
    full = apply_model_predictions(
        prepared,
        proba,
        sev_s,
        prepared["TARGET_FREQ"],
        threshold=threshold,
        config=cfg,
    )
    rows.append(
        {
            "segment_col": "_all",
            "segment": "ALL",
            "n": int(len(prepared)),
            "n_ok": True,
            "thr": float(threshold),
            "net_effect": float(full.net_effect),
            "model_effect": float(full.model_effect_total),
            "fact_effect": float(full.fact_effect_total),
        }
    )
    for col in segment_cols:
        if col not in prepared.columns:
            logger.warning("segment fin: колонки %s нет в df — skip", col)
            continue
        for seg_val, part in prepared.groupby(col, dropna=False):
            part_idx = part.index
            if len(part_idx) == 0:
                continue
            fe = apply_model_predictions(
                part,
                proba.loc[part_idx],
                sev_s.loc[part_idx],
                part["TARGET_FREQ"],
                threshold=threshold,
                config=cfg,
            )
            rows.append(
                {
                    "segment_col": col,
                    "segment": str(seg_val),
                    "n": int(len(part_idx)),
                    "n_ok": bool(len(part_idx) >= min_n),
                    "thr": float(threshold),
                    "net_effect": float(fe.net_effect),
                    "model_effect": float(fe.model_effect_total),
                    "fact_effect": float(fe.fact_effect_total),
                }
            )
    table = pd.DataFrame(rows).sort_values(
        ["segment_col", "n"], ascending=[True, False]
    )
    _display(table)
    print(
        f"Сегментный fin effect (Test_prod, τ={threshold:.2f}, "
        f"sev={'isotonic' if rg_calibrator is not None else 'raw'}): "
        f"{len(table)} строк"
    )
    return table


def run_prod_plots_and_email(
    models: ExampleDsmBundle,
    bundle: ExampleDatasetBundle,
    *,
    external_config: Any,
    send_email: bool = True,
    results_dir: Path | str | None = None,
    save_plots_zip: bool = True,
    cf_calibrator: Any | None = None,
    cf_calibration_label: str = "querulus_cal",
    fit_rg_isotonic: bool = True,
    rg_isotonic_min_samples: int = 50,
    rg_calibrator: SeverityCalibrator | None = None,
    show_figures: bool = False,
    plots_tag: str = "example",
) -> ProdPlotsResult:
    """FactorsPlot + cohort (querulus) → zip HTML; опционально email.

    CF: Fact / raw / ``cf_calibrator`` (querulus_cal); cat-ось — сырые labels из df.
    RG: Fact / raw / sev_isotonic (учится на τ-cal, если ``fit_rg_isotonic``).
    Все num+cat из DSM. OutBoxML не меняем.

    ``plots_tag`` — суффикс zip (`example` / `example_final`), чтобы ноутбуки
    не затирали архивы друг друга.
    """
    if models.dsm_cf_prod is None or models.dsm_rg_prod is None:
        raise ValueError("Нужны dsm_cf_prod / dsm_rg_prod")

    sev_cal = rg_calibrator
    if sev_cal is None and fit_rg_isotonic:
        try:
            sev_cal = fit_rg_severity_calibrator_isotonic(
                models.dsm_rg_prod,
                bundle.rg_name,
                bundle.df,
                bundle.periods,
                prefer_prod_tau=True,
                min_samples=rg_isotonic_min_samples,
            )
            print(
                f"RG severity isotonic: n_fit={sev_cal.n_fit}, "
                f"bias_before={sev_cal.bias_before:,.0f}, "
                f"bias_after={sev_cal.bias_after:,.0f}"
            )
        except ValueError as exc:
            logger.warning("RG sev_isotonic не обучен: %s", exc)
            print(f"RG sev_isotonic skip: {exc}")
            sev_cal = None

    figures = build_factors_figures(
        dsm_cf=models.dsm_cf_prod,
        cf_name=bundle.cf_name,
        dsm_rg=models.dsm_rg_prod,
        rg_name=bundle.rg_name,
        df=bundle.df,
        cf_calibrator=cf_calibrator,
        rg_calibrator=sev_cal,
    )
    print(f"FactorsPlot figures: {len(figures)}")
    if show_figures:
        for key, fig in figures.items():
            _show_figure(fig, key)

    zip_path: Path | None = None
    if save_plots_zip and figures:
        out_dir = Path(results_dir) if results_dir is not None else Path("results")
        parts = [str(plots_tag).strip()] if plots_tag else []
        if cf_calibrator is not None and cf_calibration_label:
            parts.append(cf_calibration_label)
        if sev_cal is not None:
            parts.append("sev_isotonic")
        suffix = ("_" + "_".join(p for p in parts if p)) if parts else ""
        zip_path = _save_figures_zip(
            figures,
            out_dir / f"querulus_prod_plots_{bundle.model_version}{suffix}.zip",
        )

    if send_email:
        prod_results: dict[str, Any] = {}
        prod_results.update(models.dsm_cf_prod.get_result())
        prod_results.update(models.dsm_rg_prod.get_result())
        try:
            QuerulusEMailDSResult(
                config=external_config,
                ds_manager_result=prod_results,
            ).success_mail(group_name=f"querulus_{bundle.model_version}")
            print("QuerulusEMailDSResult: письмо отправлено")
        except Exception as exc:
            logger.warning(
                "QuerulusEMailDSResult не отправлено: %s: %s",
                type(exc).__name__,
                exc,
            )

    return ProdPlotsResult(zip_path=zip_path, rg_calibrator=sev_cal)


def export_prod_service_artifacts(
    models: ExampleDsmBundle,
    bundle: ExampleDatasetBundle,
    paths: ExamplePaths,
    *,
    thresholds: ExampleThresholds,
    ensemble_pickle_name: str = SERVICE_ENSEMBLE_PICKLE,
) -> ProdExportResult:
    """Sidecar для сервиса: ``df_for_service.parquet`` + ``metadata.json``.

    Модели **не** пишет — их сохраняет ``save_prod_models_via_automl`` (AutoML).
    ``best_threshold`` в meta — τ для shadow. DQ clip — в OutBoxML feature.clip.
    """
    if models.dsm_cf_prod is None or models.dsm_rg_prod is None:
        raise ValueError("Нужны dsm_cf_prod / dsm_rg_prod")

    paths.results_dir.mkdir(parents=True, exist_ok=True)
    dq_bounds_path: Path | None = None
    ensemble_pkl = paths.results_dir / ensemble_pickle_name

    df_service = bundle.df.copy()
    df_service["preds_cf"] = predict_cf(models.dsm_cf_prod, bundle.cf_name, bundle.df)
    df_service["preds_rg"] = predict_rg(models.dsm_rg_prod, bundle.rg_name, bundle.df)
    service_df_path = paths.results_dir / "df_for_service.parquet"
    df_service.to_parquet(service_df_path, index=True)

    periods = bundle.periods
    meta_path = paths.results_dir / "metadata.json"
    meta: dict[str, Any] = {
        "model_name": "querulus",
        "model_version": bundle.model_version,
        "periods": {
            k: list(v) if isinstance(v, tuple) else v
            for k, v in periods.items()
            if k
            in {
                "parity_train_period",
                "parity_test_period",
                "prod_train_period",
                "prod_tau_cal_period",
                "prod_test_period",
                "prod_cutoff",
                "date_column",
                "cal_period",
            }
        },
        "cf_name": bundle.cf_name,
        "rg_name": bundle.rg_name,
        "best_threshold": float(thresholds.prod),
        "val_threshold": float(thresholds.parity),
        "threshold_scale": "raw",
        "threshold_source": "example_final_dsm_tau_cal",
        "calibration": None,
        "artifacts": {
            "ensemble": str(ensemble_pkl),
            "ensemble_note": "пишет AutoML save_results → копия "
            f"{ensemble_pickle_name}",
            "dq_bounds": None,
            "dq_clip_in_outboxml_configs": True,
            "data_quality_report": str(
                paths.project_root / "data" / "processed" / "data_quality_report.json"
            ),
            "df_for_service": str(service_df_path),
            "configs_dir": bundle.built.get("configs_dir"),
        },
        "preds_cf_col": "preds_cf",
        "preds_rg_col": "preds_rg",
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("=== Sidecar для сервиса (до AutoML pickle) ===")
    print(f"  df_for_service : {service_df_path}")
    print(f"    shape        : {df_service.shape[0]:,} × {df_service.shape[1]}")
    print(f"    preds_cf NA  : {df_service['preds_cf'].isna().mean():.1%}")
    print(f"    preds_rg NA  : {df_service['preds_rg'].isna().mean():.1%}")
    print(f"  metadata.json  : {meta_path}")
    print(f"    best_threshold (τ): {meta['best_threshold']:.2f}")
    print(f"    ensemble (ожидается): {ensemble_pkl}")
    print("=== Sidecar готов — дальше save_prod_models_via_automl ===")

    return ProdExportResult(
        service_df_path=service_df_path,
        meta_path=meta_path,
        ensemble_pkl=ensemble_pkl if ensemble_pkl.is_file() else None,
        dq_bounds_path=dq_bounds_path,
        meta=meta,
    )


def save_prod_models_via_automl(
    models: ExampleDsmBundle,
    *,
    results_dir: Path | str | None = None,
    stable_ensemble_name: str = SERVICE_ENSEMBLE_PICKLE,
) -> Path:
    """Сохранить CF+RG через ``AutoMLManager.save_results`` + стабильная копия.

    Пишет timestamped pickle AutoML и копирует в ``querulus_ansamble.pickle``
    для shadow. Вызывать **после** ``export_prod_service_artifacts``.
    """
    import shutil

    automl = models.dsm_cf_prod
    if automl is None:
        raise ValueError("Нужен dsm_cf_prod (AutoMLManager после fit_prod_models)")
    if not hasattr(automl, "save_results"):
        raise TypeError(
            f"dsm_cf_prod={type(automl).__name__} без save_results — нужен AutoMLManager"
        )
    results = automl.get_result()
    if not results:
        raise ValueError("Пустой get_result() — нечего сохранять")

    if results_dir is not None:
        out = Path(results_dir)
        out.mkdir(parents=True, exist_ok=True)
        automl._external_config.results_path = out

    for res in results.values():
        res.model = ensure_predictable_model(res.model)

    automl.save_results(results)
    results_path = Path(automl._external_config.results_path)
    stamped = getattr(automl.automl_results, "result_pickle_name", None)
    if not stamped:
        raise RuntimeError("AutoML save_results не выставил result_pickle_name")
    src = results_path / stamped
    if not src.is_file():
        raise FileNotFoundError(f"AutoML pickle не найден: {src}")
    dst = results_path / stable_ensemble_name
    shutil.copy2(src, dst)
    logger.info("AutoML pickle: %s → stable %s", src.name, dst.name)
    print(f"=== AutoML models saved ===\n  {src}\n  stable: {dst}")
    return dst
