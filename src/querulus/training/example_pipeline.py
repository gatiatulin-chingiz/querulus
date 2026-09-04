"""Пайплайн example.ipynb: загрузка данных, DSM, финэффект, экспорт prod."""
from __future__ import annotations

import json
import logging
import pickle
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from querulus.dataset.hadoop import load_df_final
from querulus.fin_effect import (
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
    MODEL_VERSION,
    artifacts_dir_for_version,
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
from querulus.training.outboxml_metrics import (
    display_dsm_collect_metrics,
    display_dsm_collect_metrics_cross_test,
    predict_dsm_series,
)

logger = logging.getLogger(__name__)


def _display(obj: Any) -> None:
    try:
        from IPython.display import display as ipy_display
    except ImportError:
        print(obj)
        return
    ipy_display(obj)


def _markdown(title: str) -> None:
    try:
        from IPython.display import Markdown, display as ipy_display
    except ImportError:
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
    parity: FinEffectResult | None
    prod: FinEffectResult
    compare_table: pd.DataFrame
    test_prod_idx: pd.Index


@dataclass
class ProdExportResult:
    cf_pkl: Path
    rg_pkl: Path
    ensemble_pkl: Path
    service_df_path: Path
    meta_path: Path
    dq_bounds_path: Path | None
    meta: dict[str, Any]


def resolve_example_paths(
    project_root: Path | str,
    *,
    use_synthetic: bool = False,
    results_dir: Path | str | None = None,
) -> ExamplePaths:
    """Пути parquet/Hive и каталог артефактов ``integration/results/querulus/{version}``."""
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
        else artifacts_dir_for_version(
            MODEL_VERSION, results_root=root / "integration" / "results"
        )
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
        paths.local_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_parquet(paths.local_parquet_path, index=False)
        logger.info("кэш после Hive записан в parquet: %s", paths.local_parquet_path)
    else:
        logger.info("Hive не использован; данные из файла: %s", paths.local_parquet_path)

    if paths.use_synthetic:
        paths.local_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_parquet(paths.local_parquet_path, index=False)

    df = df_raw
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
    parity = load_collect_val_threshold(project_root, training=collect_training)
    prod = load_collect_prod_threshold(project_root)
    logger.info("τ parity (Val) = %.2f; τ prod (τ-cal) = %.2f", parity, prod)
    return ExampleThresholds(parity=parity, prod=prod)


def _create_dsm(
    built: dict[str, Any],
    config_key: str,
    external_config: Any,
) -> Any:
    from outboxml.datasets_manager import DataSetsManager

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
    log_mlflow: bool = True,
) -> ExampleDsmBundle:
    """Prod-refit: CF+RG из ``config_prod.json`` (AutoMLManager + MLflow по умолчанию)."""
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


def run_test_prod_fin_effect(
    models: ExampleDsmBundle,
    bundle: ExampleDatasetBundle,
    *,
    thresholds: ExampleThresholds,
) -> TestProdCompareResult:
    """Финэффект на Test_prod (15% holdout, без τ-cal).

    Если есть parity DSM — сравнивает parity vs prod; иначе только prod
    (для ``example_final``).
    """
    test_prod_idx = bundle.periods["prod_holdout_idx"]
    logger.info(
        "Test_prod: n=%s, период %s … %s",
        len(test_prod_idx),
        bundle.periods["prod_test_period"][0],
        bundle.periods["prod_test_period"][1],
    )

    rows: list[dict[str, Any]] = []
    fe_parity: FinEffectResult | None = None
    if models.dsm_cf is not None and models.dsm_rg is not None:
        fe_parity = fin_effect_table(
            bundle.df,
            test_prod_idx,
            predict_cf(models.dsm_cf, bundle.cf_name, bundle.df.loc[test_prod_idx]),
            predict_rg(models.dsm_rg, bundle.rg_name, bundle.df.loc[test_prod_idx]),
            threshold=thresholds.parity,
            title=f"Финэффект Test_prod — parity train (τ = {thresholds.parity:.2f})",
        ).fin_effect
        rows.append(
            {
                "train": "parity",
                "train_period": (
                    f"{bundle.periods['parity_train_period'][0]} … "
                    f"{bundle.periods['parity_train_period'][1]}"
                ),
                "n_test_prod": len(test_prod_idx),
                "thr": thresholds.parity,
                "net_effect": fe_parity.net_effect,
                "model_effect": fe_parity.model_effect_total,
                "fact_effect": fe_parity.fact_effect_total,
            }
        )

    fe_prod = fin_effect_table(
        bundle.df,
        test_prod_idx,
        predict_cf(models.dsm_cf_prod, bundle.cf_name, bundle.df.loc[test_prod_idx]),
        predict_rg(models.dsm_rg_prod, bundle.rg_name, bundle.df.loc[test_prod_idx]),
        threshold=thresholds.prod,
        title=f"Финэффект Test_prod — prod train (τ = {thresholds.prod:.2f})",
    ).fin_effect
    rows.append(
        {
            "train": "prod",
            "train_period": (
                f"{bundle.periods['prod_train_period'][0]} … "
                f"{bundle.periods['prod_train_period'][1]}"
            ),
            "n_test_prod": len(test_prod_idx),
            "thr": thresholds.prod,
            "net_effect": fe_prod.net_effect,
            "model_effect": fe_prod.model_effect_total,
            "fact_effect": fe_prod.fact_effect_total,
        }
    )

    compare = pd.DataFrame(rows)
    title = (
        "Сводка: финэффект на Test_prod (parity train vs prod train)"
        if fe_parity is not None
        else "Сводка: финэффект на Test_prod (prod)"
    )
    _markdown(f"### {title}")
    _display(
        compare.style.format(
            {"net_effect": "{:,.0f}", "model_effect": "{:,.0f}", "fact_effect": "{:,.0f}"},
            na_rep="—",
        )
    )
    if fe_parity is not None:
        print(
            f"τ parity (meta val_threshold): {thresholds.parity:.2f}; "
            f"τ prod (meta best_threshold): {thresholds.prod:.2f}"
        )
    else:
        print(f"τ prod (meta best_threshold): {thresholds.prod:.2f}")
    return TestProdCompareResult(
        parity=fe_parity,
        prod=fe_prod,
        compare_table=compare,
        test_prod_idx=test_prod_idx,
    )


def _plot_features(dsm: Any, model_name: str, *, n_num: int = 6, n_cat: int = 6) -> list[str]:
    subset = dsm.get_result()[model_name].data_subset
    nums = list(subset.features_numerical or [])[:n_num]
    cats = list(subset.features_categorical or [])[:n_cat]
    return nums + cats


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


def _show_factors(export: Any, model_name: str, features: list[str], *, bins: int = 5) -> None:
    for feat in features:
        fig = export.plots(
            model_name=model_name,
            features=[feat],
            plot_type=1,
            bins_for_numerical_features=bins,
            use_exposure=False,
            only_test=True,
        )
        _show_figure(fig, f"FactorsPlot {model_name}: {feat}")


def run_prod_plots_and_email(
    models: ExampleDsmBundle,
    bundle: ExampleDatasetBundle,
    *,
    external_config: Any,
    send_email: bool = True,
) -> None:
    """FactorsPlot, cohort и опционально QuerulusEMailDSResult."""
    from outboxml.export_results import ResultExport

    export_cf = ResultExport(ds_manager=models.dsm_cf_prod, config=external_config)
    export_rg = ResultExport(ds_manager=models.dsm_rg_prod, config=external_config)
    cf_plot_feats = _plot_features(models.dsm_cf_prod, bundle.cf_name)
    rg_plot_feats = _plot_features(models.dsm_rg_prod, bundle.rg_name)
    print("FactorsPlot features CF:", cf_plot_feats)
    print("FactorsPlot features RG:", rg_plot_feats)

    _show_factors(export_cf, bundle.cf_name, cf_plot_feats)
    _show_factors(export_rg, bundle.rg_name, rg_plot_feats)

    fig_cf_cohort = export_cf.plots(
        model_name=bundle.cf_name,
        plot_type=2,
        use_exposure=False,
        only_test=True,
        cut_min_value=0.1,
        cut_max_value=0.9,
        samples=100,
        cohort_base="model",
    )
    fig_rg_cohort = export_rg.plots(
        model_name=bundle.rg_name,
        plot_type=2,
        use_exposure=False,
        only_test=True,
        cut_min_value=0.1,
        cut_max_value=0.9,
        samples=100,
        cohort_base="model",
    )
    _show_figure(fig_cf_cohort, "Cohort plot_type=2 CF")
    _show_figure(fig_rg_cohort, "Cohort plot_type=2 RG")

    if not send_email:
        return
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
        logger.warning("QuerulusEMailDSResult не отправлено: %s: %s", type(exc).__name__, exc)


def export_prod_service_artifacts(
    models: ExampleDsmBundle,
    bundle: ExampleDatasetBundle,
    paths: ExamplePaths,
    *,
    thresholds: ExampleThresholds,
) -> ProdExportResult:
    """Pickle, df_for_service, querulus_meta JSON.

    DQ-границы — в OutBoxML ``feature.clip`` (collect пишет конфиги через
    ``write_outboxml_configs``); отдельный ``querulus_dq_bounds_*.json`` не пишем.
    """
    cf_export = models.dsm_cf_prod.get_result()[bundle.cf_name].dict_for_prod_export()
    rg_export = models.dsm_rg_prod.get_result()[bundle.rg_name].dict_for_prod_export()
    cf_export["model"] = ensure_predictable_model(cf_export["model"])
    rg_export["model"] = ensure_predictable_model(rg_export["model"])

    cf_pkl = paths.results_dir / "querulus_cf_for_prod.pickle"
    rg_pkl = paths.results_dir / "querulus_rg_for_prod.pickle"
    ans_pkl = paths.results_dir / "querulus_ansamble.pickle"
    dq_bounds_path: Path | None = None

    cf_pkl.write_bytes(pickle.dumps([cf_export]))
    rg_pkl.write_bytes(pickle.dumps([rg_export]))
    ensemble = [deepcopy(cf_export), deepcopy(rg_export)]
    ans_pkl.write_bytes(pickle.dumps(ensemble))

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
        "calibration": None,
        "artifacts": {
            "cf": str(cf_pkl),
            "rg": str(rg_pkl),
            "ensemble": str(ans_pkl),
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

    print("=== Экспорт артефактов для сервиса (prod) ===")
    print(f"  frequency (CF), pickle : {cf_pkl}")
    print(f"  severity (RG), pickle    : {rg_pkl}")
    print(f"  ансамбль CF+RG, pickle   : {ans_pkl}")
    print("  DQ clip: в OutBoxML config feature.clip (data_quality_report)")
    print()
    print("  df_for_service — parquet исходного df + колонки preds_cf / preds_rg:")
    print(f"    путь      : {service_df_path}")
    print(f"    shape     : {df_service.shape[0]:,} строк × {df_service.shape[1]} колонок")
    print(f"    preds_cf  : доля NA = {df_service['preds_cf'].isna().mean():.1%}")
    print(f"    preds_rg  : доля NA = {df_service['preds_rg'].isna().mean():.1%}")
    print()
    print("  метаданные — периоды, пути артефактов, τ frequency:")
    print(f"    JSON      : {meta_path}")
    print(f"    τ (collect, frequency best_threshold): {meta['best_threshold']:.2f}")
    print("=== Готово ===")

    return ProdExportResult(
        cf_pkl=cf_pkl,
        rg_pkl=rg_pkl,
        ensemble_pkl=ans_pkl,
        service_df_path=service_df_path,
        meta_path=meta_path,
        dq_bounds_path=dq_bounds_path,
        meta=meta,
    )
