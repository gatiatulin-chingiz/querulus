"""Финэффект Querulus → Motor.agrafana.Querulus_FinEffect (wide).

Источники (prod) — это НЕ Oracle SPR:
  * monitoring — OISUU ``ВитринаСутяжность`` (или файл FIN_EFFECT_MONITORING_PATH)
  * retro      — Hive train slot=main из ``DEFAULT_DRIFT_MODELS``
    (та же таблица, что datadrift); override: FIN_EFFECT_RETRO_PATH

Расчёт — пакет ``querulus.fin_effect`` (ITT / coverage / annual / paths).

Зерно wide-таблицы: (week_start, stem, slot, filial)
  filial='(все)' → ITT / annual_* / overall coverage / path lifts
  filial=<имя>  → coverage + path shares + instruments attn

Synthetic — только в ``local/synthetic/run_fin_effect.py``.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PIPELINE_ROOT = Path(__file__).resolve().parent
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))
import _bootstrap_path  # noqa: F401 — monorepo + standalone GitLab clone

# monorepo: examples/querulus/src — на контуре пакет querulus должен быть в PYTHONPATH
_QUERULUS_SRC = _PIPELINE_ROOT.parent.parent / "src"
if _QUERULUS_SRC.is_dir() and str(_QUERULUS_SRC) not in sys.path:
    sys.path.insert(0, str(_QUERULUS_SRC))

from pipeline.config import DEFAULT_DRIFT_MODELS
from pipeline.settings import Settings, load_settings
from pipeline.db import connect_oisuu
from pipeline.write_sql import get_engine, write_append_guarded

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("fin_effect")

# Коды РезультатПроверки в витрине: −100 = вне модели, 0/1 = контроль/ручеёк
RESULT_OUT_OF_MODEL = -100
RESULT_ELIGIBLE = {0, 1, RESULT_OUT_OF_MODEL}
FILIAL_ALL = "(все)"  # агрегат «по всей сети» в wide-таблице
# Метки в Motor/Grafana: финэффект только для slot=main; stem по умолчанию «prod».
FIN_EFFECT_STEM = "prod"
FIN_EFFECT_SLOT = "main"
FIN_EFFECT_VITRINA = "[OISUU_report].[dbo].[ВитринаСутяжность]"

PILOT_FILIALS = (
    "Владимирский",
    "Кемеровский",
    "Курский",
    "Магнитогорский",
    "Мурманский",
    "Омский",
    "Пермский",
    "Петропавловск-Камчатский",
    "Уфимский",
    "Ярославский",
)

WIDE_COLS = [
    "week_start",
    "stem",
    "slot",
    "model_label",
    "filial",
    "n_incidents",
    "n_model",
    "n_control",
    "itt",
    "net",
    "coverage",
    "formula_version",
    "annual_window_yult",
    "annual_pilot_current_yult",
    "annual_pilot_full_yult",
    "annual_network_current_yult",
    "annual_network_full_yult",
    "annual_network_full_compliance_yult",
    "agreement_share_control",
    "agreement_share_model",
    "agreement_share_lift_pp",
    "pretension_share_control",
    "pretension_share_model",
    "pretension_share_lift_pp",
    "fu_share_control",
    "fu_share_model",
    "fu_share_lift_pp",
    "court_share_control",
    "court_share_model",
    "court_share_lift_pp",
    "agr_control_pct",
    "agr_model_pct",
    "gap_pp",
]

_PATH_SHARE_COLS = [
    "n_control",
    "agreement_share_control",
    "agreement_share_model",
    "agreement_share_lift_pp",
    "pretension_share_control",
    "pretension_share_model",
    "pretension_share_lift_pp",
    "fu_share_control",
    "fu_share_model",
    "fu_share_lift_pp",
    "court_share_control",
    "court_share_model",
    "court_share_lift_pp",
]

_ATTN_COLS = ["agr_control_pct", "agr_model_pct", "gap_pp"]

_ANNUAL_COLS = (
    "annual_window_yult",
    "annual_pilot_current_yult",
    "annual_pilot_full_yult",
    "annual_network_current_yult",
    "annual_network_full_yult",
    "annual_network_full_compliance_yult",
)


@dataclass(frozen=True)
class FinEffectJob:
    """Один прогон fin_effect (обычно один stem / pickle в слоте main)."""

    monitoring_df: pd.DataFrame
    retro_df: pd.DataFrame
    stem: str
    slot: str = "main"
    model_label: str = ""
    pilot_start: str | None = None
    as_of: str | None = None
    retro_as_of: str | None = None


@dataclass(frozen=True)
class FinEffectComputeResult:
    """Недельный финэффект + доля model по филиалам + доли путей + attn."""

    weekly: pd.DataFrame
    coverage_by_filial: pd.DataFrame
    path_shares: pd.DataFrame
    instruments_attention: pd.DataFrame


def compute_fin_effect_weekly(job: FinEffectJob, **kwargs: Any) -> FinEffectComputeResult:
    """Понедельный ряд ITT/Yult + coverage по филиалам + path/attn снимки."""
    from querulus.fin_effect.monitoring_report import FORMULA_VERSION
    from querulus.fin_effect.monitoring_snapshots import (
        PILOT_START_DEFAULT,
        filter_monitoring_by_application_date,
        run_weekly_monitoring_series,
    )

    pilot_start = job.pilot_start or PILOT_START_DEFAULT
    params = {
        "pilot_start": pilot_start,
        "as_of": job.as_of,
        "retro_as_of": job.retro_as_of,
        "bootstrap_iterations": 0,
        "bootstrap_compliance_iterations": 0,
        "bootstrap_n_jobs": 1,
        "min_rows": 50,
    }
    params.update(kwargs)
    logger.info(
        "fin_effect via querulus || stem=%s || formula=%s || pilot_start=%s",
        job.stem,
        FORMULA_VERSION,
        pilot_start,
    )
    series = run_weekly_monitoring_series(
        job.monitoring_df,
        job.retro_df,
        **params,
    )
    label = job.model_label or job.stem
    weekly = _weekly_frame(
        series.weekly,
        stem=job.stem,
        slot=job.slot,
        model_label=label,
        formula_version=FORMULA_VERSION,
    )
    coverage_filial = _coverage_by_filial_weekly(
        job.monitoring_df,
        week_ends=series.week_ends,
        pilot_start=pilot_start,
        stem=job.stem,
        slot=job.slot,
        model_label=label,
        formula_version=FORMULA_VERSION,
        filter_fn=filter_monitoring_by_application_date,
    )
    path_shares = _paths_frame(
        series.path_shares_latest,
        stem=job.stem,
        slot=job.slot,
        model_label=label,
        formula_version=FORMULA_VERSION,
    )
    instruments_attention = _instruments_attn_frame(
        series.instruments_attention_latest,
        stem=job.stem,
        slot=job.slot,
        model_label=label,
        formula_version=FORMULA_VERSION,
    )
    return FinEffectComputeResult(
        weekly=weekly,
        coverage_by_filial=coverage_filial,
        path_shares=path_shares,
        instruments_attention=instruments_attention,
    )


def build_wide_frame(result: FinEffectComputeResult) -> pd.DataFrame:
    """Merge weekly + filial coverage + paths + attn → one wide table."""
    keys = ["week_start", "stem", "slot", "filial"]
    weekly = result.weekly.copy()
    if not weekly.empty:
        weekly["filial"] = FILIAL_ALL
        if "n_control" not in weekly.columns:
            weekly["n_control"] = np.nan
        for col in _ATTN_COLS:
            if col not in weekly.columns:
                weekly[col] = np.nan

    filial = result.coverage_by_filial.copy()
    if not filial.empty:
        for col in (
            "itt",
            "net",
            "n_control",
            *_ANNUAL_COLS,
            *[c for c in _PATH_SHARE_COLS if c != "n_control"],
            *_ATTN_COLS,
        ):
            if col not in filial.columns:
                filial[col] = np.nan
        if "model_label" not in filial.columns and not weekly.empty:
            filial["model_label"] = weekly["model_label"].iloc[0]

    base_parts = [p for p in (weekly, filial) if p is not None and not p.empty]
    if not base_parts:
        return pd.DataFrame(columns=WIDE_COLS)
    wide = pd.concat(base_parts, ignore_index=True)

    paths = result.path_shares.copy()
    if not paths.empty:
        path_keep = keys + [c for c in _PATH_SHARE_COLS if c in paths.columns]
        # also keep n_model from paths if base has NaN for latest snapshot rows
        if "n_model" in paths.columns:
            path_keep = path_keep + ["n_model"]
        path_keep = list(dict.fromkeys(path_keep))
        paths = paths[path_keep]
        wide = wide.merge(paths, on=keys, how="left", suffixes=("", "_path"))
        for col in _PATH_SHARE_COLS:
            pcol = f"{col}_path"
            if pcol in wide.columns:
                wide[col] = wide[col].combine_first(wide[pcol])
                wide = wide.drop(columns=[pcol])
        if "n_model_path" in wide.columns:
            wide["n_model"] = wide["n_model"].combine_first(wide["n_model_path"])
            wide = wide.drop(columns=["n_model_path"])

    attn = result.instruments_attention.copy()
    if not attn.empty:
        attn_keep = keys + [c for c in _ATTN_COLS if c in attn.columns]
        attn = attn[attn_keep]
        wide = wide.merge(attn, on=keys, how="left", suffixes=("", "_attn"))
        for col in _ATTN_COLS:
            acol = f"{col}_attn"
            if acol in wide.columns:
                wide[col] = wide[col].combine_first(wide[acol])
                wide = wide.drop(columns=[acol])

    return _finalize_wide(wide)


def _finalize_wide(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if work.empty:
        return pd.DataFrame(columns=WIDE_COLS)
    work["week_start"] = pd.to_datetime(work["week_start"], errors="coerce").dt.date
    work["filial"] = work["filial"].fillna("(пусто)").astype(str)
    if "model_label" not in work.columns:
        work["model_label"] = work["stem"].astype(str)
    if "slot" not in work.columns:
        work["slot"] = "main"
    if "formula_version" not in work.columns:
        from querulus.fin_effect.monitoring_report import FORMULA_VERSION

        work["formula_version"] = FORMULA_VERSION
    for col in WIDE_COLS:
        if col not in work.columns:
            work[col] = np.nan
    numeric = [
        c
        for c in WIDE_COLS
        if c
        not in {
            "week_start",
            "stem",
            "slot",
            "model_label",
            "filial",
            "formula_version",
        }
    ]
    for col in numeric:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    return (
        work[WIDE_COLS]
        .sort_values(["week_start", "stem", "filial"])
        .reset_index(drop=True)
    )


def _weekly_frame(
    weekly: pd.DataFrame,
    *,
    stem: str,
    slot: str,
    model_label: str,
    formula_version: str,
) -> pd.DataFrame:
    if weekly.empty:
        return pd.DataFrame()

    ok = weekly
    if "status" in weekly.columns:
        ok = weekly.loc[weekly["status"].eq("ok")].copy()
    if ok.empty:
        logger.warning("fin_effect weekly: no ok rows for stem=%s", stem)
        return pd.DataFrame()

    week_col = "week_end" if "week_end" in ok.columns else "snapshot_at"
    n_rows = pd.to_numeric(ok.get("n_rows"), errors="coerce")
    n_model = pd.to_numeric(ok.get("n_model"), errors="coerce")
    itt = pd.to_numeric(ok.get("yult_effect"), errors="coerce")
    annual_window = pd.to_numeric(ok.get("annual_window_yult"), errors="coerce")
    coverage = (n_model / n_rows).where(n_rows > 0)

    return pd.DataFrame(
        {
            "week_start": pd.to_datetime(ok[week_col], errors="coerce").dt.date,
            "stem": stem,
            "slot": slot,
            "model_label": model_label,
            "n_incidents": n_rows,
            "itt": itt,
            "net": annual_window,
            "coverage": coverage,
            "formula_version": ok.get("formula_version", formula_version),
            "n_model": n_model,
            "annual_window_yult": annual_window,
            "annual_pilot_current_yult": pd.to_numeric(
                ok.get("annual_pilot_current_yult"), errors="coerce"
            ),
            "annual_pilot_full_yult": pd.to_numeric(
                ok.get("annual_pilot_full_yult"), errors="coerce"
            ),
            "annual_network_current_yult": pd.to_numeric(
                ok.get("annual_network_current_yult"), errors="coerce"
            ),
            "annual_network_full_yult": pd.to_numeric(
                ok.get("annual_network_full_yult"), errors="coerce"
            ),
            "annual_network_full_compliance_yult": pd.to_numeric(
                ok.get("annual_network_full_compliance_yult"), errors="coerce"
            ),
            "agreement_share_control": pd.to_numeric(
                ok.get("agreement_share_control"), errors="coerce"
            ),
            "agreement_share_model": pd.to_numeric(
                ok.get("agreement_share_model"), errors="coerce"
            ),
            "agreement_share_lift_pp": pd.to_numeric(
                ok.get("agreement_share_lift_pp"), errors="coerce"
            ),
            "pretension_share_control": pd.to_numeric(
                ok.get("pretension_share_control"), errors="coerce"
            ),
            "pretension_share_model": pd.to_numeric(
                ok.get("pretension_share_model"), errors="coerce"
            ),
            "pretension_share_lift_pp": pd.to_numeric(
                ok.get("pretension_share_lift_pp"), errors="coerce"
            ),
            "fu_share_control": pd.to_numeric(
                ok.get("fu_share_control"), errors="coerce"
            ),
            "fu_share_model": pd.to_numeric(ok.get("fu_share_model"), errors="coerce"),
            "fu_share_lift_pp": pd.to_numeric(
                ok.get("fu_share_lift_pp"), errors="coerce"
            ),
            "court_share_control": pd.to_numeric(
                ok.get("court_share_control"), errors="coerce"
            ),
            "court_share_model": pd.to_numeric(
                ok.get("court_share_model"), errors="coerce"
            ),
            "court_share_lift_pp": pd.to_numeric(
                ok.get("court_share_lift_pp"), errors="coerce"
            ),
        }
    )


def _paths_frame(
    paths: pd.DataFrame,
    *,
    stem: str,
    slot: str,
    model_label: str,
    formula_version: str,
) -> pd.DataFrame:
    if paths is None or paths.empty:
        return pd.DataFrame()
    week_col = "week_end" if "week_end" in paths.columns else "week_start"
    out = paths.copy()
    out["week_start"] = pd.to_datetime(out[week_col], errors="coerce").dt.date
    out["stem"] = stem
    out["slot"] = slot
    out["model_label"] = model_label
    out["formula_version"] = formula_version
    out["filial"] = out["filial"].fillna("(пусто)").astype(str)
    for col in _PATH_SHARE_COLS + ["n_model"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _instruments_attn_frame(
    attn: pd.DataFrame,
    *,
    stem: str,
    slot: str,
    model_label: str,
    formula_version: str,
) -> pd.DataFrame:
    if attn is None or attn.empty:
        return pd.DataFrame()
    out = attn.copy()
    week_col = "week_end" if "week_end" in out.columns else "week_start"
    out["week_start"] = pd.to_datetime(out[week_col], errors="coerce").dt.date
    out["stem"] = stem
    out["slot"] = slot
    out["model_label"] = model_label
    out["formula_version"] = formula_version
    rename = {
        "share_none_control_pct": "agr_control_pct",
        "share_none_model_pct": "agr_model_pct",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    out["filial"] = out["filial"].fillna("(пусто)").astype(str)
    for col in _ATTN_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _coverage_by_filial_weekly(
    monitoring_df: pd.DataFrame,
    *,
    week_ends: list,
    pilot_start: str,
    stem: str,
    slot: str,
    model_label: str,
    formula_version: str,
    filter_fn,
) -> pd.DataFrame:
    work = monitoring_df.copy()
    result_col = _first_col(
        work, ("РезультатПроверки", "result_check", "ResultCheck")
    )
    filial_col = _first_col(work, ("Филиал", "Filial", "filial", "FILIAL"))
    if result_col is None or filial_col is None:
        logger.warning(
            "coverage by filial skipped: no РезультатПроверки/Филиал for stem=%s",
            stem,
        )
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for week_end in week_ends:
        subset = filter_fn(work, end=week_end, start=pilot_start)
        if subset.empty:
            continue
        result = pd.to_numeric(subset[result_col], errors="coerce")
        eligible = subset.loc[result.isin(RESULT_ELIGIBLE)].copy()
        if eligible.empty:
            continue
        eligible["_is_model"] = (
            pd.to_numeric(eligible[result_col], errors="coerce") != RESULT_OUT_OF_MODEL
        )
        eligible["_filial"] = (
            eligible[filial_col].astype("string").fillna("(пусто)").astype(str)
        )
        for filial, part in eligible.groupby("_filial", dropna=False):
            n = len(part)
            n_model = int(part["_is_model"].sum())
            rows.append(
                {
                    "week_start": pd.Timestamp(week_end).date(),
                    "stem": stem,
                    "slot": slot,
                    "model_label": model_label,
                    "filial": str(filial),
                    "n_incidents": n,
                    "n_model": n_model,
                    "coverage": (n_model / n) if n else np.nan,
                    "formula_version": formula_version,
                }
            )
    return pd.DataFrame(rows)


def _first_col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((n for n in names if n in df.columns), None)


# --- extract (prod: OISUU vittrina + Hive train as retro) ----------------------


def _read_table(path: Path) -> pd.DataFrame:
    """csv или parquet по суффиксу."""
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _main_hive_table() -> str:
    """Hive train для slot=main — тот же, что datadrift (DEFAULT_DRIFT_MODELS)."""
    override = os.environ.get("FIN_EFFECT_RETRO_HIVE", "").strip()
    if override:
        return override
    for model in DEFAULT_DRIFT_MODELS:
        if str(model.get("slot", "")) == FIN_EFFECT_SLOT:
            hive = str(model.get("hive", "")).strip()
            if hive:
                return hive
    raise ValueError(
        f"No hive for slot={FIN_EFFECT_SLOT!r} in DEFAULT_DRIFT_MODELS "
        "(or set FIN_EFFECT_RETRO_HIVE / FIN_EFFECT_RETRO_PATH)."
    )


def _load_retro(settings: Settings) -> pd.DataFrame:
    """Ретро: файл (override) или Hive train main как у datadrift."""
    if settings.fin_effect_retro_path:
        df = _read_table(Path(settings.fin_effect_retro_path))
        logger.info(
            "fin_effect retro ← file %s || rows=%s",
            settings.fin_effect_retro_path,
            len(df),
        )
        return df

    from pipeline.datadrift import load_reference_dataset

    hive = _main_hive_table()
    logger.info("fin_effect retro ← Hive %s (same as datadrift train)", hive)
    return load_reference_dataset(settings, hive_table=hive)


def _load_monitoring_vitrina(settings: Settings) -> pd.DataFrame:
    """Витрина сутяжности из OISUU (не SPR-логи).

    Override: FIN_EFFECT_MONITORING_PATH — готовый csv/parquet (отладка / снимок).
    enrich_incident_path_flags — флаги ФУ/суда на уровне инцидента (если есть querulus).
    """
    if settings.fin_effect_monitoring_path:
        df = _read_table(Path(settings.fin_effect_monitoring_path))
        logger.info(
            "fin_effect monitoring ← file %s || rows=%s",
            settings.fin_effect_monitoring_path,
            len(df),
        )
        return df

    sql = f"SELECT * FROM {FIN_EFFECT_VITRINA}"
    conn = connect_oisuu(settings)
    try:
        df = pd.read_sql(sql, conn)
    finally:
        conn.close()
    try:
        from querulus.fin_effect.excel_explore import enrich_incident_path_flags

        df = enrich_incident_path_flags(df)
    except Exception as exc:  # noqa: BLE001 — на контуре без querulus src не валим весь шаг
        logger.warning("enrich_incident_path_flags skipped: %s", exc)
    logger.info(
        "fin_effect monitoring ← OISUU %s || rows=%s",
        FIN_EFFECT_VITRINA,
        len(df),
    )
    return df


def load_fin_effect_jobs(settings: Settings) -> list[FinEffectJob]:
    """Собрать jobs: одна витрина × список stem (история смен main).

    По умолчанию stem/slot — константы ``FIN_EFFECT_STEM`` / ``FIN_EFFECT_SLOT``.
    Опционально ``FIN_EFFECT_STEMS`` (csv) для истории смен.
    Retro: Hive train main (как datadrift) или ``FIN_EFFECT_RETRO_PATH``.
    """
    monitoring = _load_monitoring_vitrina(settings)
    retro = _load_retro(settings)
    stems_raw = os.environ.get("FIN_EFFECT_STEMS", "").strip()
    if stems_raw:
        stems = [s.strip() for s in stems_raw.split(",") if s.strip()]
    else:
        stems = [FIN_EFFECT_STEM]
    labels_raw = os.environ.get("FIN_EFFECT_MODEL_LABELS", "").strip()
    labels = [x.strip() for x in labels_raw.split(",")] if labels_raw else []
    pilot_start = os.environ.get("FIN_EFFECT_PILOT_START") or None
    as_of = os.environ.get("FIN_EFFECT_AS_OF") or None
    retro_as_of = os.environ.get("FIN_EFFECT_RETRO_AS_OF") or None
    jobs: list[FinEffectJob] = []
    for i, stem in enumerate(stems):
        label = labels[i] if i < len(labels) and labels[i] else stem
        jobs.append(
            FinEffectJob(
                monitoring_df=monitoring.copy(),
                retro_df=retro.copy(),
                stem=stem,
                slot=FIN_EFFECT_SLOT,
                model_label=label,
                pilot_start=pilot_start,
                as_of=as_of,
                retro_as_of=retro_as_of,
            )
        )
    return jobs


def main() -> int:
    """Точка входа: extract → weekly snapshots → wide append в Motor."""
    settings = load_settings()
    logger.info(
        "fin_effect ← OISUU vittrina + Hive/file retro → %s.%s",
        settings.schema,
        settings.table_fin_effect,
    )
    jobs = load_fin_effect_jobs(settings)
    results = [compute_fin_effect_weekly(job) for job in jobs]
    frames = [build_wide_frame(res) for res in results]
    wide = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=WIDE_COLS)
    )
    wide = _finalize_wide(wide)
    if wide.empty:
        logger.warning("Empty fin_effect wide report")
        return 0

    engine = get_engine(settings.grafana_sqlalchemy_url)
    n = write_append_guarded(
        wide,
        engine=engine,
        table=settings.table_fin_effect,
        schema=settings.schema,
        guard_cols=["week_start", "stem", "slot", "filial"],
    )
    formula = str(wide["formula_version"].iloc[0])
    n_all = int((wide["filial"] == FILIAL_ALL).sum())
    n_fil = int((wide["filial"] != FILIAL_ALL).sum())
    logger.info(
        "fin_effect done || rows=%s || filial=(все):%s other:%s || formula=%s",
        n,
        n_all,
        n_fil,
        formula,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
