"""Fin-effect via querulus → Querulus_FinEffect wide (MSSQL agrafana).

Grain: (week_start, stem, slot, filial)
  filial='(все)' → ITT / annual_* / overall coverage / path lifts
  filial=<имя>  → coverage + path shares + instruments attn
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
_MONITORING_ROOT = _PIPELINE_ROOT.parent
if str(_MONITORING_ROOT) not in sys.path:
    sys.path.insert(0, str(_MONITORING_ROOT))

_QUERULUS_SRC = _MONITORING_ROOT.parent / "src"
if _QUERULUS_SRC.is_dir() and str(_QUERULUS_SRC) not in sys.path:
    sys.path.insert(0, str(_QUERULUS_SRC))

from pipeline.config import Settings, load_settings
from pipeline.write_sql import get_engine, write_replace

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("fin_effect")

RESULT_OUT_OF_MODEL = -100
RESULT_ELIGIBLE = {0, 1, RESULT_OUT_OF_MODEL}
FILIAL_ALL = "(все)"

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

# Модели, которые когда-либо были в слоте main (финэффект только для main).
# Сейчас в бою одна; в истории / synthetic — несколько stem с slot=main.
_MAIN_MODELS = (
    {
        "stem": "querulus_ansamble_2026_04_18_v1",
        "label": "querulus_ansamble_2026_04_18_v1",
        "seed": 42,
    },
    {
        "stem": "UU_querulus_2026_11_05_12_44_24",
        "label": "UU_querulus_2026_11_05_12_44_24",
        "seed": 99,
    },
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


# --- extract (synthetic / prod) -------------------------------------------------


def load_fin_effect_jobs(settings: Settings) -> list[FinEffectJob]:
    if settings.source_mode == "synthetic":
        return _load_synthetic_jobs(settings.synthetic_data_dir)
    return _load_prod_jobs(settings)


def _load_synthetic_jobs(data_dir: Path) -> list[FinEffectJob]:
    """Несколько stem с slot=main — как смена main в проде во времени."""
    mon_path = data_dir / "fin_effect_monitoring.csv"
    retro_path = data_dir / "fin_effect_retro.csv"
    data_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[FinEffectJob] = []
    for i, model in enumerate(_MAIN_MODELS):
        if i == 0 and mon_path.is_file() and retro_path.is_file():
            monitoring = pd.read_csv(mon_path)
            retro = pd.read_csv(retro_path)
        else:
            monitoring, retro = build_synthetic_frames(seed=int(model["seed"]))
            if i == 0:
                monitoring.to_csv(mon_path, index=False)
                retro.to_csv(retro_path, index=False)
        jobs.append(
            FinEffectJob(
                monitoring_df=monitoring.copy(),
                retro_df=retro.copy(),
                stem=model["stem"],
                slot="main",
                model_label=model["label"],
            )
        )
    return jobs


def _load_prod_jobs(settings: Settings) -> list[FinEffectJob]:
    """Все stem, которые когда-либо были main.

    FIN_EFFECT_MONITORING_PATH / FIN_EFFECT_RETRO_PATH — витрины.
    FIN_EFFECT_STEMS — через запятую (предпочтительно); иначе FIN_EFFECT_STEM.
    FIN_EFFECT_MODEL_LABELS — опционально, через запятую, 1:1 к stems.
    """
    mon_path = os.environ.get("FIN_EFFECT_MONITORING_PATH", "").strip()
    retro_path = os.environ.get("FIN_EFFECT_RETRO_PATH", "").strip()
    if not mon_path or not retro_path:
        raise NotImplementedError(
            "prod_fin_effect: задай FIN_EFFECT_MONITORING_PATH и "
            "FIN_EFFECT_RETRO_PATH (csv/parquet). "
            f"SOURCE_MODE={settings.source_mode}"
        )
    monitoring = _read_table(Path(mon_path))
    retro = _read_table(Path(retro_path))
    stems_raw = os.environ.get("FIN_EFFECT_STEMS", "").strip()
    if stems_raw:
        stems = [s.strip() for s in stems_raw.split(",") if s.strip()]
    else:
        stems = [os.environ.get("FIN_EFFECT_STEM", "prod").strip() or "prod"]
    labels_raw = os.environ.get("FIN_EFFECT_MODEL_LABELS", "").strip()
    labels = (
        [x.strip() for x in labels_raw.split(",")]
        if labels_raw
        else []
    )
    slot = os.environ.get("FIN_EFFECT_SLOT", "main").strip() or "main"
    if slot != "main":
        raise ValueError(
            f"финэффект считается только для slot=main, получено {slot!r}"
        )
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
                slot=slot,
                model_label=label,
                pilot_start=pilot_start,
                as_of=as_of,
                retro_as_of=retro_as_of,
            )
        )
    return jobs


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_synthetic_frames(
    *,
    n_monitoring: int = 400,
    n_retro: int = 800,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monitoring = _build_monitoring_claims(n_rows=n_monitoring, seed=seed)
    today = pd.Timestamp.today().normalize()
    pilot_start = pd.Timestamp("2026-04-20")
    span = max((today - pilot_start).days, 30)
    rng = np.random.default_rng(seed)
    monitoring["ДатаЗаявления"] = pilot_start + pd.to_timedelta(
        rng.integers(0, span, len(monitoring)), unit="D"
    )
    monitoring["Дата вызова модели сутяжности"] = monitoring["ДатаЗаявления"]
    retro = _build_synthetic_retro(n_rows=n_retro, seed=seed + 1)
    return monitoring, retro


def _build_monitoring_claims(*, n_rows: int, seed: int) -> pd.DataFrame:
    if n_rows < 20:
        raise ValueError("n_rows должен быть не меньше 20")
    rng = np.random.default_rng(seed)
    result = rng.choice([RESULT_OUT_OF_MODEL, 0, 1], size=n_rows)
    payout = (result == 1) & (rng.random(n_rows) < 0.6)
    agreement = (result == 1) & (rng.random(n_rows) < 0.5)
    today = pd.Timestamp.today().normalize()
    psr = np.where(rng.random(n_rows) < 0.08, rng.uniform(1_000, 30_000, n_rows), 0.0)
    frame = pd.DataFrame(
        {
            "Убыток": [f"SYN-{index:06d}" for index in range(n_rows)],
            "НомерИнцидент": [f"INC-{index // 2:06d}" for index in range(n_rows)],
            "Филиал": rng.choice(PILOT_FILIALS, size=n_rows),
            "ФормаВозмещения": np.where(agreement, "Соглашение", "Денежная"),
            "УбытокСтатус": "Первичный",
            "ТипОбъектаАвтотранспорт": 1,
            "РезультатПроверки": result,
            "СуммаПлатежа": rng.uniform(30_000, 250_000, n_rows),
            "СуммаКВыплате": rng.uniform(30_000, 250_000, n_rows),
            "СуммаОсновногоДолгаЗаявлено": rng.uniform(20_000, 300_000, n_rows),
            "Сумма рекомендованная к доплате по модулю": np.where(
                result == 1, rng.uniform(10_000, 80_000, n_rows), 0.0
            ),
            "Иные затраты": np.where(payout, rng.uniform(5_000, 60_000, n_rows), 0.0),
            "Выплата по модели": payout.astype(int),
            "Заключено соглашение": agreement.astype(int),
            "ВызовМодельСутяжность": np.where(
                result != RESULT_OUT_OF_MODEL,
                1,
                (rng.random(n_rows) < 0.35).astype(int),
            ),
            "Дата вызова модели сутяжности": today
            - pd.to_timedelta(rng.integers(0, 120, n_rows), unit="D"),
            "ДатаЗаявления": today
            - pd.to_timedelta(rng.integers(0, 120, n_rows), unit="D"),
            "Cумма выплаты по претензии": psr,
            "Сумма выплат по ФУ": 0.0,
            "Сумма выплаты по суду": 0.0,
            "ЕМР+20": ((rng.random(n_rows) < 0.12) & agreement).astype(int),
            "ПереплатаЕМР+(безКВ)": 0.0,
            "Выплата из лимитов директоров филиалов": (
                (rng.random(n_rows) < 0.10) & agreement
            ).astype(int),
            "СогласованнаяСуммаКВ": 0.0,
        }
    )
    frame["СуммаКВыплате"] = frame["СуммаПлатежа"]
    frame["ПереплатаЕМР+(безКВ)"] = np.where(
        frame["ЕМР+20"].eq(1), rng.uniform(3_000, 40_000, n_rows), 0.0
    )
    frame["СогласованнаяСуммаКВ"] = np.where(
        frame["Выплата из лимитов директоров филиалов"].eq(1),
        rng.uniform(5_000, 50_000, n_rows),
        0.0,
    )
    return frame


def _build_synthetic_retro(*, n_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    nonpilot = ("Московский", "Санкт-Петербургский", "Казанский", "Новосибирский")
    filials = list(PILOT_FILIALS) + list(nonpilot)
    end = pd.Timestamp("2025-06-30")
    start = end - pd.DateOffset(years=2)
    days = (end - start).days
    dates = start + pd.to_timedelta(rng.integers(0, max(days, 1), n_rows), unit="D")
    amount = rng.uniform(20_000, 200_000, n_rows)
    return pd.DataFrame(
        {
            "FILIAL": rng.choice(filials, size=n_rows),
            "LOSS_DATE": dates,
            "TARGET_FREQ_AMOUNT": amount,
            "RECOVEREDMAINDEBT_LAST_INST_SUM": amount * rng.uniform(0.3, 0.9, n_rows),
            "Сумма_взыскано_по_ФУ": np.where(
                rng.random(n_rows) < 0.1, amount * 0.2, 0.0
            ),
            "Суммы_взыскано_по_иску": np.where(
                rng.random(n_rows) < 0.05, amount * 0.3, 0.0
            ),
        }
    )


def main() -> int:
    settings = load_settings()
    logger.info(
        "SOURCE_MODE=%s || engine=querulus.fin_effect → %s (wide)",
        settings.source_mode,
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
    n = write_replace(
        wide,
        engine=engine,
        table=settings.table_fin_effect,
        schema=settings.schema,
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
