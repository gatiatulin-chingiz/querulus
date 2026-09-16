"""Лог снимков финэффекта и понедельный ряд с декомпозицией дельт."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from querulus.fin_effect.excel_explore import resolve_column
from querulus.fin_effect.excel_monitoring import (
    MonitoringEffectResult,
    estimate_monitoring_effect,
)
from querulus.fin_effect.monitoring_report import FORMULA_VERSION, _CSS, _table

PILOT_START_DEFAULT = "2026-04-20"
SNAPSHOT_FILENAME = "fin_effect_snapshots.csv"
WEEKLY_HTML_FILENAME = "fin_effect_weekly.html"
TOP_LOSS_DRIVERS_PER_KIND = 5

SNAPSHOT_COLUMNS = [
    "snapshot_at",
    "t_calc",
    "formula_version",
    "source_label",
    "observation_start",
    "observation_end",
    "n_rows",
    "n_control",
    "n_model",
    "n_filials",
    "max_age_days",
    "yfact_effect",
    "yfact_ci_low",
    "yfact_ci_high",
    "yfact_unstratified",
    "y365_effect",
    "y365_ci_low",
    "y365_ci_high",
    "y365_unstratified",
    "annual_pilot_full_y365",
    "annual_pilot_full_y365_ci_low",
    "annual_pilot_full_y365_ci_high",
    "annual_network_full_y365",
    "annual_network_full_y365_ci_low",
    "annual_network_full_y365_ci_high",
    "n_pilot_eligible_year",
    "network_multiplier",
    "payment_to_pay_mismatch_rows",
    "note",
]


def _quality_map(result: MonitoringEffectResult) -> dict[str, Any]:
    return result.data_quality.set_index("metric")["value"].to_dict()


def _effect_row(result: MonitoringEffectResult, horizon: str) -> dict[str, float]:
    effects = result.effect_summary.set_index("horizon")
    if horizon not in effects.index:
        return {
            "effect": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "unstratified": np.nan,
        }
    row = effects.loc[horizon]
    return {
        "effect": float(row["effect_per_case"]),
        "ci_low": float(row["ci_low"]) if "ci_low" in row and pd.notna(row["ci_low"]) else np.nan,
        "ci_high": float(row["ci_high"]) if "ci_high" in row and pd.notna(row["ci_high"]) else np.nan,
        "unstratified": float(row["unstratified_effect"]),
    }


def snapshot_row_from_result(
    result: MonitoringEffectResult,
    *,
    source_label: str = "",
    note: str = "",
    formula_version: str = FORMULA_VERSION,
    snapshot_at: str | None = None,
) -> dict[str, Any]:
    """Одна строка лога снимка из MonitoringEffectResult."""
    quality = _quality_map(result)
    fact = _effect_row(result, "fact")
    y365 = _effect_row(result, "365")
    annual = result.annual_summary.set_index("horizon")
    ann = annual.loc["365"] if "365" in annual.index else None
    return {
        "snapshot_at": snapshot_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "t_calc": result.t_calc.date().isoformat(),
        "formula_version": formula_version,
        "source_label": source_label,
        "observation_start": quality.get("observation_start"),
        "observation_end": quality.get("observation_end"),
        "n_rows": int(quality.get("n_rows", len(result.frame))),
        "n_control": int(quality.get("n_control", 0)),
        "n_model": int(quality.get("n_model", 0)),
        "n_filials": int(quality.get("n_filials", 0)),
        "max_age_days": int(quality.get("max_age_days", 0)),
        "yfact_effect": fact["effect"],
        "yfact_ci_low": fact["ci_low"],
        "yfact_ci_high": fact["ci_high"],
        "yfact_unstratified": fact["unstratified"],
        "y365_effect": y365["effect"],
        "y365_ci_low": y365["ci_low"],
        "y365_ci_high": y365["ci_high"],
        "y365_unstratified": y365["unstratified"],
        "annual_pilot_full_y365": (
            float(ann["annual_pilot_full"]) if ann is not None else np.nan
        ),
        "annual_pilot_full_y365_ci_low": (
            float(ann["annual_pilot_full_ci_low"]) if ann is not None else np.nan
        ),
        "annual_pilot_full_y365_ci_high": (
            float(ann["annual_pilot_full_ci_high"]) if ann is not None else np.nan
        ),
        "annual_network_full_y365": (
            float(ann["annual_network_full"]) if ann is not None else np.nan
        ),
        "annual_network_full_y365_ci_low": (
            float(ann["annual_network_full_ci_low"]) if ann is not None else np.nan
        ),
        "annual_network_full_y365_ci_high": (
            float(ann["annual_network_full_ci_high"]) if ann is not None else np.nan
        ),
        "n_pilot_eligible_year": (
            float(ann["N_pilot_eligible_year"]) if ann is not None else np.nan
        ),
        "network_multiplier": (
            float(ann["network_multiplier"]) if ann is not None else np.nan
        ),
        "payment_to_pay_mismatch_rows": int(
            quality.get("payment_to_pay_mismatch_rows", 0)
        ),
        "note": note,
    }


def append_snapshot_log(
    result: MonitoringEffectResult,
    path: str | Path,
    *,
    source_label: str = "",
    note: str = "",
) -> Path:
    """Дописать снимок в CSV-лог (создаёт файл при первом запуске)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = snapshot_row_from_result(
        result, source_label=source_label, note=note
    )
    frame = pd.DataFrame([row], columns=SNAPSHOT_COLUMNS)
    if path.exists():
        prev = pd.read_csv(path)
        for col in SNAPSHOT_COLUMNS:
            if col not in prev.columns:
                prev[col] = np.nan
        frame = pd.concat([prev[SNAPSHOT_COLUMNS], frame], ignore_index=True)
    frame.to_csv(path, index=False)
    return path


def load_snapshot_log(path: str | Path) -> pd.DataFrame:
    """Прочитать CSV-лог снимков."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    return pd.read_csv(path)


def _week_ends(
    *,
    pilot_start: str | pd.Timestamp,
    as_of: str | pd.Timestamp,
    freq: str = "W-SUN",
) -> list[pd.Timestamp]:
    start = pd.Timestamp(pilot_start).normalize()
    end = pd.Timestamp(as_of).normalize()
    if end < start:
        raise ValueError("as_of раньше pilot_start")
    ends = pd.date_range(start=start, end=end, freq=freq)
    out = [pd.Timestamp(value).normalize() for value in ends]
    if not out or out[-1] < end:
        out.append(end)
    # убрать дубликат, если end уже воскресенье
    uniq: list[pd.Timestamp] = []
    for value in out:
        if not uniq or uniq[-1] != value:
            uniq.append(value)
    return uniq


def filter_monitoring_by_application_date(
    monitoring_df: pd.DataFrame,
    *,
    end: str | pd.Timestamp,
    start: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Оставить строки с ДатаЗаявления в [start, end]."""
    col = resolve_column(monitoring_df, "application_date")
    if col is None:
        raise KeyError("Не найдена ДатаЗаявления для фильтра среза")
    dates = pd.to_datetime(monitoring_df[col], errors="coerce")
    end_ts = pd.Timestamp(end).normalize()
    mask = dates.notna() & (dates.dt.normalize() <= end_ts)
    if start is not None:
        start_ts = pd.Timestamp(start).normalize()
        mask = mask & (dates.dt.normalize() >= start_ts)
    return monitoring_df.loc[mask].copy()


def _filial_contributions(filial_effects: pd.DataFrame, horizon: str) -> pd.DataFrame:
    part = filial_effects.loc[
        filial_effects["horizon"].astype(str).eq(horizon)
    ].copy()
    if part.empty:
        return pd.DataFrame(
            columns=["filial", "n", "effect", "weight", "contribution"]
        )
    part = part.dropna(subset=["effect_control_minus_model"])
    total_n = float(part["n"].sum())
    part["weight"] = part["n"] / total_n if total_n else np.nan
    part["contribution"] = part["weight"] * part["effect_control_minus_model"]
    return part.rename(
        columns={"effect_control_minus_model": "effect"}
    )[["filial", "n", "effect", "weight", "contribution"]]


def _group_means_fact(result: MonitoringEffectResult) -> dict[str, float]:
    """Средние Yfact и N по control/model на горизонте fact."""
    part = result.group_summary.loc[
        result.group_summary["horizon"].astype(str).eq("fact")
    ]
    out = {
        "mean_control": np.nan,
        "mean_model": np.nan,
        "n_control": 0.0,
        "n_model": 0.0,
    }
    for _, row in part.iterrows():
        group = str(row["group"])
        if group in ("control", "model"):
            out[f"mean_{group}"] = float(row["mean_cost"])
            out[f"n_{group}"] = float(row["n"])
    return out


def _unit_level_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Одна строка на инцидент (единица ITT): группа, Yfact, флаги."""
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "incident_id",
                "loss_ids",
                "group",
                "filial",
                "yfact",
                "paid",
                "od",
                "age_days",
                "agreement",
                "has_pretension",
                "has_fu",
                "has_court",
                "application_date",
            ]
        )
    work = frame.copy()
    work["_has_pretension"] = work["_psr_pretension"].fillna(0).gt(0)
    work["_has_fu"] = work["_psr_fu"].fillna(0).gt(0)
    work["_has_court"] = work["_psr_court"].fillna(0).gt(0)
    # после схлопывания строка уже = инцидент
    out = pd.DataFrame(
        {
            "incident_id": work["_incident"].astype("string"),
            "loss_ids": work["_loss"].astype("string"),
            "group": work["_group"],
            "filial": work["_filial"],
            "yfact": work["Yfact"],
            "paid": work["_paid_to_date"],
            "od": work["_od"],
            "age_days": work["_age_days"],
            "agreement": work["_agreement"].fillna(False).astype(bool),
            "has_pretension": work["_has_pretension"].fillna(False).astype(bool),
            "has_fu": work["_has_fu"].fillna(False).astype(bool),
            "has_court": work["_has_court"].fillna(False).astype(bool),
            "application_date": pd.to_datetime(
                work["_application_date"], errors="coerce"
            ).dt.date.astype("string"),
        }
    )
    return out.reset_index(drop=True)


def _severity_flags(row: pd.Series) -> str:
    parts: list[str] = []
    if bool(row.get("agreement")):
        parts.append("соглашение")
    if bool(row.get("has_pretension")):
        parts.append("претензия")
    if bool(row.get("has_fu")):
        parts.append("ФУ")
    if bool(row.get("has_court")):
        parts.append("суд")
    od = row.get("od")
    if pd.notna(od) and float(od) > 0:
        parts.append(f"OD={float(od):,.0f}₽".replace(",", " "))
    age = row.get("age_days")
    if pd.notna(age):
        parts.append(f"age={int(age)}д")
    return ", ".join(parts)


def _group_delta_table(
    weekly: pd.DataFrame,
    group_means_by_week: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Δ mean control/model и вклад в Δ ITT (без стратификации по филиалам)."""
    rows: list[dict[str, Any]] = []
    ok_weeks = (
        weekly.loc[weekly["status"].eq("ok"), "week_end"].astype(str).tolist()
        if "status" in weekly.columns
        else []
    )
    for prev_key, key in zip(ok_weeks, ok_weeks[1:]):
        prev = group_means_by_week.get(prev_key)
        cur = group_means_by_week.get(key)
        if prev is None or cur is None:
            continue
        delta_control = (
            float(cur["mean_control"]) - float(prev["mean_control"])
            if pd.notna(cur["mean_control"]) and pd.notna(prev["mean_control"])
            else np.nan
        )
        delta_model = (
            float(cur["mean_model"]) - float(prev["mean_model"])
            if pd.notna(cur["mean_model"]) and pd.notna(prev["mean_model"])
            else np.nan
        )
        # ITT = mean_c − mean_m → ΔITT ≈ Δmean_c − Δmean_m
        contrib_control = delta_control
        contrib_model = (
            -delta_model if pd.notna(delta_model) else np.nan
        )
        delta_itt_unstrat = (
            contrib_control + contrib_model
            if pd.notna(contrib_control) and pd.notna(contrib_model)
            else np.nan
        )
        week_row = weekly.loc[weekly["week_end"].astype(str).eq(key)]
        delta_yfact = (
            float(week_row.iloc[0]["delta_yfact"])
            if not week_row.empty and "delta_yfact" in week_row.columns
            and pd.notna(week_row.iloc[0].get("delta_yfact"))
            else np.nan
        )
        if pd.notna(delta_control) and abs(delta_control) >= abs(
            delta_model if pd.notna(delta_model) else 0.0
        ):
            leader = "control"
            leader_dir = (
                "подорожал (+ к ITT)"
                if pd.notna(delta_control) and delta_control > 0
                else "подешевел (− к ITT)"
                if pd.notna(delta_control) and delta_control < 0
                else "без изменения"
            )
        else:
            leader = "model"
            leader_dir = (
                "подорожал (− к ITT)"
                if pd.notna(delta_model) and delta_model > 0
                else "подешевел (+ к ITT)"
                if pd.notna(delta_model) and delta_model < 0
                else "без изменения"
            )
        def _fmt(value: Any) -> str:
            if pd.isna(value):
                return "—"
            return f"{float(value):+,.0f}".replace(",", " ")

        read = (
            f"Сильнее потянул {leader} ({leader_dir}): "
            f"control Δmean={_fmt(delta_control)}₽ → вклад {_fmt(contrib_control)}; "
            f"model Δmean={_fmt(delta_model)}₽ → вклад {_fmt(contrib_model)}"
        )
        rows.append(
            {
                "week_end": key,
                "prev_week_end": prev_key,
                "mean_control_prev": prev["mean_control"],
                "mean_control_cur": cur["mean_control"],
                "delta_mean_control": delta_control,
                "n_control_prev": prev["n_control"],
                "n_control_cur": cur["n_control"],
                "mean_model_prev": prev["mean_model"],
                "mean_model_cur": cur["mean_model"],
                "delta_mean_model": delta_model,
                "n_model_prev": prev["n_model"],
                "n_model_cur": cur["n_model"],
                "itt_contrib_control": contrib_control,
                "itt_contrib_model": contrib_model,
                "delta_itt_unstratified": delta_itt_unstrat,
                "delta_yfact_stratified": delta_yfact,
                "dominant_group": leader,
                "readout": read,
            }
        )
    return pd.DataFrame(rows)


def _loss_drivers_table(
    weekly: pd.DataFrame,
    loss_by_week: dict[str, pd.DataFrame],
    group_means_by_week: dict[str, dict[str, float]],
    *,
    top_n: int = TOP_LOSS_DRIVERS_PER_KIND,
) -> pd.DataFrame:
    """Топ новых инцидентов и топ shared с Δ оплаты по вкладу в Δ ITT."""
    rows: list[dict[str, Any]] = []
    ok_weeks = (
        weekly.loc[weekly["status"].eq("ok"), "week_end"].astype(str).tolist()
        if "status" in weekly.columns
        else []
    )
    for prev_key, key in zip(ok_weeks, ok_weeks[1:]):
        cur_tbl = loss_by_week.get(key)
        prev_tbl = loss_by_week.get(prev_key)
        prev_means = group_means_by_week.get(prev_key)
        cur_means = group_means_by_week.get(key)
        if cur_tbl is None or prev_tbl is None or prev_means is None or cur_means is None:
            continue
        if cur_tbl.empty or "incident_id" not in cur_tbl.columns:
            continue
        prev_ids = set(prev_tbl["incident_id"].astype(str).tolist())
        cur_ids = set(cur_tbl["incident_id"].astype(str).tolist())
        new_ids = cur_ids - prev_ids
        shared_ids = cur_ids & prev_ids
        prev_paid = prev_tbl.set_index(
            prev_tbl["incident_id"].astype(str)
        )["paid"]
        cur_indexed = cur_tbl.set_index(cur_tbl["incident_id"].astype(str))

        week_rows: list[dict[str, Any]] = []
        for incident_id in new_ids:
            rec = cur_indexed.loc[incident_id]
            if isinstance(rec, pd.DataFrame):
                rec = rec.iloc[0]
            group = str(rec["group"])
            yfact = float(rec["yfact"]) if pd.notna(rec["yfact"]) else np.nan
            mean_g = float(prev_means.get(f"mean_{group}", np.nan))
            n_g = float(prev_means.get(f"n_{group}", 0.0))
            if pd.notna(yfact) and pd.notna(mean_g) and n_g >= 0:
                delta_mean = (yfact - mean_g) / (n_g + 1.0)
                sign = 1.0 if group == "control" else -1.0
                pull = sign * delta_mean
            else:
                pull = np.nan
            week_rows.append(
                {
                    "week_end": key,
                    "prev_week_end": prev_key,
                    "driver_kind": "new_incident",
                    "incident_id": incident_id,
                    "loss_ids": rec.get("loss_ids"),
                    "group": group,
                    "filial": rec["filial"],
                    "application_date": rec.get("application_date"),
                    "yfact": yfact,
                    "paid_prev": np.nan,
                    "paid_cur": float(rec["paid"]) if pd.notna(rec["paid"]) else np.nan,
                    "delta_paid": np.nan,
                    "estimated_itt_pull": pull,
                    "severity": _severity_flags(rec),
                }
            )

        for incident_id in shared_ids:
            rec = cur_indexed.loc[incident_id]
            if isinstance(rec, pd.DataFrame):
                rec = rec.iloc[0]
            if incident_id not in prev_paid.index:
                continue
            paid_cur = float(rec["paid"]) if pd.notna(rec["paid"]) else np.nan
            paid_prev_v = prev_paid.loc[incident_id]
            paid_prev_f = float(paid_prev_v) if pd.notna(paid_prev_v) else np.nan
            if pd.isna(paid_cur) or pd.isna(paid_prev_f):
                continue
            delta_paid = paid_cur - paid_prev_f
            if abs(delta_paid) < 1.0:
                continue
            group = str(rec["group"])
            n_g = float(cur_means.get(f"n_{group}", 0.0))
            if n_g > 0:
                sign = 1.0 if group == "control" else -1.0
                pull = sign * (delta_paid / n_g)
            else:
                pull = np.nan
            week_rows.append(
                {
                    "week_end": key,
                    "prev_week_end": prev_key,
                    "driver_kind": "payment_update",
                    "incident_id": incident_id,
                    "loss_ids": rec.get("loss_ids"),
                    "group": group,
                    "filial": rec["filial"],
                    "application_date": rec.get("application_date"),
                    "yfact": float(rec["yfact"]) if pd.notna(rec["yfact"]) else np.nan,
                    "paid_prev": paid_prev_f,
                    "paid_cur": paid_cur,
                    "delta_paid": delta_paid,
                    "estimated_itt_pull": pull,
                    "severity": _severity_flags(rec),
                }
            )

        if not week_rows:
            continue
        frame = pd.DataFrame(week_rows)
        for kind in ("new_incident", "payment_update"):
            part = frame.loc[frame["driver_kind"].eq(kind)].copy()
            if part.empty:
                continue
            part = part.reindex(
                part["estimated_itt_pull"].abs().sort_values(ascending=False).index
            ).head(top_n)
            rows.extend(part.to_dict("records"))
    return pd.DataFrame(rows)


@dataclass
class WeeklySeriesResult:
    """Понедельный ряд и декомпозиция."""

    weekly: pd.DataFrame
    filial_deltas: pd.DataFrame
    group_deltas: pd.DataFrame
    loss_drivers: pd.DataFrame
    week_ends: list[pd.Timestamp]
    note: str


def run_weekly_monitoring_series(
    monitoring_df: pd.DataFrame,
    retro_df: pd.DataFrame,
    *,
    pilot_start: str | pd.Timestamp = PILOT_START_DEFAULT,
    as_of: str | pd.Timestamp | None = None,
    residual_share: float = 0.07,
    discount_rate: float = 0.12,
    lookback_years: float | None = 2.0,
    retro_as_of: str | pd.Timestamp | None = None,
    bootstrap_iterations: int = 0,
    bootstrap_compliance_iterations: int = 0,
    bootstrap_n_jobs: int | None = 1,
    freq: str = "W-SUN",
    min_rows: int = 50,
) -> WeeklySeriesResult:
    """Кумулятивные срезы по week_end: заявления ≤ week_end, t_calc=week_end.

    Платежи — как в текущей витрине (не архив на ту дату). Bootstrap по умолчанию
    выключен для скорости; CI в ряду будут пустыми, пока iterations>0.
    """
    as_of_ts = (
        pd.Timestamp(as_of).normalize()
        if as_of is not None
        else pd.Timestamp.today().normalize()
    )
    ends = _week_ends(pilot_start=pilot_start, as_of=as_of_ts, freq=freq)
    weekly_rows: list[dict[str, Any]] = []
    contrib_by_week: dict[str, pd.DataFrame] = {}
    loss_sets: dict[str, set[Any]] = {}
    paid_by_week: dict[str, pd.Series] = {}
    loss_by_week: dict[str, pd.DataFrame] = {}
    group_means_by_week: dict[str, dict[str, float]] = {}

    for week_end in ends:
        subset = filter_monitoring_by_application_date(
            monitoring_df,
            end=week_end,
            start=pilot_start,
        )
        if len(subset) < min_rows:
            weekly_rows.append(
                {
                    "week_end": week_end.date().isoformat(),
                    "status": "skipped_small_n",
                    "n_rows": len(subset),
                    "yfact_effect": np.nan,
                    "y365_effect": np.nan,
                    "reason": f"n<{min_rows}",
                }
            )
            continue
        try:
            result = estimate_monitoring_effect(
                subset,
                retro_df,
                t_calc=week_end,
                residual_share=residual_share,
                discount_rate=discount_rate,
                lookback_years=lookback_years,
                retro_as_of=retro_as_of,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_compliance_iterations=bootstrap_compliance_iterations,
                bootstrap_n_jobs=bootstrap_n_jobs,
            )
        except Exception as exc:  # noqa: BLE001 — ряд не должен падать целиком
            weekly_rows.append(
                {
                    "week_end": week_end.date().isoformat(),
                    "status": "error",
                    "n_rows": len(subset),
                    "yfact_effect": np.nan,
                    "y365_effect": np.nan,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        snap = snapshot_row_from_result(
            result,
            note=f"weekly cumulative ≤ {week_end.date().isoformat()}",
            snapshot_at=week_end.date().isoformat(),
        )
        key = week_end.date().isoformat()
        snap["week_end"] = key
        snap["status"] = "ok"
        snap["reason"] = ""
        weekly_rows.append(snap)

        contrib_by_week[key] = _filial_contributions(result.filial_effects, "fact")
        loss_level = _unit_level_table(result.frame)
        loss_by_week[key] = loss_level
        group_means_by_week[key] = _group_means_fact(result)
        loss_sets[key] = set(loss_level["incident_id"].astype(str).tolist())
        paid_by_week[key] = loss_level.set_index(
            loss_level["incident_id"].astype(str)
        )["paid"]

    weekly = pd.DataFrame(weekly_rows)
    empty = WeeklySeriesResult(
        weekly=weekly,
        filial_deltas=pd.DataFrame(),
        group_deltas=pd.DataFrame(),
        loss_drivers=pd.DataFrame(),
        week_ends=ends,
        note="Пустой ряд",
    )
    if weekly.empty:
        return empty

    weekly = _add_week_deltas(weekly, loss_sets, paid_by_week)
    filial_deltas = _filial_delta_table(weekly, contrib_by_week)
    group_deltas = _group_delta_table(weekly, group_means_by_week)
    loss_drivers = _loss_drivers_table(
        weekly, loss_by_week, group_means_by_week
    )
    note = (
        "Кумулятивные срезы на текущих платежах витрины; "
        "ретроспектива ≠ архивный снимок на ту дату. "
        "Вклад отдельного убытка в Δ ITT — диагностическая оценка "
        "без стратификации по филиалам."
    )
    return WeeklySeriesResult(
        weekly=weekly,
        filial_deltas=filial_deltas,
        group_deltas=group_deltas,
        loss_drivers=loss_drivers,
        week_ends=ends,
        note=note,
    )


def _add_week_deltas(
    weekly: pd.DataFrame,
    loss_sets: dict[str, set[Any]],
    paid_by_week: dict[str, pd.Series],
) -> pd.DataFrame:
    out = weekly.copy()
    for col in (
        "delta_yfact",
        "delta_y365",
        "delta_n_rows",
        "n_new_losses",
        "n_shared_losses",
        "shared_mean_paid_delta",
        "ci_yfact_includes_0",
        "why_moved",
    ):
        if col not in out.columns:
            out[col] = np.nan if col != "why_moved" else ""

    ok_mask = (
        out["status"].eq("ok")
        if "status" in out.columns
        else pd.Series(True, index=out.index)
    )
    prev_i: Any | None = None
    for i, row in out.iterrows():
        if not bool(ok_mask.loc[i]):
            continue
        yf = row.get("yfact_effect")
        yf_lo = row.get("yfact_ci_low")
        yf_hi = row.get("yfact_ci_high")
        if pd.notna(yf_lo) and pd.notna(yf_hi):
            out.at[i, "ci_yfact_includes_0"] = bool(yf_lo <= 0 <= yf_hi)
        if prev_i is None:
            out.at[i, "why_moved"] = "базовая неделя ряда"
            prev_i = i
            continue
        prev = out.loc[prev_i]
        out.at[i, "delta_yfact"] = (
            float(yf) - float(prev["yfact_effect"])
            if pd.notna(yf) and pd.notna(prev["yfact_effect"])
            else np.nan
        )
        out.at[i, "delta_y365"] = (
            float(row["y365_effect"]) - float(prev["y365_effect"])
            if pd.notna(row.get("y365_effect")) and pd.notna(prev.get("y365_effect"))
            else np.nan
        )
        out.at[i, "delta_n_rows"] = (
            float(row["n_rows"]) - float(prev["n_rows"])
            if pd.notna(row.get("n_rows")) and pd.notna(prev.get("n_rows"))
            else np.nan
        )
        key = str(row["week_end"])
        prev_key = str(prev["week_end"])
        cur_losses = loss_sets.get(key, set())
        prev_losses = loss_sets.get(prev_key, set())
        shared = cur_losses & prev_losses
        new = cur_losses - prev_losses
        out.at[i, "n_new_losses"] = len(new)
        out.at[i, "n_shared_losses"] = len(shared)
        if shared and key in paid_by_week and prev_key in paid_by_week:
            cur_paid = paid_by_week[key].reindex(list(shared))
            prev_paid = paid_by_week[prev_key].reindex(list(shared))
            out.at[i, "shared_mean_paid_delta"] = float(
                (cur_paid - prev_paid).mean()
            )
        parts = []
        dy = out.at[i, "delta_yfact"]
        if pd.notna(dy):
            parts.append(f"ΔYfact={dy:+,.0f}₽".replace(",", " "))
        dn = out.at[i, "delta_n_rows"]
        if pd.notna(dn):
            parts.append(f"ΔN={dn:+.0f}")
        nn = out.at[i, "n_new_losses"]
        if pd.notna(nn):
            parts.append(f"new={int(nn)}")
        spd = out.at[i, "shared_mean_paid_delta"]
        if pd.notna(spd):
            parts.append(f"shared_paidΔ={spd:+,.0f}₽".replace(",", " "))
        out.at[i, "why_moved"] = "; ".join(parts)
        prev_i = i
    return out


def _filial_delta_table(
    weekly: pd.DataFrame,
    contrib_by_week: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ok_weeks = (
        weekly.loc[weekly["status"].eq("ok"), "week_end"].astype(str).tolist()
        if "status" in weekly.columns
        else []
    )
    for prev_key, key in zip(ok_weeks, ok_weeks[1:]):
        prev = contrib_by_week.get(prev_key)
        cur = contrib_by_week.get(key)
        if prev is None or cur is None or prev.empty or cur.empty:
            continue
        merged = cur.merge(prev, on="filial", how="outer", suffixes=("_cur", "_prev"))
        for col in (
            "n_cur",
            "n_prev",
            "effect_cur",
            "effect_prev",
            "contribution_cur",
            "contribution_prev",
        ):
            if col not in merged.columns:
                merged[col] = np.nan
        merged["n_cur"] = merged["n_cur"].fillna(0)
        merged["n_prev"] = merged["n_prev"].fillna(0)
        merged["contribution_cur"] = merged["contribution_cur"].fillna(0.0)
        merged["contribution_prev"] = merged["contribution_prev"].fillna(0.0)
        merged["delta_contribution"] = (
            merged["contribution_cur"] - merged["contribution_prev"]
        )
        top = merged.reindex(
            merged["delta_contribution"].abs().sort_values(ascending=False).index
        ).head(5)
        for rec in top.to_dict("records"):
            rows.append(
                {
                    "week_end": key,
                    "prev_week_end": prev_key,
                    "filial": rec["filial"],
                    "n_prev": rec["n_prev"],
                    "n_cur": rec["n_cur"],
                    "effect_prev": rec.get("effect_prev"),
                    "effect_cur": rec.get("effect_cur"),
                    "delta_contribution": rec["delta_contribution"],
                }
            )
    return pd.DataFrame(rows)


def save_weekly_outputs(
    series: WeeklySeriesResult,
    data_dir: str | Path,
    *,
    weekly_html_name: str | None = None,
) -> Path:
    """Записать общий ``fin_effect_weekly.html`` в ``data_dir`` (без CSV)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    html_path = data_dir / (weekly_html_name or WEEKLY_HTML_FILENAME)
    html_path.write_text(build_weekly_html(series), encoding="utf-8")
    return html_path


def build_weekly_html(series: WeeklySeriesResult) -> str:
    """HTML: понедельные показатели, Δ control/model, топ убытков и филиалов."""
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    show_cols = [
        c
        for c in (
            "week_end",
            "status",
            "n_rows",
            "n_control",
            "n_model",
            "yfact_effect",
            "delta_yfact",
            "y365_effect",
            "delta_y365",
            "ci_yfact_includes_0",
            "n_new_losses",
            "shared_mean_paid_delta",
            "annual_pilot_full_y365",
            "annual_network_full_y365",
            "why_moved",
            "reason",
        )
        if c in series.weekly.columns
    ]
    weekly_show = series.weekly[show_cols].copy()
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8"/>
  <title>Понедельный финэффект</title>
  <style>{_CSS}</style>
</head>
<body>
<div class="page">
  <h1>Понедельный ряд финансового эффекта</h1>
  <p class="sub">Сформировано {escape(generated)} · версия {FORMULA_VERSION}</p>
  <div class="card">
    <p>{escape(series.note)}</p>
    <p>Стартовое окно пилота → week_end (кумулятивно). Частота: W-SUN + финальный as_of.</p>
  </div>
  <h2>Главные показатели</h2>
  <div class="card">
    {_table(
      weekly_show,
      columns={
        "week_end": "Дата конца недели / финальный срез as_of",
        "status": "Статус расчёта среза: ok, skipped или error",
        "n_rows": "Число убытков в кумулятивном окне на эту дату",
        "n_control": "Число убытков в группе control",
        "n_model": "Число убытков в группе model",
        "yfact_effect": (
            "ITT по Yfact: взвешенная разность mean(control)−mean(model), ₽/убыток"
        ),
        "delta_yfact": "Изменение ITT Yfact относительно предыдущей недели, ₽/убыток",
        "y365_effect": (
            "ITT по Y365 (Yfact + NPV хвоста ПСР), ₽/убыток"
        ),
        "delta_y365": "Изменение ITT Y365 относительно предыдущей недели, ₽/убыток",
        "ci_yfact_includes_0": (
            "Проходит ли 95% CI ITT Yfact через 0 (эффект статистически не подтверждён)"
        ),
        "n_new_losses": "Сколько новых убытков добавилось к предыдущей неделе",
        "shared_mean_paid_delta": (
            "Δ средней СуммаПлатежа по LossID, которые есть и на прошлой, и на текущей неделе"
        ),
        "annual_pilot_full_y365": (
            "Сценарий годового эффекта пилота при full rollout, горизонт Y365, ₽/год"
        ),
        "annual_network_full_y365": (
            "Сценарий годового эффекта сети при full rollout, горизонт Y365, ₽/год"
        ),
        "why_moved": "Кратко, за счёт чего уехала неделя к неделе",
        "reason": "Текст ошибки или причина skip",
      },
      rows=[
        "Одна строка = кумулятивный срез пилота от старта до week_end.",
      ],
    )}
  </div>
  <h2>Куда уехали control и model</h2>
  <div class="card">
    <p>ITT ≈ mean(control) − mean(model). Поэтому рост среднего control
    увеличивает ITT, рост среднего model — уменьшает. Вклад в Δ ITT без
    стратификации по филиалам: Δmean_control и −Δmean_model.</p>
    {_table(
      series.group_deltas,
      columns={
        "week_end": "Конец текущей недели",
        "prev_week_end": "Конец предыдущей недели",
        "mean_control_prev": "Средний Yfact control на прошлой неделе, ₽",
        "mean_control_cur": "Средний Yfact control на текущей неделе, ₽",
        "delta_mean_control": "Изменение среднего Yfact control, ₽/убыток",
        "n_control_prev": "N control на прошлой неделе",
        "n_control_cur": "N control на текущей неделе",
        "mean_model_prev": "Средний Yfact model на прошлой неделе, ₽",
        "mean_model_cur": "Средний Yfact model на текущей неделе, ₽",
        "delta_mean_model": "Изменение среднего Yfact model, ₽/убыток",
        "n_model_prev": "N model на прошлой неделе",
        "n_model_cur": "N model на текущей неделе",
        "itt_contrib_control": (
            "Вклад control в Δ ITT (= Δmean_control); «+» — control подорожал"
        ),
        "itt_contrib_model": (
            "Вклад model в Δ ITT (= −Δmean_model); «+» — model подешевел"
        ),
        "delta_itt_unstratified": "Сумма вкладов control+model (без филиалов), ₽",
        "delta_yfact_stratified": "Фактический Δ ITT Yfact со стратификацией, ₽",
        "dominant_group": "Какая группа сильнее потянула неделю по |Δmean|",
        "readout": "Краткая расшифровка знаков и вкладов",
      },
      rows=[
        "Одна строка на пару соседних недель с status=ok.",
      ],
    )}
  </div>
  <h2>Топ инцидентов по вкладу в Δ ITT</h2>
  <div class="card">
    <p>Диагностика: до {TOP_LOSS_DRIVERS_PER_KIND} новых инцидентов и до
    {TOP_LOSS_DRIVERS_PER_KIND} обновлений оплаты с наибольшим
    |estimated_itt_pull|. Знак: control «+» увеличивает ITT при росте
    расхода; model «+» увеличивает ITT при падении расхода model.</p>
    {_table(
      series.loss_drivers,
      columns={
        "week_end": "Конец текущей недели",
        "prev_week_end": "Конец предыдущей недели",
        "driver_kind": (
            "Тип драйвера: new_incident (новый инцидент) или payment_update (Δ оплаты)"
        ),
        "incident_id": "Номер инцидента",
        "loss_ids": "Номера убытков внутри инцидента (через «; »)",
        "group": "Группа: control или model",
        "filial": "Филиал",
        "application_date": "Дата заявления",
        "yfact": "Yfact инцидента на текущем срезе, ₽",
        "paid_prev": "СуммаПлатежа на прошлой неделе, ₽ (только payment_update)",
        "paid_cur": "СуммаПлатежа на текущей неделе, ₽",
        "delta_paid": "Изменение СуммаПлатежа, ₽",
        "estimated_itt_pull": (
            "Оценка вклада в Δ ITT, ₽: для new_incident — sign×(Yfact−mean_group)/(N+1); "
            "для payment_update — sign×Δpaid/N"
        ),
        "severity": (
            "Тяжесть / путь: соглашение, претензия, ФУ, суд, OD, возраст (дни)"
        ),
      },
      rows=[
        "До 5 new_incident и до 5 payment_update с наибольшим |вкладом| на пару недель.",
      ],
    )}
  </div>
  <h2>Топ филиалов по вкладу в Δ Yfact</h2>
  <div class="card">
    <p>Δ вклада филиала = вклад на текущей неделе − вклад на предыдущей
    (вклад = вес филиала × локальный ITT Yfact). Показаны филиалы с наибольшим |Δ|.</p>
    {_table(
      series.filial_deltas,
      columns={
        "week_end": "Конец текущей недели (кумулятивный срез)",
        "prev_week_end": "Конец предыдущей недели для сравнения",
        "filial": "Филиал",
        "n_prev": "Число убытков филиала на предыдущей неделе",
        "n_cur": "Число убытков филиала на текущей неделе",
        "effect_prev": (
            "Локальный ITT Yfact филиала на предыдущей неделе "
            "(mean control − mean model), ₽/убыток"
        ),
        "effect_cur": (
            "Локальный ITT Yfact филиала на текущей неделе "
            "(mean control − mean model), ₽/убыток"
        ),
        "delta_contribution": (
            "Изменение вклада филиала в стратифицированный ITT Yfact, ₽ "
            "(положительное — филиал усилил экономию)"
        ),
      },
      rows=[
        "До 5 филиалов с наибольшим |Δ вклада| на каждую пару соседних недель.",
      ],
    )}
  </div>
</div>
</body>
</html>
"""


__all__ = [
    "PILOT_START_DEFAULT",
    "SNAPSHOT_FILENAME",
    "TOP_LOSS_DRIVERS_PER_KIND",
    "WEEKLY_HTML_FILENAME",
    "WeeklySeriesResult",
    "append_snapshot_log",
    "build_weekly_html",
    "filter_monitoring_by_application_date",
    "load_snapshot_log",
    "run_weekly_monitoring_series",
    "save_weekly_outputs",
    "snapshot_row_from_result",
]
