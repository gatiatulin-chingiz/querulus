"""Диагностика вызревания таргета: отчёт, месячные ряды, иски, «дыры» после maturity."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from querulus import PROJECT_ROOT
from querulus.dataset.preprocess.filters import load_dataset_filters
from querulus.dataset.preprocess.maturity import (
    MATURITY_REPORT_NAME,
    resolve_snapshot_date,
)

logger = logging.getLogger("querulus.dataset")

_T0_DEFAULT = "PAYMENT_ORDER_DATE_TIME"
_CLAIMS_ARTIFACT = "target_3_claims.parquet"


def _processed_dir(project_root: Path | None = None) -> Path:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    return root / "data" / "processed"


def _raw_dir(project_root: Path | None = None) -> Path:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    return root / "data" / "raw"


def load_maturity_report(
    project_root: Path | None = None,
) -> dict[str, Any] | None:
    """Загрузить ``target_maturity_report.json``, если есть."""
    path = _processed_dir(project_root) / MATURITY_REPORT_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def maturity_policy_summary(
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Текущая политика из dataset_filters + S / mature_until."""
    filters = load_dataset_filters()
    cfg = filters.get("target_maturity") or {}
    snapshot, source = resolve_snapshot_date(filters)
    if cfg.get("horizon_months_no_court") is not None:
        horizon = int(cfg["horizon_months_no_court"])
    elif cfg.get("horizon_months") is not None:
        horizon = int(cfg["horizon_months"])
    else:
        horizon = 36
    mature_until = (snapshot - pd.DateOffset(months=horizon)).normalize()
    victim = filters.get("victim") or {}
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "t0_column": str(cfg.get("t0_column") or _T0_DEFAULT),
        "horizon_months_no_court": horizon,
        "snapshot_date": str(snapshot.date()),
        "snapshot_source": source,
        "mature_until": str(pd.Timestamp(mature_until).date()),
        "victim_date_from": victim.get("date_from"),
        "victim_date_to": victim.get("date_to"),
    }


def _month_start(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.to_period("M").dt.to_timestamp()


def monthly_incident_stats(
    df: pd.DataFrame,
    *,
    t0_column: str = _T0_DEFAULT,
    freq_col: str = "TARGET_FREQ",
    sev_col: str = "TARGET_SEV",
) -> pd.DataFrame:
    """Инциденты / TARGET_FREQ / severity по месяцу T0."""
    if t0_column not in df.columns:
        raise KeyError(f"Нет колонки T0: {t0_column}")
    work = df.copy()
    work["_month"] = _month_start(work[t0_column])
    work = work.loc[work["_month"].notna()]
    if "INCIDENT_NUMBER" in work.columns:
        n_inc = work.groupby("_month")["INCIDENT_NUMBER"].nunique()
    else:
        n_inc = work.groupby("_month").size()
    out = pd.DataFrame({"n_rows": work.groupby("_month").size(), "n_incidents": n_inc})
    if freq_col in work.columns:
        freq = pd.to_numeric(work[freq_col], errors="coerce").fillna(0).astype(int)
        out["n_freq1"] = work.assign(_f=freq).groupby("_month")["_f"].sum()
        out["freq1_rate"] = out["n_freq1"] / out["n_rows"].clip(lower=1)
    if sev_col in work.columns:
        sev = pd.to_numeric(work[sev_col], errors="coerce").fillna(0)
        out["sev_sum"] = work.assign(_s=sev).groupby("_month")["_s"].sum()
        out["sev_mean_pos"] = (
            work.assign(_s=sev)
            .loc[lambda d: d["_s"] > 0]
            .groupby("_month")["_s"]
            .mean()
        )
    return out.sort_index().fillna(0)


def yearly_cohort_stats(
    df: pd.DataFrame,
    *,
    t0_column: str = _T0_DEFAULT,
    freq_col: str = "TARGET_FREQ",
) -> pd.DataFrame:
    """Когорты по году T0."""
    if t0_column not in df.columns:
        raise KeyError(f"Нет колонки T0: {t0_column}")
    year = pd.to_datetime(df[t0_column], errors="coerce").dt.year
    work = df.assign(_year=year).loc[lambda d: d["_year"].notna()]
    rows: list[dict[str, Any]] = []
    for y, part in work.groupby("_year"):
        row: dict[str, Any] = {
            "year": int(y),
            "n_rows": int(len(part)),
            "n_incidents": (
                int(part["INCIDENT_NUMBER"].nunique())
                if "INCIDENT_NUMBER" in part.columns
                else int(len(part))
            ),
        }
        if freq_col in part.columns:
            f = pd.to_numeric(part[freq_col], errors="coerce").fillna(0)
            row["n_freq1"] = int(f.eq(1).sum())
            row["freq1_rate"] = float(f.eq(1).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def load_claims_frame(project_root: Path | None = None) -> pd.DataFrame | None:
    path = _raw_dir(project_root) / _CLAIMS_ARTIFACT
    if not path.is_file():
        return None
    return pd.read_parquet(path)


def claims_per_incident_table(
    claims: pd.DataFrame,
    *,
    incident_ids: pd.Series | None = None,
) -> pd.DataFrame:
    """Число строк исков на инцидент (сырой target_3_claims)."""
    work = claims.copy()
    work.columns = [str(c).upper() for c in work.columns]
    if "INCIDENTNUMBER" in work.columns and "INCIDENT_NUMBER" not in work.columns:
        work["INCIDENT_NUMBER"] = work["INCIDENTNUMBER"]
    if "INCIDENT_NUMBER" not in work.columns:
        raise KeyError("В claims нет INCIDENT_NUMBER")
    if incident_ids is not None:
        keys = set(incident_ids.dropna().astype(str).str.strip())
        work = work.loc[work["INCIDENT_NUMBER"].astype(str).str.strip().isin(keys)]
    counts = (
        work.groupby("INCIDENT_NUMBER", dropna=False)
        .size()
        .rename("n_claim_rows")
        .reset_index()
    )
    return counts


def _display_report(report: dict[str, Any] | None, policy: dict[str, Any]) -> None:
    display(Markdown("### Политика вызревания"))
    display(pd.DataFrame([policy]).T.rename(columns={0: "value"}))
    if report is None:
        display(
            Markdown(
                "_Нет `target_maturity_report.json` — пересоберите с "
                '`DATA_SOURCE="pipeline"` или откройте отчёт после collect._'
            )
        )
        return
    display(Markdown("### Отчёт maturity (последний прогон)"))
    keys = [
        "n_before",
        "n_after",
        "n_kept_branch_a",
        "n_kept_branch_b_only",
        "n_dropped_horizon",
        "n_dropped_open_court",
        "n_dropped_t0_missing",
        "snapshot_date",
        "mature_until",
        "horizon_months_no_court",
        "policy",
    ]
    rows = {k: report.get(k) for k in keys}
    display(pd.Series(rows, name="report").to_frame())
    if report.get("n_before") and report.get("n_after") is not None:
        before = int(report["n_before"])
        after = int(report["n_after"])
        dropped = before - after
        pct = 100.0 * dropped / before if before else 0.0
        print(f"Итого: {before:,} → {after:,} (−{dropped:,}, {pct:.1f}%)")
    lags = report.get("lag_days_target_freq_1") or {}
    if lags.get("n"):
        display(Markdown("### Лаг T0 → последняя дата иска (TARGET_FREQ=1)"))
        display(pd.Series(lags, name="lag_days").to_frame())


def _plot_monthly_counts(
    monthly: pd.DataFrame,
    *,
    mature_until: str | None,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax0, ax1 = axes
    ax0.bar(monthly.index, monthly["n_incidents"], width=20, color="#4C78A8", alpha=0.85)
    ax0.set_ylabel("инциденты")
    ax0.set_title(title)
    if mature_until:
        cut = pd.Timestamp(mature_until)
        ax0.axvline(cut, color="#E45756", ls="--", lw=1.5, label=f"mature_until={mature_until}")
        ax0.legend(loc="upper right")
    if "freq1_rate" in monthly.columns:
        ax1.plot(monthly.index, monthly["freq1_rate"] * 100, color="#F58518", marker="o", ms=3)
        ax1.set_ylabel("TARGET_FREQ=1, %")
        ax1.set_ylim(0, max(5, float(monthly["freq1_rate"].max() * 100) * 1.15))
    ax1.set_xlabel("месяц T0 (PAYMENT_ORDER_DATE_TIME)")
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()


def _plot_claims_hist(counts: pd.DataFrame) -> None:
    if counts.empty:
        print("Нет строк claims для гистограммы")
        return
    fig, ax = plt.subplots(figsize=(8, 3.5))
    vals = counts["n_claim_rows"].clip(upper=20)
    ax.hist(vals, bins=np.arange(0.5, 21.5, 1), color="#54A24B", alpha=0.85, edgecolor="white")
    ax.set_xlabel("строк исков на инцидент (clip≤20)")
    ax.set_ylabel("число инцидентов (в df)")
    ax.set_title("Распределение числа исков на инцидент (после maturity)")
    fig.tight_layout()
    plt.show()
    print(
        f"claims/incident: mean={counts['n_claim_rows'].mean():.2f}, "
        f"p50={counts['n_claim_rows'].median():.0f}, "
        f"p90={counts['n_claim_rows'].quantile(0.9):.0f}, "
        f"max={counts['n_claim_rows'].max()}, "
        f"без исков в claims: см. долю ниже"
    )


def _gap_months(monthly: pd.DataFrame) -> pd.DataFrame:
    """Месяцы с нулём или резким провалом относительно соседей."""
    if monthly.empty:
        return monthly
    full_idx = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    series = monthly["n_incidents"].reindex(full_idx, fill_value=0)
    gaps = series[series == 0]
    out = gaps.rename("n_incidents").to_frame()
    out["note"] = "zero_incidents"
    return out


def run_maturity_eda(
    df: pd.DataFrame,
    *,
    project_root: Path | str | None = None,
    freq_col: str = "TARGET_FREQ",
    sev_col: str = "TARGET_SEV",
) -> dict[str, Any]:
    """Статистика и графики вызревания сразу после загрузки df в collect.

    Показывает:
    - политику и ``target_maturity_report.json``;
    - инциденты и долю TARGET_FREQ=1 по месяцам T0;
    - годовые когорты;
    - месяцы-дыры (0 инцидентов);
    - распределение числа исков на инцидент (если есть claims parquet).
    """
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    policy = maturity_policy_summary(root)
    report = load_maturity_report(root)
    t0 = policy["t0_column"]
    mature_until = (report or {}).get("mature_until") or policy["mature_until"]

    _display_report(report, policy)

    monthly = monthly_incident_stats(df, t0_column=t0, freq_col=freq_col, sev_col=sev_col)
    display(Markdown("### Инциденты и TARGET_FREQ по месяцам T0 (после maturity)"))
    _fmt = {
        "n_rows": "{:,.0f}",
        "n_incidents": "{:,.0f}",
        "n_freq1": "{:,.0f}",
        "freq1_rate": "{:.1%}",
        "sev_sum": "{:,.0f}",
        "sev_mean_pos": "{:,.0f}",
    }
    display(monthly.tail(24).style.format({k: v for k, v in _fmt.items() if k in monthly.columns}, na_rep="—"))
    _plot_monthly_counts(
        monthly,
        mature_until=str(mature_until) if mature_until else None,
        title="Инциденты по месяцу T0 (после maturity); красная линия = mature_until ветки B",
    )

    display(Markdown("### Когорты по году T0"))
    yearly = yearly_cohort_stats(df, t0_column=t0, freq_col=freq_col)
    display(yearly.style.format({"freq1_rate": "{:.1%}", "n_rows": "{:,}", "n_incidents": "{:,}", "n_freq1": "{:,}"}))

    gaps = _gap_months(monthly)
    display(Markdown("### Дырки: месяцы без инцидентов в финальном df"))
    if gaps.empty:
        print("Пустых месяцев в диапазоне T0 нет")
    else:
        display(gaps)
        print(f"Пустых месяцев: {len(gaps)}")

    # Хвост после mature_until — только ветка A; полезно смотреть плотность и freq rate
    if mature_until and t0 in df.columns:
        cut = pd.Timestamp(mature_until)
        t0s = pd.to_datetime(df[t0], errors="coerce")
        recent = df.loc[t0s > cut]
        older = df.loc[t0s <= cut]
        display(Markdown("### Хвост после mature_until (только ветка A могла выжить)"))
        print(
            f"T0 ≤ {cut.date()}: n={len(older):,} | "
            f"T0 > {cut.date()}: n={len(recent):,} "
            f"(доля хвоста {100 * len(recent) / max(len(df), 1):.1f}%)"
        )
        if freq_col in df.columns and len(recent):
            r = pd.to_numeric(recent[freq_col], errors="coerce").fillna(0).eq(1).mean()
            o = (
                pd.to_numeric(older[freq_col], errors="coerce").fillna(0).eq(1).mean()
                if len(older)
                else float("nan")
            )
            print(f"TARGET_FREQ=1 rate: older={o:.1%} | recent(A-only)={r:.1%}")

    claims = load_claims_frame(root)
    claims_counts = None
    if claims is None:
        print(f"Нет {_CLAIMS_ARTIFACT} — пропуск гистограммы исков")
    elif "INCIDENT_NUMBER" not in df.columns:
        print("Нет INCIDENT_NUMBER в df — пропуск гистограммы исков")
    else:
        display(Markdown("### Иски на инцидент (target_3_claims ∩ df)"))
        claims_counts = claims_per_incident_table(
            claims, incident_ids=df["INCIDENT_NUMBER"]
        )
        covered = claims_counts["INCIDENT_NUMBER"].nunique()
        total = df["INCIDENT_NUMBER"].nunique()
        print(
            f"Инцидентов в df с ≥1 строкой claims: {covered:,} / {total:,} "
            f"({100 * covered / max(total, 1):.1f}%)"
        )
        _plot_claims_hist(claims_counts)
        # месячное число исков по get-date / period если есть
        c = claims.copy()
        c.columns = [str(x).upper() for x in c.columns]
        if "INCIDENTNUMBER" in c.columns and "INCIDENT_NUMBER" not in c.columns:
            c["INCIDENT_NUMBER"] = c["INCIDENTNUMBER"]
        period_col = next(
            (
                col
                for col in ("CLAIMEDVALUEPERIOD", "INCOMINGCLAIMGETDATE", "INCOMING_CLAIM_GET_DATE")
                if col in c.columns
            ),
            None,
        )
        if period_col and "INCIDENT_NUMBER" in c.columns:
            keep_ids = set(df["INCIDENT_NUMBER"].dropna().astype(str).str.strip())
            c = c.loc[c["INCIDENT_NUMBER"].astype(str).str.strip().isin(keep_ids)]
            c["_m"] = _month_start(c[period_col])
            by_m = c.groupby("_m").size().rename("n_claim_rows")
            fig, ax = plt.subplots(figsize=(12, 3.2))
            ax.bar(by_m.index, by_m.values, width=20, color="#B279A2", alpha=0.85)
            ax.set_title("Строки исков по месяцу даты иска (только инциденты из df)")
            ax.set_ylabel("claim rows")
            if mature_until:
                ax.axvline(pd.Timestamp(mature_until), color="#E45756", ls="--", lw=1.5)
            fig.autofmt_xdate()
            fig.tight_layout()
            plt.show()

    return {
        "policy": policy,
        "report": report,
        "monthly": monthly,
        "yearly": yearly,
        "gap_months": gaps,
        "claims_per_incident": claims_counts,
    }
