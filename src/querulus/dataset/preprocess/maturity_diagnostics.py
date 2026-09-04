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
_PRETENSIONS_ARTIFACT = "df_pretensions.parquet"
_PAYMENTS_ARTIFACT = "df_payments.parquet"
_LAG_PERCENTILES: tuple[int, ...] = (50, 60, 70, 80, 90, 95, 99)


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
    """Текущая политика из dataset_filters + S / silent_until / cooloff_until."""
    filters = load_dataset_filters()
    cfg = filters.get("target_maturity") or {}
    snapshot, source = resolve_snapshot_date(filters)
    if cfg.get("silent_horizon_months_fu_court") is not None:
        silent_court = int(cfg["silent_horizon_months_fu_court"])
    elif cfg.get("silent_horizon_months") is not None:
        silent_court = int(cfg["silent_horizon_months"])
    elif cfg.get("horizon_months_no_court") is not None:
        silent_court = int(cfg["horizon_months_no_court"])
    elif cfg.get("horizon_months") is not None:
        silent_court = int(cfg["horizon_months"])
    else:
        silent_court = 24
    silent_pret = int(cfg.get("silent_horizon_months_pretension") or 6)
    cooloff = int(cfg.get("cooloff_months_after_psr") or 24)
    pret_until = (snapshot - pd.DateOffset(months=silent_pret)).normalize()
    silent_until = (snapshot - pd.DateOffset(months=silent_court)).normalize()
    cooloff_until = (snapshot - pd.DateOffset(months=cooloff)).normalize()
    victim = filters.get("victim") or {}
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "t0_column": str(cfg.get("t0_column") or _T0_DEFAULT),
        "silent_horizon_months_pretension": silent_pret,
        "silent_horizon_months_fu_court": silent_court,
        "silent_horizon_months": silent_court,
        "cooloff_months_after_psr": cooloff,
        "drop_psr_before_t0": bool(cfg.get("drop_psr_before_t0", True)),
        "horizon_months_no_court": silent_court,
        "snapshot_date": str(snapshot.date()),
        "snapshot_source": source,
        "mature_until": str(pd.Timestamp(silent_until).date()),
        "silent_until": str(pd.Timestamp(silent_until).date()),
        "silent_pretension_until": str(pd.Timestamp(pret_until).date()),
        "cooloff_until": str(pd.Timestamp(cooloff_until).date()),
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


def load_pretensions_frame(project_root: Path | None = None) -> pd.DataFrame | None:
    path = _raw_dir(project_root) / _PRETENSIONS_ARTIFACT
    if not path.is_file():
        return None
    return pd.read_parquet(path)


def load_payments_frame(project_root: Path | None = None) -> pd.DataFrame | None:
    path = _raw_dir(project_root) / _PAYMENTS_ARTIFACT
    if not path.is_file():
        return None
    return pd.read_parquet(path)


def _normalize_incident_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).upper() for c in out.columns]
    if "INCIDENTNUMBER" in out.columns and "INCIDENT_NUMBER" not in out.columns:
        out["INCIDENT_NUMBER"] = out["INCIDENTNUMBER"]
    return out


def _incident_key(series: pd.Series) -> pd.Series:
    num = pd.to_numeric(series, errors="coerce")
    out = series.astype(str).str.strip()
    int_like = num.notna() & ((num % 1) == 0)
    out = out.copy()
    out.loc[int_like] = num.loc[int_like].astype("int64").astype(str)
    return out


def first_payment_by_incident(payments: pd.DataFrame) -> pd.Series:
    """Минимум даты выплаты по инциденту."""
    work = _normalize_incident_col(payments)
    date_col = next(
        (
            c
            for c in (
                "PAYMENT_DATETIME",
                "PAYMENTDATETIME",
                "PAYMENT_DATE_TIME",
                "PAYMENTDATETIME",
            )
            if c in work.columns
        ),
        None,
    )
    if date_col is None:
        for c in work.columns:
            if "PAYMENT" in c and "DATE" in c:
                date_col = c
                break
    if "INCIDENT_NUMBER" not in work.columns or date_col is None:
        raise KeyError("В payments нужны INCIDENT_NUMBER и дата выплаты")
    work["_pay"] = pd.to_datetime(work[date_col], errors="coerce")
    work["_inc"] = _incident_key(work["INCIDENT_NUMBER"])
    return work.groupby("_inc", sort=False)["_pay"].min().rename("first_payment_datetime")


def pretension_lag_frame(
    df: pd.DataFrame,
    pretensions: pd.DataFrame,
    *,
    payments: pd.DataFrame | None = None,
    t0_column: str = _T0_DEFAULT,
) -> pd.DataFrame:
    """Лаг (дни): дата претензии − база; база = первая выплата, иначе T0.

    Одна строка на претензию инцидента из ``df``.
    """
    if "INCIDENT_NUMBER" not in df.columns:
        raise KeyError("В df нет INCIDENT_NUMBER")
    if t0_column not in df.columns:
        raise KeyError(f"В df нет T0: {t0_column}")

    base = df[["INCIDENT_NUMBER", t0_column]].copy()
    base["_inc"] = _incident_key(base["INCIDENT_NUMBER"])
    base["_t0"] = pd.to_datetime(base[t0_column], errors="coerce")
    base = base.drop_duplicates("_inc", keep="first")

    if payments is not None and len(payments):
        try:
            first_pay = first_payment_by_incident(payments)
            base = base.merge(first_pay, how="left", left_on="_inc", right_index=True)
        except KeyError as exc:
            logger.warning("payments без нужных колонок: %s", exc)
            base["first_payment_datetime"] = pd.NaT
    else:
        base["first_payment_datetime"] = pd.NaT

    base["_base"] = base["first_payment_datetime"].fillna(base["_t0"])
    base["base_source"] = np.where(
        base["first_payment_datetime"].notna(), "payment", "t0"
    )

    pret = _normalize_incident_col(pretensions)
    get_col = next(
        (
            c
            for c in (
                "PRETENSION_GET_DATE",
                "PRETENSIONGETDATE",
                "PRETENSION_DATE",
                "PRETENSIONDATE",
            )
            if c in pret.columns
        ),
        None,
    )
    if "INCIDENT_NUMBER" not in pret.columns or get_col is None:
        raise KeyError("В pretensions нужны INCIDENT_NUMBER и PRETENSION_GET_DATE")
    pret["_inc"] = _incident_key(pret["INCIDENT_NUMBER"])
    pret["_pret_date"] = pd.to_datetime(pret[get_col], errors="coerce")
    pret = pret.loc[pret["_pret_date"].notna(), ["_inc", "_pret_date"]].copy()

    keep = set(base["_inc"])
    pret = pret.loc[pret["_inc"].isin(keep)]
    merged = pret.merge(
        base[["_inc", "_base", "_t0", "first_payment_datetime", "base_source"]],
        on="_inc",
        how="inner",
    )
    merged["lag_days"] = (merged["_pret_date"] - merged["_base"]).dt.days
    # первая претензия на инцидент
    first = (
        merged.sort_values(["_inc", "_pret_date"])
        .drop_duplicates("_inc", keep="first")
        .copy()
    )
    first["which"] = "first_pretension"
    merged = merged.copy()
    merged["which"] = "all_pretensions"
    return pd.concat([first, merged], ignore_index=True)


def lag_percentile_table(lag_days: pd.Series) -> pd.Series:
    clean = pd.to_numeric(lag_days, errors="coerce").dropna()
    clean = clean[clean >= 0]
    out: dict[str, float | int] = {"n": int(len(clean))}
    if clean.empty:
        for p in _LAG_PERCENTILES:
            out[f"p{p}"] = float("nan")
        out["mean"] = float("nan")
        return pd.Series(out, name="lag_days")
    for p in _LAG_PERCENTILES:
        out[f"p{p}"] = float(clean.quantile(p / 100.0))
    out["mean"] = float(clean.mean())
    out["share_within_90d"] = float((clean <= 90).mean())
    out["share_within_180d"] = float((clean <= 180).mean())
    out["share_within_365d"] = float((clean <= 365).mean())
    return pd.Series(out, name="lag_days")


def _plot_pretension_lags(lags: pd.DataFrame) -> None:
    first = lags.loc[lags["which"] == "first_pretension", "lag_days"]
    first = pd.to_numeric(first, errors="coerce").dropna()
    first_pos = first[first >= 0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    ax0, ax1 = axes
    if first_pos.empty:
        ax0.set_title("Нет неотрицательных лагов (первая претензия)")
    else:
        clip = first_pos.clip(upper=max(730, float(first_pos.quantile(0.99))))
        ax0.hist(clip, bins=40, color="#4C78A8", alpha=0.85, edgecolor="white")
        ax0.set_title("Лаг до первой претензии (дни, clip≈p99)")
        ax0.set_xlabel("дни после выплаты / T0")
        ax0.set_ylabel("претензии (инциденты)")
    if first_pos.empty:
        ax1.set_title("CDF недоступен")
    else:
        xs = np.sort(first_pos.to_numpy())
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax1.plot(xs, ys * 100, color="#F58518")
        ax1.set_xlim(0, min(730, float(xs.max()) if len(xs) else 730))
        ax1.set_xlabel("дни")
        ax1.set_ylabel("доля, %")
        ax1.set_title("CDF: первая претензия после выплаты/T0")
        ax1.axvline(90, color="#E45756", ls="--", lw=1, label="90д")
        ax1.axvline(180, color="#54A24B", ls="--", lw=1, label="180д")
        ax1.axvline(365, color="#B279A2", ls="--", lw=1, label="365д")
        ax1.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    plt.show()


def display_pretension_lag_eda(
    df: pd.DataFrame,
    *,
    project_root: Path | None = None,
    t0_column: str = _T0_DEFAULT,
) -> pd.DataFrame | None:
    """Аналитика лага претензий относительно выплаты (fallback T0)."""
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    pret = load_pretensions_frame(root)
    if pret is None:
        print(
            f"Нет {_PRETENSIONS_ARTIFACT} — пропуск лага претензий "
            "(нужен raw parquet; при pipeline с enrich/USE_SQL он появляется)."
        )
        return None
    payments = load_payments_frame(root)
    try:
        lags = pretension_lag_frame(
            df, pret, payments=payments, t0_column=t0_column
        )
    except KeyError as exc:
        print(f"Лаг претензий недоступен: {exc}")
        return None

    display(Markdown("### Лаг претензий после выплаты (если нет выплаты → T0)"))
    n_pay = int((lags["base_source"] == "payment").sum())
    n_t0 = int((lags["base_source"] == "t0").sum())
    print(
        f"База даты: payment={n_pay:,} строк претензий, t0-fallback={n_t0:,}"
        + ("" if payments is not None else f" (нет {_PAYMENTS_ARTIFACT} — всё через T0)")
    )
    neg = int((pd.to_numeric(lags["lag_days"], errors="coerce") < 0).sum())
    if neg:
        print(f"Отрицательный лаг (претензия раньше базы): {neg:,} — в перцентили не входят")

    for label, which in (
        ("Первая претензия на инцидент", "first_pretension"),
        ("Все претензии", "all_pretensions"),
    ):
        part = lags.loc[lags["which"] == which, "lag_days"]
        display(Markdown(f"#### {label}"))
        stats = lag_percentile_table(part)
        display(stats.to_frame())

    _plot_pretension_lags(lags)
    return lags


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
        "enabled",
        "n_before",
        "n_after",
        "n_kept_silent",
        "n_kept_cooloff",
        "n_dropped_still_maturing",
        "n_dropped_open_court",
        "n_dropped_psr_before_t0",
        "n_dropped_t0_missing",
        "n_with_psr_event",
        "n_without_psr_event",
        "silent_horizon_months_pretension",
        "silent_horizon_months_fu_court",
        "cooloff_months_after_psr",
        "drop_psr_before_t0",
        "snapshot_date",
        "silent_until",
        "silent_pretension_until",
        "cooloff_until",
        "policy",
        "lag_definition",
        "pretensions_loaded",
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
    - распределение числа исков на инцидент (если есть claims parquet);
    - лаг претензий после выплаты (fallback T0).
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
    display(monthly.style.format({k: v for k, v in _fmt.items() if k in monthly.columns}, na_rep="—"))
    _plot_monthly_counts(
        monthly,
        mature_until=str(mature_until) if mature_until else None,
        title="Инциденты по месяцу T0 (после maturity); красная линия = silent_until",
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

    # Хвост после silent_until: без ПСР сюда не попасть; только cooloff после события ПСР
    if mature_until and t0 in df.columns:
        cut = pd.Timestamp(mature_until)
        t0s = pd.to_datetime(df[t0], errors="coerce")
        recent = df.loc[t0s > cut]
        older = df.loc[t0s <= cut]
        display(
            Markdown(
                "### Хвост после silent_until "
                "(без ПСР сюда не попасть — только охлаждение после события)"
            )
        )
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
            print(f"TARGET_FREQ=1 rate: older={o:.1%} | recent(cooloff)={r:.1%}")

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

    pretension_lags = display_pretension_lag_eda(df, project_root=root, t0_column=t0)

    return {
        "policy": policy,
        "report": report,
        "monthly": monthly,
        "yearly": yearly,
        "gap_months": gaps,
        "claims_per_incident": claims_counts,
        "pretension_lags": pretension_lags,
    }
