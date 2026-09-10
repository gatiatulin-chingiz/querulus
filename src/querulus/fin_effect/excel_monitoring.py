"""Единая ITT-оценка финансового эффекта по витрине мониторинга."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from querulus.fin_effect.excel_explore import (
    VITRINA_TABLE_DEFAULT,
    _to_numeric,
    analytics_base_mask,
    load_monitoring_frame,
    resolve_column,
    resolve_model_payout_loss_column,
)
from querulus.fin_effect.monitoring_analytics import (
    agreement_mask,
    compare_path_shares,
    filial_path_shares,
    shares_as_percent,
)

FU_FEE_DEFAULT = 100_000.0
COURT_FEE_DEFAULT = 15_000.0
RETRO_AS_OF_DEFAULT = "2025-06-30"
RESULT_OUT_OF_MODEL = -100
HORIZONS = (365, 1095)
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

LOSS_CANDIDATES = ("Убыток", "LOSS_NUMBER", "LossNumber")
INCIDENT_CANDIDATES = (
    "НомерИнцидент",
    "НомерИнцидента",
    "INCIDENT_NUMBER",
)
RETRO_DATE_CANDIDATES = (
    "PAYMENT_ORDER_DATE_TIME",
    "INCOMING_CLAIM_GET_DATE",
    "INCOMING_CLAIM_GET_DATE_1",
    "LOSS_DATE",
)
RETRO_FILIAL_CANDIDATES = ("FILIAL", "Филиал")
PSR_AMOUNT_CANDIDATES = {
    "pretension": (
        "Cумма выплаты по претензии",
        "Сумма выплаты по претензии",
        "TARGET_FREQ_PRET_AMOUNT",
    ),
    "fu": (
        "Cумма выплат по ФУ",
        "Сумма выплат по ФУ",
        "Сумма_взыскано_по_ФУ",
    ),
    "court": ("Cумма выплаты по суду", "Сумма выплаты по суду", "Суммы_взыскано_по_иску"),
}


@dataclass(frozen=True)
class GroupRetroPriors:
    """Терминальные коэффициенты ПСР для одной группы филиалов."""

    group: str
    filials: tuple[str, ...]
    n_rows: int
    n_positive: int
    n_positive_with_od: int
    p_ultimate: float
    k_ultimate: float
    mean_positive_psr: float
    mean_psr_all: float
    p_fu_given_psr: float
    p_court_given_psr: float
    fu_fee: float = FU_FEE_DEFAULT
    court_fee: float = COURT_FEE_DEFAULT

    @property
    def expected_fee(self) -> float:
        """Средние организационные расходы ФУ/суда на один ПСР."""
        return (
            self.p_fu_given_psr * self.fu_fee
            + self.p_court_given_psr * self.court_fee
        )

    def as_row(self) -> dict[str, Any]:
        """Представить коэффициенты строкой отчёта."""
        return {
            "group": self.group,
            "n_filials": len(self.filials),
            "filials": ", ".join(self.filials) if self.group == "pilot" else "all other",
            "n_rows": self.n_rows,
            "n_positive": self.n_positive,
            "n_positive_with_od": self.n_positive_with_od,
            "n_positive_without_retro_od": (
                self.n_positive - self.n_positive_with_od
            ),
            "p_U": self.p_ultimate,
            "k_U": self.k_ultimate,
            "m_U": self.mean_positive_psr,
            "mean_PSR_all": self.mean_psr_all,
            "p_FU_given_PSR": self.p_fu_given_psr,
            "p_court_given_PSR": self.p_court_given_psr,
            "e_U": self.expected_fee,
        }


@dataclass(frozen=True)
class TerminalRetroPriors:
    """Терминальные priors пилотных и непилотных филиалов."""

    pilot: GroupRetroPriors
    nonpilot: GroupRetroPriors
    date_column: str | None
    window_start: str | None
    window_end: str | None
    lookback_years: float | None

    def table(self) -> pd.DataFrame:
        """Таблица коэффициентов для HTML."""
        return pd.DataFrame([self.pilot.as_row(), self.nonpilot.as_row()])


@dataclass
class MonitoringEffectResult:
    """Все результаты единой методики для HTML-отчёта."""

    frame: pd.DataFrame
    priors: TerminalRetroPriors
    effect_summary: pd.DataFrame
    group_summary: pd.DataFrame
    filial_effects: pd.DataFrame
    compliance_a: pd.DataFrame
    compliance_b: pd.DataFrame
    sensitivity: pd.DataFrame
    annual_summary: pd.DataFrame
    seasonality: pd.DataFrame
    data_quality: pd.DataFrame
    path_shares: pd.DataFrame
    filial_path_shares: pd.DataFrame
    attention_filials: pd.DataFrame
    contract: dict[str, str]
    t_calc: pd.Timestamp
    discount_rate: float
    residual_share: float
    bootstrap_iterations: int
    bootstrap_compliance_iterations: int
    warnings: list[str] = field(default_factory=list)


def format_money(value: float) -> str:
    """Форматировать сумму без научной нотации."""
    return f"{float(value):,.2f}".replace(",", " ")


def _first_existing(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    return next((name for name in names if name in df.columns), None)


def _required_existing(
    df: pd.DataFrame,
    names: Iterable[str],
    label: str,
) -> str:
    column = _first_existing(df, names)
    if column is None:
        raise KeyError(f"Не найдена колонка {label}. Проверены: {tuple(names)}")
    return column


def _required_alias(df: pd.DataFrame, alias: str, label: str) -> str:
    column = resolve_column(df, alias)
    if column is None:
        raise KeyError(f"Не найдена колонка {label} (alias={alias})")
    return column


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.casefold()


def _resolve_retro_date(df: pd.DataFrame) -> str | None:
    return _first_existing(df, RETRO_DATE_CANDIDATES)


def filter_retro_lookback(
    df: pd.DataFrame,
    *,
    lookback_years: float | None = 2.0,
    as_of: str | pd.Timestamp | None = RETRO_AS_OF_DEFAULT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ограничить финальный incident-level датафрейм ретро-окном."""
    date_column = _resolve_retro_date(df)
    meta = {
        "date_column": date_column,
        "window_start": None,
        "window_end": None,
        "lookback_years": lookback_years,
    }
    if date_column is None or not lookback_years:
        return df.copy(), meta

    dates = pd.to_datetime(df[date_column], errors="coerce")
    end = pd.Timestamp(as_of) if as_of is not None else dates.max()
    if pd.isna(end):
        raise ValueError("В ретро-данных нет валидной даты для окна коэффициентов")
    start = end - pd.Timedelta(days=365.25 * float(lookback_years))
    mask = dates.between(start, end, inclusive="both")
    work = df.loc[mask].copy()
    if work.empty:
        raise ValueError("Ретро-окно не содержит строк")
    meta.update(
        {
            "window_start": start.date().isoformat(),
            "window_end": pd.Timestamp(end).date().isoformat(),
        }
    )
    return work, meta


def _group_retro_priors(
    df: pd.DataFrame,
    *,
    group: str,
    filials: tuple[str, ...],
    amount_col: str,
    od_col: str,
    fu_col: str,
    court_col: str,
) -> GroupRetroPriors:
    amount = _to_numeric(df[amount_col]).fillna(0.0).clip(lower=0.0)
    od = _to_numeric(df[od_col])
    fu = _to_numeric(df[fu_col]).fillna(0.0)
    court = _to_numeric(df[court_col]).fillna(0.0)
    positive = amount > 0
    positive_with_od = positive & od.gt(0)

    n_rows = int(len(df))
    n_positive = int(positive.sum())
    if n_rows == 0:
        raise ValueError(f"Нет строк для ретро-группы {group}")
    if n_positive == 0:
        raise ValueError(f"Нет положительных ПСР для ретро-группы {group}")

    od_sum = float(od[positive_with_od].sum())
    k_ultimate = (
        float(amount[positive_with_od].sum() / od_sum)
        if od_sum > 0
        else 0.0
    )
    is_court = positive & court.gt(0)
    is_fu = positive & fu.gt(0) & ~is_court
    return GroupRetroPriors(
        group=group,
        filials=filials,
        n_rows=n_rows,
        n_positive=n_positive,
        n_positive_with_od=int(positive_with_od.sum()),
        p_ultimate=float(n_positive / n_rows),
        k_ultimate=k_ultimate,
        mean_positive_psr=float(amount[positive].mean()),
        mean_psr_all=float(amount.mean()),
        p_fu_given_psr=float(is_fu.sum() / n_positive),
        p_court_given_psr=float(is_court.sum() / n_positive),
    )


def compute_terminal_priors(
    retro_df: pd.DataFrame,
    *,
    pilot_filials: Iterable[str],
    lookback_years: float | None = 2.0,
    as_of: str | pd.Timestamp | None = RETRO_AS_OF_DEFAULT,
) -> TerminalRetroPriors:
    """Посчитать p_U, k_U, m_U и e_U из финального incident-level df."""
    work, meta = filter_retro_lookback(
        retro_df,
        lookback_years=lookback_years,
        as_of=as_of,
    )
    filial_col = _required_existing(work, RETRO_FILIAL_CANDIDATES, "FILIAL")
    amount_col = _required_existing(work, ("TARGET_FREQ_AMOUNT",), "TARGET_FREQ_AMOUNT")
    od_col = _required_existing(
        work,
        ("RECOVEREDMAINDEBT_LAST_INST_SUM",),
        "RECOVEREDMAINDEBT_LAST_INST_SUM",
    )
    fu_col = _required_existing(
        work,
        ("Сумма_взыскано_по_ФУ",),
        "Сумма_взыскано_по_ФУ",
    )
    court_col = _required_existing(
        work,
        ("Суммы_взыскано_по_иску",),
        "Суммы_взыскано_по_иску",
    )

    pilot_names = tuple(sorted({str(value).strip() for value in pilot_filials}))
    pilot_norm = {value.casefold() for value in pilot_names}
    is_pilot = _normalize_text(work[filial_col]).isin(pilot_norm)
    pilot_df = work.loc[is_pilot]
    nonpilot_df = work.loc[~is_pilot & work[filial_col].notna()]
    nonpilot_names = tuple(
        sorted(nonpilot_df[filial_col].dropna().astype(str).str.strip().unique())
    )

    kwargs = {
        "amount_col": amount_col,
        "od_col": od_col,
        "fu_col": fu_col,
        "court_col": court_col,
    }
    return TerminalRetroPriors(
        pilot=_group_retro_priors(
            pilot_df,
            group="pilot",
            filials=pilot_names,
            **kwargs,
        ),
        nonpilot=_group_retro_priors(
            nonpilot_df,
            group="nonpilot",
            filials=nonpilot_names,
            **kwargs,
        ),
        date_column=meta["date_column"],
        window_start=meta["window_start"],
        window_end=meta["window_end"],
        lookback_years=lookback_years,
    )


def _prepare_monitoring_contract(
    df: pd.DataFrame,
    *,
    t_calc: str | pd.Timestamp | None,
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    result_col = _required_alias(df, "result_check", "РезультатПроверки")
    filial_col = _required_alias(df, "filial", "Филиал")
    payment_col = _required_alias(df, "payment", "СуммаПлатежа")
    to_pay_col = _required_alias(df, "to_pay", "СуммаКВыплате")
    od_col = _required_alias(df, "od_claimed", "СуммаОсновногоДолгаЗаявлено")
    recommended_col = _required_alias(
        df,
        "recommended",
        "Сумма рекомендованная к доплате по модели",
    )
    payout_col = resolve_model_payout_loss_column(df)
    if payout_col is None:
        raise KeyError("Не найдена колонка Выплата по модели")
    loss_col = _required_existing(df, LOSS_CANDIDATES, "номера убытка")
    incident_col = _required_existing(df, INCIDENT_CANDIDATES, "номера инцидента")
    call_date_col = _required_alias(
        df,
        "model_call_date",
        "Дата вызова модели сутяжности",
    )
    application_col = _required_alias(df, "application_date", "ДатаЗаявления")
    psr_columns = {
        key: _required_existing(df, candidates, f"ПСР: {key}")
        for key, candidates in PSR_AMOUNT_CANDIDATES.items()
    }

    base = analytics_base_mask(df, filial_scope="pilot")
    result = _to_numeric(df[result_col])
    eligible = base & result.isin([0, 1, RESULT_OUT_OF_MODEL])
    work = df.loc[eligible].copy()
    if work.empty:
        raise ValueError("После model/control и базовых фильтров нет строк")

    work["_result"] = _to_numeric(work[result_col])
    work["_group"] = np.where(
        work["_result"].eq(RESULT_OUT_OF_MODEL),
        "control",
        "model",
    )
    work["_filial"] = work[filial_col].astype("string").fillna("(пусто)")
    work["_loss"] = work[loss_col]
    work["_incident"] = work[incident_col]
    work["_paid_missing"] = _to_numeric(work[payment_col]).isna()
    work["_od_missing"] = _to_numeric(work[od_col]).isna()
    work["_paid_to_date"] = _to_numeric(work[payment_col]).fillna(0.0).clip(lower=0.0)
    work["_to_pay"] = _to_numeric(work[to_pay_col]).fillna(0.0).clip(lower=0.0)
    work["_od"] = _to_numeric(work[od_col])
    work["_recommended_extra"] = (
        _to_numeric(work[recommended_col]).fillna(0.0).clip(lower=0.0)
    )
    work["_payout_by_model"] = (
        _to_numeric(work[payout_col]).fillna(0.0).eq(1)
    )
    work["_agreement"] = agreement_mask(work)
    for key, column in psr_columns.items():
        work[f"_psr_{key}"] = _to_numeric(work[column]).fillna(0.0).clip(lower=0.0)
    work["_observed_psr"] = work[
        ["_psr_pretension", "_psr_fu", "_psr_court"]
    ].sum(axis=1)

    call_dates = pd.to_datetime(work[call_date_col], errors="coerce")
    application_dates = pd.to_datetime(work[application_col], errors="coerce")
    work["_t0"] = call_dates.fillna(application_dates)
    work["_application_date"] = application_dates.fillna(work["_t0"])
    if work["_t0"].isna().any():
        raise ValueError("Есть строки без t0: нет даты вызова модели и ДатаЗаявления")

    calc_date = (
        pd.Timestamp(t_calc).normalize()
        if t_calc is not None
        else pd.Timestamp.today().normalize()
    )
    work["_age_days"] = (calc_date - work["_t0"].dt.normalize()).dt.days
    warnings: list[str] = []
    if work["_age_days"].lt(0).any():
        warnings.append("Есть t0 позже t_calc; age_days для них ограничен нулём.")
        work["_age_days"] = work["_age_days"].clip(lower=0)
    if work["_age_days"].ge(min(HORIZONS)).any():
        n_old = int(work["_age_days"].ge(min(HORIZONS)).sum())
        raise ValueError(
            f"{n_old} строк имеют age_days >= 365. "
            "Методика требует отдельного наблюдаемого Y365 и останавливает расчёт."
        )

    contract = {
        "loss": loss_col,
        "incident": incident_col,
        "result": result_col,
        "filial": filial_col,
        "payment": payment_col,
        "to_pay_diagnostic_only": to_pay_col,
        "od": od_col,
        "recommended_extra": recommended_col,
        "payout_by_model": payout_col,
        "agreement": resolve_column(df, "agreement") or "agreement_mask",
        "t0_primary": call_date_col,
        "t0_fallback": application_col,
        "psr_pretension": psr_columns["pretension"],
        "psr_fu": psr_columns["fu"],
        "psr_court": psr_columns["court"],
    }
    return work, contract, warnings


def _add_outcomes(
    frame: pd.DataFrame,
    priors: GroupRetroPriors,
    *,
    residual_share: float,
    discount_rate: float,
    compliance_100: bool = False,
) -> pd.DataFrame:
    work = frame.copy()
    od_available = work["_od"].gt(0)
    work["_ultimate_base"] = np.where(
        od_available,
        work["_od"].fillna(0.0) * priors.k_ultimate,
        priors.mean_positive_psr,
    )
    work["_expected_open_psr"] = priors.p_ultimate * (
        work["_ultimate_base"] + priors.expected_fee
    )

    if compliance_100:
        force = (
            work["_group"].eq("model")
            & work["_result"].eq(1)
            & ~work["_payout_by_model"]
        )
        model_one = work["_group"].eq("model") & work["_result"].eq(1)
        work["_forced_extra"] = work["_recommended_extra"].where(force, 0.0)
        work["Yfact_100"] = work["_paid_to_date"] + work["_forced_extra"]
        residual = pd.Series(1.0, index=work.index)
        residual.loc[work["_agreement"]] = residual_share
        residual.loc[model_one] = residual_share
        prefix = "_100"
        fact_column = "Yfact_100"
    else:
        work["_forced_extra"] = 0.0
        work["Yfact"] = work["_paid_to_date"]
        residual = pd.Series(1.0, index=work.index)
        residual.loc[work["_agreement"]] = residual_share
        prefix = ""
        fact_column = "Yfact"

    remaining = (
        residual * work["_expected_open_psr"] - work["_observed_psr"]
    ).clip(lower=0.0)
    work[f"_remaining_nominal{prefix}"] = remaining
    for horizon in HORIZONS:
        remaining_days = (horizon - work["_age_days"]).clip(lower=0.0)
        midpoint_days = remaining_days / 2.0
        discount_factor = (1.0 + discount_rate) ** (midpoint_days / 365.0)
        column = f"Y{horizon}{prefix}"
        work[column] = work[fact_column] + remaining / discount_factor
        work[f"_midpoint_days_{horizon}{prefix}"] = midpoint_days
    return work


def _summaries(
    frame: pd.DataFrame,
    *,
    suffix: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes = {
        "fact": f"Yfact{suffix}",
        "365": f"Y365{suffix}",
        "1095": f"Y1095{suffix}",
    }
    group_rows: list[dict[str, Any]] = []
    filial_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []

    for horizon, column in outcomes.items():
        for group in ("control", "model"):
            part = frame.loc[frame["_group"].eq(group)]
            if part.empty:
                continue
            group_rows.append(
                {
                    "horizon": horizon,
                    "group": group,
                    "n": len(part),
                    "sum_cost": float(part[column].sum()),
                    "mean_cost": float(part[column].mean()),
                }
            )

        valid_filials: list[dict[str, Any]] = []
        for filial, part in frame.groupby("_filial", observed=True):
            means = part.groupby("_group", observed=True)[column].mean()
            row = {
                "horizon": horizon,
                "filial": filial,
                "n": len(part),
                "n_control": int(part["_group"].eq("control").sum()),
                "n_model": int(part["_group"].eq("model").sum()),
                "mean_control": float(means.get("control", np.nan)),
                "mean_model": float(means.get("model", np.nan)),
            }
            row["effect_control_minus_model"] = (
                row["mean_control"] - row["mean_model"]
            )
            filial_rows.append(row)
            if pd.notna(row["effect_control_minus_model"]):
                valid_filials.append(row)

        if not valid_filials:
            raise ValueError(
                f"Нет филиалов с model и control для горизонта {horizon}"
            )
        weight_total = sum(int(row["n"]) for row in valid_filials)
        effect = sum(
            float(row["effect_control_minus_model"]) * int(row["n"])
            for row in valid_filials
        ) / weight_total
        overall = frame.groupby("_group", observed=True)[column].mean()
        effect_rows.append(
            {
                "horizon": horizon,
                "effect_per_case": float(effect),
                "unstratified_effect": float(
                    overall.get("control", np.nan) - overall.get("model", np.nan)
                ),
                "n_weighted": weight_total,
                "n_filials": len(valid_filials),
            }
        )

    return (
        pd.DataFrame(group_rows),
        pd.DataFrame(filial_rows).sort_values(["horizon", "filial"]).reset_index(
            drop=True
        ),
        pd.DataFrame(effect_rows),
    )


def _cluster_resample(
    frame: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    groups = [part for _, part in frame.groupby("_incident", dropna=False)]
    picks = rng.integers(0, len(groups), size=len(groups))
    sampled = [groups[index].copy() for index in picks]
    return pd.concat(sampled, ignore_index=True)


def _bootstrap_ci(
    current: pd.DataFrame,
    retro: pd.DataFrame,
    *,
    pilot_filials: tuple[str, ...],
    lookback_years: float | None,
    as_of: str | pd.Timestamp | None,
    residual_share: float,
    discount_rate: float,
    iterations: int,
    seed: int,
    compliance_100: bool = False,
    progress_desc: str = "bootstrap ITT",
) -> pd.DataFrame:
    if iterations <= 0:
        return pd.DataFrame(columns=["horizon", "ci_low", "ci_high", "n_bootstrap"])
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None  # type: ignore[assignment]

    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"fact": [], "365": [], "1095": []}
    iterator = range(iterations)
    if tqdm is not None:
        iterator = tqdm(iterator, total=iterations, desc=progress_desc, leave=True)
    suffix = "_100" if compliance_100 else ""
    for _ in iterator:
        retro_sample = retro.iloc[
            rng.integers(0, len(retro), size=len(retro))
        ].reset_index(drop=True)
        try:
            priors = compute_terminal_priors(
                retro_sample,
                pilot_filials=pilot_filials,
                lookback_years=lookback_years,
                as_of=as_of,
            )
            current_sample = _cluster_resample(current, rng)
            outcomes = _add_outcomes(
                current_sample,
                priors.pilot,
                residual_share=residual_share,
                discount_rate=discount_rate,
                compliance_100=compliance_100,
            )
            _, _, effects = _summaries(outcomes, suffix=suffix)
        except (KeyError, ValueError, ZeroDivisionError):
            continue
        for row in effects.to_dict("records"):
            values[str(row["horizon"])].append(float(row["effect_per_case"]))

    rows = []
    for horizon, sample in values.items():
        if sample:
            low, high = np.quantile(sample, [0.025, 0.975])
        else:
            low, high = np.nan, np.nan
        rows.append(
            {
                "horizon": horizon,
                "ci_low": float(low),
                "ci_high": float(high),
                "n_bootstrap": len(sample),
            }
        )
    return pd.DataFrame(rows)


def _compliance_a(frame: pd.DataFrame) -> pd.DataFrame:
    model_one = frame.loc[
        frame["_group"].eq("model") & frame["_result"].eq(1)
    ].copy()
    model_one["_compliance"] = np.where(
        model_one["_payout_by_model"],
        "complied",
        "not_complied",
    )
    rows: list[dict[str, Any]] = []
    for status in ("complied", "not_complied"):
        part = model_one.loc[model_one["_compliance"].eq(status)]
        if part.empty:
            continue
        n = len(part)
        rows.append(
            {
                "horizon": "fact",
                "compliance": status,
                "n": n,
                "agreement_share": float(part["_agreement"].mean() * 100) if n else np.nan,
                "mean_cost": float(part["Yfact"].mean()),
                "descriptive_only": True,
            }
        )
    return pd.DataFrame(rows)


def _attention_filials(
    filial_paths: pd.DataFrame,
    filial_effects: pd.DataFrame,
) -> pd.DataFrame:
    """Филиалы с agreement(control)>agreement(model) или отрицательным ITT."""
    rows: list[dict[str, Any]] = []
    agreement_gap: dict[str, float] = {}
    if (
        not filial_paths.empty
        and {"filial", "segment", "agreement_share"}.issubset(filial_paths.columns)
    ):
        pivot = (
            filial_paths.pivot_table(
                index="filial",
                columns="segment",
                values="agreement_share",
                aggfunc="first",
            )
            .rename_axis(None, axis=1)
            .reset_index()
        )
        for row in pivot.to_dict("records"):
            filial = str(row["filial"])
            control = row.get("control", np.nan)
            model = row.get("model", np.nan)
            if pd.notna(control) and pd.notna(model):
                agreement_gap[filial] = float(control) - float(model)

    if filial_effects.empty:
        return pd.DataFrame(
            columns=[
                "filial",
                "agreement_share_control",
                "agreement_share_model",
                "agreement_gap_pp",
                "negative_effect_horizons",
                "min_effect_control_minus_model",
                "flags",
            ]
        )

    for filial, part in filial_effects.groupby("filial", dropna=False):
        filial_key = str(filial)
        gap = agreement_gap.get(filial_key, np.nan)
        effects = {
            str(row["horizon"]): float(row["effect_control_minus_model"])
            for row in part.to_dict("records")
            if pd.notna(row.get("effect_control_minus_model"))
        }
        negative = [h for h, value in effects.items() if value < 0]
        flags: list[str] = []
        if pd.notna(gap) and gap > 0:
            flags.append("agreement_control_gt_model")
        if negative:
            flags.append("negative_itt")
        if not flags:
            continue
        control_share = np.nan
        model_share = np.nan
        if filial_key in agreement_gap or not filial_paths.empty:
            match = filial_paths.loc[filial_paths["filial"].astype(str).eq(filial_key)]
            if not match.empty:
                control_row = match.loc[match["segment"].astype(str).eq("control")]
                model_row = match.loc[match["segment"].astype(str).eq("model")]
                if not control_row.empty:
                    control_share = float(control_row.iloc[0]["agreement_share"])
                if not model_row.empty:
                    model_share = float(model_row.iloc[0]["agreement_share"])
        rows.append(
            {
                "filial": filial_key,
                "agreement_share_control": control_share,
                "agreement_share_model": model_share,
                "agreement_gap_pp": gap,
                "negative_effect_horizons": ", ".join(negative) if negative else "",
                "min_effect_control_minus_model": (
                    min(effects.values()) if effects else np.nan
                ),
                "flags": ", ".join(flags),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "filial",
                "agreement_share_control",
                "agreement_share_model",
                "agreement_gap_pp",
                "negative_effect_horizons",
                "min_effect_control_minus_model",
                "flags",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["flags", "filial"])
        .reset_index(drop=True)
    )


def _sensitivity(
    frame: pd.DataFrame,
    priors: GroupRetroPriors,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for discount_rate in (0.08, 0.12, 0.16):
        for residual_share in (0.00, 0.07, 0.15):
            outcomes = _add_outcomes(
                frame,
                priors,
                residual_share=residual_share,
                discount_rate=discount_rate,
            )
            _, _, effects = _summaries(outcomes)
            for row in effects.to_dict("records"):
                rows.append(
                    {
                        "discount_rate": discount_rate,
                        "residual_share": residual_share,
                        "horizon": row["horizon"],
                        "effect_per_case": row["effect_per_case"],
                    }
                )
    return pd.DataFrame(rows)


def _seasonal_scaling(
    retro_df: pd.DataFrame,
    current: pd.DataFrame,
    priors: TerminalRetroPriors,
    effects: pd.DataFrame,
    compliance_effects: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    warnings: list[str] = []
    date_col = _resolve_retro_date(retro_df)
    filial_col = _required_existing(retro_df, RETRO_FILIAL_CANDIDATES, "FILIAL")
    if date_col is None:
        raise KeyError("Нет даты ретро для сезонной экстраполяции")
    dates = pd.to_datetime(retro_df[date_col], errors="coerce")
    pilot_norm = {value.casefold() for value in priors.pilot.filials}
    is_pilot = _normalize_text(retro_df[filial_col]).isin(pilot_norm)
    season_source = pd.DataFrame(
        {
            "date": dates,
            "is_pilot": is_pilot,
        }
    ).dropna(subset=["date"])
    pilot_source = season_source.loc[season_source["is_pilot"]].copy()
    pilot_source["year"] = pilot_source["date"].dt.year
    pilot_source["month"] = pilot_source["date"].dt.month
    complete_years = [
        int(year)
        for year, part in pilot_source.groupby("year")
        if part["month"].nunique() == 12
    ]
    if complete_years:
        pilot_source = pilot_source.loc[pilot_source["year"].isin(complete_years)]
    else:
        warnings.append(
            "Для сезонности нет полного ретро-года; использованы все доступные годы."
        )
    seasonal_counts = (
        pilot_source.groupby(["year", "month"])
        .size()
        .rename("n_month")
        .reset_index()
    )
    annual_counts = (
        pilot_source.groupby("year").size().rename("n_year").reset_index()
    )
    seasonal_counts = seasonal_counts.merge(annual_counts, on="year", how="left")
    seasonal_counts["share"] = (
        seasonal_counts["n_month"] / seasonal_counts["n_year"]
    )
    shares = seasonal_counts.groupby("month")["share"].mean()
    shares = shares.reindex(range(1, 13), fill_value=0.0)
    if shares.sum() <= 0:
        raise ValueError("Не удалось рассчитать сезонные доли")
    shares = shares / shares.sum()

    observed_dates = current["_application_date"].dropna()
    observed_start = observed_dates.min().normalize()
    observed_end = observed_dates.max().normalize()
    coverage = {month: 0.0 for month in range(1, 13)}
    for period in pd.period_range(observed_start, observed_end, freq="M"):
        month_start = max(observed_start, period.start_time.normalize())
        month_end = min(observed_end, period.end_time.normalize())
        observed_days = max(0, (month_end - month_start).days + 1)
        coverage[period.month] += observed_days / period.days_in_month
    seasonal_exposure = sum(
        coverage[month] * float(shares.loc[month]) for month in range(1, 13)
    )
    if seasonal_exposure <= 0:
        raise ValueError("Сезонная экспозиция равна нулю")

    n_observed = len(current)
    share_model = float(current["_group"].eq("model").mean())
    n_pilot_eligible_year = n_observed / seasonal_exposure
    n_model_year_current = share_model * n_pilot_eligible_year
    n_model_year_full = n_pilot_eligible_year

    volume_source = season_source.assign(year=season_source["date"].dt.year)
    if complete_years:
        volume_source = volume_source.loc[
            volume_source["year"].isin(complete_years)
        ]
    counts = volume_source.groupby(
        ["year", "is_pilot"]
    ).size().unstack(fill_value=0)
    pilot_annual = float(counts.get(True, pd.Series(dtype=float)).mean())
    nonpilot_annual = float(counts.get(False, pd.Series(dtype=float)).mean())
    volume_ratio = nonpilot_annual / pilot_annual if pilot_annual > 0 else 0.0
    amount_col = _required_existing(
        retro_df,
        ("TARGET_FREQ_AMOUNT",),
        "TARGET_FREQ_AMOUNT",
    )
    amount = _to_numeric(retro_df[amount_col]).fillna(0.0).clip(lower=0.0)
    valid_filial = retro_df[filial_col].notna()
    pilot_mean_psr = float(amount[is_pilot].mean())
    nonpilot_mean_psr = float(amount[~is_pilot & valid_filial].mean())
    risk_ratio = (
        nonpilot_mean_psr / pilot_mean_psr if pilot_mean_psr > 0 else 0.0
    )
    network_multiplier = 1.0 + volume_ratio * risk_ratio

    seasonal_rows = [
        {
            "month": month,
            "retro_share": float(shares.loc[month]),
            "observed_coverage": coverage[month],
            "exposure_contribution": coverage[month] * float(shares.loc[month]),
        }
        for month in range(1, 13)
    ]
    effect_idx = effects.set_index("horizon")
    effect_map = effect_idx["effect_per_case"].to_dict()
    ci_low_map = (
        effect_idx["ci_low"].to_dict()
        if "ci_low" in effect_idx.columns
        else {}
    )
    ci_high_map = (
        effect_idx["ci_high"].to_dict()
        if "ci_high" in effect_idx.columns
        else {}
    )
    compliance_map = compliance_effects.set_index("horizon")[
        "effect_per_case"
    ].to_dict()
    annual_rows = []
    for horizon in ("fact", "365", "1095"):
        effect = float(effect_map[horizon])
        effect_100 = float(compliance_map[horizon])
        scale_full = n_model_year_full * network_multiplier
        scale_pilot_full = n_model_year_full
        scale_pilot_current = n_model_year_current
        ci_low = float(ci_low_map.get(horizon, np.nan))
        ci_high = float(ci_high_map.get(horizon, np.nan))
        annual_rows.append(
            {
                "horizon": horizon,
                "effect_per_case": effect,
                "effect_100_compliance": effect_100,
                "seasonal_exposure": seasonal_exposure,
                "N_pilot_eligible_year": n_pilot_eligible_year,
                "N_pilot_model_year_current": n_model_year_current,
                "N_pilot_model_year_full": n_model_year_full,
                "volume_ratio_nonpilot": volume_ratio,
                "risk_ratio_nonpilot": risk_ratio,
                "network_multiplier": network_multiplier,
                "annual_pilot_current": effect * scale_pilot_current,
                "annual_pilot_full": effect * scale_pilot_full,
                "annual_pilot_full_ci_low": ci_low * scale_pilot_full,
                "annual_pilot_full_ci_high": ci_high * scale_pilot_full,
                "effect_per_case_nonpilot": effect * risk_ratio,
                "annual_network_full": effect * scale_full,
                "annual_network_full_ci_low": ci_low * scale_full,
                "annual_network_full_ci_high": ci_high * scale_full,
                "annual_network_full_compliance": (
                    effect_100 * scale_full
                ),
            }
        )
    return pd.DataFrame(annual_rows), pd.DataFrame(seasonal_rows), warnings


def _data_quality(
    frame: pd.DataFrame,
    contract: dict[str, str],
) -> pd.DataFrame:
    rows_per_loss = frame.groupby("_loss", dropna=False).size()
    duplicate_losses = rows_per_loss.gt(1)
    payment = frame["_paid_to_date"]
    unique_loss_payment = (
        frame[["_loss", "_paid_to_date"]]
        .drop_duplicates()["_paid_to_date"]
        .sum()
    )
    multiple_psr = (
        frame[["_psr_pretension", "_psr_fu", "_psr_court"]].gt(0).sum(axis=1) > 1
    )
    metrics = {
        "n_rows": len(frame),
        "n_unique_losses": frame["_loss"].nunique(dropna=False),
        "losses_with_multiple_rows": int(duplicate_losses.sum()),
        "extra_rows_vs_one_row_per_loss": int(
            (rows_per_loss - 1).clip(lower=0).sum()
        ),
        "exact_duplicate_rows": int(frame.duplicated(keep=False).sum()),
        "payment_raw_sum": float(payment.sum()),
        "payment_possible_inflation": float(payment.sum() - unique_loss_payment),
        "payment_minus_to_pay_sum": float(
            (frame["_paid_to_date"] - frame["_to_pay"]).sum()
        ),
        "payment_to_pay_mismatch_rows": int(
            frame["_paid_to_date"].ne(frame["_to_pay"]).sum()
        ),
        "missing_payment": int(frame["_paid_missing"].sum()),
        "missing_od": int(frame["_od_missing"].sum()),
        "rows_with_multiple_psr_components": int(multiple_psr.sum()),
        "observed_psr_gt_paid": int(
            frame["_observed_psr"].gt(frame["_paid_to_date"]).sum()
        ),
        "max_age_days": int(frame["_age_days"].max()),
        "observation_start": frame["_application_date"].min().date().isoformat(),
        "observation_end": frame["_application_date"].max().date().isoformat(),
        "n_model": int(frame["_group"].eq("model").sum()),
        "n_control": int(frame["_group"].eq("control").sum()),
        "n_filials": frame["_filial"].nunique(dropna=False),
    }
    rows = [{"metric": key, "value": value} for key, value in metrics.items()]
    rows.extend(
        {
            "metric": f"column:{key}",
            "value": value,
        }
        for key, value in contract.items()
    )
    return pd.DataFrame(rows)


def estimate_monitoring_effect(
    monitoring_df: pd.DataFrame,
    retro_df: pd.DataFrame,
    *,
    t_calc: str | pd.Timestamp | None = None,
    residual_share: float = 0.07,
    discount_rate: float = 0.12,
    lookback_years: float | None = 2.0,
    retro_as_of: str | pd.Timestamp | None = RETRO_AS_OF_DEFAULT,
    bootstrap_iterations: int = 1000,
    bootstrap_compliance_iterations: int = 200,
    bootstrap_seed: int = 42,
) -> MonitoringEffectResult:
    """Выполнить ITT, compliance-сценарии, CI и сезонную экстраполяцию."""
    if not 0 <= residual_share <= 1:
        raise ValueError("residual_share должен быть в [0, 1]")
    if discount_rate <= -1:
        raise ValueError("discount_rate должен быть больше -1")

    current, contract, warnings = _prepare_monitoring_contract(
        monitoring_df,
        t_calc=t_calc,
    )
    pilot_filials = PILOT_FILIALS
    priors = compute_terminal_priors(
        retro_df,
        pilot_filials=pilot_filials,
        lookback_years=lookback_years,
        as_of=retro_as_of,
    )
    current = _add_outcomes(
        current,
        priors.pilot,
        residual_share=residual_share,
        discount_rate=discount_rate,
    )
    group_summary, filial_effects, effects = _summaries(current)

    ci = _bootstrap_ci(
        current,
        retro_df,
        pilot_filials=pilot_filials,
        lookback_years=lookback_years,
        as_of=retro_as_of,
        residual_share=residual_share,
        discount_rate=discount_rate,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
        progress_desc="bootstrap ITT",
    )
    effects = effects.merge(ci, on="horizon", how="left")

    compliance_a = _compliance_a(current)
    current_100 = _add_outcomes(
        current,
        priors.pilot,
        residual_share=residual_share,
        discount_rate=discount_rate,
        compliance_100=True,
    )
    _, _, compliance_b = _summaries(current_100, suffix="_100")
    compliance_b["scenario"] = "100% compliance for model result=1"
    compliance_ci = _bootstrap_ci(
        current,
        retro_df,
        pilot_filials=pilot_filials,
        lookback_years=lookback_years,
        as_of=retro_as_of,
        residual_share=residual_share,
        discount_rate=discount_rate,
        iterations=bootstrap_compliance_iterations,
        seed=bootstrap_seed + 1,
        compliance_100=True,
        progress_desc="bootstrap compliance-100",
    )
    compliance_b = compliance_b.merge(compliance_ci, on="horizon", how="left")

    sensitivity = _sensitivity(current, priors.pilot)
    annual, seasonality, season_warnings = _seasonal_scaling(
        retro_df,
        current,
        priors,
        effects,
        compliance_b,
    )
    warnings.extend(season_warnings)
    for group_priors in (priors.pilot, priors.nonpilot):
        if group_priors.n_positive < 30:
            warnings.append(
                f"В ретро-группе {group_priors.group} только "
                f"{group_priors.n_positive} положительных ПСР (<30)."
            )
    warnings.append(
        "Масштабирование на сеть является сценарным, а не экспериментальным."
    )
    missing_strata = set(current["_filial"].astype(str)) - set(
        filial_effects.dropna(subset=["effect_control_minus_model"])["filial"].astype(str)
    )
    if missing_strata:
        warnings.append(
            "Без пары model/control и исключены из ITT-взвешивания филиалы: "
            + ", ".join(sorted(missing_strata))
        )
    data_quality = _data_quality(current, contract)
    quality = data_quality.set_index("metric")["value"].to_dict()
    if int(quality["observed_psr_gt_paid"]) > 0:
        warnings.append(
            "Есть строки, где observed_PSR больше СуммаПлатежа; "
            "проверьте состав кассового факта."
        )
    if int(quality["losses_with_multiple_rows"]) > 0:
        warnings.append(
            "Есть повторяющиеся номера убытков. Они не удалены согласно контракту."
        )
    if int(quality["payment_to_pay_mismatch_rows"]) > 0:
        warnings.append(
            "СуммаПлатежа отличается от СуммаКВыплате хотя бы в одной строке; "
            "в outcome используется только СуммаПлатежа."
        )

    path_shares = shares_as_percent(
        compare_path_shares(monitoring_df, filial_scope="pilot", variant=1)
    )
    filial_paths = shares_as_percent(
        filial_path_shares(monitoring_df, filial_scope="pilot", variant=1)
    )
    attention = _attention_filials(filial_paths, filial_effects)
    if not attention.empty:
        warnings.append(
            "Филиалы на внимании (agreement control>model и/или отрицательный ITT): "
            + ", ".join(attention["filial"].astype(str).tolist())
        )

    calc_date = (
        pd.Timestamp(t_calc).normalize()
        if t_calc is not None
        else pd.Timestamp.today().normalize()
    )
    return MonitoringEffectResult(
        frame=current_100,
        priors=priors,
        effect_summary=effects,
        group_summary=group_summary,
        filial_effects=filial_effects,
        compliance_a=compliance_a,
        compliance_b=compliance_b,
        sensitivity=sensitivity,
        annual_summary=annual,
        seasonality=seasonality,
        data_quality=data_quality,
        path_shares=path_shares,
        filial_path_shares=filial_paths,
        attention_filials=attention,
        contract=contract,
        t_calc=calc_date,
        discount_rate=discount_rate,
        residual_share=residual_share,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_compliance_iterations=bootstrap_compliance_iterations,
        warnings=warnings,
    )


def build_synthetic_claims_excel(
    n_rows: int = 300,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Сохранить совместимость synthetic-источника загрузчика мониторинга."""
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
            "СуммаОсновногоДолгаЗаявлено": rng.uniform(
                20_000,
                300_000,
                n_rows,
            ),
            "Сумма рекомендованная к доплате по модулю": np.where(
                result == 1,
                rng.uniform(10_000, 80_000, n_rows),
                0.0,
            ),
            "Выплата по модели": payout.astype(int),
            "Заключено соглашение": agreement.astype(int),
            "Дата вызова модели сутяжности": today - pd.to_timedelta(
                rng.integers(0, 120, n_rows),
                unit="D",
            ),
            "ДатаЗаявления": today - pd.to_timedelta(
                rng.integers(0, 120, n_rows),
                unit="D",
            ),
            "Cумма выплаты по претензии": psr,
            "Сумма выплат по ФУ": 0.0,
            "Сумма выплаты по суду": 0.0,
        }
    )
    frame["СуммаКВыплате"] = frame["СуммаПлатежа"]
    return frame


__all__ = [
    "build_synthetic_claims_excel",
    "COURT_FEE_DEFAULT",
    "FU_FEE_DEFAULT",
    "GroupRetroPriors",
    "HORIZONS",
    "MonitoringEffectResult",
    "PILOT_FILIALS",
    "RETRO_AS_OF_DEFAULT",
    "RESULT_OUT_OF_MODEL",
    "TerminalRetroPriors",
    "VITRINA_TABLE_DEFAULT",
    "compute_terminal_priors",
    "estimate_monitoring_effect",
    "filter_retro_lookback",
    "format_money",
    "load_monitoring_frame",
]
