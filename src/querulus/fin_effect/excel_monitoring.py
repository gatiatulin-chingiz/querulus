"""Единая ITT-оценка финансового эффекта по витрине мониторинга."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from querulus.fin_effect.excel_explore import (
    ALLOWED_REFUND_FORM_NEEDLES,
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
RESULT_ELIGIBLE = (0, 1, RESULT_OUT_OF_MODEL)
WRITEOFF_REFUND_NEEDLES = ("списан",)
# NPV-окно для lifetime-хвоста ПСР (калибровка ultimate + дисконт до ~3 лет)
ULTIMATE_HORIZON_DAYS = 1095
HORIZON_SPECS: tuple[tuple[str, int], ...] = (("ult", ULTIMATE_HORIZON_DAYS),)
HORIZONS = tuple(days for _, days in HORIZON_SPECS)
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
    recommended_extra_summary: pd.DataFrame
    payment_descriptives: pd.DataFrame
    sample_size_guidance: pd.DataFrame
    contract: dict[str, str]
    t_calc: pd.Timestamp
    discount_rate: float
    residual_share: float
    bootstrap_iterations: int
    bootstrap_compliance_iterations: int
    bootstrap_samples: pd.DataFrame = field(default_factory=pd.DataFrame)
    bootstrap_compliance_samples: pd.DataFrame = field(default_factory=pd.DataFrame)
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


def _series_mode(series: pd.Series) -> Any:
    """Мода; при ничьей — первое из самых частых значений."""
    clean = series.dropna()
    if clean.empty:
        return np.nan
    if pd.api.types.is_string_dtype(clean) or clean.dtype == object:
        text = clean.astype("string").str.strip()
        text = text[text.notna() & text.ne("") & text.str.casefold().ne("nan")]
        if text.empty:
            return np.nan
        return text.value_counts().index[0]
    return clean.value_counts().index[0]


def _collapse_result_check(series: pd.Series) -> float:
    """РезультатПроверки на инцидент: Null игнорируем, непустой 0/1/−100 размазываем.

    Если в инциденте есть −100 и Null → −100 (control на весь инцидент).
    Аналогично для 0/1. При конфликте нескольких непустых — мода,
    ничья: −100, затем 1, затем 0.
    """
    num = pd.to_numeric(series, errors="coerce")
    valid = num[num.isin(RESULT_ELIGIBLE)]
    if valid.empty:
        return np.nan
    counts = valid.value_counts()
    top = counts[counts == counts.max()].index.astype(float).tolist()
    for preferred in (float(RESULT_OUT_OF_MODEL), 1.0, 0.0):
        if preferred in top:
            return preferred
    return float(top[0])


def _is_writeoff_refund(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.casefold()
    mask = pd.Series(False, index=series.index)
    for needle in WRITEOFF_REFUND_NEEDLES:
        mask = mask | text.str.contains(needle, na=False)
    return mask


def _is_allowed_refund(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.casefold()
    mask = pd.Series(False, index=series.index)
    for needle in ALLOWED_REFUND_FORM_NEEDLES:
        mask = mask | text.str.contains(needle, na=False)
    return mask


def _aggregate_row_group(
    group: pd.DataFrame,
    *,
    result_col: str,
    incident_col: str,
    loss_col: str,
) -> pd.Series:
    """Схлопывание строк: текст/категории — мода, числа — сумма, даты — min.

    Номера инцидента/убытка — идентификаторы: не сумма и не «взять один» наугад.
    Инцидент = ключ группы; убытки = список уникальных номеров через «; ».
    """
    out: dict[str, Any] = {}
    id_cols = {incident_col, loss_col}
    for column in group.columns:
        series = group[column]
        if column == incident_col:
            vals = series.dropna().astype("string").str.strip()
            vals = vals[vals.ne("") & vals.str.casefold().ne("nan")]
            out[column] = vals.iloc[0] if not vals.empty else np.nan
            continue
        if column == loss_col:
            vals = (
                series.dropna()
                .astype("string")
                .str.strip()
            )
            vals = vals[vals.ne("") & vals.str.casefold().ne("nan")]
            uniq = list(dict.fromkeys(vals.tolist()))
            out[column] = "; ".join(uniq) if uniq else np.nan
            continue
        if column == result_col:
            out[column] = _collapse_result_check(series)
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            out[column] = series.min()
            continue
        if pd.api.types.is_bool_dtype(series):
            out[column] = bool(series.fillna(False).any())
            continue
        if pd.api.types.is_numeric_dtype(series):
            out[column] = series.sum(min_count=1)
            continue
        # object/string и смешанные: пробуем числовую сумму, иначе мода
        # (идентификаторы уже обработаны выше)
        as_num = pd.to_numeric(series, errors="coerce")
        if as_num.notna().any() and as_num.isna().sum() <= series.isna().sum():
            non_null_raw = series.dropna()
            if not non_null_raw.empty and as_num.notna().mean() >= 0.8:
                out[column] = as_num.sum(min_count=1)
                continue
        out[column] = _series_mode(series)
    out["_n_rows_collapsed"] = len(group)
    out["_loss_ids"] = out.get(loss_col, np.nan)
    _ = id_cols
    return pd.Series(out)


def dedupe_monitoring_by_loss(
    df: pd.DataFrame,
    *,
    loss_col: str,
    result_col: str,
) -> tuple[pd.DataFrame, int]:
    """Одна строка на номер убытка (keep first; без суммирования дублей)."""
    _ = result_col  # совместимость сигнатуры / будущие правила конфликта
    if loss_col not in df.columns:
        raise KeyError(f"Нет колонки убытка: {loss_col}")
    n_before = len(df)
    out = df.drop_duplicates(subset=[loss_col], keep="first").copy()
    out = out.reset_index(drop=True)
    out["_n_rows_collapsed"] = 1
    return out, n_before - len(out)


def collapse_monitoring_to_incident(
    df: pd.DataFrame,
    *,
    incident_col: str,
    loss_col: str,
    result_col: str,
    refund_col: str | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Схлопнуть убытки в инцидент.

    Списанные по форме возмещения не входят в схлопывание (исключаются).
    Остальные: ResultПроверки размазывается с непустых 0/1/−100; числа — sum,
    текст — mode.
    """
    work = df.copy()
    stats = {
        "n_before_incident_collapse": len(work),
        "n_writeoff_excluded": 0,
        "n_incidents_after_collapse": 0,
        "n_multi_loss_incidents": 0,
    }
    if refund_col is not None and refund_col in work.columns:
        writeoff = _is_writeoff_refund(work[refund_col])
        stats["n_writeoff_excluded"] = int(writeoff.sum())
        # списанные не схлопываем и не оставляем в ITT-популяции
        work = work.loc[~writeoff].copy()
        # на всякий случай оставляем только разрешённые формы
        allowed = _is_allowed_refund(work[refund_col])
        work = work.loc[allowed].copy()

    if work.empty:
        return work, stats

    multi = work.groupby(incident_col, dropna=False)[loss_col].transform(
        "nunique"
    )
    stats["n_multi_loss_incidents"] = int(
        work.loc[multi.gt(1), incident_col].nunique(dropna=False)
    )
    parts = [
        _aggregate_row_group(
            part,
            result_col=result_col,
            incident_col=incident_col,
            loss_col=loss_col,
        )
        for incident_key, part in work.groupby(incident_col, dropna=False, sort=False)
    ]
    out = pd.DataFrame(parts).reset_index(drop=True)
    # ключ группы надёжнее любой агрегации колонки
    if not out.empty:
        keys = [
            key
            for key, _ in work.groupby(incident_col, dropna=False, sort=False)
        ]
        out[incident_col] = keys
    stats["n_incidents_after_collapse"] = len(out)
    return out, stats


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
        return df, meta

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


def _resolve_n_jobs(n_jobs: int | None) -> int:
    import os

    cpu = os.cpu_count() or 2
    if n_jobs is None or n_jobs < 0:
        return max(1, cpu - 1)
    return max(1, int(n_jobs))


_BOOTSTRAP_WORKER: dict[str, Any] = {}


def _bootstrap_worker_init(payload: dict[str, Any]) -> None:
    """Инициализация процесса/потока: общие данные без pickle на каждую итерацию."""
    _BOOTSTRAP_WORKER.clear()
    _BOOTSTRAP_WORKER.update(payload)
    current = payload["current"]
    _BOOTSTRAP_WORKER["incident_groups"] = [
        part for _, part in current.groupby("_incident", dropna=False)
    ]


def _pilot_priors_fast(
    work: pd.DataFrame,
    *,
    pilot_norm: set[str],
    filial_col: str,
    amount_col: str,
    od_col: str,
    fu_col: str,
    court_col: str,
    pilot_names: tuple[str, ...],
) -> GroupRetroPriors:
    is_pilot = _normalize_text(work[filial_col]).isin(pilot_norm)
    pilot_df = work.loc[is_pilot]
    return _group_retro_priors(
        pilot_df,
        group="pilot",
        filials=pilot_names,
        amount_col=amount_col,
        od_col=od_col,
        fu_col=fu_col,
        court_col=court_col,
    )


def _bootstrap_one_seed(seed: int) -> dict[str, float] | None:
    """Одна bootstrap-итерация; использует `_BOOTSTRAP_WORKER`."""
    ctx = _BOOTSTRAP_WORKER
    rng = np.random.default_rng(seed)
    retro = ctx["retro"]
    groups: list[pd.DataFrame] = ctx["incident_groups"]
    try:
        retro_sample = retro.iloc[rng.integers(0, len(retro), size=len(retro))]
        pilot = _pilot_priors_fast(
            retro_sample,
            pilot_norm=ctx["pilot_norm"],
            filial_col=ctx["filial_col"],
            amount_col=ctx["amount_col"],
            od_col=ctx["od_col"],
            fu_col=ctx["fu_col"],
            court_col=ctx["court_col"],
            pilot_names=ctx["pilot_names"],
        )
        picks = rng.integers(0, len(groups), size=len(groups))
        current_sample = pd.concat([groups[index] for index in picks], ignore_index=True)
        outcomes = _add_outcomes(
            current_sample,
            pilot,
            residual_share=ctx["residual_share"],
            discount_rate=ctx["discount_rate"],
            compliance_100=ctx["compliance_100"],
        )
        _, _, effects = _summaries(outcomes, suffix=ctx["suffix"])
    except (KeyError, ValueError, ZeroDivisionError):
        return None
    return {
        str(row["horizon"]): float(row["effect_per_case"])
        for row in effects.to_dict("records")
    }


def _cluster_resample(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    *,
    incident_groups: list[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    groups = incident_groups or [
        part for _, part in frame.groupby("_incident", dropna=False)
    ]
    picks = rng.integers(0, len(groups), size=len(groups))
    return pd.concat([groups[index] for index in picks], ignore_index=True)


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
    n_jobs: int | None = -1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    empty_ci = pd.DataFrame(columns=["horizon", "ci_low", "ci_high", "n_bootstrap"])
    empty_samples = pd.DataFrame(columns=["horizon", "effect_per_case"])
    if iterations <= 0:
        return empty_ci, empty_samples

    # Ретро уже режем один раз: внутри bootstrap не повторяем lookback.
    retro_window, _ = filter_retro_lookback(
        retro,
        lookback_years=lookback_years,
        as_of=as_of,
    )
    filial_col = _required_existing(retro_window, RETRO_FILIAL_CANDIDATES, "FILIAL")
    amount_col = _required_existing(
        retro_window, ("TARGET_FREQ_AMOUNT",), "TARGET_FREQ_AMOUNT"
    )
    od_col = _required_existing(
        retro_window,
        ("RECOVEREDMAINDEBT_LAST_INST_SUM",),
        "RECOVEREDMAINDEBT_LAST_INST_SUM",
    )
    fu_col = _required_existing(
        retro_window, ("Сумма_взыскано_по_ФУ",), "Сумма_взыскано_по_ФУ"
    )
    court_col = _required_existing(
        retro_window, ("Суммы_взыскано_по_иску",), "Суммы_взыскано_по_иску"
    )
    pilot_names = tuple(sorted({str(value).strip() for value in pilot_filials}))
    payload = {
        "current": current,
        "retro": retro_window,
        "residual_share": residual_share,
        "discount_rate": discount_rate,
        "compliance_100": compliance_100,
        "suffix": "_100" if compliance_100 else "",
        "pilot_names": pilot_names,
        "pilot_norm": {value.casefold() for value in pilot_names},
        "filial_col": filial_col,
        "amount_col": amount_col,
        "od_col": od_col,
        "fu_col": fu_col,
        "court_col": court_col,
    }
    seeds = [int(seed) + i for i in range(iterations)]
    workers = _resolve_n_jobs(n_jobs)
    results: list[dict[str, float] | None]

    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None  # type: ignore[assignment]

    if workers == 1:
        _bootstrap_worker_init(payload)
        iterator: Any = seeds
        if tqdm is not None:
            iterator = tqdm(seeds, total=iterations, desc=progress_desc, leave=True)
        results = [_bootstrap_one_seed(item) for item in iterator]
    else:
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

        def _collect(executor_cls: Any, init: bool) -> list[dict[str, float] | None]:
            collected: list[dict[str, float] | None] = []
            kwargs: dict[str, Any] = {"max_workers": workers}
            if init:
                kwargs["initializer"] = _bootstrap_worker_init
                kwargs["initargs"] = (payload,)
            else:
                _bootstrap_worker_init(payload)
            with executor_cls(**kwargs) as pool:
                futures = [pool.submit(_bootstrap_one_seed, item) for item in seeds]
                done = as_completed(futures)
                if tqdm is not None:
                    done = tqdm(
                        done,
                        total=iterations,
                        desc=f"{progress_desc}×{workers}",
                        leave=True,
                    )
                for future in done:
                    collected.append(future.result())
            return collected

        try:
            # Процессы лучше для pandas (GIL); данные в initializer один раз на worker.
            results = _collect(ProcessPoolExecutor, init=True)
        except Exception:
            results = _collect(ThreadPoolExecutor, init=False)

    values: dict[str, list[float]] = {"ult": []}
    for item in results:
        if not item:
            continue
        for horizon, value in item.items():
            values.setdefault(horizon, []).append(float(value))

    rows = []
    sample_rows: list[dict[str, Any]] = []
    for horizon, sample in values.items():
        if sample:
            low, high = np.quantile(sample, [0.025, 0.975])
            for value in sample:
                sample_rows.append(
                    {"horizon": horizon, "effect_per_case": float(value)}
                )
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
    return pd.DataFrame(rows), pd.DataFrame(sample_rows)


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
    other_costs_col = resolve_column(df, "other_costs")
    loss_col = _required_existing(df, LOSS_CANDIDATES, "номера убытка")
    incident_col = _required_existing(df, INCIDENT_CANDIDATES, "номера инцидента")
    call_date_col = _required_alias(
        df,
        "model_call_date",
        "Дата вызова модели сутяжности",
    )
    application_col = _required_alias(df, "application_date", "ДатаЗаявления")
    refund_col = resolve_column(df, "refund_form")
    psr_columns = {
        key: _required_existing(df, candidates, f"ПСР: {key}")
        for key, candidates in PSR_AMOUNT_CANDIDATES.items()
    }
    warnings: list[str] = []

    # 1) дедуп по номеру убытка
    work, n_dup_rows = dedupe_monitoring_by_loss(
        df, loss_col=loss_col, result_col=result_col
    )
    if n_dup_rows:
        warnings.append(
            f"Дедупликация по убытку: убрано/схлопнуто {n_dup_rows} лишних строк."
        )

    # даты до фильтров — иначе groupby может оставить object
    work[call_date_col] = pd.to_datetime(work[call_date_col], errors="coerce")
    work[application_col] = pd.to_datetime(work[application_col], errors="coerce")

    # 2) базовые фильтры без Result (Null оставляем — размажется с соседей по инциденту)
    base = analytics_base_mask(work, filial_scope="pilot")
    work = work.loc[base].copy()
    if work.empty:
        raise ValueError("После базовых фильтров нет строк")

    # 3) схлоп на инцидент; списанные по форме возмещения не входят
    work, collapse_stats = collapse_monitoring_to_incident(
        work,
        incident_col=incident_col,
        loss_col=loss_col,
        result_col=result_col,
        refund_col=refund_col,
    )
    if collapse_stats["n_writeoff_excluded"]:
        warnings.append(
            "Списанные по форме возмещения не схлопывались и исключены: "
            f"{collapse_stats['n_writeoff_excluded']} убытков."
        )
    if collapse_stats["n_multi_loss_incidents"]:
        warnings.append(
            "Инцидентов с >1 убытком до схлопывания: "
            f"{collapse_stats['n_multi_loss_incidents']}."
        )
    if work.empty:
        raise ValueError("После схлопывания на инцидент нет строк")

    # 4) только model/control после размазывания Result
    work["_result"] = _to_numeric(work[result_col])
    eligible = work["_result"].isin(list(RESULT_ELIGIBLE))
    n_dropped_null_result = int((~eligible).sum())
    work = work.loc[eligible].copy()
    if n_dropped_null_result:
        warnings.append(
            "Инцидентов без Result∈{0,1,−100} после схлопывания (все Null/прочее): "
            f"{n_dropped_null_result}."
        )
    if work.empty:
        raise ValueError("После model/control нет строк (нет Result 0/1/−100)")

    work["_group"] = np.where(
        work["_result"].eq(RESULT_OUT_OF_MODEL),
        "control",
        "model",
    )
    work["_filial"] = work[filial_col].astype("string").fillna("(пусто)")
    # единица анализа — инцидент; _loss хранит список LossID через «; »
    work["_incident"] = work[incident_col].astype("string")
    work["_loss"] = work[loss_col].astype("string")
    work["_n_losses_collapsed"] = (
        _to_numeric(work["_n_rows_collapsed"]).fillna(1).astype(int)
        if "_n_rows_collapsed" in work.columns
        else pd.Series(1, index=work.index, dtype=int)
    )
    work["_paid_missing"] = _to_numeric(work[payment_col]).isna()
    work["_od_missing"] = _to_numeric(work[od_col]).isna()
    work["_paid_to_date"] = _to_numeric(work[payment_col]).fillna(0.0).clip(lower=0.0)
    work["_to_pay"] = _to_numeric(work[to_pay_col]).fillna(0.0).clip(lower=0.0)
    work["_od"] = _to_numeric(work[od_col])
    work["_recommended_extra"] = (
        _to_numeric(work[recommended_col]).fillna(0.0).clip(lower=0.0)
    )
    work["_payout_by_model"] = _to_numeric(work[payout_col]).fillna(0.0).gt(0)
    # «Иные затраты» = сумма доплаты по модели (флаг «Выплата по модели» — 0/1)
    if other_costs_col is not None:
        work["_model_payout_amount"] = (
            _to_numeric(work[other_costs_col]).fillna(0.0).clip(lower=0.0)
        )
    else:
        work["_model_payout_amount"] = 0.0
        warnings.append(
            "Нет колонки «Иные затраты» — Σ выплаты по модели в описании = 0."
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
    if work["_age_days"].lt(0).any():
        warnings.append("Есть t0 позже t_calc; age_days для них ограничен нулём.")
        work["_age_days"] = work["_age_days"].clip(lower=0)
    if work["_age_days"].ge(ULTIMATE_HORIZON_DAYS).any():
        n_old = int(work["_age_days"].ge(ULTIMATE_HORIZON_DAYS).sum())
        warnings.append(
            f"{n_old} строк имеют age_days >= {ULTIMATE_HORIZON_DAYS}: "
            "для них midpoint NPV хвоста = 0 (Yult = paid + remaining)."
        )

    contract = {
        "unit": "incident_after_loss_dedupe",
        "loss": loss_col,
        "incident": incident_col,
        "result": result_col,
        "filial": filial_col,
        "payment": payment_col,
        "to_pay_diagnostic_only": to_pay_col,
        "od": od_col,
        "recommended_extra": recommended_col,
        "payout_by_model": payout_col,
        "model_payout_amount": other_costs_col or "Иные затраты",
        "agreement": resolve_column(df, "agreement") or "agreement_mask",
        "t0_primary": call_date_col,
        "t0_fallback": application_col,
        "psr_pretension": psr_columns["pretension"],
        "psr_fu": psr_columns["fu"],
        "psr_court": psr_columns["court"],
        "n_rows_after_loss_dedupe_removed": str(n_dup_rows),
        "n_writeoff_excluded": str(collapse_stats["n_writeoff_excluded"]),
        "n_multi_loss_incidents": str(collapse_stats["n_multi_loss_incidents"]),
        "n_incidents_after_collapse": str(
            collapse_stats["n_incidents_after_collapse"]
        ),
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
        work["_paid_base"] = work["_paid_to_date"] + work["_forced_extra"]
        residual = pd.Series(1.0, index=work.index)
        residual.loc[work["_agreement"]] = residual_share
        residual.loc[model_one] = residual_share
        prefix = "_100"
    else:
        work["_forced_extra"] = 0.0
        work["_paid_base"] = work["_paid_to_date"]
        residual = pd.Series(1.0, index=work.index)
        residual.loc[work["_agreement"]] = residual_share
        prefix = ""

    remaining = (
        residual * work["_expected_open_psr"] - work["_observed_psr"]
    ).clip(lower=0.0)
    work[f"_remaining_nominal{prefix}"] = remaining
    for name, horizon_days in HORIZON_SPECS:
        remaining_days = (horizon_days - work["_age_days"]).clip(lower=0.0)
        midpoint_days = remaining_days / 2.0
        discount_factor = (1.0 + discount_rate) ** (midpoint_days / 365.0)
        column = f"Y{name}{prefix}"
        work[column] = work["_paid_base"] + remaining / discount_factor
        work[f"_midpoint_days_{name}{prefix}"] = midpoint_days
    return work


def _summaries(
    frame: pd.DataFrame,
    *,
    suffix: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes = {
        "ult": f"Yult{suffix}",
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
                "horizon": "ult",
                "compliance": status,
                "n": n,
                "agreement_share": float(part["_agreement"].mean() * 100) if n else np.nan,
                "mean_cost": float(part["Yult"].mean()),
                "descriptive_only": True,
            }
        )
    return pd.DataFrame(rows)


def _recommended_extra_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Объём доплат: Σ «Иные затраты» (выплата по модели) и рекомендация.

    Описательная таблица: не ITT и не экономия от доплат.
    """
    segments: list[tuple[str, pd.Series]] = [
        ("control", frame["_group"].eq("control")),
        ("model", frame["_group"].eq("model")),
        (
            "model_result_1",
            frame["_group"].eq("model") & frame["_result"].eq(1),
        ),
        (
            "model_result_0",
            frame["_group"].eq("model") & frame["_result"].eq(0),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, mask in segments:
        part = frame.loc[mask]
        n = len(part)
        if n == 0:
            rows.append(
                {
                    "segment": name,
                    "n": 0,
                    "n_recommended_gt0": 0,
                    "sum_recommended_extra": 0.0,
                    "sum_model_payout": 0.0,
                    "n_paid_gt0": 0,
                    "share_paid_gt0": np.nan,
                    "n_recommended_and_paid": 0,
                    "sum_recommended_paid": 0.0,
                    "sum_recommended_unpaid": 0.0,
                    "descriptive_only": True,
                }
            )
            continue
        rec = part["_recommended_extra"]
        model_pay = part["_model_payout_amount"]
        paid_flag = part["_paid_to_date"].gt(0)
        rec_flag = rec.gt(0)
        rows.append(
            {
                "segment": name,
                "n": n,
                "n_recommended_gt0": int(rec_flag.sum()),
                "sum_recommended_extra": float(rec.sum()),
                "sum_model_payout": float(model_pay.sum()),
                "n_paid_gt0": int(paid_flag.sum()),
                "share_paid_gt0": float(paid_flag.mean() * 100),
                "n_recommended_and_paid": int((rec_flag & paid_flag).sum()),
                "sum_recommended_paid": float(rec.loc[paid_flag].sum()),
                "sum_recommended_unpaid": float(rec.loc[~paid_flag].sum()),
                "descriptive_only": True,
            }
        )
    return pd.DataFrame(rows)


def _payment_descriptives(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean/median СуммаПлатежа по группам и Иные затраты при выплате по модели=1."""
    rows: list[dict[str, Any]] = []
    for group in ("control", "model"):
        part = frame.loc[frame["_group"].eq(group)]
        n = len(part)
        paid = part["_paid_to_date"] if n else pd.Series(dtype=float)
        rows.append(
            {
                "metric": "СуммаПлатежа",
                "segment": group,
                "filter": "all",
                "n": n,
                "mean": float(paid.mean()) if n else np.nan,
                "median": float(paid.median()) if n else np.nan,
            }
        )

    paid_by_model = frame.loc[frame["_payout_by_model"]]
    for group in ("control", "model", "all"):
        if group == "all":
            part = paid_by_model
        else:
            part = paid_by_model.loc[paid_by_model["_group"].eq(group)]
        n = len(part)
        extra = part["_model_payout_amount"] if n else pd.Series(dtype=float)
        rows.append(
            {
                "metric": "Иные затраты",
                "segment": group,
                "filter": "Выплата по модели = 1",
                "n": n,
                "mean": float(extra.mean()) if n else np.nan,
                "median": float(extra.median()) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _sample_size_guidance(effects: pd.DataFrame) -> pd.DataFrame:
    """Во сколько раз вырастить N, чтобы 95% CI ITT не содержал 0.

    Приближение: полуширина CI сжимается как 1/√k при росте выборки в k раз
    (та же доля model/control и тот же эффект на инцидент).
    """
    if effects.empty or "ult" not in set(effects["horizon"].astype(str)):
        return pd.DataFrame()
    row = effects.loc[effects["horizon"].astype(str).eq("ult")].iloc[0]
    effect = float(row["effect_per_case"])
    ci_low = float(row["ci_low"]) if pd.notna(row.get("ci_low")) else np.nan
    ci_high = float(row["ci_high"]) if pd.notna(row.get("ci_high")) else np.nan
    n_now = float(row["n_weighted"]) if pd.notna(row.get("n_weighted")) else np.nan
    half = (
        (ci_high - ci_low) / 2.0
        if pd.notna(ci_low) and pd.notna(ci_high)
        else np.nan
    )
    already = bool(pd.notna(ci_low) and ci_low > 0)
    if already:
        multiplier = 1.0
        note = "95% CI уже выше 0 — текущего N достаточно по этому критерию."
    elif not pd.notna(effect) or effect <= 0:
        multiplier = np.nan
        note = (
            "Точечный ITT ≤ 0: рост N сам по себе не сделает CI выше 0 "
            "(нужен устойчивый положительный эффект)."
        )
    elif not pd.notna(half) or half <= 0:
        multiplier = np.nan
        note = "Нет валидного bootstrap CI — оценку роста N посчитать нельзя."
    else:
        # нужно effect − half/√k > 0 → k > (half/effect)^2
        multiplier = float((half / effect) ** 2)
        note = (
            "При том же ITT и доле model/control: N_need = N * k, "
            "чтобы нижняя граница 95% CI стала > 0 (приближение 1/sqrt(N))."
        )
    n_need = (
        float(n_now * multiplier)
        if pd.notna(n_now) and pd.notna(multiplier)
        else np.nan
    )
    return pd.DataFrame(
        [
            {
                "horizon": "ult",
                "n_current": n_now,
                "effect_per_case": effect,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_halfwidth": half,
                "already_excludes_0": already,
                "n_multiplier": multiplier,
                "n_needed": n_need,
                "note": note,
            }
        ]
    )


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
    for horizon in ("ult",):
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
                "annual_window": effect * n_observed,
                "annual_window_ci_low": ci_low * n_observed,
                "annual_window_ci_high": ci_high * n_observed,
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
        "n_unique_incidents": frame["_incident"].nunique(dropna=False),
        "unit_is_incident": True,
        "mean_losses_per_incident": float(
            frame["_n_losses_collapsed"].mean()
        )
        if "_n_losses_collapsed" in frame.columns
        else 1.0,
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
    bootstrap_n_jobs: int | None = -1,
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

    ci, bootstrap_samples = _bootstrap_ci(
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
        n_jobs=bootstrap_n_jobs,
    )
    effects = effects.merge(ci, on="horizon", how="left")

    compliance_a = _compliance_a(current)
    recommended_extra_summary = _recommended_extra_summary(current)
    payment_descriptives = _payment_descriptives(current)
    sample_size_guidance = _sample_size_guidance(effects)
    current_100 = _add_outcomes(
        current,
        priors.pilot,
        residual_share=residual_share,
        discount_rate=discount_rate,
        compliance_100=True,
    )
    _, _, compliance_b = _summaries(current_100, suffix="_100")
    compliance_b["scenario"] = "100% compliance for model result=1"
    compliance_ci, bootstrap_compliance_samples = _bootstrap_ci(
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
        n_jobs=bootstrap_n_jobs,
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
        recommended_extra_summary=recommended_extra_summary,
        payment_descriptives=payment_descriptives,
        sample_size_guidance=sample_size_guidance,
        contract=contract,
        t_calc=calc_date,
        discount_rate=discount_rate,
        residual_share=residual_share,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_compliance_iterations=bootstrap_compliance_iterations,
        bootstrap_samples=bootstrap_samples,
        bootstrap_compliance_samples=bootstrap_compliance_samples,
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
            "Иные затраты": np.where(
                payout,
                rng.uniform(5_000, 60_000, n_rows),
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
    "collapse_monitoring_to_incident",
    "COURT_FEE_DEFAULT",
    "dedupe_monitoring_by_loss",
    "FU_FEE_DEFAULT",
    "GroupRetroPriors",
    "HORIZONS",
    "HORIZON_SPECS",
    "ULTIMATE_HORIZON_DAYS",
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
