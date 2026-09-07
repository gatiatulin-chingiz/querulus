"""Оценка финэффекта по Excel-выгрузке (фаза 2 мониторинга).

Без факта ПСР: expected_psr = precision × Σ_I (ОД_заявлено×k + e_fee),
cost = Σ_I СуммаКВыплате, e_fee = p_fu×fu_fee + p_court×court_fee.
I = общие фильтры ∧ ВызовМодельСутяжность=1 ∧ Выплата по модели=1.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from querulus.fin_effect.excel_explore import (
    COLUMN_ALIASES,
    _to_numeric,
    analytics_base_mask,
    intervention_mask,
    load_excel,
    resolve_column,
)

FU_FEE_DEFAULT = 100_000.0
COURT_FEE_DEFAULT = 15_000.0
# Конец окна ретро для k / p_fu / p_court / psr_share (2 года назад от этой даты).
RETRO_AS_OF_DEFAULT = "2025-06-30"


@dataclass(frozen=True)
class RetroPriors:
    """Ретро-приоры для оценки на выгрузке."""

    precision: float
    k: float
    p_pret: float
    p_fu: float
    p_court: float
    fu_fee: float = FU_FEE_DEFAULT
    court_fee: float = COURT_FEE_DEFAULT
    lookback_years: float | None = None
    date_column: str | None = None
    window_start: str | None = None
    window_end: str | None = None
    n_rows: int = 0
    n_pos: int = 0
    psr_share: float = 0.0

    def expected_fee(self) -> float:
        """Средний пакет взносов: ФУ 100k или суд 15k (без двойного 100k)."""
        return self.p_fu * self.fu_fee + self.p_court * self.court_fee

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MonitoringEffectResult:
    """Итог оценки на Excel."""

    frame: pd.DataFrame
    n_intervention: int
    sum_od: float
    sum_paid: float
    expected_psr: float
    cost: float
    net: float
    e_fee: float
    priors: RetroPriors
    od_column: str
    cost_column: str

    @property
    def paid_column(self) -> str:
        """Обратная совместимость: колонка ОД для ×k."""
        return self.od_column


def load_retro_priors(path: str | Path) -> RetroPriors:
    """Загрузить priors из JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return RetroPriors(
        precision=float(data["precision"]),
        k=float(data["k"]),
        p_pret=float(data["p_pret"]),
        p_fu=float(data["p_fu"]),
        p_court=float(data["p_court"]),
        fu_fee=float(data.get("fu_fee", FU_FEE_DEFAULT)),
        court_fee=float(data.get("court_fee", COURT_FEE_DEFAULT)),
        lookback_years=(
            float(data["lookback_years"])
            if data.get("lookback_years") is not None
            else None
        ),
        date_column=data.get("date_column"),
        window_start=data.get("window_start"),
        window_end=data.get("window_end"),
        n_rows=int(data.get("n_rows", 0)),
        n_pos=int(data.get("n_pos", 0)),
        psr_share=float(data.get("psr_share", 0.0)),
    )


def save_retro_priors(priors: RetroPriors, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(priors.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


RETRO_DATE_CANDIDATES = (
    "PAYMENT_ORDER_DATE_TIME",
    "INCOMING_CLAIM_GET_DATE",
    "INCOMING_CLAIM_GET_DATE_1",
    "LOSS_DATE",
    "ДатаСобытия",
    "ДатаЗаявления",
)


def resolve_retro_date_column(
    df: pd.DataFrame,
    date_col: str | None = None,
) -> str | None:
    """Колонка даты для окна ретро (T0 выплаты / заявление)."""
    if date_col and date_col in df.columns:
        return date_col
    for name in RETRO_DATE_CANDIDATES:
        if name in df.columns:
            return name
    return None


def filter_retro_lookback(
    df: pd.DataFrame,
    *,
    lookback_years: float = 2.0,
    date_col: str | None = None,
    as_of: pd.Timestamp | str | None = RETRO_AS_OF_DEFAULT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Оставить строки за последние ``lookback_years`` лет по дате T0.

    Конец окна — ``as_of`` (по умолчанию ``2025-06-30``) или max(дата),
    если ``as_of`` явно None и дат нет валидных.
    """
    resolved = resolve_retro_date_column(df, date_col)
    meta: dict[str, Any] = {
        "lookback_years": float(lookback_years),
        "date_column": resolved,
        "window_start": None,
        "window_end": None,
        "applied": False,
        "n_before": int(len(df)),
        "n_after": int(len(df)),
    }
    if resolved is None or lookback_years <= 0:
        return df, meta

    dates = pd.to_datetime(df[resolved], errors="coerce")
    if as_of is None:
        end = dates.max()
    else:
        end = pd.Timestamp(as_of)
    if pd.isna(end):
        return df, meta

    start = end - pd.DateOffset(years=lookback_years)
    mask = dates.notna() & (dates >= start) & (dates <= end)
    out = df.loc[mask].copy()
    meta.update(
        {
            "window_start": start.strftime("%Y-%m-%d"),
            "window_end": pd.Timestamp(end).strftime("%Y-%m-%d"),
            "applied": True,
            "n_after": int(len(out)),
        }
    )
    return out, meta


def compute_retro_priors(
    df: pd.DataFrame,
    *,
    precision: float | None = None,
    threshold: float = 0.5,
    proba_col: str = "preds_cf",
    freq_col: str = "TARGET_FREQ",
    amount_col: str = "TARGET_FREQ_AMOUNT",
    od_col: str = "RECOVEREDMAINDEBT_LAST_INST_SUM",
    pret_col: str = "TARGET_FREQ_PRET_AMOUNT",
    fu_col: str = "Сумма_взыскано_по_ФУ",
    court_col: str = "Суммы_взыскано_по_иску",
    fu_fee: float = FU_FEE_DEFAULT,
    court_fee: float = COURT_FEE_DEFAULT,
    lookback_years: float = 2.0,
    date_col: str | None = None,
    as_of: pd.Timestamp | str | None = RETRO_AS_OF_DEFAULT,
) -> RetroPriors:
    """Посчитать priors на зрелом ретро-кадре с таргетами.

    По умолчанию ``k``, ``psr_share`` и доли путей (p_pret/p_fu/p_court)
    считаются на окне **2 года до 2025-06-30** по ``PAYMENT_ORDER_DATE_TIME``
    (т.е. примерно 2023-06-30 … 2025-06-30). ``precision``: если задан —
    как есть; иначе из preds_cf vs TARGET_FREQ (нет preds → 0.5).
    """
    work, window = filter_retro_lookback(
        df,
        lookback_years=lookback_years,
        date_col=date_col,
        as_of=as_of,
    )
    y = _to_numeric(work[freq_col]).fillna(0).astype(int)
    if precision is not None:
        precision_val = float(precision)
        if not 0.0 <= precision_val <= 1.0:
            raise ValueError(f"precision должен быть в [0, 1], получено {precision_val}")
    elif proba_col in work.columns:
        proba = _to_numeric(work[proba_col])
        pred = (proba >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        precision_val = float(tp / (tp + fp)) if (tp + fp) else 0.0
    else:
        precision_val = 0.5

    pos = y == 1
    n_rows = int(len(work))
    n_pos = int(pos.sum())
    psr_share = float(n_pos / n_rows) if n_rows else 0.0

    amount = _to_numeric(work[amount_col]).fillna(0.0) if amount_col in work.columns else pd.Series(0.0, index=work.index)
    od = _to_numeric(work[od_col]).fillna(0.0) if od_col in work.columns else pd.Series(0.0, index=work.index)
    od_pos = pos & (od > 0)
    k = float(amount[od_pos].sum() / od[od_pos].sum()) if od_pos.any() and od[od_pos].sum() > 0 else 1.0

    pret = _to_numeric(work[pret_col]).fillna(0.0) if pret_col in work.columns else pd.Series(0.0, index=work.index)
    fu = _to_numeric(work[fu_col]).fillna(0.0) if fu_col in work.columns else pd.Series(0.0, index=work.index)
    court = (
        _to_numeric(work[court_col]).fillna(0.0)
        if court_col in work.columns
        else pd.Series(0.0, index=work.index)
    )

    if n_pos == 0:
        p_pret, p_fu, p_court = 1.0, 0.0, 0.0
    else:
        is_court = pos & (court > 0)
        is_fu = pos & (fu > 0) & ~is_court
        is_pret = pos & ~is_court & ~is_fu & ((pret > 0) | (amount > 0))
        # остаток позитивов без явных сумм — в pret_only
        covered = is_court | is_fu | is_pret
        is_pret = is_pret | (pos & ~covered)
        p_court = float(is_court.sum() / n_pos)
        p_fu = float(is_fu.sum() / n_pos)
        p_pret = float(is_pret.sum() / n_pos)
        total = p_pret + p_fu + p_court
        if total > 0:
            p_pret, p_fu, p_court = p_pret / total, p_fu / total, p_court / total

    return RetroPriors(
        precision=precision_val,
        k=k,
        p_pret=p_pret,
        p_fu=p_fu,
        p_court=p_court,
        fu_fee=fu_fee,
        court_fee=court_fee,
        lookback_years=float(lookback_years) if lookback_years else None,
        date_column=window.get("date_column"),
        window_start=window.get("window_start"),
        window_end=window.get("window_end"),
        n_rows=n_rows,
        n_pos=n_pos,
        psr_share=psr_share,
    )


def resolve_paid_column(
    df: pd.DataFrame,
    paid_col: str | None = None,
) -> str:
    """Выбрать колонку paid: явная константа или эвристика по алиасам."""
    if paid_col:
        if paid_col in df.columns:
            return paid_col
        resolved = resolve_column(df, paid_col)
        if resolved is not None:
            return resolved
        raise KeyError(f"PAID_COL не найден: {paid_col}")
    # Explore: дефолтный ОД → заявлено.
    for key in ("od_claimed", "od_to_pay", "od_paid", "to_pay", "payment", "recommended"):
        col = resolve_column(df, key)
        if col is not None:
            return col
    raise KeyError("Не найдена колонка ОД / к выплате / платёж / рекомендованная")


def format_money(x: float) -> str:
    """Число без научной нотации, с пробелами тысяч."""
    return f"{float(x):,.2f}".replace(",", " ")


def _as_bool01(series: pd.Series) -> pd.Series:
    num = _to_numeric(series)
    # NA / -999 (explore fill) → 0
    num = num.where(~num.isin([-999, -100]), np.nan)
    return (num.fillna(0) > 0).astype(int)


def model_used_mask(df: pd.DataFrame) -> pd.Series:
    """Сегмент «модель использовали»: вызов модели или есть РезультатПроверки."""
    call = resolve_column(df, "model_call")
    if call is not None:
        return _as_bool01(df[call]).astype(bool)
    result = resolve_column(df, "result_check")
    if result is not None:
        num = _to_numeric(df[result])
        return num.notna() & ~num.isin([-999])
    payout = resolve_column(df, "model_payout")
    if payout is not None:
        return _as_bool01(df[payout]).astype(bool)
    raise KeyError("Нет колонки вызова модели / РезультатПроверки / выплаты по модели")


def model_positive_mask(df: pd.DataFrame) -> pd.Series:
    """С моделью внутри общих фильтров: вызов модели ∧ выплата по модели.

    Без фильтра соглашения — иначе agreement_share всегда 100%.
    """
    mask = analytics_base_mask(df)
    for key in ("model_call", "model_payout"):
        col = resolve_column(df, key)
        if col is None:
            return pd.Series(False, index=df.index)
        mask = mask & _to_numeric(df[col]).fillna(0).eq(1)
    return mask


def agreement_mask(df: pd.DataFrame) -> pd.Series:
    """Соглашение: флаг или ФормаВозмещения содержит «соглашен»."""
    flag = resolve_column(df, "agreement")
    form = resolve_column(df, "refund_form")
    out = pd.Series(False, index=df.index)
    if flag is not None:
        out = out | _as_bool01(df[flag]).astype(bool)
    if form is not None:
        text = df[form].fillna("").astype(str).str.casefold()
        out = out | text.str.contains("соглашен", na=False)
    return out


def pretension_mask(df: pd.DataFrame) -> pd.Series:
    """Претензия: ЕстьПретензияВИнциденте (или запасные алиасы)."""
    col = resolve_column(df, "pretension")
    if col is None:
        return pd.Series(False, index=df.index)
    return _as_bool01(df[col]).astype(bool)


def fu_mask(df: pd.DataFrame) -> pd.Series:
    """Обращение к ФУ."""
    col = resolve_column(df, "fu_flag")
    if col is None:
        return pd.Series(False, index=df.index)
    return _as_bool01(df[col]).astype(bool)


def court_mask(df: pd.DataFrame) -> pd.Series:
    """Обращение к суду."""
    col = resolve_column(df, "court_flag")
    if col is None:
        return pd.Series(False, index=df.index)
    return _as_bool01(df[col]).astype(bool)


def compare_agreement_pretension_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Доли соглашений и претензий: с моделью vs без — внутри общих фильтров.

    Сегмент ``with_model`` = analytics_base ∧ ВызовМодель=1 ∧ ВыплатаПоМодели=1.
    ``without_model`` = analytics_base ∧ не with_model.
    """
    base = analytics_base_mask(df)
    used = model_positive_mask(df)
    agr = agreement_mask(df)
    pret = pretension_mask(df)

    rows: list[dict[str, Any]] = []
    for label, mask in (
        ("with_model", used),
        ("without_model", base & ~used),
    ):
        n = int(mask.sum())
        rows.append(
            {
                "segment": label,
                "n": n,
                "agreement_share": float(agr[mask].mean()) if n else np.nan,
                "pretension_share": float(pret[mask].mean()) if n else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    used_row = table.loc[table["segment"] == "with_model"].iloc[0]
    ctrl_row = table.loc[table["segment"] == "without_model"].iloc[0]
    lift = pd.DataFrame(
        [
            {
                "segment": "lift_pp (with - without)",
                "n": np.nan,
                "agreement_share": used_row["agreement_share"] - ctrl_row["agreement_share"],
                "pretension_share": used_row["pretension_share"] - ctrl_row["pretension_share"],
            },
            {
                "segment": "lift_rel (with / without - 1)",
                "n": np.nan,
                "agreement_share": (
                    used_row["agreement_share"] / ctrl_row["agreement_share"] - 1.0
                    if ctrl_row["agreement_share"] and ctrl_row["agreement_share"] > 0
                    else np.nan
                ),
                "pretension_share": (
                    used_row["pretension_share"] / ctrl_row["pretension_share"] - 1.0
                    if ctrl_row["pretension_share"] and ctrl_row["pretension_share"] > 0
                    else np.nan
                ),
            },
        ]
    )
    return pd.concat([table, lift], ignore_index=True)


def resolve_effect_date_series(df: pd.DataFrame) -> pd.Series:
    """Дата для окна экстраполяции (заявление / событие / вызов модели)."""
    col = resolve_column(df, "application_date")
    if col is None:
        col = resolve_column(df, "model_call_date")
    if col is None:
        for name in (
            "ДатаЗаявления",
            "ДатаСобытия",
            "Дата вызова модели сутяжности",
        ):
            if name in df.columns:
                col = name
                break
    if col is None:
        raise KeyError("Нет даты для экстраполяции (ДатаЗаявления / вызов модели)")
    return pd.to_datetime(df[col], errors="coerce")


def resolve_model_call_date_series(df: pd.DataFrame) -> pd.Series | None:
    """Колонка даты вызова модели, если есть."""
    col = resolve_column(df, "model_call_date")
    if col is None:
        for name in (
            "Дата вызова модели сутяжности",
            "Дата_выхода_модели_ступенчатая",
            "Дата вызова модуля суррогативности",
        ):
            if name in df.columns:
                col = name
                break
    if col is None:
        return None
    return pd.to_datetime(df[col], errors="coerce")


def infer_model_start(df: pd.DataFrame) -> pd.Timestamp:
    """Начало работы модели = min даты вызова модели (где модель использовали).

    Fallback: min даты вызова по всем непустым; иначе min ДатаЗаявления в окне.
    """
    call_dates = resolve_model_call_date_series(df)
    if call_dates is not None:
        try:
            used = model_used_mask(df)
            among_used = call_dates[used].dropna()
            if not among_used.empty:
                return pd.Timestamp(among_used.min())
        except KeyError:
            pass
        all_calls = call_dates.dropna()
        if not all_calls.empty:
            return pd.Timestamp(all_calls.min())
    # fallback — начало окна выгрузки
    dates = resolve_effect_date_series(df).dropna()
    if dates.empty:
        raise ValueError("Не удалось вывести MODEL_START из дат Excel")
    return pd.Timestamp(dates.min())


def extrapolate_to_year(
    effect: MonitoringEffectResult,
    df: pd.DataFrame,
    *,
    model_start: str | pd.Timestamp | None = None,
    as_of: str | pd.Timestamp | None = None,
    year_days: float = 365.0,
) -> dict[str, Any]:
    """Экстраполяция net/cost/expected_psr на year_days с начала работы модели.

    ``model_start=None`` → ``infer_model_start(df)`` из дат Excel.
    Скорость: net / дней в окне Excel (по датам строк).
    """
    dates = resolve_effect_date_series(df)
    valid = dates.dropna()
    if valid.empty:
        raise ValueError("Нет валидных дат в Excel для экстраполяции")
    sample_start = valid.min()
    sample_end = valid.max()
    sample_days = max(1.0, float((sample_end - sample_start).days) + 1.0)

    if model_start is None:
        start = infer_model_start(df)
        model_start_source = "excel"
    else:
        start = pd.Timestamp(model_start)
        model_start_source = "manual"
    end = pd.Timestamp(as_of) if as_of is not None else sample_end
    elapsed_days = max(1.0, float((end - start).days) + 1.0)

    daily_net = effect.net / sample_days
    daily_cost = effect.cost / sample_days
    daily_psr = effect.expected_psr / sample_days

    return {
        "sample_start": sample_start.date().isoformat(),
        "sample_end": sample_end.date().isoformat(),
        "sample_days": round(sample_days, 1),
        "model_start": start.date().isoformat(),
        "model_start_source": model_start_source,
        "as_of": end.date().isoformat(),
        "elapsed_days_since_model_start": round(elapsed_days, 1),
        "net_sample": round(effect.net, 2),
        "net_per_day": round(daily_net, 2),
        "net_since_model_start": round(daily_net * elapsed_days, 2),
        "net_annual_365": round(daily_net * year_days, 2),
        "cost_annual_365": round(daily_cost * year_days, 2),
        "expected_psr_annual_365": round(daily_psr * year_days, 2),
        "scale_sample_to_365": round(year_days / sample_days, 4),
    }


def format_summary_dict(summary: dict[str, Any]) -> dict[str, Any]:
    """Копия summary с денежными полями без e-нотации."""
    money_keys = {
        "sum_od",
        "sum_paid",
        "e_fee",
        "expected_psr",
        "cost",
        "net",
        "net_sample",
        "net_per_day",
        "net_since_model_start",
        "net_annual_365",
        "cost_annual_365",
        "expected_psr_annual_365",
    }
    out: dict[str, Any] = {}
    for k, v in summary.items():
        if k in money_keys and isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = format_money(v)
        else:
            out[k] = v
    return out


def format_sensitivity_table(sens: pd.DataFrame) -> pd.DataFrame:
    """Чувствительность с обычными числами (не 1.2e+06)."""
    out = sens.copy()
    for col in ("net", "expected_psr", "k", "precision"):
        if col in out.columns:
            out[col] = out[col].map(
                lambda x: round(float(x), 2) if pd.notna(x) else x
            )
    return out


def estimate_monitoring_effect(
    df: pd.DataFrame,
    priors: RetroPriors,
    *,
    od_col: str | None = None,
    cost_col: str | None = None,
    paid_col: str | None = None,
) -> MonitoringEffectResult:
    """Оценка net на выгрузке.

    expected_psr = precision × Σ_I (ОД_заявлено × k + e_fee)
    cost         = Σ_I СуммаКВыплате
    net          = expected_psr − cost

    ``paid_col`` — устаревший алиас для ``od_col``.
    """
    if od_col is None:
        od_col = paid_col
    if od_col is None:
        od_col = "СуммаОсновногоДолгаЗаявлено"
    if cost_col is None:
        cost_col = "СуммаКВыплате"

    od_name = resolve_paid_column(df, od_col)
    try:
        cost_name = resolve_paid_column(df, cost_col)
    except KeyError:
        cost_name = resolve_column(df, "to_pay") or resolve_column(df, "payment") or od_name

    mask = intervention_mask(df)
    od = _to_numeric(df[od_name]).fillna(0.0)
    cost_s = _to_numeric(df[cost_name]).fillna(0.0)
    od_i = od.where(mask, 0.0)
    cost_i = cost_s.where(mask, 0.0)
    e_fee = priors.expected_fee()
    sum_od = float(od_i.sum())
    sum_paid = float(cost_i.sum())
    n_i = int(mask.sum())
    expected_psr = float(priors.precision * ((od_i * priors.k).sum() + n_i * e_fee))
    cost = sum_paid
    net = expected_psr - cost

    frame = df.copy()
    frame["_intervention"] = mask.astype(int)
    frame["_od"] = od
    frame["_od_i"] = od_i
    frame["_cost_i"] = cost_i
    frame["_expected_psr_row"] = np.where(
        mask,
        priors.precision * (od_i * priors.k + e_fee),
        0.0,
    )

    return MonitoringEffectResult(
        frame=frame,
        n_intervention=n_i,
        sum_od=sum_od,
        sum_paid=sum_paid,
        expected_psr=expected_psr,
        cost=cost,
        net=net,
        e_fee=e_fee,
        priors=priors,
        od_column=od_name,
        cost_column=cost_name,
    )


def sensitivity_table(
    df: pd.DataFrame,
    priors: RetroPriors,
    *,
    od_col: str | None = None,
    cost_col: str | None = None,
    paid_col: str | None = None,
    rel_delta: float = 0.2,
) -> pd.DataFrame:
    """Чувствительность net к ±rel_delta по k и precision."""
    if od_col is None:
        od_col = paid_col
    rows: list[dict[str, Any]] = []
    base = estimate_monitoring_effect(
        df, priors, od_col=od_col, cost_col=cost_col
    )
    rows.append(
        {
            "scenario": "base",
            "precision": priors.precision,
            "k": priors.k,
            "net": base.net,
            "expected_psr": base.expected_psr,
        }
    )
    for name, p_mult, k_mult in (
        ("precision -20%", 1 - rel_delta, 1.0),
        ("precision +20%", 1 + rel_delta, 1.0),
        ("k -20%", 1.0, 1 - rel_delta),
        ("k +20%", 1.0, 1 + rel_delta),
    ):
        adj = RetroPriors(
            precision=max(0.0, min(1.0, priors.precision * p_mult)),
            k=max(0.0, priors.k * k_mult),
            p_pret=priors.p_pret,
            p_fu=priors.p_fu,
            p_court=priors.p_court,
            fu_fee=priors.fu_fee,
            court_fee=priors.court_fee,
        )
        res = estimate_monitoring_effect(
            df, adj, od_col=od_col, cost_col=cost_col
        )
        rows.append(
            {
                "scenario": name,
                "precision": adj.precision,
                "k": adj.k,
                "net": res.net,
                "expected_psr": res.expected_psr,
            }
        )
    return pd.DataFrame(rows)


def default_demo_priors() -> RetroPriors:
    """Демо-приоры, пока нет ретро-parquet на контуре."""
    return RetroPriors(
        precision=0.55,
        k=1.35,
        p_pret=0.45,
        p_fu=0.30,
        p_court=0.25,
        fu_fee=FU_FEE_DEFAULT,
        court_fee=COURT_FEE_DEFAULT,
        lookback_years=2.0,
        date_column="PAYMENT_ORDER_DATE_TIME",
        window_start=None,
        window_end=None,
        n_rows=0,
        n_pos=0,
        psr_share=0.0,
    )


_FILIALS = (
    "Курский",
    "Ярославский",
    "Пермский",
    "Архангельский",
    "Омский",
    "Владимирский",
    "Магнитогорский",
    "Марийский",
)


def build_synthetic_claims_excel(
    n_rows: int = 300,
    *,
    seed: int = 42,
    intervention_rate: float = 0.12,
) -> pd.DataFrame:
    """Синтетика под общие фильтры и I = вызов модели ∧ выплата по модели."""
    if n_rows < 50:
        raise ValueError("n_rows >= 50")
    rng = np.random.default_rng(seed)
    # Большинство — разрешённые филиалы; немного исключённых для проверки фильтра.
    allowed_filials = [f for f in _FILIALS if "Архангельск" not in f and "Марийск" not in f]
    excluded_filials = [f for f in _FILIALS if f not in allowed_filials]
    filial = np.array(
        [
            rng.choice(excluded_filials if rng.random() < 0.08 else allowed_filials)
            for _ in range(n_rows)
        ]
    )
    zones = np.array([f"Зона ф-ла {f}" for f in filial])
    incident = 11_000_000 + rng.integers(0, 900_000, size=n_rows)

    event = pd.to_datetime("2024-03-01") + pd.to_timedelta(
        rng.integers(0, 120, size=n_rows), unit="D"
    )
    apply = event + pd.to_timedelta(rng.integers(1, 15, size=n_rows), unit="D")

    # Базовые фильтры: почти все строки проходят; чуть шума.
    auto_obj = np.ones(n_rows, dtype=int)
    auto_obj[rng.choice(n_rows, size=max(1, n_rows // 25), replace=False)] = 0
    loss_status = np.array(["Первичный"] * n_rows, dtype=object)
    loss_status[rng.choice(n_rows, size=max(1, n_rows // 30), replace=False)] = "Повторный"

    forms = np.array(
        rng.choice(
            ["Денежная", "Ремонт", "Соглашение", "Отказ"],
            size=n_rows,
            p=[0.45, 0.30, 0.18, 0.07],
        )
    )

    # I: вызов модели + выплата по модели (на разрешённых строках)
    n_model = max(2, int(round(n_rows * intervention_rate * 1.8)))
    model_call = np.zeros(n_rows, dtype=int)
    model_pay = np.zeros(n_rows, dtype=int)
    result = np.zeros(n_rows, dtype=int)
    candidate = np.array(
        [
            (f in allowed_filials)
            and (a == 1)
            and str(s).casefold().startswith("первич")
            and any(x in str(form).casefold() for x in ("денежн", "ремонт", "соглашен"))
            for f, a, s, form in zip(filial, auto_obj, loss_status, forms)
        ]
    )
    cand_idx = np.where(candidate)[0]
    if len(cand_idx) == 0:
        cand_idx = np.arange(n_rows)
    pick = rng.choice(cand_idx, size=min(n_model, len(cand_idx)), replace=False)
    model_call[pick] = 1
    model_pay[pick] = 1
    result[pick] = 1
    # часть — вызвали модель, но без выплаты по модели
    extra_call = rng.choice(
        np.where((model_call == 0) & candidate)[0],
        size=min(max(len(pick), 1), int(((model_call == 0) & candidate).sum()) or 1),
        replace=False,
    ) if ((model_call == 0) & candidate).any() else np.array([], dtype=int)
    model_call[extra_call] = 1

    with_model = (model_call == 1) & (model_pay == 1)
    is_i = with_model & candidate
    agreement = np.zeros(n_rows, dtype=int)
    agreement[with_model] = (rng.random(int(with_model.sum())) < 0.55).astype(int)
    agreement_extra = (model_call == 1) & ~with_model & (rng.random(n_rows) < 0.20)
    agreement = np.where(agreement | agreement_extra, 1, agreement)

    pret = (rng.random(n_rows) < np.where(model_call == 1, 0.08, 0.03)).astype(int)
    fu_flag = (rng.random(n_rows) < np.where(pret == 1, 0.25, 0.02)).astype(int)
    court_flag = (rng.random(n_rows) < np.where(fu_flag == 1, 0.35, 0.01)).astype(int)

    recommended = np.zeros(n_rows, dtype=float)
    recommended[with_model] = rng.uniform(15_000, 120_000, size=int(with_model.sum()))

    wear = rng.uniform(5_000, 200_000, size=n_rows)
    other = np.where(rng.random(n_rows) < 0.15, rng.uniform(1_000, 40_000, size=n_rows), 0.0)
    to_pay = wear + other
    payment = to_pay.copy()

    claimed_od = rng.uniform(10_000, 250_000, size=n_rows)
    claimed_uts = np.where(rng.random(n_rows) < 0.2, rng.uniform(1_000, 30_000, size=n_rows), 0.0)
    claimed_evac = np.zeros(n_rows)
    claimed_storage = np.zeros(n_rows)
    od_claimed = claimed_od + claimed_uts + claimed_evac + claimed_storage
    od_to_pay = claimed_od * rng.uniform(0.4, 1.0, size=n_rows)

    payment[is_i] = recommended[is_i]
    od_to_pay[is_i] = recommended[is_i]
    to_pay[is_i] = recommended[is_i]
    od_claimed[is_i] = recommended[is_i] * rng.uniform(0.9, 1.3, size=int(is_i.sum()))
    wear[is_i] = recommended[is_i] * 0.85
    other[is_i] = recommended[is_i] * 0.15
    forms[is_i] = rng.choice(["Денежная", "Ремонт", "Соглашение"], size=int(is_i.sum()))

    products = rng.choice(
        ["Традиционное ОСАГО", "Прямое ОСАГО (с 1 марта 2009)"],
        size=n_rows,
    )
    applicants = rng.choice(
        [
            "Потерпевший",
            "Юрист с потерпевшим",
            "Опытный юрист",
            "Представитель (не автоюрист)",
            "Выгодоприобретатель",
        ],
        size=n_rows,
    )

    df = pd.DataFrame(
        {
            "НомерИнцидента": incident,
            "ЗонаУрегулирования": zones,
            "Филиал": filial,
            "ДатаЗаявления": apply,
            "ДатаСобытия": event,
            "Дата вызова модели сутяжности": np.where(
                model_call == 1,
                apply - pd.to_timedelta(rng.integers(0, 5, size=n_rows), unit="D"),
                pd.NaT,
            ),
            "Продукт": products,
            "Тип заявителя": applicants,
            "ВозрастЗаявителя": rng.integers(18, 75, size=n_rows),
            "СпособПолученияЗаявления": rng.choice(
                ["Лично", "Электронное обращение"], size=n_rows
            ),
            "ОформленоГИБДД": rng.integers(0, 2, size=n_rows),
            "Категория ТС потерпевшего": rng.choice(
                ["Автомобили легковые", "Мотоциклы", "Микроавтобусы"], size=n_rows
            ),
            "Тип владельца транспортного средства": rng.choice(
                ["Физ. Лицо", "Юр. Лицо"], size=n_rows, p=[0.85, 0.15]
            ),
            "ФормаВозмещения": forms,
            "УбытокСтатус": loss_status,
            "ТипОбъектаАвтотранспорт": auto_obj,
            "РезультатПроверки": result,
            "ВызовМодельСутяжность": model_call,
            "Заключено соглашение": agreement,
            "Выплата по модели в Инциденте": model_pay,
            "ЕстьПретензияВИнциденте": pret,
            "Обращение к ФУ": fu_flag,
            "Обращение к суду": court_flag,
            "Сумма рекомендованная к доплате по модулю": recommended,
            "СуммаОсновногоДолгаЗаявлено": od_claimed,
            "СуммаОсновногоДолгаКВыплате": od_to_pay,
            "СтоимостьСУчетомИзноса": wear,
            "Иные затраты": other,
            "СуммаКВыплате": to_pay,
            "СуммаПлатежа": payment,
            "СуммаОД": od_to_pay,
            "СуммаОУ": payment,
            "Сумма выплаты по претензии": 0.0,
            "Кол-во претензий": pret,
            "Сумма выплат по ФУ": 0.0,
            "Сумма выплаты по суду": 0.0,
            "ОсновныеВыплатыЗаявлено_Основной долг": claimed_od,
            "ОсновныеВыплатыЗаявлено_Утрата товарной стоимости": claimed_uts,
            "ОсновныеВыплатыЗаявлено_Затраты на эвакуацию ТС": claimed_evac,
            "ОсновныеВыплатыЗаявлено_Затраты на хранение": claimed_storage,
            "ОсновныеВыплатыСуммаОплаченоДолиВыплата": od_to_pay,
        }
    )
    return df


def write_synthetic_claims_excel(
    path: str | Path,
    *,
    n_rows: int = 300,
    seed: int = 42,
) -> Path:
    """Сгенерировать и сохранить синтетический xlsx."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = build_synthetic_claims_excel(n_rows=n_rows, seed=seed)
    df.to_excel(path, index=False, engine="openpyxl")
    return path


# Реэкспорт для тетрадок
__all__ = [
    "COLUMN_ALIASES",
    "COURT_FEE_DEFAULT",
    "FU_FEE_DEFAULT",
    "MonitoringEffectResult",
    "RETRO_AS_OF_DEFAULT",
    "RETRO_DATE_CANDIDATES",
    "RetroPriors",
    "agreement_mask",
    "analytics_base_mask",
    "build_synthetic_claims_excel",
    "compare_agreement_pretension_by_model",
    "compute_retro_priors",
    "court_mask",
    "default_demo_priors",
    "estimate_monitoring_effect",
    "extrapolate_to_year",
    "filter_retro_lookback",
    "format_money",
    "format_sensitivity_table",
    "format_summary_dict",
    "fu_mask",
    "infer_model_start",
    "intervention_mask",
    "load_excel",
    "load_retro_priors",
    "model_positive_mask",
    "model_used_mask",
    "pretension_mask",
    "resolve_paid_column",
    "resolve_retro_date_column",
    "save_retro_priors",
    "sensitivity_table",
    "write_synthetic_claims_excel",
]
