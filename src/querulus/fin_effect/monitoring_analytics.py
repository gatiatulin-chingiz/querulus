"""Сегменты и аналитика мониторинга: Bpilot / Bam, варианты 1 и 2."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from querulus.fin_effect.excel_explore import (
    INCIDENT_COURT_COL,
    INCIDENT_FU_COL,
    FilialScope,
    _to_numeric,
    analytics_base_mask,
    enrich_incident_path_flags,
    resolve_column,
    resolve_model_payout_loss_column,
)

RESULT_OUT_OF_MODEL = -100

VARIANT2_LABELS = {
    "ignored_zero_paid": "вызов=1, result=0, выплата=1 (результат проигнорирован)",
    "out_of_model_paid": "вызов=1, result=−100, выплата=1 (вне модели)",
    "applied_one_paid": "вызов=1, result=1, выплата=1 (модель применена)",
    "recommended_unpaid": "вызов=1, result=1, выплата=0 (рекомендация без доплаты)",
}


def _as_bool01(series: pd.Series) -> pd.Series:
    num = _to_numeric(series)
    num = num.where(~num.isin([-999, -100]), np.nan)
    return (num.fillna(0) > 0).astype(int)


def result_check_bucket(series: pd.Series) -> pd.Series:
    num = _to_numeric(series)
    out = pd.Series("other", index=series.index, dtype=object)
    out = out.mask(num.eq(0), "model_0")
    out = out.mask(num.eq(1), "model_1")
    out = out.mask(num.eq(RESULT_OUT_OF_MODEL), "out_of_model")
    out = out.mask(num.isna(), "missing")
    return out


def model_rucheek_mask(
    df: pd.DataFrame,
    *,
    filial_scope: FilialScope = "pilot",
) -> pd.Series:
    """Модель работала / ручеёк: РезультатПроверки ∈ {0, 1}."""
    base = analytics_base_mask(df, filial_scope=filial_scope)
    result_col = resolve_column(df, "result_check")
    if result_col is None:
        return pd.Series(False, index=df.index)
    return base & _to_numeric(df[result_col]).isin([0, 1])


def control_mask(
    df: pd.DataFrame,
    *,
    filial_scope: FilialScope = "pilot",
) -> pd.Series:
    """Контроль: РезультатПроверки = −100."""
    base = analytics_base_mask(df, filial_scope=filial_scope)
    result_col = resolve_column(df, "result_check")
    if result_col is None:
        return pd.Series(False, index=df.index)
    return base & _to_numeric(df[result_col]).eq(RESULT_OUT_OF_MODEL)


def model_payout_loss_mask(df: pd.DataFrame) -> pd.Series:
    col = resolve_model_payout_loss_column(df)
    if col is None:
        return pd.Series(False, index=df.index)
    return _to_numeric(df[col]).fillna(0).eq(1)


def model_call_flag_mask(df: pd.DataFrame) -> pd.Series:
    col = resolve_column(df, "model_call")
    if col is None:
        return pd.Series(False, index=df.index)
    return _to_numeric(df[col]).fillna(0).eq(1)


def benefit_mask(
    df: pd.DataFrame,
    *,
    filial_scope: FilialScope = "pilot",
) -> pd.Series:
    """Устаревшая маска: result=1 ∧ выплата=1 (для совместимости)."""
    base = analytics_base_mask(df, filial_scope=filial_scope)
    result_col = resolve_column(df, "result_check")
    if result_col is None:
        return pd.Series(False, index=df.index)
    return (
        base
        & _to_numeric(df[result_col]).fillna(-999).eq(1)
        & model_payout_loss_mask(df)
    )


def cost_mask_variant1(
    df: pd.DataFrame,
    *,
    filial_scope: FilialScope = "pilot",
) -> pd.Series:
    """Устаревшая cost-маска v1: ручеёк 0/1 с выплатой."""
    base = analytics_base_mask(df, filial_scope=filial_scope)
    result_col = resolve_column(df, "result_check")
    if result_col is None:
        return pd.Series(False, index=df.index)
    return (
        base
        & _to_numeric(df[result_col]).isin([0, 1])
        & model_payout_loss_mask(df)
    )


def variant2_case_masks(
    df: pd.DataFrame,
    *,
    filial_scope: FilialScope = "pilot",
) -> dict[str, pd.Series]:
    base = analytics_base_mask(df, filial_scope=filial_scope)
    call = model_call_flag_mask(df)
    pay = model_payout_loss_mask(df)
    result_col = resolve_column(df, "result_check")
    empty = pd.Series(False, index=df.index)
    if result_col is None:
        return {k: empty for k in VARIANT2_LABELS}
    result = _to_numeric(df[result_col])
    return {
        "ignored_zero_paid": base & call & result.eq(0) & pay,
        "out_of_model_paid": base & call & result.eq(RESULT_OUT_OF_MODEL) & pay,
        "applied_one_paid": base & call & result.eq(1) & pay,
        "recommended_unpaid": base & call & result.eq(1) & ~pay,
    }


def agreement_mask(df: pd.DataFrame) -> pd.Series:
    flag = resolve_column(df, "agreement")
    form = resolve_column(df, "refund_form")
    out = pd.Series(False, index=df.index)
    if flag is not None:
        out = out | _as_bool01(df[flag]).astype(bool)
    if form is not None:
        text = df[form].fillna("").astype(str).str.casefold()
        out = out | text.str.contains("соглашен", na=False)
    return out


def segment_111_mask(
    df: pd.DataFrame,
    *,
    filial_scope: FilialScope = "pilot",
) -> pd.Series:
    """Сегмент 111: result=1 ∧ выплата по модели=1 ∧ соглашение=1."""
    base = analytics_base_mask(df, filial_scope=filial_scope)
    result_col = resolve_column(df, "result_check")
    if result_col is None:
        return pd.Series(False, index=df.index)
    return (
        base
        & _to_numeric(df[result_col]).fillna(-999).eq(1)
        & model_payout_loss_mask(df)
        & agreement_mask(df)
    )


def pretension_mask(df: pd.DataFrame) -> pd.Series:
    col = resolve_column(df, "pretension")
    if col is None:
        return pd.Series(False, index=df.index)
    return _as_bool01(df[col]).astype(bool)


def _share_row(
    mask: pd.Series,
    *,
    agr: pd.Series,
    pret: pd.Series,
    fu: pd.Series,
    court: pd.Series,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    n = int(mask.sum())
    row: dict[str, Any] = {
        "n": n,
        "agreement_share": float(agr[mask].mean()) if n else np.nan,
        "pretension_share": float(pret[mask].mean()) if n else np.nan,
        "fu_incident_share": float(fu[mask].mean()) if n else np.nan,
        "court_incident_share": float(court[mask].mean()) if n else np.nan,
    }
    if extra:
        row = {**extra, **row}
    return row


def path_share_table(
    df: pd.DataFrame,
    segments: list[tuple[str, pd.Series]],
    *,
    lift_from: str | None = None,
    lift_to: str | None = None,
) -> pd.DataFrame:
    if INCIDENT_FU_COL not in df.columns or INCIDENT_COURT_COL not in df.columns:
        df = enrich_incident_path_flags(df)
    agr = agreement_mask(df)
    pret = pretension_mask(df)
    fu = _as_bool01(df[INCIDENT_FU_COL]).astype(bool)
    court = _as_bool01(df[INCIDENT_COURT_COL]).astype(bool)

    rows: list[dict[str, Any]] = []
    by_label: dict[str, dict[str, Any]] = {}
    for label, mask in segments:
        row = _share_row(
            mask, agr=agr, pret=pret, fu=fu, court=court, extra={"segment": label}
        )
        rows.append(row)
        by_label[label] = row
    table = pd.DataFrame(rows)

    share_cols = (
        "agreement_share",
        "pretension_share",
        "fu_incident_share",
        "court_incident_share",
    )
    if lift_from and lift_to and lift_from in by_label and lift_to in by_label:
        a, b = by_label[lift_from], by_label[lift_to]
        lift_pp: dict[str, Any] = {
            "segment": f"lift_pp ({lift_from} - {lift_to})",
            "n": np.nan,
        }
        lift_rel: dict[str, Any] = {
            "segment": f"lift_rel ({lift_from} / {lift_to} - 1)",
            "n": np.nan,
        }
        for col in share_cols:
            xa, xb = a[col], b[col]
            lift_pp[col] = (
                float(xa - xb) if pd.notna(xa) and pd.notna(xb) else np.nan
            )
            lift_rel[col] = (
                float(xa / xb - 1.0)
                if pd.notna(xa) and pd.notna(xb) and xb
                else np.nan
            )
        table = pd.concat(
            [table, pd.DataFrame([lift_pp, lift_rel])], ignore_index=True
        )
    return table


def compare_path_shares(
    df: pd.DataFrame,
    *,
    filial_scope: FilialScope = "pilot",
    variant: int = 1,
) -> pd.DataFrame:
    if variant == 1:
        return path_share_table(
            df,
            [
                ("model", model_rucheek_mask(df, filial_scope=filial_scope)),
                ("control", control_mask(df, filial_scope=filial_scope)),
            ],
            lift_from="model",
            lift_to="control",
        )
    cases = variant2_case_masks(df, filial_scope=filial_scope)
    worked = cases["applied_one_paid"]
    not_worked = (
        cases["ignored_zero_paid"]
        | cases["out_of_model_paid"]
        | cases["recommended_unpaid"]
    )
    return path_share_table(
        df,
        [
            ("applied_one_paid", cases["applied_one_paid"]),
            ("ignored_zero_paid", cases["ignored_zero_paid"]),
            ("out_of_model_paid", cases["out_of_model_paid"]),
            ("recommended_unpaid", cases["recommended_unpaid"]),
            ("model_worked", worked),
            ("model_not_worked", not_worked),
        ],
        lift_from="model_worked",
        lift_to="model_not_worked",
    )


def shares_as_percent(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        name = str(col)
        if (
            name.endswith("_share")
            or name.startswith("share_")
            or name.endswith("_of_base")
            or name.endswith("_of_called")
        ):
            out[col] = (pd.to_numeric(out[col], errors="coerce") * 100).round(2)
    return out


def _resolve_amount_col(df: pd.DataFrame, amount_col: str | None) -> str:
    if amount_col and amount_col in df.columns:
        return amount_col
    for key in ("to_pay", "payment", "other_costs", "od_claimed"):
        col = resolve_column(df, key)
        if col is not None:
            return col
    raise KeyError("Нет денежной колонки для статистики выплат")


def _money_stats(values: pd.Series) -> dict[str, float]:
    s = _to_numeric(values).dropna()
    n = int(len(s))
    if n == 0:
        return {
            "n_amount": 0,
            "sum": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "n_amount": n,
        "sum": float(s.sum()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def filial_usage_stats(
    df: pd.DataFrame,
    *,
    filial_scope: FilialScope = "all",
) -> pd.DataFrame:
    base = analytics_base_mask(df, filial_scope=filial_scope)
    filial_col = resolve_column(df, "filial")
    result_col = resolve_column(df, "result_check")
    if filial_col is None:
        return pd.DataFrame()
    buckets = (
        result_check_bucket(df[result_col])
        if result_col is not None
        else pd.Series("missing", index=df.index)
    )
    pay = model_payout_loss_mask(df)
    rows: list[dict[str, Any]] = []
    for filial, idx in df.loc[base].groupby(filial_col, dropna=False).groups.items():
        mask = base & df.index.isin(idx)
        n = int(mask.sum())
        b = buckets[mask]
        n_model = int(b.isin(["model_0", "model_1"]).sum())
        n_control = int((b == "out_of_model").sum())
        n_applied = int((mask & pay & b.eq("model_1")).sum())
        rows.append(
            {
                "filial": filial,
                "n": n,
                "n_model_rucheek": n_model,
                "model_rucheek_share": float(n_model / n) if n else np.nan,
                "n_control": n_control,
                "control_share": float(n_control / n) if n else np.nan,
                "n_applied_result1_paid": n_applied,
                "applied_share": float(n_applied / n) if n else np.nan,
                "share_result_0": float((b == "model_0").mean()) if n else np.nan,
                "share_result_1": float((b == "model_1").mean()) if n else np.nan,
                "share_out_of_model": float((b == "out_of_model").mean()) if n else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("n", ascending=False).reset_index(drop=True) if not out.empty else out


def filial_path_shares(
    df: pd.DataFrame,
    *,
    filial_scope: FilialScope = "pilot",
    variant: int = 1,
) -> pd.DataFrame:
    if INCIDENT_FU_COL not in df.columns or INCIDENT_COURT_COL not in df.columns:
        df = enrich_incident_path_flags(df)
    filial_col = resolve_column(df, "filial")
    if filial_col is None:
        return pd.DataFrame()
    base = analytics_base_mask(df, filial_scope=filial_scope)
    agr = agreement_mask(df)
    pret = pretension_mask(df)
    fu = _as_bool01(df[INCIDENT_FU_COL]).astype(bool)
    court = _as_bool01(df[INCIDENT_COURT_COL]).astype(bool)
    rows: list[dict[str, Any]] = []
    for filial, idx in df.loc[base].groupby(filial_col, dropna=False).groups.items():
        in_f = base & df.index.isin(idx)
        if variant == 1:
            segs = (
                ("model", in_f & model_rucheek_mask(df, filial_scope=filial_scope)),
                ("control", in_f & control_mask(df, filial_scope=filial_scope)),
            )
        else:
            cases = variant2_case_masks(df, filial_scope=filial_scope)
            segs = tuple((name, in_f & m) for name, m in cases.items())
        for label, mask in segs:
            rows.append(
                _share_row(
                    mask,
                    agr=agr,
                    pret=pret,
                    fu=fu,
                    court=court,
                    extra={"filial": filial, "segment": label},
                )
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["filial", "segment"]).reset_index(drop=True) if not out.empty else out


def result_distribution(
    df: pd.DataFrame,
    *,
    filial_scope: FilialScope = "pilot",
    amount_col: str | None = None,
) -> pd.DataFrame:
    base = analytics_base_mask(df, filial_scope=filial_scope)
    result_col = resolve_column(df, "result_check")
    if result_col is None:
        return pd.DataFrame()
    amount = _resolve_amount_col(df, amount_col)
    buckets = result_check_bucket(df[result_col])
    n_base = int(base.sum())
    order = ("model_0", "model_1", "model_rucheek", "control", "other", "missing")
    rows: list[dict[str, Any]] = []
    mapping = (
        ("model_0", base & buckets.eq("model_0")),
        ("model_1", base & buckets.eq("model_1")),
        ("model_rucheek", base & buckets.isin(["model_0", "model_1"])),
        ("control", base & buckets.eq("out_of_model")),
        ("other", base & buckets.eq("other")),
        ("missing", base & buckets.eq("missing")),
    )
    for label, mask in mapping:
        n = int(mask.sum())
        rows.append(
            {
                "bucket": label,
                "n": n,
                "share_of_base": float(n / n_base) if n_base else np.nan,
                "amount_col": amount,
                **_money_stats(df.loc[mask, amount]),
            }
        )
    out = pd.DataFrame(rows)
    out["_ord"] = out["bucket"].map({b: i for i, b in enumerate(order)})
    return out.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)


def payment_stats(
    df: pd.DataFrame,
    *,
    filial_scope: FilialScope = "pilot",
    amount_col: str | None = None,
    variant: int = 1,
) -> pd.DataFrame:
    base = analytics_base_mask(df, filial_scope=filial_scope)
    amount = _resolve_amount_col(df, amount_col)
    if variant == 1:
        segments = [
            ("model", model_rucheek_mask(df, filial_scope=filial_scope)),
            ("control", control_mask(df, filial_scope=filial_scope)),
            ("base_all", base),
        ]
    else:
        cases = variant2_case_masks(df, filial_scope=filial_scope)
        segments = list(cases.items()) + [("base_all", base)]
    rows = []
    for label, mask in segments:
        rows.append(
            {
                "segment": label,
                "amount_col": amount,
                "n": int(mask.sum()),
                **_money_stats(df.loc[mask, amount]),
            }
        )
    return pd.DataFrame(rows)


def case_diagnostics(
    df: pd.DataFrame,
    *,
    filial_scope: FilialScope = "pilot",
) -> pd.DataFrame:
    base = analytics_base_mask(df, filial_scope=filial_scope)
    cases = variant2_case_masks(df, filial_scope=filial_scope)
    filial_col = resolve_column(df, "filial")

    def _row(slice_name: str, filial: Any, mask_base: pd.Series) -> dict[str, Any]:
        return {
            "slice": slice_name,
            "filial": filial,
            "n_base": int(mask_base.sum()),
            "n_applied_one_paid": int((mask_base & cases["applied_one_paid"]).sum()),
            "n_ignored_zero_paid": int((mask_base & cases["ignored_zero_paid"]).sum()),
            "n_out_of_model_paid": int((mask_base & cases["out_of_model_paid"]).sum()),
            "n_recommended_unpaid": int((mask_base & cases["recommended_unpaid"]).sum()),
        }

    rows = [_row("total", None, base)]
    if filial_col is not None:
        for filial, idx in df.loc[base].groupby(filial_col, dropna=False).groups.items():
            rows.append(_row("filial", filial, base & df.index.isin(idx)))
    return pd.DataFrame(rows)


def build_variant_analytics(
    df: pd.DataFrame,
    *,
    amount_col: str | None = None,
    variant: int = 1,
) -> dict[str, pd.DataFrame]:
    amount = _resolve_amount_col(df, amount_col)
    return {
        "segments_bpilot": shares_as_percent(
            compare_path_shares(df, filial_scope="pilot", variant=variant)
        ),
        "filial_usage_all": shares_as_percent(filial_usage_stats(df, filial_scope="all")),
        "filial_shares_bpilot": shares_as_percent(
            filial_path_shares(df, filial_scope="pilot", variant=variant)
        ),
        "result_distribution_bpilot": shares_as_percent(
            result_distribution(df, filial_scope="pilot", amount_col=amount)
        ),
        "payments_bpilot": payment_stats(
            df, filial_scope="pilot", amount_col=amount, variant=variant
        ),
        "cases_bpilot": case_diagnostics(df, filial_scope="pilot"),
        "segments_bam": shares_as_percent(
            compare_path_shares(df, filial_scope="am", variant=variant)
        ),
        "filial_usage_bam": shares_as_percent(filial_usage_stats(df, filial_scope="am")),
        "filial_shares_bam": shares_as_percent(
            filial_path_shares(df, filial_scope="am", variant=variant)
        ),
        "result_distribution_bam": shares_as_percent(
            result_distribution(df, filial_scope="am", amount_col=amount)
        ),
        "payments_bam": payment_stats(
            df, filial_scope="am", amount_col=amount, variant=variant
        ),
        "cases_bam": case_diagnostics(df, filial_scope="am"),
    }
