"""Сравнение legacy vs icnl связок таргетов и баз фин. эффекта."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from querulus.fin_effect.calculator import (
    FinEffectResult,
    add_premiums_column,
    prepare_effect_frame,
    run_fin_effect_from_training,
)
from querulus.fin_effect.config import FinEffectConfig
from querulus.fin_effect.resolve import resolve_fin_effect_config
from querulus.fin_effect.summary import create_summary_table


def plain_floats(df: pd.DataFrame | None, decimals: int = 2) -> pd.DataFrame | None:
    """Округлить числовые колонки — без scientific notation в display."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in out.select_dtypes(include=["number"]).columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(decimals)
    return out


@dataclass(frozen=True)
class StackCompareReport:
    """Набор таблиц сравнения двух связок таргетов."""

    severity_targets: pd.DataFrame
    severity_predictions: pd.DataFrame | None
    fact_bases: pd.DataFrame
    premiums: pd.DataFrame
    model_quadrants_legacy: pd.DataFrame | None
    model_quadrants_new: pd.DataFrame | None
    summary_itogo_legacy: pd.DataFrame | None
    summary_itogo_new: pd.DataFrame | None
    legacy_result: FinEffectResult | None = None
    new_result: FinEffectResult | None = None


def compare_severity_targets(
    df: pd.DataFrame,
    *,
    legacy_col: str = "TARGET_3_SEV",
    new_col: str = "TARGET_SEV",
    key: str = "INCIDENT_NUMBER",
    top_n: int = 20,
) -> pd.DataFrame:
    """Сводка отличий фактических TARGET_3_SEV и TARGET_SEV."""
    if legacy_col not in df.columns or new_col not in df.columns:
        raise KeyError(f"Нужны колонки {legacy_col!r} и {new_col!r}")

    a = pd.to_numeric(df[legacy_col], errors="coerce").fillna(0)
    b = pd.to_numeric(df[new_col], errors="coerce").fillna(0)
    diff = b - a
    abs_diff = diff.abs()
    both_pos = (a > 0) | (b > 0)

    rows = [
        {"metric": "n", "value": float(len(df))},
        {"metric": f"{legacy_col}_sum", "value": float(a.sum())},
        {"metric": f"{new_col}_sum", "value": float(b.sum())},
        {"metric": "sum_diff_new_minus_legacy", "value": float(diff.sum())},
        {"metric": "sum_diff_pct_vs_legacy", "value": float(diff.sum() / a.sum()) if a.sum() else np.nan},
        {"metric": "exact_match_n", "value": float((abs_diff <= 1e-6).sum())},
        {"metric": "exact_match_pct", "value": float((abs_diff <= 1e-6).mean())},
        {"metric": "mismatch_n", "value": float((abs_diff > 1e-6).sum())},
        {"metric": "corr", "value": float(a.corr(b)) if len(df) > 1 else np.nan},
        {"metric": "mae", "value": float(abs_diff.mean())},
        {"metric": "mae_where_either_pos", "value": float(abs_diff[both_pos].mean()) if both_pos.any() else np.nan},
        {"metric": "median_abs_diff_where_mismatch", "value": float(abs_diff[abs_diff > 1e-6].median()) if (abs_diff > 1e-6).any() else 0.0},
    ]
    summary = pd.DataFrame(rows)

    top = pd.DataFrame(
        {
            key: df[key] if key in df.columns else df.index,
            legacy_col: a,
            new_col: b,
            "diff_new_minus_legacy": diff,
            "abs_diff": abs_diff,
        }
    )
    top = top.loc[abs_diff > 1e-6].sort_values("abs_diff", ascending=False).head(top_n)
    summary = plain_floats(summary)
    summary.attrs["top_mismatches"] = plain_floats(top)
    return summary


def compare_severity_predictions(
    *,
    y_true_legacy: pd.Series | np.ndarray,
    y_pred_legacy: pd.Series | np.ndarray,
    y_true_new: pd.Series | np.ndarray,
    y_pred_new: pd.Series | np.ndarray,
    label_legacy: str = "TARGET_3_SEV",
    label_new: str = "TARGET_SEV",
) -> pd.DataFrame:
    """Метрики прогнозов severity для двух таргетов на одном индексе строк."""
    yt_l = np.asarray(y_true_legacy, dtype=float)
    yp_l = np.asarray(y_pred_legacy, dtype=float)
    yt_n = np.asarray(y_true_new, dtype=float)
    yp_n = np.asarray(y_pred_new, dtype=float)
    if not (len(yt_l) == len(yp_l) == len(yt_n) == len(yp_n)):
        raise ValueError("Длины y_true/y_pred для двух связок должны совпадать")

    def _block(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
        err = y_pred - y_true
        return [
            {"stack": name, "metric": "n", "value": float(len(y_true))},
            {"stack": name, "metric": "y_true_sum", "value": float(np.nansum(y_true))},
            {"stack": name, "metric": "y_pred_sum", "value": float(np.nansum(y_pred))},
            {"stack": name, "metric": "bias_pred_minus_true", "value": float(np.nanmean(err))},
            {"stack": name, "metric": "mae", "value": float(np.nanmean(np.abs(err)))},
            {"stack": name, "metric": "rmse", "value": float(np.sqrt(np.nanmean(err**2)))},
            {
                "stack": name,
                "metric": "corr_pred_true",
                "value": float(pd.Series(y_true).corr(pd.Series(y_pred))),
            },
        ]

    rows = _block(f"legacy({label_legacy})", yt_l, yp_l) + _block(f"new({label_new})", yt_n, yp_n)
    rows.extend(
        [
            {
                "stack": "pred_vs_pred",
                "metric": "corr_pred_legacy_vs_pred_new",
                "value": float(pd.Series(yp_l).corr(pd.Series(yp_n))),
            },
            {
                "stack": "pred_vs_pred",
                "metric": "mae_pred_legacy_vs_pred_new",
                "value": float(np.nanmean(np.abs(yp_l - yp_n))),
            },
            {
                "stack": "pred_vs_pred",
                "metric": "sum_pred_legacy",
                "value": float(np.nansum(yp_l)),
            },
            {
                "stack": "pred_vs_pred",
                "metric": "sum_pred_new",
                "value": float(np.nansum(yp_n)),
            },
            {
                "stack": "pred_vs_pred",
                "metric": "sum_diff_new_minus_legacy",
                "value": float(np.nansum(yp_n - yp_l)),
            },
        ]
    )
    return plain_floats(pd.DataFrame(rows))


def compare_fact_bases(df: pd.DataFrame) -> pd.DataFrame:
    """Денежная база факта: legacy ПСР vs icnl TARGET_FREQ_AMOUNT."""
    legacy_parts = {
        "Сумма_выплат_по_претензиям": _col(df, "Сумма_выплат_по_претензиям"),
        "Сумма_взыскано_по_ФУ": _col(df, "Сумма_взыскано_по_ФУ"),
        "Суммы_взыскано_по_иску": _col(df, "Суммы_взыскано_по_иску"),
    }
    legacy_base = sum(legacy_parts.values())
    icnl_claims = _col(df, "TARGET_FREQ_CLAIMS_AMOUNT")
    icnl_pret = _col(df, "TARGET_FREQ_PRET_AMOUNT")
    icnl_amount = _col(df, "TARGET_FREQ_AMOUNT")
    if (icnl_amount == 0).all() and ((icnl_claims + icnl_pret) > 0).any():
        icnl_amount = icnl_claims + icnl_pret

    rows = [
        {"side": "legacy_psr", "part": name, "sum": float(series.sum()), "nonzero_n": int((series > 0).sum())}
        for name, series in legacy_parts.items()
    ]
    rows.append(
        {
            "side": "legacy_psr",
            "part": "BASE (= pret+FU+court)",
            "sum": float(legacy_base.sum()),
            "nonzero_n": int((legacy_base > 0).sum()),
        }
    )
    rows.extend(
        [
            {
                "side": "icnl",
                "part": "TARGET_FREQ_CLAIMS_AMOUNT",
                "sum": float(icnl_claims.sum()),
                "nonzero_n": int((icnl_claims > 0).sum()),
            },
            {
                "side": "icnl",
                "part": "TARGET_FREQ_PRET_AMOUNT",
                "sum": float(icnl_pret.sum()),
                "nonzero_n": int((icnl_pret > 0).sum()),
            },
            {
                "side": "icnl",
                "part": "BASE (= TARGET_FREQ_AMOUNT)",
                "sum": float(icnl_amount.sum()),
                "nonzero_n": int((icnl_amount > 0).sum()),
            },
            {
                "side": "diff",
                "part": "icnl_base - legacy_base",
                "sum": float((icnl_amount - legacy_base).sum()),
                "nonzero_n": int(((icnl_amount - legacy_base).abs() > 1e-6).sum()),
            },
            {
                "side": "diff",
                "part": "ratio_icnl_over_legacy",
                "sum": float(icnl_amount.sum() / legacy_base.sum()) if legacy_base.sum() else np.nan,
                "nonzero_n": np.nan,
            },
        ]
    )
    return plain_floats(pd.DataFrame(rows))


def compare_premiums(df: pd.DataFrame, config: FinEffectConfig | None = None) -> pd.DataFrame:
    """Взносы: тот же триггер ФУ из ПСР; суд. взнос по apply_court_fee."""
    config = config or FinEffectConfig()
    fu = _col(df, config.fu_fee_trigger_column)
    claims = _col(df, config.freq_claims_amount_column)
    premiums = add_premiums_column(df, config)
    fu_fee_rows = fu > 0
    court_fee_rows = (claims > 0) & bool(config.apply_court_fee)

    rows = [
        {
            "metric": "fu_trigger_col",
            "value": config.fu_fee_trigger_column,
            "note": "и legacy, и icnl используют эту колонку",
        },
        {"metric": "fu_fee_amount", "value": config.fu_fee_amount, "note": ""},
        {"metric": "fu_trigger_n", "value": int(fu_fee_rows.sum()), "note": "Сумма_взыскано_по_ФУ > 0"},
        {
            "metric": "fu_fee_total",
            "value": float(fu_fee_rows.sum() * config.fu_fee_amount),
            "note": "",
        },
        {"metric": "apply_court_fee", "value": int(config.apply_court_fee), "note": "по умолчанию 0"},
        {"metric": "court_fee_amount", "value": config.court_fee_amount, "note": ""},
        {"metric": "court_fee_n", "value": int(court_fee_rows.sum()), "note": "если apply_court_fee"},
        {
            "metric": "court_fee_total",
            "value": float(court_fee_rows.sum() * config.court_fee_amount),
            "note": "",
        },
        {"metric": "premiums_sum", "value": float(premiums.sum()), "note": "колонка Взносы"},
        {
            "metric": "legacy_fact_with_premiums",
            "value": float(
                (
                    _col(df, "Сумма_выплат_по_претензиям")
                    + _col(df, "Сумма_взыскано_по_ФУ")
                    + _col(df, "Суммы_взыскано_по_иску")
                    + premiums
                ).sum()
            ),
            "note": "legacy fin_effect_fact",
        },
        {
            "metric": "icnl_fact_with_premiums",
            "value": float((_col(df, "TARGET_FREQ_AMOUNT") + premiums).sum()),
            "note": "icnl fin_effect_fact",
        },
    ]
    for row in rows:
        if isinstance(row["value"], (int, float, np.integer, np.floating)) and not isinstance(
            row["value"], (bool, np.bool_)
        ):
            row["value"] = round(float(row["value"]), 2)
    return pd.DataFrame(rows)


def model_quadrant_breakdown(
    result: FinEffectResult,
    config: FinEffectConfig,
) -> pd.DataFrame:
    """Заполненная таблица pred × fact → fin_effect_model."""
    frame = result.frame
    freq = config.frequency_target_column
    sev = config.severity_target_column
    pred = frame["pred_freq"].astype(int)
    fact = pd.to_numeric(frame[freq], errors="coerce").fillna(0).astype(int)
    y_pred_sev = pd.to_numeric(frame["pred_sev"], errors="coerce").fillna(0)
    y_true_sev = pd.to_numeric(frame[sev], errors="coerce").fillna(0)
    # После negate_fact_for_report fact отрицательный; для формулы нужны |base| = -fact.
    fact_signed = pd.to_numeric(frame["fin_effect_fact"], errors="coerce").fillna(0)
    base_sum = -fact_signed if config.negate_fact_for_report else fact_signed
    model = pd.to_numeric(frame["fin_effect_model"], errors="coerce").fillna(0)

    rows: list[dict] = []
    specs = [
        (0, 0, "pred=0, fact=0", "-base_sum", None),
        (0, 1, "pred=0, fact=1", "-base_sum", None),
        (1, 0, "pred=1, fact=0", "-y_pred_sev - base_sum", None),
        (1, 1, "pred=1, fact=1, pred_sev≥true_sev", "-y_pred_sev", "over"),
        (1, 1, "pred=1, fact=1, pred_sev<true_sev", "-base_sum", "under"),
    ]
    for pred_v, fact_v, label, formula, branch in specs:
        mask = (pred == pred_v) & (fact == fact_v)
        if branch == "over":
            mask = mask & (y_pred_sev >= y_true_sev)
        elif branch == "under":
            mask = mask & (y_pred_sev < y_true_sev)
        g = frame.loc[mask]
        rows.append(
            {
                "pred": pred_v,
                "fact": fact_v,
                "case": label,
                "formula": formula,
                "n": int(mask.sum()),
                "sum_base_sum": float(base_sum.loc[mask].sum()),
                "sum_y_pred_sev": float(y_pred_sev.loc[mask].sum()),
                "sum_y_true_sev": float(y_true_sev.loc[mask].sum()),
                "sum_fin_effect_model": float(model.loc[mask].sum()),
            }
        )
    return plain_floats(pd.DataFrame(rows))


def summary_itogo_breakdown(
    result: FinEffectResult,
    config: FinEffectConfig,
) -> pd.DataFrame:
    """Таблица pred × fact → ИТОГО (как в create_summary_table).

    Порядок строк/колонок совпадает с model_quadrant_breakdown:
    pred, fact; строки (0,0) → (0,1) → (1,0) → (1,1).
    """
    summary = create_summary_table(result.frame, config)
    # Ключ (pred, fact) — как в model_quadrant_breakdown.
    mapping = {
        (0, 0): "0",
        (0, 1): "fin_effect_fact",
        (1, 0): "model − fact",
        (1, 1): "fin_effect_model",
    }
    out = summary.rename(
        columns={
            "Факт": "fact",
            "Классификация": "pred",
            "ИТОГО": "itogo",
            "ФИН. ЭФФЕКТ МОДЕЛЬ": "fin_effect_model",
            "ФИН. ЭФФЕКТ ФАКТ": "fin_effect_fact",
            "Количество инцидентов с иными взысканиями": "n",
        }
    )
    out["pred"] = out["pred"].astype(int)
    out["fact"] = out["fact"].astype(int)
    out = out.sort_values(["pred", "fact"], kind="mergesort").reset_index(drop=True)
    out["formula_itogo"] = [
        mapping[(int(r.pred), int(r.fact))] for r in out.itertuples(index=False)
    ]
    cols = [
        "pred",
        "fact",
        "formula_itogo",
        "n",
        "fin_effect_model",
        "fin_effect_fact",
        "itogo",
    ]
    return plain_floats(out[[c for c in cols if c in out.columns]])


def run_dual_stack_compare(
    df: pd.DataFrame,
    training_legacy: object,
    training_new: object,
    *,
    split: str = "test",
    legacy_freq: str = "TARGET_2",
    legacy_sev: str = "TARGET_3_SEV",
    new_freq: str = "TARGET_FREQ",
    new_sev: str = "TARGET_SEV",
    loaded_from_checkpoint: bool = True,
    legacy_dataset: bool | None = None,
) -> StackCompareReport:
    """Полное сравнение двух обученных связок + базы/взносы/severity-факты."""
    cfg_legacy = resolve_fin_effect_config(
        df,
        frequency_target=legacy_freq,
        severity_target=legacy_sev,
        loaded_from_checkpoint=loaded_from_checkpoint,
        legacy_dataset=legacy_dataset,
        fact_mode="legacy_psr",
    )
    cfg_new = resolve_fin_effect_config(
        df,
        frequency_target=new_freq,
        severity_target=new_sev,
        loaded_from_checkpoint=loaded_from_checkpoint,
        legacy_dataset=legacy_dataset,
        fact_mode="icnl",
    )

    legacy_res = run_fin_effect_from_training(
        df, training_legacy, split=split, config=cfg_legacy
    )
    new_res = run_fin_effect_from_training(
        df, training_new, split=split, config=cfg_new
    )

    # Предсказания на пересечении индексов test (если совпали сплиты по дате — индексы равны).
    sev_preds: pd.DataFrame | None
    try:
        idx = legacy_res.frame.index.intersection(new_res.frame.index)
        sev_preds = compare_severity_predictions(
            y_true_legacy=legacy_res.frame.loc[idx, legacy_sev],
            y_pred_legacy=legacy_res.frame.loc[idx, "pred_sev"],
            y_true_new=new_res.frame.loc[idx, new_sev],
            y_pred_new=new_res.frame.loc[idx, "pred_sev"],
            label_legacy=legacy_sev,
            label_new=new_sev,
        )
    except Exception:
        sev_preds = None

    return StackCompareReport(
        severity_targets=compare_severity_targets(df, legacy_col=legacy_sev, new_col=new_sev),
        severity_predictions=sev_preds,
        fact_bases=compare_fact_bases(df),
        premiums=compare_premiums(df, cfg_new),
        model_quadrants_legacy=model_quadrant_breakdown(legacy_res, cfg_legacy),
        model_quadrants_new=model_quadrant_breakdown(new_res, cfg_new),
        summary_itogo_legacy=summary_itogo_breakdown(legacy_res, cfg_legacy),
        summary_itogo_new=summary_itogo_breakdown(new_res, cfg_new),
        legacy_result=legacy_res,
        new_result=new_res,
    )


def fact_only_compare_report(df: pd.DataFrame) -> StackCompareReport:
    """Сравнение без обучения: факты severity, база, взносы."""
    # прогреть взносы / fact для прозрачности (не используется дальше напрямую)
    prepare_effect_frame(df, FinEffectConfig(fact_mode="icnl"))
    return StackCompareReport(
        severity_targets=compare_severity_targets(df),
        severity_predictions=None,
        fact_bases=compare_fact_bases(df),
        premiums=compare_premiums(df),
        model_quadrants_legacy=None,
        model_quadrants_new=None,
        summary_itogo_legacy=None,
        summary_itogo_new=None,
    )


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
