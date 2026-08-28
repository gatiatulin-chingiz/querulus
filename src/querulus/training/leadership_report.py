"""Короткий HTML для руководителя: старая vs новая модель, цифры из collect."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from querulus import PROJECT_ROOT
from querulus.fin_effect.calculator import FinEffectResult
from querulus.training.config import TrainingConfig
from querulus.training.feature_labels import feature_ru_name
from querulus.training.pipeline import TrainingArtifacts
from querulus.training.selected_features import (
    PROD_FREQUENCY_FEATURES,
    PROD_SEVERITY_FEATURES,
)
from querulus.training.splits import DateSplitParts, default_inner_periods_from_train
from querulus.training.stack_eval import StackEvalReport
from querulus.training.train_loop import TrainLoopResult
from querulus.training.triple_stack import TripleStackResult

DEFAULT_HTML_PATH = PROJECT_ROOT / "notebooks" / "leadership_models_report.html"
_FREQ_METRIC_COLS = ("stack", "split", "y_true", "n", "pr_auc", "roc_auc", "gini", "shift")
_SEV_METRIC_COLS = ("stack", "split", "y_true", "n", "mae", "rmse", "r2", "shift")
_SHARE_RENAME = {
    "stack": "модель",
    "n": "дел 1–1",
    "n_covered": "дел, где pred_sev ≥ TARGET_SEV",
    "share_rows_covered": "доля дел с полным покрытием",
    "share_amount_covered": "доля суммы TARGET_SEV, покрытая pred_sev",
    "share_under": "доля недобора суммы",
}

_CSS = """
:root {
  --bg: #f7f5f1; --surface: #fff; --ink: #1a1a1a; --muted: #5c5c5c;
  --line: #ddd8ce; --accent: #0d5c63; --accent-soft: #e6f2f3;
  --old: #6b4c7a; --old-soft: #f3edf5; --new: #1a6b4a; --new-soft: #e8f5ef;
  --warn: #9a6b00; --warn-soft: #fff8e6; --code-bg: #eeece7; --pending: #8a8680;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 13px;
  line-height: 1.35;
  color: var(--ink);
  background: var(--bg);
}
.page { max-width: 980px; margin: 0 auto; padding: 0.7rem 1rem 1.2rem; }
h1 { margin: 0 0 0.2rem; font-size: 1.35rem; color: var(--accent); }
.sub { margin: 0 0 0.45rem; color: var(--muted); font-size: 0.85rem; }
.meta-row { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-bottom: 0.55rem; }
.pill {
  font-size: 0.75rem; padding: 0.12rem 0.5rem; border-radius: 999px;
  background: var(--accent-soft); color: var(--accent); border: 1px solid #c5dfe1;
}
.pill.pending { background: var(--warn-soft); color: var(--warn); border-color: #e8d9a8; }
section {
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 6px; padding: 0.55rem 0.75rem 0.65rem; margin-bottom: 0.55rem;
}
h2 { margin: 0 0 0.35rem; font-size: 1rem; color: var(--accent); }
h3 { margin: 0.45rem 0 0.25rem; font-size: 0.9rem; }
p { margin: 0.25rem 0; }
.compare-hero { display: grid; grid-template-columns: 1fr 1fr; gap: 0.45rem; margin-bottom: 0.55rem; }
@media (max-width: 720px) { .compare-hero { grid-template-columns: 1fr; } }
.hero-card { border-radius: 6px; padding: 0.55rem 0.7rem; border: 1px solid var(--line); }
.hero-card.old { background: var(--old-soft); }
.hero-card.new { background: var(--new-soft); }
.hero-card h3 { margin: 0 0 0.3rem; }
.hero-card.old h3 { color: var(--old); }
.hero-card.new h3 { color: var(--new); }
.hero-card ul { margin: 0; padding-left: 1.05rem; }
.hero-card li { margin: 0.15rem 0; }
table { width: 100%; border-collapse: collapse; font-size: 0.8rem; margin: 0.3rem 0; }
th, td { border: 1px solid var(--line); padding: 0.28rem 0.4rem; text-align: left; vertical-align: top; }
th { background: var(--accent-soft); color: var(--accent); }
th.col-old { background: var(--old-soft); color: var(--old); }
th.col-new { background: var(--new-soft); color: var(--new); }
td.topic { font-weight: 600; white-space: nowrap; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.pending { color: var(--pending); font-style: italic; }
.kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.35rem; margin: 0.35rem 0; }
@media (max-width: 900px) { .kpi-row { grid-template-columns: 1fr 1fr; } }
.kpi { border: 1px solid var(--line); border-radius: 6px; padding: 0.4rem 0.5rem; background: #fafaf8; }
.kpi .k { font-size: 0.72rem; color: var(--muted); display: block; }
.kpi .v { font-weight: 700; }
.kpi .h { font-size: 0.72rem; color: var(--muted); }
.note { background: var(--warn-soft); border: 1px solid #e8d9a8; border-radius: 5px; padding: 0.4rem 0.55rem; font-size: 0.8rem; }
code { font-family: Consolas, "Courier New", monospace; font-size: 0.86em; background: var(--code-bg); padding: 0 0.2em; border-radius: 3px; }
.feat-list { margin: 0; padding-left: 1rem; }
.formula { font-family: Consolas, "Courier New", monospace; font-size: 0.78rem; }
@media print {
  body { background: #fff; font-size: 11px; }
  .page { max-width: none; padding: 0; }
  section, .hero-card { break-inside: avoid; }
}
"""


def _dash(value: Any) -> str:
    if value is None or value == "" or (isinstance(value, float) and np.isnan(value)):
        return '<span class="pending">—</span>'
    return escape(str(value))


def _int(value: int | float | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return '<span class="pending">—</span>'
    return f"{int(value):,}".replace(",", " ")


def _pct(value: float | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return '<span class="pending">—</span>'
    return f"{100 * float(value):.{digits}f}%"


def _pct100(value: float | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return '<span class="pending">—</span>'
    return f"{float(value):.{digits}f}%"


def _num(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return '<span class="pending">—</span>'
    try:
        number = float(value)
    except (TypeError, ValueError):
        return escape(str(value))
    if abs(number - round(number)) < 1e-9:
        return f"{int(round(number)):,}".replace(",", " ")
    if abs(number) >= 100:
        return f"{number:,.2f}".replace(",", " ")
    return f"{number:.{digits}f}"


def _rub(value: float | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return '<span class="pending">—</span>'
    number = int(round(float(value)))
    text = f"{abs(number):,}".replace(",", " ")
    return f"−{text} ₽" if number < 0 else f"{text} ₽"


def _period(start: str | None, end: str | None) -> str:
    if not start or not end:
        return "—"
    return f"{start} → {end}"


def _feature_list_html(names: Iterable[str]) -> str:
    items = list(names)
    if not items:
        return '<p class="pending">после блока B</p>'
    return (
        '<ul class="feat-list">'
        + "".join(
            f"<li><code>{escape(name)}</code> — {escape(feature_ru_name(name))}</li>"
            for name in items
        )
        + "</ul>"
    )


def _df_html(
    frame: pd.DataFrame | None,
    *,
    columns: Iterable[str] | None = None,
    rename: dict[str, str] | None = None,
    frac_cols: Iterable[str] = (),
    pct100_cols: Iterable[str] = (),
    money_cols: Iterable[str] = (),
    empty: str = "после прогона collect",
) -> str:
    if frame is None or getattr(frame, "empty", True):
        return f'<p class="pending">{escape(empty)}</p>'
    work = frame.copy()
    if columns:
        keep = [col for col in columns if col in work.columns]
        work = work[keep]
    if rename:
        work = work.rename(columns={old: new for old, new in rename.items() if old in work.columns})
        frac_cols = tuple(rename.get(col, col) for col in frac_cols)
        pct100_cols = tuple(rename.get(col, col) for col in pct100_cols)
        money_cols = tuple(rename.get(col, col) for col in money_cols)
    frac = set(frac_cols)
    pct100 = set(pct100_cols)
    money = set(money_cols)
    cols = list(work.columns)
    head = "".join(f"<th>{escape(str(col))}</th>" for col in cols)
    rows: list[str] = []
    for _, row in work.iterrows():
        cells: list[str] = []
        for col in cols:
            val = row[col]
            missing = val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val)
            if missing:
                cells.append('<td class="num pending">—</td>')
            elif col in money:
                cells.append(f'<td class="num">{_rub(float(val))}</td>')
            elif col in frac:
                cells.append(f'<td class="num">{_pct(float(val), 2)}</td>')
            elif col in pct100:
                cells.append(f'<td class="num">{_pct100(float(val), 2)}</td>')
            elif isinstance(val, (int, np.integer)):
                cells.append(f'<td class="num">{_int(int(val))}</td>')
            elif isinstance(val, (float, np.floating)):
                cells.append(f'<td class="num">{_num(float(val))}</td>')
            else:
                cells.append(f"<td>{escape(str(val))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _kpi(label: str, value: str, hint: str = "") -> str:
    hint_html = f'<span class="h">{escape(hint)}</span>' if hint else ""
    return (
        f'<div class="kpi"><span class="k">{escape(label)}</span>'
        f'<span class="v">{value}</span>{hint_html}</div>'
    )


@dataclass
class LeadershipReportData:
    generated_at: str
    df: pd.DataFrame | None = None
    config: TrainingConfig | None = None
    train_loop: TrainLoopResult | None = None
    triple: TripleStackResult | None = None
    stack_eval: StackEvalReport | None = None
    fin_effect_new: FinEffectResult | None = None
    fin_effect_legacy: FinEffectResult | None = None
    freq_metrics_at_05: pd.DataFrame | None = None
    freq_metrics_at_best: pd.DataFrame | None = None
    production_threshold: float | None = None
    production_net_effect: float | None = None
    subtitle: str = ""


def _new_training(data: LeadershipReportData) -> TrainingArtifacts | None:
    if data.train_loop is not None:
        return data.train_loop.training
    return None


def _holdout(data: LeadershipReportData) -> pd.DataFrame | None:
    if data.df is None:
        return None
    if data.train_loop is not None:
        return data.df.loc[data.train_loop.splits.test]
    return data.df


def _match_binary(frame: pd.DataFrame, left: str, right: str) -> tuple[int | None, float | None, float | None]:
    if left not in frame.columns or right not in frame.columns:
        return None, None, None
    a = pd.to_numeric(frame[left], errors="coerce").fillna(0).astype(int)
    b = pd.to_numeric(frame[right], errors="coerce").fillna(0).astype(int)
    n = int(len(frame))
    if n == 0:
        return 0, None, None
    exact = a == b
    row_pct = float(exact.mean())
    if "TARGET_FREQ_AMOUNT" in frame.columns:
        weights = pd.to_numeric(frame["TARGET_FREQ_AMOUNT"], errors="coerce").fillna(0).abs()
    else:
        legacy_cols = [
            c
            for c in (
                "Сумма_выплат_по_претензиям",
                "Сумма_взыскано_по_ФУ",
                "Суммы_взыскано_по_иску",
            )
            if c in frame.columns
        ]
        if legacy_cols:
            weights = frame[legacy_cols].apply(pd.to_numeric, errors="coerce").fillna(0).abs().sum(axis=1)
        else:
            weights = pd.Series(1.0, index=frame.index)
    w_sum = float(weights.sum())
    amt_pct = float(weights[exact].sum() / w_sum) if w_sum > 0 else None
    return n, row_pct, amt_pct


def _match_float(frame: pd.DataFrame, left: str, right: str) -> tuple[int | None, float | None, float | None]:
    if left not in frame.columns or right not in frame.columns:
        return None, None, None
    a = pd.to_numeric(frame[left], errors="coerce").fillna(0)
    b = pd.to_numeric(frame[right], errors="coerce").fillna(0)
    n = int(len(frame))
    if n == 0:
        return 0, None, None
    exact = (a - b).abs() <= 1e-6
    row_pct = float(exact.mean())
    weights = pd.concat([a.abs(), b.abs()], axis=1).max(axis=1)
    w_sum = float(weights.sum())
    amt_pct = float(weights[exact].sum() / w_sum) if w_sum > 0 else None
    return n, row_pct, amt_pct


def _status_pills(data: LeadershipReportData) -> str:
    flags = [
        ("датасет", data.df is not None),
        ("блок B", data.train_loop is not None),
        ("C2+", data.stack_eval is not None),
        ("C3", data.fin_effect_new is not None),
    ]
    html = [
        f'<span class="{"pill" if ok else "pill pending"}">{escape(name)}: {"есть" if ok else "нет"}</span>'
        for name, ok in flags
    ]
    html.append(f'<span class="pill">{escape(data.generated_at)}</span>')
    return "".join(html)


def _split_table(data: LeadershipReportData) -> str:
    cfg = data.config or TrainingConfig()
    splits: DateSplitParts | None = None if data.train_loop is None else data.train_loop.splits
    train_period = cfg.train_period
    test_period = cfg.test_period
    val_period = cfg.val_period
    cal_period = cfg.cal_period
    if val_period is None or cal_period is None:
        try:
            train_core, val_period, cal_period = default_inner_periods_from_train(train_period)
        except ValueError:
            train_core, val_period, cal_period = train_period, ("—", "—"), ("—", "—")
    else:
        train_core = train_period

    def _n(part: str) -> str:
        if splits is None:
            return _dash(None)
        return _int(len(getattr(splits, part)))

    rows = [
        ("Train", _period(*train_core), _n("train")),
        ("Val", _period(*val_period), _n("val")),
        ("Test", _period(*test_period), _n("test")),
    ]
    body = "".join(
        f"<tr><td class='topic'>{escape(name)}</td><td>{escape(period)}</td>"
        f"<td class='num'>{n}</td></tr>"
        for name, period, n in rows
    )
    return (
        "<table><thead><tr><th>Выборка</th><th>Период</th><th class='num'>Строк</th>"
        "</tr></thead><tbody>"
        f"{body}</tbody></table>"
        "<p>Старая модель: сплит по <code>LOSS_DATE_TIME</code>. "
        f"Новая: по <code>{escape(cfg.date_column)}</code>.</p>"
    )


def _targets_section(data: LeadershipReportData) -> str:
    holdout = _holdout(data)
    n_freq = pct_freq = pct_freq_amt = n_sev = pct_sev = pct_sev_amt = None
    if holdout is not None:
        n_freq, pct_freq, pct_freq_amt = _match_binary(holdout, "TARGET_2", "TARGET_FREQ")
        n_sev, pct_sev, pct_sev_amt = _match_float(holdout, "TARGET_3_SEV", "TARGET_SEV")
    return f"""
      <table>
        <thead>
          <tr>
            <th></th>
            <th class="col-old">Старая</th>
            <th class="col-new">Новая</th>
            <th>Совпадение на Test</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td class="topic">Frequency</td>
            <td>
              <code>TARGET_2</code> = 1[{{претензии + ФУ + иск}} &gt; 0]<br/>
              <code>Datamart.oisuu81_t_ПСР</code>
              + <code>dbo.oisuu81_t_Pretensions</code>
            </td>
            <td>
              <code>TARGET_FREQ</code> = 1[{{взыскано с госпошлиной на последней инстанции + претензии}} &gt; 0]<br/>
              <code>Datamart.oisuu81_t_IncomingClaimNewLogicByInst</code>
              + <code>dbo.oisuu81_t_Pretensions</code>
            </td>
            <td class="num">
              по строкам: {_pct(pct_freq, 2)}<br/>
              по суммам: {_pct(pct_freq_amt, 2)}<br/>
              <span class="pending">n={_int(n_freq)}</span>
            </td>
          </tr>
          <tr>
            <td class="topic">Severity</td>
            <td>
              <code>TARGET_3_SEV</code> = последнее ненулевое (ОД + УТС + износ) среди инстанций 1…5<br/>
              колонки <code>RECOVERED*</code> из той же таблицы исков
            </td>
            <td>
              <code>TARGET_SEV</code> = сумма (ОД + УТС + износ) на последней принятой инстанции + доплаты претензий<br/>
              та же таблица исков, инстанция по <code>ClaimedValuePeriod</code>
            </td>
            <td class="num">
              по строкам: {_pct(pct_sev, 2)}<br/>
              по суммам: {_pct(pct_sev_amt, 2)}<br/>
              <span class="pending">n={_int(n_sev)}</span>
            </td>
          </tr>
        </tbody>
      </table>
      <p>
        Совпадение по строкам — доля инцидентов с равными метками/значениями.
        Совпадение по суммам — доля веса <code>TARGET_FREQ_AMOUNT</code> (frequency)
        или max(|старое|, |новое|) (severity), на которой метки/значения совпали.
      </p>
      <p>
        <code>TARGET_3_SEV</code> и <code>TARGET_SEV</code> — одни компоненты (ОД + УТС + износ), но считаются иначе:
        в старой берётся последнее ненулевое значение по слотам инстанций 1…5;
        в новой — последняя принятая инстанция по <code>ClaimedValuePeriod</code>, плюс доплаты претензий (в старой их нет).
      </p>
    """


def _features_section(data: LeadershipReportData) -> str:
    new_tr = _new_training(data)
    new_freq = list(new_tr.frequency_features) if new_tr is not None else []
    new_sev = list(new_tr.severity_features) if new_tr is not None else []
    return f"""
      <table>
        <thead>
          <tr><th></th><th class="col-old">Старая (MVP)</th><th class="col-new">Новая (MVP → PSI → SHAP → noise-cut)</th></tr>
        </thead>
        <tbody>
          <tr>
            <td class="topic">Frequency n={_int(len(PROD_FREQUENCY_FEATURES))} / {_int(len(new_freq)) if new_freq else "—"}</td>
            <td>{_feature_list_html(PROD_FREQUENCY_FEATURES)}</td>
            <td>{_feature_list_html(new_freq)}</td>
          </tr>
          <tr>
            <td class="topic">Severity n={_int(len(PROD_SEVERITY_FEATURES))} / {_int(len(new_sev)) if new_sev else "—"}</td>
            <td>{_feature_list_html(PROD_SEVERITY_FEATURES)}</td>
            <td>{_feature_list_html(new_sev)}</td>
          </tr>
        </tbody>
      </table>
    """


def _metrics_section(data: LeadershipReportData) -> str:
    eval_report = data.stack_eval
    freq = None if eval_report is None else eval_report.frequency_metrics
    sev = None if eval_report is None else getattr(eval_report, "severity_metrics", None)
    return f"""
      <h3>Frequency, Test (C2+)</h3>
      {_df_html(freq, columns=_FREQ_METRIC_COLS, empty="нужен C2+")}
      <h3>Severity, Test (C2+)</h3>
      {_df_html(sev, columns=_SEV_METRIC_COLS, empty="нужен C2+")}
      <p><code>shift</code> для frequency = число предсказанных единиц / число фактических единиц; для severity = сумма прогноза / сумма факта.</p>
    """


def _overlap_section(data: LeadershipReportData) -> str:
    eval_report = data.stack_eval
    labels = None if eval_report is None else eval_report.label_agreement
    disagree = None if eval_report is None else eval_report.pred_freq_disagree
    share = None if eval_report is None else eval_report.coverage_share
    label_cols = (
        "pair",
        "common_n",
        "exact_match_n",
        "exact_match_pct",
        "exact_match_pct_by_amount",
    )
    return f"""
      <h3>TARGET_2 vs TARGET_FREQ</h3>
      {_df_html(
          labels,
          columns=label_cols,
          pct100_cols=("exact_match_pct", "exact_match_pct_by_amount"),
          empty="нужен C2+",
      )}
      <h3>pred_freq old vs new</h3>
      {_df_html(disagree, empty="нужен C2+")}
      <h3>Покрытие планки TARGET_SEV (строки 1–1, классификация new)</h3>
      {_df_html(
          share,
          rename=_SHARE_RENAME,
          frac_cols=("share_rows_covered", "share_amount_covered", "share_under"),
          empty="нужен C2+",
      )}
    """


def _fin_effect_section(data: LeadershipReportData) -> str:
    old_fe = data.fin_effect_legacy
    new_fe = data.fin_effect_new
    return f"""
      <div class="kpi-row">
        {_kpi("Чистый эффект, старый", _rub(None if old_fe is None else old_fe.net_effect), "блок A")}
        {_kpi("Расход модели, старый", _rub(None if old_fe is None else old_fe.model_effect_total), "")}
        {_kpi("Порог, новый", _dash(None if new_fe is None else f"{new_fe.best_threshold:.2f}"), "Val → Test")}
        {_kpi("Чистый эффект, новый", _rub(None if new_fe is None else new_fe.net_effect), "C3 Test")}
      </div>
      <div class="kpi-row">
        {_kpi("Расход факта, новый", _rub(None if new_fe is None else new_fe.fact_effect_total), "")}
        {_kpi("Расход модели, новый", _rub(None if new_fe is None else new_fe.model_effect_total), "")}
      </div>
      <table>
        <thead>
          <tr>
            <th>Ситуация</th>
            <th class="col-old">Старый</th>
            <th class="col-new">Новый</th>
          </tr>
        </thead>
        <tbody>
          <tr><td class="topic">0–0</td><td class="formula">−(ПСР + взносы)</td><td class="formula">0</td></tr>
          <tr><td class="topic">0–1</td><td class="formula">−pred_sev − (ПСР + взносы)</td><td class="formula">−pred_sev</td></tr>
          <tr><td class="topic">1–0</td><td class="formula">−(ПСР + взносы)</td><td class="formula">−(ПСР + взносы)</td></tr>
          <tr><td class="topic">1–1, pred ≥ T</td><td class="formula">−pred_sev</td><td class="formula">−pred_sev</td></tr>
          <tr>
            <td class="topic">1–1, pred &lt; T</td>
            <td class="formula">−(ПСР + взносы)</td>
            <td class="formula">−(ПСР × (1 − pred_sev / T) + взносы)</td>
          </tr>
        </tbody>
      </table>
      <p class="formula">чистый эффект = расход<sub>модель</sub> − расход<sub>факт</sub></p>
    """


def render_leadership_html(data: LeadershipReportData | None = None) -> str:
    data = data or LeadershipReportData(generated_at=datetime.now().strftime("%d.%m.%Y %H:%M"))
    n = None if data.df is None else int(len(data.df))
    n_html = _int(n) if n is not None else _dash(None)
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Querulus — старая vs новая модель</title>
  <style>{_CSS}</style>
</head>
<body>
<div class="page">
  <h1>Querulus: старая vs новая модель</h1>
  <p class="sub">Сравнение прод Litigant и кандидата Querulus. Цифры — Test collect. Строк df: {n_html}.</p>
  <div class="meta-row">{_status_pills(data)}</div>

  <div class="compare-hero">
    <div class="hero-card old">
      <h3>Старая (прод)</h3>
      <ul>
        <li><code>TARGET_2</code>; <code>TARGET_3_SEV</code> = ОД+УТС+износ</li>
        <li>Факт денег: претензии + ФУ + иск + взносы</li>
        <li>Фичи: MVP</li>
        <li>Обучение: без HPO</li>
        <li>Сплит: <code>LOSS_DATE_TIME</code></li>
      </ul>
    </div>
    <div class="hero-card new">
      <h3>Новая (кандидат)</h3>
      <ul>
        <li><code>TARGET_FREQ</code>; <code>TARGET_SEV</code> = ОД+УТС+износ</li>
        <li>Факт денег: претензии + ФУ + иск + взносы</li>
        <li>Фичи: MVP → PSI → SHAP → noise-cut</li>
        <li>Обучение: с HPO</li>
        <li>Сплит: <code>PAYMENT_ORDER_DATE_TIME</code></li>
      </ul>
    </div>
  </div>

  <section>
    <h2>1. Таргеты</h2>
    {_targets_section(data)}
  </section>
  <section>
    <h2>2. Выборки</h2>
    {_split_table(data)}
  </section>
  <section>
    <h2>3. Фичи</h2>
    {_features_section(data)}
  </section>
  <section>
    <h2>4. Метрики C2+</h2>
    {_metrics_section(data)}
  </section>
  <section>
    <h2>5. Пересечение решений</h2>
    {_overlap_section(data)}
  </section>
  <section>
    <h2>6. Финансовый эффект</h2>
    {_fin_effect_section(data)}
  </section>
</div>
</body>
</html>
"""


def collect_report_data(
    *,
    df: pd.DataFrame | None = None,
    config: TrainingConfig | None = None,
    train_loop: TrainLoopResult | None = None,
    triple: TripleStackResult | None = None,
    stack_eval: StackEvalReport | None = None,
    fin_effect_new: FinEffectResult | None = None,
    fin_effect_legacy: FinEffectResult | None = None,
    freq_metrics_at_05: pd.DataFrame | None = None,
    freq_metrics_at_best: pd.DataFrame | None = None,
    production_threshold: float | None = None,
    production_net_effect: float | None = None,
    subtitle: str = "",
) -> LeadershipReportData:
    if fin_effect_legacy is None and triple is not None and triple.fin_effects:
        fin_effect_legacy = triple.fin_effects.get("legacy")
    return LeadershipReportData(
        generated_at=datetime.now().strftime("%d.%m.%Y %H:%M"),
        df=df,
        config=config,
        train_loop=train_loop,
        triple=triple,
        stack_eval=stack_eval,
        fin_effect_new=fin_effect_new,
        fin_effect_legacy=fin_effect_legacy,
        freq_metrics_at_05=freq_metrics_at_05,
        freq_metrics_at_best=freq_metrics_at_best,
        production_threshold=production_threshold,
        production_net_effect=production_net_effect,
        subtitle=subtitle,
    )


def export_leadership_html(
    data: LeadershipReportData | None = None,
    *,
    path: str | Path | None = None,
) -> Path:
    output = Path(path) if path is not None else DEFAULT_HTML_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_leadership_html(data), encoding="utf-8")
    return output


def export_leadership_html_from_collect(
    namespace: dict[str, Any],
    *,
    path: str | Path | None = None,
) -> Path:
    triple = namespace.get("triple")
    fe_prod = namespace.get("fe_prod_cal")
    prod_thr = namespace.get("production_training_best_threshold")
    data = collect_report_data(
        df=namespace.get("df"),
        config=namespace.get("loop_cfg") or namespace.get("TRAINING_CONFIG"),
        train_loop=namespace.get("train_loop_result"),
        triple=triple if isinstance(triple, TripleStackResult) else None,
        stack_eval=namespace.get("stack_eval_b"),
        fin_effect_new=namespace.get("fin_effect_b"),
        freq_metrics_at_05=namespace.get("freq_metrics_at_05"),
        freq_metrics_at_best=namespace.get("freq_metrics_at_best"),
        production_threshold=None if prod_thr is None else float(prod_thr),
        production_net_effect=None if fe_prod is None else float(fe_prod.net_effect),
    )
    return export_leadership_html(data, path=path)
