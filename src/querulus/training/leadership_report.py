"""HTML-отчёт для руководителя: изменения с июня 2026 и сравнение моделей.

Без артефактов collect — костяк с «—». После прогона collect подставляются
выборки, фичи, метрики, покрытие регрессий и фин. эффект.
"""
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

_CSS = """
:root {
  --bg: #f7f5f1; --surface: #fff; --ink: #1a1a1a; --muted: #5c5c5c;
  --line: #ddd8ce; --accent: #0d5c63; --accent-soft: #e6f2f3;
  --old: #6b4c7a; --old-soft: #f3edf5; --new: #1a6b4a; --new-soft: #e8f5ef;
  --warn: #9a6b00; --warn-soft: #fff8e6; --code-bg: #eeece7;
  --pending: #8a8680;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 15px;
  line-height: 1.5;
  color: var(--ink);
  background:
    radial-gradient(ellipse at 8% 0%, #e8f0ef 0%, transparent 42%),
    radial-gradient(ellipse at 92% 8%, #f0e9df 0%, transparent 38%),
    var(--bg);
}
header, main, nav, footer {
  max-width: 1080px;
  margin-left: auto;
  margin-right: auto;
  padding-left: 1.25rem;
  padding-right: 1.25rem;
}
header { padding-top: 2rem; padding-bottom: 1rem; }
header h1 {
  margin: 0 0 0.35rem;
  font-size: 1.7rem;
  color: var(--accent);
  letter-spacing: -0.02em;
}
header .subtitle { margin: 0 0 1rem; color: var(--muted); max-width: 54rem; }
.meta-row { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.pill {
  display: inline-block;
  font-size: 0.82rem;
  padding: 0.2rem 0.65rem;
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--accent);
  border: 1px solid #c5dfe1;
}
.pill.pending { background: var(--warn-soft); color: var(--warn); border-color: #e8d9a8; }
nav {
  display: flex; flex-wrap: wrap; gap: 0.5rem 1rem;
  margin-bottom: 1rem; position: sticky; top: 0; z-index: 10;
  background: rgba(247, 245, 241, 0.92);
  backdrop-filter: blur(6px);
  padding-top: 0.5rem; padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--line);
}
nav a { color: var(--accent); text-decoration: none; font-size: 0.9rem; }
nav a:hover { text-decoration: underline; }
main { padding-bottom: 2rem; }
section {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 1rem 1.2rem 1.15rem;
  margin-bottom: 1rem;
}
h2 {
  margin: 0 0 0.65rem;
  font-size: 1.15rem;
  color: var(--accent);
  border-bottom: 2px solid var(--accent-soft);
  padding-bottom: 0.3rem;
}
h3 { margin: 1rem 0 0.4rem; font-size: 1rem; }
p { margin: 0.4rem 0; }
ul { margin: 0.35rem 0 0.7rem; padding-left: 1.15rem; }
li { margin: 0.3rem 0; }
.lead-box {
  background: linear-gradient(135deg, var(--accent-soft) 0%, #f8faf9 100%);
  border: 1px solid #b8d8db;
  border-radius: 8px;
  padding: 1rem 1.15rem;
  margin-bottom: 1rem;
}
.lead-box strong { color: var(--accent); }
.compare-hero {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.75rem;
  margin-bottom: 1rem;
}
@media (max-width: 720px) { .compare-hero { grid-template-columns: 1fr; } }
.hero-card {
  border-radius: 8px;
  padding: 1rem 1.1rem;
  border: 1px solid var(--line);
}
.hero-card.old { background: var(--old-soft); border-color: #d8c9df; }
.hero-card.new { background: var(--new-soft); border-color: #b8dcc8; }
.hero-card h3 { margin: 0 0 0.5rem; font-size: 1.05rem; }
.hero-card.old h3 { color: var(--old); }
.hero-card.new h3 { color: var(--new); }
.hero-card ul { margin-bottom: 0; font-size: 0.92rem; }
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.86rem;
  margin: 0.5rem 0;
}
th, td {
  border: 1px solid var(--line);
  padding: 0.45rem 0.6rem;
  text-align: left;
  vertical-align: top;
}
th { background: var(--accent-soft); color: var(--accent); font-weight: 600; }
th.col-old { background: var(--old-soft); color: var(--old); }
th.col-new { background: var(--new-soft); color: var(--new); }
tr:nth-child(even) td { background: #fafaf8; }
td.topic { font-weight: 600; width: 18%; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
td.pending, .pending { color: var(--pending); font-style: italic; }
.kpi-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.55rem;
  margin: 0.75rem 0;
}
@media (max-width: 900px) { .kpi-row { grid-template-columns: 1fr 1fr; } }
.kpi {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 0.65rem 0.75rem;
  background: #fafaf8;
}
.kpi .k { font-size: 0.78rem; color: var(--muted); display: block; }
.kpi .v { font-size: 1.05rem; font-weight: 700; }
.kpi .h { font-size: 0.78rem; color: var(--muted); }
.note {
  background: var(--warn-soft);
  border: 1px solid #e8d9a8;
  border-radius: 6px;
  padding: 0.65rem 0.85rem;
  font-size: 0.9rem;
  margin-top: 0.65rem;
}
.note strong { color: var(--warn); }
.timeline { list-style: none; padding: 0; margin: 0.4rem 0; }
.timeline li {
  display: grid;
  grid-template-columns: 7.5rem 1fr;
  gap: 0.75rem;
  padding: 0.55rem 0;
  border-bottom: 1px solid var(--line);
}
.timeline .when { font-weight: 700; color: var(--accent); }
.feat-list { margin: 0; padding-left: 1.1rem; font-size: 0.86rem; }
.feat-list code { font-size: 0.82em; }
.delta-add { color: var(--new); }
.delta-drop { color: #8b3a3a; text-decoration: line-through; }
code {
  font-family: Consolas, "Courier New", monospace;
  font-size: 0.86em;
  background: var(--code-bg);
  padding: 0.05em 0.3em;
  border-radius: 3px;
}
footer {
  padding-bottom: 2rem;
  font-size: 0.82rem;
  color: var(--muted);
  text-align: center;
}
@media print {
  body { background: #fff; }
  nav { position: static; }
  section, .lead-box, .hero-card { break-inside: avoid; }
}
"""


def _dash(value: Any) -> str:
    if value is None or value == "":
        return '<span class="pending">—</span>'
    if isinstance(value, float) and np.isnan(value):
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


def _feature_item(name: str, *, kind: str = "") -> str:
    ru = feature_ru_name(name)
    cls = f" class=\"{kind}\"" if kind else ""
    return f"<li{cls}><code>{escape(name)}</code> — {escape(ru)}</li>"


def _feature_list_html(names: Iterable[str], *, kind: str = "") -> str:
    items = list(names)
    if not items:
        return '<p class="pending">Список появится после прогона collect (блок B).</p>'
    return "<ul class=\"feat-list\">" + "".join(_feature_item(n, kind=kind) for n in items) + "</ul>"


def _df_html(
    frame: pd.DataFrame | None,
    *,
    money_cols: Iterable[str] = (),
    pct_cols: Iterable[str] = (),
    empty: str = "Таблица появится после прогона collect.",
) -> str:
    if frame is None or getattr(frame, "empty", True):
        return f'<p class="pending">{escape(empty)}</p>'
    money = set(money_cols)
    pct = set(pct_cols)
    cols = list(frame.columns)
    head = "".join(f"<th>{escape(str(c))}</th>" for c in cols)
    rows: list[str] = []
    for _, row in frame.iterrows():
        cells: list[str] = []
        for col in cols:
            val = row[col]
            if col in money:
                cells.append(f'<td class="num">{_rub(None if pd.isna(val) else float(val))}</td>')
            elif col in pct:
                cells.append(f'<td class="num">{_pct(None if pd.isna(val) else float(val), 2)}</td>')
            elif isinstance(val, (int, np.integer)):
                cells.append(f'<td class="num">{_int(int(val))}</td>')
            elif isinstance(val, (float, np.floating)):
                cells.append(f'<td class="num">{_num(float(val))}</td>')
            else:
                cells.append(f"<td>{escape(str(val)) if val is not None else '—'}</td>")
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
    """Снимок прогона collect для HTML. Все поля опциональны."""

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


def _legacy_training(data: LeadershipReportData) -> TrainingArtifacts | None:
    triple = data.triple
    if triple is None:
        return None
    return triple.trainings.get("legacy")


def _new_training(data: LeadershipReportData) -> TrainingArtifacts | None:
    if data.train_loop is not None:
        return data.train_loop.training
    return None


def _dataset_kpis(data: LeadershipReportData) -> str:
    df = data.df
    cfg = data.config or TrainingConfig()
    date_col = cfg.date_column
    n = None if df is None else int(len(df))
    date_min = date_max = None
    rate_2 = rate_f = None
    if df is not None and date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        if dates.notna().any():
            date_min = dates.min().strftime("%Y-%m-%d")
            date_max = dates.max().strftime("%Y-%m-%d")
    if df is not None:
        if "TARGET_2" in df.columns:
            rate_2 = float(pd.to_numeric(df["TARGET_2"], errors="coerce").fillna(0).mean())
        if "TARGET_FREQ" in df.columns:
            rate_f = float(pd.to_numeric(df["TARGET_FREQ"], errors="coerce").fillna(0).mean())
    return (
        '<div class="kpi-row">'
        + _kpi("Строк в датасете", _int(n) if n is not None else _dash(None), "после сборки / checkpoint")
        + _kpi("Окно поручений", _period(date_min, date_max), date_col)
        + _kpi("Доля TARGET_2", _pct(rate_2), "старая метка сутяжника")
        + _kpi("Доля TARGET_FREQ", _pct(rate_f), "новая метка доплаты")
        + "</div>"
    )


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
        index = getattr(splits, part, None)
        if index is None:
            return _dash(None)
        return _int(len(index))

    rows = [
        ("Train", _period(*train_core), _n("train"), "обучение frequency / severity"),
        ("Val", _period(*val_period), _n("val"), "early-stop, HPO, порог финэффекта"),
        ("Cal", _period(*cal_period), _n("cal"), "калибровка вероятности"),
        ("Test (holdout)", _period(*test_period), _n("test"), "C2+, C3, отчёт для решения"),
    ]
    body = "".join(
        f"<tr><td class='topic'>{escape(name)}</td>"
        f"<td>{escape(period)}</td>"
        f"<td class='num'>{n}</td>"
        f"<td>{escape(role)}</td></tr>"
        for name, period, n, role in rows
    )
    return (
        "<table><thead><tr>"
        "<th>Выборка</th><th>Период</th><th class='num'>Строк</th><th>Зачем</th>"
        "</tr></thead><tbody>"
        f"{body}</tbody></table>"
        f"<p>Якорь сплита: <code>{escape(cfg.date_column)}</code>. "
        "Старая прод-модель училась без отдельного Val (train + test по дате убытка).</p>"
    )


def _features_section(data: LeadershipReportData) -> str:
    new_tr = _new_training(data)
    old_freq = list(PROD_FREQUENCY_FEATURES)
    old_sev = list(PROD_SEVERITY_FEATURES)
    new_freq = list(new_tr.frequency_features) if new_tr is not None else []
    new_sev = list(new_tr.severity_features) if new_tr is not None else []

    def _delta(old: list[str], new: list[str]) -> str:
        if not new:
            return '<p class="pending">Дельта фич появится после блока B.</p>'
        added = [f for f in new if f not in old]
        dropped = [f for f in old if f not in new]
        kept = [f for f in new if f in old]
        bits = [
            f"осталось из прода: <strong>{len(kept)}</strong>",
            f"добавлено: <strong>{len(added)}</strong>",
            f"убрано из прода: <strong>{len(dropped)}</strong>",
        ]
        html = "<p>" + "; ".join(bits) + ".</p>"
        if added:
            html += "<p>Добавлены:</p>" + _feature_list_html(added, kind="delta-add")
        if dropped:
            html += "<p>Убраны из прод-пула:</p>" + _feature_list_html(dropped, kind="delta-drop")
        return html

    n_new_f = _int(len(new_freq)) if new_freq else _dash(None)
    n_new_s = _int(len(new_sev)) if new_sev else _dash(None)
    return f"""
      <div class="kpi-row">
        {_kpi("Freq, старая", _int(len(old_freq)), "фиксированный прод-пул")}
        {_kpi("Freq, новая", n_new_f, "после FS / HPO")}
        {_kpi("Sev, старая", _int(len(old_sev)), "в т.ч. номинал AMOUNT_REPAIR")}
        {_kpi("Sev, новая", n_new_s, "часто VALUE_BEFORE в ценах 2022")}
      </div>
      <table>
        <thead>
          <tr><th>Задача</th><th class="col-old">Старая (прод)</th><th class="col-new">Новая (кандидат)</th></tr>
        </thead>
        <tbody>
          <tr>
            <td class="topic">Frequency</td>
            <td>{_feature_list_html(old_freq)}</td>
            <td>{_feature_list_html(new_freq)}</td>
          </tr>
          <tr>
            <td class="topic">Severity</td>
            <td>{_feature_list_html(old_sev)}</td>
            <td>{_feature_list_html(new_sev)}</td>
          </tr>
        </tbody>
      </table>
      <h3>Что изменилось в составе фич</h3>
      <h4>Frequency</h4>
      {_delta(old_freq, new_freq)}
      <h4>Severity</h4>
      {_delta(old_sev, new_sev)}
    """


def _metrics_section(data: LeadershipReportData) -> str:
    eval_report = data.stack_eval
    freq_cmp = None if eval_report is None else eval_report.frequency_metrics
    new_tr = _new_training(data)
    sev_table = None if new_tr is None else new_tr.severity_metrics_table
    return f"""
      <p>
        Классификация в C2+: порог 0.5. Каждая модель на <strong>своей</strong> метке.
        Строка «legacy @ TARGET_FREQ» — старые фичи и гиперпараметры, переобученные на новую метку:
        отделяет эффект таргета от эффекта новой модели.
      </p>
      <h3>Frequency на holdout (C2+)</h3>
      {_df_html(freq_cmp, empty="Нужны блок A (legacy) и блок B+C2+.")}
      <h3>Frequency новой модели @0.5 и @пороге финэффекта</h3>
      <p>Порог 0.5</p>
      {_df_html(data.freq_metrics_at_05, empty="Появится после C3.")}
      <p>Лучший порог Val</p>
      {_df_html(data.freq_metrics_at_best, empty="Появится после C3.")}
      <h3>Severity новой модели (raw fit)</h3>
      {_df_html(sev_table, empty="Появится после fit блока B.")}
    """


def _overlap_section(data: LeadershipReportData) -> str:
    eval_report = data.stack_eval
    labels = None if eval_report is None else eval_report.label_agreement
    disagree = None if eval_report is None else eval_report.pred_freq_disagree
    coverage = None if eval_report is None else eval_report.coverage
    share = None if eval_report is None else eval_report.coverage_share
    return f"""
      <p>
        Регрессии сравниваем не «каждая на своих pred_freq» — иначе наборы строк разные и MAE несравним.
        Обе severity считаем на <strong>классификации новой модели</strong>.
        Доля планки — сколько <code>TARGET_SEV</code> покрыто <code>pred_sev</code> на строках 1–1.
      </p>
      <h3>Сверка меток TARGET_2 vs TARGET_FREQ (holdout)</h3>
      {_df_html(labels, pct_cols=("exact_match_pct",), empty="Нужен C2+.")}
      <h3>Расхождение pred_freq старой и новой классификации</h3>
      {_df_html(disagree, empty="Нужен C2+.")}
      <h3>Покрытие ₽: обе регрессии, классификация new</h3>
      {_df_html(coverage, money_cols=("amount",), empty="Нужен C2+.")}
      <h3>Доля покрытой планки TARGET_SEV (строки 1–1)</h3>
      {_df_html(
          share,
          pct_cols=("share_rows_covered", "share_amount_covered", "share_under"),
          empty="Нужен C2+.",
      )}
      <p class="note">
        <strong>Как читать:</strong> строка <code>new − legacy</code> в доле планки — выигрыш новой severity
        при одинаковых решениях «платить / не платить». Квадранты 0–1 / 1–0 показывают ложную тревогу и пропуск.
      </p>
    """


def _fin_effect_section(data: LeadershipReportData) -> str:
    new_fe = data.fin_effect_new
    old_fe = data.fin_effect_legacy
    return f"""
      <div class="kpi-row">
        {_kpi("Порог new (Val→Test)", _dash(None if new_fe is None else f"{new_fe.best_threshold:.2f}"), "C3 holdout")}
        {_kpi("Чистый эффект new", _rub(None if new_fe is None else new_fe.net_effect), "model − fact")}
        {_kpi("Чистый эффект legacy", _rub(None if old_fe is None else old_fe.net_effect), "блок A, свои правила факта")}
        {_kpi("Порог прода", _dash(None if data.production_threshold is None else f"{data.production_threshold:.2f}"), "cal freq + cal sev")}
      </div>
      <div class="kpi-row">
        {_kpi("Расход факта new", _rub(None if new_fe is None else new_fe.fact_effect_total), "holdout Test, icnl")}
        {_kpi("Расход модели new", _rub(None if new_fe is None else new_fe.model_effect_total), "coverage / icnl")}
        {_kpi("Расход факта legacy", _rub(None if old_fe is None else old_fe.fact_effect_total), "блок A, ПСР")}
        {_kpi("Чистый эффект прода", _rub(data.production_net_effect), "на cal_prod, если PROD прогнан")}
      </div>
      <h3>Формулы расхода модели (знак минус = расход)</h3>
      <table>
        <thead>
          <tr>
            <th>Ситуация</th>
            <th class="col-old">Старый (ПСР / legacy)</th>
            <th class="col-new">Новый (coverage / icnl)</th>
          </tr>
        </thead>
        <tbody>
          <tr><td class="topic">0–0 нет→нет</td><td>−(ПСР + взносы)</td><td>0</td></tr>
          <tr><td class="topic">0–1 ложная тревога</td><td>−pred_sev − (ПСР + взносы)</td><td>−pred_sev</td></tr>
          <tr><td class="topic">1–0 пропуск</td><td>−(ПСР + взносы)</td><td>−(ПСР + взносы)</td></tr>
          <tr><td class="topic">1–1 хватило</td><td>−pred_sev</td><td>−pred_sev</td></tr>
          <tr><td class="topic">1–1 не хватило</td><td>−(ПСР + взносы)</td>
            <td>−(ПСР × (1 − pred_sev / TARGET_SEV) + взносы)</td></tr>
        </tbody>
      </table>
      <p class="note">
        <strong>Не сравнивайте рубли C3 new и блок A legacy напрямую.</strong>
        У них разные базы факта и разные формулы. Для решения «какая модель лучше»
        смотрите C2+ (метрики + покрытие + доля планки) на одном holdout.
        Детальный разбор формул: <code>fin_effect_detailed.html</code>.
      </p>
    """


def _status_pills(data: LeadershipReportData) -> str:
    has_df = data.df is not None
    has_b = data.train_loop is not None
    has_c2 = data.stack_eval is not None
    has_c3 = data.fin_effect_new is not None
    has_prod = data.production_threshold is not None
    pills = [
        ("датасет", has_df),
        ("блок B", has_b),
        ("C2+", has_c2),
        ("C3 финэффект", has_c3),
        ("PROD", has_prod),
    ]
    html: list[str] = []
    for label, ok in pills:
        cls = "pill" if ok else "pill pending"
        mark = "есть" if ok else "нет прогона"
        html.append(f'<span class="{cls}">{escape(label)}: {mark}</span>')
    html.append(f'<span class="pill">{escape(data.generated_at)}</span>')
    html.append('<span class="pill">collect.ipynb</span>')
    return "".join(html)


def render_leadership_html(data: LeadershipReportData | None = None) -> str:
    """Полный HTML. Без data — костяк с пустыми цифрами."""
    data = data or LeadershipReportData(generated_at=datetime.now().strftime("%d.%m.%Y %H:%M"))
    sub = data.subtitle or (
        "Укрупнённо: что изменилось с июня 2026 и чем новая модель Querulus отличается от прод Litigant."
    )
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Querulus — отчёт для руководителя: старая vs новая модель</title>
  <style>{_CSS}</style>
</head>
<body>
  <header>
    <h1>Querulus: что изменилось с июня 2026</h1>
    <p class="subtitle">{escape(sub)}</p>
    <div class="meta-row">{_status_pills(data)}</div>
  </header>
  <nav>
    <a href="#summary">Выжимка</a>
    <a href="#changes">С июня</a>
    <a href="#models">Модели</a>
    <a href="#samples">Выборки</a>
    <a href="#targets">Таргеты</a>
    <a href="#features">Фичи</a>
    <a href="#metrics">Метрики</a>
    <a href="#overlap">Регрессии</a>
    <a href="#fineffect">Финэффект</a>
  </nav>
  <main>
    <div class="lead-box" id="summary">
      <strong>Главное.</strong>
      С июня 2026 моделирование вынесено из Litigant в Querulus: новые таргеты из судебных требований
      и доплат, датасет по дате поручения на выплату с вызреванием, рублёвые фичи в ценах 2022,
      отбор фич и HPO, калибровка, единая формула финэффекта с покрытием планки.
      Решение «переходить ли на новую модель» — по блоку C2+ на holdout, а не по черновым метрикам блока A.
    </div>

    <div class="compare-hero">
      <div class="hero-card old">
        <h3>Старая (Litigant / прод)</h3>
        <ul>
          <li>Частота: <code>TARGET_2</code> — сутяжник по ПСР</li>
          <li>Сумма: <code>TARGET_3_SEV</code> — ОД + УТС + износ</li>
          <li>Факт денег: претензии + ФУ + иск + взносы</li>
          <li>Фичи: фиксированный пул (~10 freq, ~4 sev)</li>
          <li>Обучение: CatBoost 375 / 100 iter, без FS/HPO в проде</li>
          <li>Окно: дата убытка с 2022</li>
        </ul>
      </div>
      <div class="hero-card new">
        <h3>Новая (Querulus / кандидат)</h3>
        <ul>
          <li>Частота: <code>TARGET_FREQ</code> — будет ли доплата после УУ</li>
          <li>Сумма: <code>TARGET_SEV</code> — сумма доплаты, ₽</li>
          <li>Факт денег: <code>TARGET_FREQ_AMOUNT</code> + взносы ФУ</li>
          <li>Фичи: MVP → PSI → SHAP → noise-cut → HPO</li>
          <li>Обучение: блок B + калибровка freq/sev</li>
          <li>Окно: поручение на выплату с 2020, вызревание H мес.</li>
        </ul>
      </div>
    </div>

    <section id="changes">
      <h2>1. Что сделали с июня 2026 (укрупнённо)</h2>
      <ul class="timeline">
        <li>
          <span class="when">Июнь</span>
          <span>Вынесли Querulus в отдельный пайплайн: сборка датасета, кэш raw parquet, первый train frequency/severity в <code>collect</code>. Старт истории моделирования в этом репозитории — 26 июня.</span>
        </li>
        <li>
          <span class="when">Начало июля</span>
          <span>Новая постановка задачи: метки из судебных требований (icnl), не из ПСР. Появились <code>TARGET_FREQ</code> / <code>TARGET_SEV</code>, факт финэффекта на сумме требований, фичи <code>FE_*</code>, мотор-only victim, вызревание от даты поручения.</span>
        </li>
        <li>
          <span class="when">Середина июля</span>
          <span>Честный as-of: история претензий/судов на дату поручения, дефляция в цены 2022, person-фичи, отсев leakage. Сверка старых и новых меток. Три стека: legacy / new / new_claims.</span>
        </li>
        <li>
          <span class="when">Конец июля — август</span>
          <span>Прод-контур обучения: отбор фич, HPO (Optuna/MLflow), калибровка вероятности и severity, порог финэффекта на Val → отчёт на Test. Честное сравнение C2+: свои фичи у каждой модели, обе регрессии на классификации new.</span>
        </li>
        <li>
          <span class="when">Август</span>
          <span>Формула покрытия планки вместо «всё или ничего». Выгрузка в OutBoxML (версия <code>2_*</code>) для сервиса. HTML-отчёты для бизнеса. Production-refit с калибровкой на свежем хвосте.</span>
        </li>
      </ul>
      <p class="note">
        <strong>Не путать с дневным журналом.</strong>
        Здесь только крупные сдвиги постановки. Детали по дням — в <code>notebooks/CHANGELOG.md</code>.
      </p>
    </section>

    <section id="models">
      <h2>2. Две модели рядом</h2>
      <table>
        <thead>
          <tr><th></th><th class="col-old">Legacy (прод)</th><th class="col-new">New (кандидат)</th></tr>
        </thead>
        <tbody>
          <tr><td class="topic">Алгоритм</td><td>CatBoost classifier + regressor</td><td>CatBoost classifier + regressor</td></tr>
          <tr><td class="topic">Где учится</td><td>Блок A, стек <code>legacy</code></td><td>Блок B, <code>train_loop_new</code></td></tr>
          <tr><td class="topic">Итерации</td><td>375 freq / 100 sev</td><td>После HPO (или те же defaults)</td></tr>
          <tr><td class="topic">Калибровка</td><td>Нет в проде</td><td>Isotonic frequency; isotonic severity на cal</td></tr>
          <tr><td class="topic">Порог «платить»</td><td>Подбор по старому ИТОГО</td><td>Максимум чистого финэффекта на Val, проверка на Test</td></tr>
        </tbody>
      </table>
    </section>

    <section id="samples">
      <h2>3. Датасет и выборки</h2>
      {_dataset_kpis(data)}
      {_split_table(data)}
      <table>
        <thead>
          <tr><th>Тема</th><th class="col-old">Было</th><th class="col-new">Стало</th></tr>
        </thead>
        <tbody>
          <tr><td class="topic">Зерно</td><td>Инцидент; TARGET_2 на событие</td><td>Одна строка на инцидент; первичный убыток = min LOSS_NUMBER</td></tr>
          <tr><td class="topic">Фильтр дат</td><td>LOSS_DATE_TIME с 2022</td><td>PAYMENT_ORDER_DATE_TIME с 2020, верхняя граница по вызреванию</td></tr>
          <tr><td class="topic">Вызревание</td><td>Неявное / по дате убытка</td><td>Поручение + H месяцев ≤ дата среза; без «Не принято» по иску</td></tr>
          <tr><td class="topic">Деньги в фичах</td><td>Номинальные рубли</td><td>ИПЦ Росстата → *_REAL_2022</td></tr>
        </tbody>
      </table>
    </section>

    <section id="targets">
      <h2>4. Таргеты</h2>
      <table>
        <thead>
          <tr><th>Задача</th><th class="col-old">Старая</th><th class="col-new">Новая</th></tr>
        </thead>
        <tbody>
          <tr>
            <td class="topic">Frequency</td>
            <td><code>TARGET_2</code> — был ли сутяжник по ПСР (претензии + ФУ + иск)</td>
            <td><code>TARGET_FREQ</code> — была ли доплата после первичного УУ (иски + претензии с выплатой)</td>
          </tr>
          <tr>
            <td class="topic">Severity</td>
            <td><code>TARGET_3_SEV</code> — ОД + УТС + износ (планка Litigant)</td>
            <td><code>TARGET_SEV</code> — сумма доплаты last instance, ₽</td>
          </tr>
          <tr>
            <td class="topic">База факта</td>
            <td>ПСР: претензии + ФУ + иск</td>
            <td><code>TARGET_FREQ_AMOUNT</code> — требования icnl + претензии</td>
          </tr>
        </tbody>
      </table>
      <p class="note">
        <strong>Важно:</strong> <code>TARGET_2</code> и <code>TARGET_FREQ</code> — разные определения.
        Совпадение меток на holdout — в разделе «Регрессии / сверка». Нельзя учить новую модель на старой метке и наоборот без оговорки.
      </p>
    </section>

    <section id="features">
      <h2>5. Признаки</h2>
      {_features_section(data)}
    </section>

    <section id="metrics">
      <h2>6. Метрики</h2>
      {_metrics_section(data)}
    </section>

    <section id="overlap">
      <h2>7. Как пересекаются классификации и регрессии</h2>
      {_overlap_section(data)}
    </section>

    <section id="fineffect">
      <h2>8. Финансовый эффект</h2>
      {_fin_effect_section(data)}
    </section>
  </main>
  <footer>
    Querulus · отчёт для руководителя · цифры из последнего прогона collect ·
    подробнее: dataset_sources.html, fin_effect_detailed.html, CHANGELOG.md
  </footer>
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
    """Собрать снимок из объектов collect (пропущенное = None)."""
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
    """Записать HTML. Без data — костяк без цифр прогона."""
    output = Path(path) if path is not None else DEFAULT_HTML_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_leadership_html(data), encoding="utf-8")
    return output


def export_leadership_html_from_collect(
    namespace: dict[str, Any],
    *,
    path: str | Path | None = None,
) -> Path:
    """Снять снимок из globals() collect и перезаписать HTML."""
    triple = namespace.get("triple")
    train_loop = namespace.get("train_loop_result")
    fe_prod = namespace.get("fe_prod_cal")
    prod_thr = namespace.get("production_training_best_threshold")
    data = collect_report_data(
        df=namespace.get("df"),
        config=namespace.get("loop_cfg") or namespace.get("TRAINING_CONFIG"),
        train_loop=train_loop,
        triple=triple if isinstance(triple, TripleStackResult) else None,
        stack_eval=namespace.get("stack_eval_b"),
        fin_effect_new=namespace.get("fin_effect_b"),
        freq_metrics_at_05=namespace.get("freq_metrics_at_05"),
        freq_metrics_at_best=namespace.get("freq_metrics_at_best"),
        production_threshold=None if prod_thr is None else float(prod_thr),
        production_net_effect=(
            None if fe_prod is None else float(getattr(fe_prod, "net_effect", float("nan")))
        ),
        subtitle=(
            "Цифры из текущего прогона collect: выборки, фичи, C2+, C3 и production — "
            "что успело посчитаться."
        ),
    )
    return export_leadership_html(data, path=path)
