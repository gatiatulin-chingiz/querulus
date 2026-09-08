"""HTML-отчёт по мониторингу финэффекта (витрина MSSQL + ретро-приоры)."""
from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from querulus.fin_effect.excel_monitoring import (
    MonitoringEffectResult,
    format_money,
)


_CSS = """
:root {
  --bg: #f6f4ef; --surface: #fff; --ink: #1c1c1c; --muted: #5a5a5a;
  --line: #ddd6c8; --accent: #0b5f66; --soft: #e7f3f4; --code: #f0ece4;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 15px;
  line-height: 1.45;
  color: var(--ink);
  background: var(--bg);
}
.page { max-width: 1180px; margin: 0 auto; padding: 1.2rem 1.1rem 2rem; }
h1 { margin: 0 0 0.3rem; font-size: 1.45rem; color: var(--accent); }
h2 { margin: 1.35rem 0 0.4rem; font-size: 1.12rem; color: var(--accent); }
.sub { color: var(--muted); margin: 0 0 0.85rem; font-size: 0.92rem; }
.card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 0.75rem 0.95rem;
  margin: 0.45rem 0;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 0.45rem;
  margin: 0.5rem 0 0.7rem;
}
.stat {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 0.5rem 0.6rem;
}
.stat span { display: block; color: var(--muted); font-size: 0.76rem; }
.stat b { font-variant-numeric: tabular-nums; }
table {
  width: 100%;
  border-collapse: collapse;
  background: var(--surface);
  border: 1px solid var(--line);
  font-size: 0.82rem;
  margin: 0.35rem 0 0.5rem;
}
.scroll {
  overflow-x: auto;
  margin: 0.35rem 0 0.5rem;
}
th, td {
  border: 1px solid var(--line);
  padding: 0.35rem 0.45rem;
  text-align: left;
  vertical-align: top;
}
th { background: var(--soft); color: var(--accent); }
code, pre {
  font-family: Consolas, "Courier New", monospace;
  font-size: 0.82rem;
}
pre {
  background: var(--code);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 0.65rem 0.75rem;
  overflow-x: auto;
  white-space: pre-wrap;
  margin: 0.35rem 0;
}
.formula {
  background: linear-gradient(135deg, var(--soft), #f7fbfb);
  border: 1px solid #c7dfe1;
  border-left: 4px solid var(--accent);
  border-radius: 0 8px 8px 0;
  padding: 0.75rem 0.9rem;
  margin: 0.55rem 0;
  overflow-x: auto;
}
.formula-title {
  color: var(--muted);
  font-size: 0.76rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  margin-bottom: 0.35rem;
  text-transform: uppercase;
}
.equation {
  color: #153f43;
  font-family: Cambria, "Times New Roman", serif;
  font-size: 1.05rem;
  line-height: 1.65;
  white-space: nowrap;
}
.equation strong { color: var(--accent); }
.equation .op { padding: 0 0.2em; }
.equation-note {
  border-top: 1px solid #d4e6e7;
  color: var(--muted);
  font-size: 0.82rem;
  margin-top: 0.45rem;
  padding-top: 0.4rem;
}
.conditions { display: grid; gap: 0.3rem; }
.condition {
  align-items: baseline;
  display: grid;
  gap: 0.65rem;
  grid-template-columns: minmax(130px, auto) 1fr;
}
.condition b { color: var(--accent); font-family: Consolas, monospace; }
@media (max-width: 620px) {
  .condition { grid-template-columns: 1fr; gap: 0; }
  .equation { font-size: 0.95rem; }
}
ul { margin: 0.3rem 0 0.45rem 1.15rem; padding: 0; }
li { margin: 0.15rem 0; }
.muted { color: var(--muted); font-size: 0.9rem; }
"""

# Семантика только ключевых колонок (без повтора фильтров/формул).
COLUMN_SEMANTICS: list[dict[str, str]] = [
    {
        "column": "РезультатПроверки",
        "source": "Case _Fld11690 → 0/1/−100",
        "meaning": "ручеёк: 0/1 в модели, −100 вне",
        "example": "1",
    },
    {
        "column": "СуммаОсновногоДолгаЗаявлено",
        "source": "РасчетВыплаты._Fld18877",
        "meaning": "ОД для ×k",
        "example": "85 000",
    },
    {
        "column": "СуммаКВыплате",
        "source": "док. СуммаВыплаты (не 1С-СуммаКВыплате!)",
        "meaning": "кандидат cost",
        "example": "42 300",
    },
    {
        "column": "Иные затраты",
        "source": "док. поле СуммаКВыплате",
        "meaning": "кандидат cost; на нём loss-флаг выплаты по модели",
        "example": "42 300",
    },
    {
        "column": "СуммаПлатежа",
        "source": "sum(Payments)",
        "meaning": "факт кассы, кандидат cost",
        "example": "42 300",
    },
    {
        "column": "ЕстьПретензияВИнциденте",
        "source": "pretensions по инциденту",
        "meaning": "доля претензий",
        "example": "1",
    },
    {
        "column": "ЕстьОбращениеКФУВИнциденте",
        "source": "max(Обращение к ФУ) по инциденту",
        "meaning": "доля ФУ (на первичных loss-флаг ≈0)",
        "example": "1",
    },
    {
        "column": "ЕстьОбращениеКСудуВИнциденте",
        "source": "max(Обращение к суду) по инциденту",
        "meaning": "доля судов",
        "example": "0",
    },
]


def _pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"


def _money(x: float) -> str:
    return format_money(float(x))


def _fmt_share(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    v = float(x)
    # сегменты в тетрадке часто уже в %
    if abs(v) > 1.5:
        return f"{v:.2f}%"
    return _pct(v)


def _df_to_html_table(df: pd.DataFrame) -> str:
    if df is None or getattr(df, "empty", True):
        return "<p class='muted'>(нет данных)</p>"
    try:
        html = df.to_html(index=False, border=0, escape=True)
        return f'<div class="scroll">{html}</div>'
    except Exception:
        return f"<pre>{escape(str(df))}</pre>"


def _analytics_block(
    analytics: dict[str, Any] | None,
    key: str,
    caption: str,
) -> str:
    if not analytics or key not in analytics:
        return ""
    frame = analytics[key]
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return f"<p class='muted'>{escape(caption)}: нет данных</p>"
    return f"<p><b>{escape(caption)}</b></p>{_df_to_html_table(frame)}"


def _segment_example(segments: Any | None) -> str:
    """Короткий пример долей из таблицы сегментов прогона."""
    if not isinstance(segments, pd.DataFrame) or segments.empty:
        return (
            "пример: with_model n=40, agreement=55%, pretension=12%, "
            "fu_incident=8%, court_incident=5%; "
            "without_model n=200, agreement=20%, pretension=4%, "
            "fu_incident=3%, court_incident=2%; "
            "lift_pp agreement = +35 п.п."
        )
    try:
        wm = segments.loc[segments["segment"] == "with_model"].iloc[0]
        wo = segments.loc[segments["segment"] == "without_model"].iloc[0]
    except Exception:
        return "см. таблицу сегментов ниже"

    def g(row: pd.Series, col: str) -> str:
        return _fmt_share(row[col]) if col in row.index else "—"

    return (
        f"with_model n={int(wm['n']) if pd.notna(wm.get('n')) else '—'}: "
        f"agreement={g(wm, 'agreement_share')}, "
        f"pretension={g(wm, 'pretension_share')}, "
        f"fu={g(wm, 'fu_incident_share')}, "
        f"court={g(wm, 'court_incident_share')}; "
        f"without_model n={int(wo['n']) if pd.notna(wo.get('n')) else '—'}: "
        f"agreement={g(wo, 'agreement_share')}, "
        f"pretension={g(wo, 'pretension_share')}, "
        f"fu={g(wo, 'fu_incident_share')}, "
        f"court={g(wo, 'court_incident_share')}"
    )


def _fin_effect_example(effect: MonitoringEffectResult) -> str:
    p = effect.priors
    n = effect.n_intervention
    fees_total = n * effect.e_fee
    od_term = effect.sum_od * p.k
    raw = od_term + fees_total
    return (
        f"n_I={n}; e_fee={_money(effect.e_fee)} "
        f"(= {_pct(p.p_fu)}×{_money(p.fu_fee)} + {_pct(p.p_court)}×{_money(p.court_fee)}); "
        f"Σ ОД×k = {_money(od_term)}; Σ e_fee = {_money(fees_total)}; "
        f"expected_psr = {p.precision:.2f}×{_money(raw)} = {_money(effect.expected_psr)}; "
        f"cost({effect.cost_column})={_money(effect.cost)}; "
        f"net={_money(effect.net)}"
    )


def build_monitoring_html(
    effect: MonitoringEffectResult,
    *,
    annual: dict[str, Any] | None = None,
    segments: Any | None = None,
    cost_reconcile: Any | None = None,
    analytics: dict[str, Any] | None = None,
    source_label: str = "[OISUU_report].[dbo].[ВитринаСутяжность]",
    title: str = "Мониторинг фин. эффекта Querulus",
) -> str:
    """Собрать ёмкий HTML без дублирования одних и тех же блоков."""
    p = effect.priors
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    window = (
        f"{p.window_start or '?'} … {p.window_end or '?'}"
        if p.window_start or p.window_end
        else "окно не задано"
    )

    semantics_rows = "".join(
        "<tr>"
        f"<td><code>{escape(r['column'])}</code></td>"
        f"<td>{escape(r['source'])}</td>"
        f"<td>{escape(r['meaning'])}</td>"
        f"<td>{escape(r['example'])}</td>"
        "</tr>"
        for r in COLUMN_SEMANTICS
    )

    cost_block = ""
    if cost_reconcile is not None:
        note = ""
        if isinstance(cost_reconcile, pd.DataFrame) and not cost_reconcile.empty:
            note = str(cost_reconcile.iloc[0].get("suggestion_note", "") or "")
        # не тащим suggestion_note в каждую строку таблицы
        show = cost_reconcile
        if isinstance(show, pd.DataFrame):
            drop_cols = [c for c in ("suggested_cost_col", "suggestion_note") if c in show.columns]
            if drop_cols:
                show = show.drop(columns=drop_cols)
        cost_block = f"""
        <h2>3. Выбор cost на I</h2>
        <div class="card">
          <p class="muted">Сверка на I: <code>СуммаКВыплате</code> / <code>СуммаПлатежа</code> /
          <code>Иные затраты</code> (колонки <code>СуммаВыплаты</code> в витрине нет).
          <b>Итог:</b> {escape(note)} → в расчёте cost =
          <code>{escape(effect.cost_column)}</code>.</p>
          {_df_to_html_table(show)}
        </div>
        """

    annual_line = ""
    if annual:
        annual_line = (
            f"<p class='muted'>Экстраполяция: "
            f"net/день={escape(str(annual.get('net_per_day', '—')))}, "
            f"net_365={escape(str(annual.get('net_annual_365', '—')))}, "
            f"sample_days={escape(str(annual.get('sample_days', '—')))}, "
            f"model_start={escape(str(annual.get('model_start', '—')))}."
            f"</p>"
        )

    seg_block = ""
    if segments is not None:
        seg_block = f"""
        <h2>5. Результат долей (этот прогон, main / ручеёк)</h2>
        <div class="card">
          {_df_to_html_table(segments)}
        </div>
        """

    ext = ""
    if analytics:
        ext = f"""
  <h2>7. Филиалы: использование модели</h2>
  <div class="card">
    <p class="muted">База: форма/первичный/авто; филиал не режем (включая Арх/Марийск).
    with_model = вызов ∧ выплата по модели; ручеёк = доли РезультатПроверки 0/1/−100.</p>
    {_analytics_block(analytics, "filial_usage", "Все филиалы")}
  </div>

  <h2>8. Филиалы: доли путей (main)</h2>
  <div class="card">
    <p class="muted">Только ручеёк (без Арх/Марийск). Сегменты with_model / without_model.</p>
    {_analytics_block(analytics, "filial_shares_main", "Доли по филиалам")}
  </div>

  <h2>9. Ручеёк: РезультатПроверки</h2>
  <div class="card">
    <div class="formula">
      <div class="formula-title">Сегментация ручейка</div>
      <div class="conditions">
        <div class="condition"><b>model_0 / model_1</b><span>РезультатПроверки ∈ {{0, 1}} — в модели</span></div>
        <div class="condition"><b>out_of_model</b><span>РезультатПроверки = −100 — вне модели</span></div>
        <div class="condition"><b>in_model_0_1</b><span>model_0 ∪ model_1</span></div>
      </div>
    </div>
    {_analytics_block(analytics, "result_distribution_main", "Распределение + выплаты (main)")}
  </div>

  <h2>10. Выплаты с моделью и без (main)</h2>
  <div class="card">
    <p class="muted">with_model = вызов∧выплата; rucheek_* — по РезультатПроверки.
    Метрики: n, sum, mean, median, p25, p75, min, max.</p>
    {_analytics_block(analytics, "payments_main", "Выплаты")}
  </div>

  <h2>11. Вызов модели без использования (main)</h2>
  <div class="card">
    <div class="formula">
      <div class="formula-title">Вызвали, но не использовали</div>
      <div class="equation"><strong>called_not_used</strong> <span class="op">=</span>
      (ВызовМодельСутяжность = 1) <span class="op">∧</span>
      (Выплата по модели в Инциденте ≠ 1)</div>
      <div class="equation-note">Расчёт выполняется внутри B<sub>main</sub>.</div>
    </div>
    {_analytics_block(analytics, "called_not_used_main", "Итого и по филиалам")}
  </div>

  <h2>12. Пилот Архангельск / Марийск (~100% модель)</h2>
  <div class="card">
    <p class="muted">Тот же набор метрик на filial_scope=pilot (только эти филиалы).</p>
    {_analytics_block(analytics, "segments_pilot", "Доли путей (агрегат)")}
    {_analytics_block(analytics, "filial_usage_pilot", "Использование модели")}
    {_analytics_block(analytics, "filial_shares_pilot", "Доли по филиалам")}
    {_analytics_block(analytics, "result_distribution_pilot", "Ручеёк / −100")}
    {_analytics_block(analytics, "payments_pilot", "Выплаты")}
    {_analytics_block(analytics, "called_not_used_pilot", "Вызов без использования")}
  </div>
"""

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{escape(title)}</title>
  <style>{_CSS}</style>
</head>
<body>
<div class="page">
  <h1>{escape(title)}</h1>
  <p class="sub">{escape(generated)} · источник <code>{escape(source_label)}</code> ·
  ретро-окно {escape(window)} · precision={p.precision:.2f}, k={p.k:.4f},
  p_fu={_pct(p.p_fu)}, p_court={_pct(p.p_court)}, p_pret={_pct(p.p_pret)},
  e_fee={_money(effect.e_fee)}, psr_share={_pct(p.psr_share)}</p>

  <div class="grid">
    <div class="stat"><span>n_I</span><b>{effect.n_intervention}</b></div>
    <div class="stat"><span>expected_psr</span><b>{_money(effect.expected_psr)}</b></div>
    <div class="stat"><span>cost</span><b>{_money(effect.cost)}</b></div>
    <div class="stat"><span>net</span><b>{_money(effect.net)}</b></div>
  </div>
  {annual_line}

  <h2>1. Контуры фильтрации (один раз)</h2>
  <div class="card">
    <div class="formula">
      <div class="formula-title">Контуры выборок</div>
      <div class="conditions">
        <div class="condition"><b>B<sub>main</sub></b><span>базовые фильтры; филиал ∉ {{Архангельский, Марийский}} · ручеёк ~50%</span></div>
        <div class="condition"><b>B<sub>pilot</sub></b><span>базовые фильтры; филиал ∈ {{Архангельский, Марийский}} · модель ~100%</span></div>
        <div class="condition"><b>I</b><span>B<sub>main</sub> ∩ {{вызов модели = 1}} ∩ {{выплата по модели = 1}}</span></div>
        <div class="condition"><b>W</b><span>B ∩ {{вызов модели = 1}} ∩ {{выплата по модели = 1}}</span></div>
        <div class="condition"><b>U</b><span>B ∖ W</span></div>
      </div>
      <div class="equation-note">Базовые фильтры: форма возмещения ∈ {{денежная, ремонт, соглашение}}; первичный убыток; автотранспорт = 1.</div>
    </div>
    <p class="muted">ФУ/суд для долей: сначала
    <code>ЕстьОбращениеКФУ/СудуВИнциденте = max(флаг) по НомерИнцидент</code>,
    иначе на первичных строках loss-флаги почти всегда 0.
    <code>p_fu</code>/<code>p_court</code> для e_fee — только с ретро, не с витрины.
    Ретро-окно по умолчанию: 2 года до <code>2025-06-30</code>.</p>
  </div>

  <h2>2. Финэффект</h2>
  <div class="card">
    <div class="formula">
      <div class="formula-title">Расчёт ожидаемого эффекта</div>
      <div class="equation"><strong>e<sub>fee</sub></strong> <span class="op">=</span>
      p<sub>fu</sub> × 100 000 <span class="op">+</span> p<sub>court</sub> × 15 000</div>
      <div class="equation"><strong>expected_psr</strong> <span class="op">=</span>
      precision × (Σ<sub>i∈I</sub> OD<sub>i</sub> × k <span class="op">+</span> n<sub>I</sub> × e<sub>fee</sub>)</div>
      <div class="equation"><strong>cost</strong> <span class="op">=</span>
      Σ<sub>i∈I</sub> {escape(effect.cost_column)}<sub>i</sub></div>
      <div class="equation"><strong>net</strong> <span class="op">=</span>
      expected_psr <span class="op">−</span> cost</div>
      <div class="equation-note">OD = <code>{escape(effect.od_column)}</code>; I — контур финэффекта.</div>
    </div>
    <p><b>Пример этого прогона:</b> {escape(_fin_effect_example(effect))}</p>
  </div>

  {cost_block}

  <h2>4. Доли соглашений / претензий / ФУ / суда</h2>
  <div class="card">
    <div class="formula">
      <div class="formula-title">Доли на сегменте S ∈ {{W, U}}</div>
      <div class="equation"><strong>share<sub>event</sub>(S)</strong> <span class="op">=</span>
      n(event = 1 ∩ S) / n(S)</div>
      <div class="equation"><strong>lift<sub>pp</sub></strong> <span class="op">=</span>
      share(W) <span class="op">−</span> share(U)</div>
      <div class="equation"><strong>lift<sub>rel</sub></strong> <span class="op">=</span>
      share(W) / share(U) <span class="op">−</span> 1</div>
      <div class="equation-note">event ∈ {{соглашение, претензия, ФУ, суд}}. ФУ и суд агрегированы на уровень инцидента.</div>
    </div>
    <p><b>Пример этого прогона:</b> {escape(_segment_example(segments))}</p>
  </div>

  {seg_block}

  {ext}

  <h2>13. Колонки витрины (справка)</h2>
  <div class="card">
    <table>
      <thead>
        <tr><th>Колонка</th><th>Источник ETL</th><th>Зачем</th><th>Пример</th></tr>
      </thead>
      <tbody>{semantics_rows}</tbody>
    </table>
    <p class="muted">В витрине нет отдельной <code>СуммаВыплаты</code>: она уже лежит
    под именем <code>СуммаКВыплате</code>.
    <code>РезультатПроверки</code>: 0/1 — ручеёк, −100 — вне модели.</p>
  </div>
</div>
</body>
</html>
"""


def write_monitoring_html(
    effect: MonitoringEffectResult,
    path: str | Path,
    *,
    annual: dict[str, Any] | None = None,
    segments: Any | None = None,
    cost_reconcile: Any | None = None,
    analytics: dict[str, Any] | None = None,
    source_label: str = "[OISUU_report].[dbo].[ВитринаСутяжность]",
    title: str = "Мониторинг фин. эффекта Querulus",
) -> Path:
    """Записать HTML-отчёт на диск."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html = build_monitoring_html(
        effect,
        annual=annual,
        segments=segments,
        cost_reconcile=cost_reconcile,
        analytics=analytics,
        source_label=source_label,
        title=title,
    )
    path.write_text(html, encoding="utf-8")
    return path