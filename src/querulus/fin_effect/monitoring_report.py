"""HTML-отчёты мониторинга: вариант 1 (ручеёк vs контроль) и вариант 2 (кейсы)."""
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
from querulus.fin_effect.monitoring_analytics import VARIANT2_LABELS


_CSS = """
:root {
  --bg: #f6f4ef; --surface: #fff; --ink: #1c1c1c; --muted: #5a5a5a;
  --line: #ddd6c8; --accent: #0b5f66; --soft: #e7f3f4; --code: #f0ece4;
  --example: #fff8e8; --example-line: #e6d7a8;
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
h3 { margin: 0.85rem 0 0.3rem; font-size: 0.98rem; color: #2a4f53; }
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
.scroll { overflow-x: auto; margin: 0.35rem 0 0.5rem; }
th, td {
  border: 1px solid var(--line);
  padding: 0.35rem 0.45rem;
  text-align: left;
  vertical-align: top;
}
th { background: var(--soft); color: var(--accent); }
code {
  font-family: Consolas, "Courier New", monospace;
  font-size: 0.82rem;
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
.example {
  background: var(--example);
  border: 1px solid var(--example-line);
  border-left: 4px solid #b08900;
  border-radius: 0 8px 8px 0;
  padding: 0.7rem 0.85rem;
  margin: 0.45rem 0 0.7rem;
}
.example-title {
  color: #7a5c00;
  font-size: 0.76rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  margin-bottom: 0.3rem;
  text-transform: uppercase;
}
.filters {
  display: grid;
  gap: 0.35rem;
}
.filter-row {
  display: grid;
  gap: 0.55rem;
  grid-template-columns: minmax(120px, auto) 1fr;
  align-items: baseline;
}
.filter-row b { color: var(--accent); font-family: Consolas, monospace; }
.legend {
  background: #faf9f6;
  border: 1px dashed var(--line);
  border-radius: 6px;
  padding: 0.55rem 0.7rem;
  margin: 0.35rem 0 0.55rem;
  font-size: 0.86rem;
}
.legend ul { margin: 0.2rem 0 0 1.1rem; padding: 0; }
.legend li { margin: 0.12rem 0; }
.muted { color: var(--muted); font-size: 0.9rem; }
@media (max-width: 620px) {
  .filter-row { grid-template-columns: 1fr; gap: 0; }
  .equation { font-size: 0.95rem; white-space: normal; }
}
"""


def _pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"


def _money(x: float) -> str:
    return format_money(float(x))


def _df_to_html_table(df: pd.DataFrame) -> str:
    if df is None or getattr(df, "empty", True):
        return "<p class='muted'>(нет данных)</p>"
    try:
        return f'<div class="scroll">{df.to_html(index=False, border=0, escape=True)}</div>'
    except Exception:
        return f"<pre>{escape(str(df))}</pre>"


def _block_filters(title: str, rows: list[tuple[str, str]]) -> str:
    body = "".join(
        f'<div class="filter-row"><b>{escape(k)}</b><span>{v}</span></div>'
        for k, v in rows
    )
    return f"""
    <div class="formula">
      <div class="formula-title">{escape(title)}</div>
      <div class="filters">{body}</div>
    </div>
    """


def _block_formula(title: str, equations: list[str], note: str = "") -> str:
    eqs = "".join(f'<div class="equation">{eq}</div>' for eq in equations)
    note_html = f'<div class="equation-note">{note}</div>' if note else ""
    return f"""
    <div class="formula">
      <div class="formula-title">{escape(title)}</div>
      {eqs}
      {note_html}
    </div>
    """


def _block_example(title: str, text: str) -> str:
    return f"""
    <div class="example">
      <div class="example-title">{escape(title)}</div>
      <div>{text}</div>
    </div>
    """


def _block_legend(rows: str, cols: str) -> str:
    return f"""
    <div class="legend">
      <b>Легенда строк</b>
      <ul>{rows}</ul>
      <b>Легенда колонок</b>
      <ul>{cols}</ul>
    </div>
    """


def _table_section(
    heading: str,
    filters_html: str,
    formula_html: str,
    example_html: str,
    legend_html: str,
    table: Any,
) -> str:
    return f"""
  <h2>{escape(heading)}</h2>
  <div class="card">
    {filters_html}
    {formula_html}
    {example_html}
    {legend_html}
    {_df_to_html_table(table) if table is not None else ""}
  </div>
"""


def _fin_example(effect: MonitoringEffectResult) -> str:
    p = effect.priors
    n = effect.n_intervention
    fees_total = n * effect.e_fee
    od_term = effect.sum_od * p.k
    raw = od_term + fees_total
    return (
        f"n<sub>I</sub>={n}; "
        f"e<sub>fee</sub>={_money(effect.e_fee)} "
        f"(= {_pct(p.p_fu)}×{_money(p.fu_fee)} + {_pct(p.p_court)}×{_money(p.court_fee)}); "
        f"Σ OD×k = {_money(od_term)}; Σ e<sub>fee</sub> = {_money(fees_total)}; "
        f"expected_psr = {p.precision:.2f}×{_money(raw)} = {_money(effect.expected_psr)}; "
        f"cost = {_money(effect.cost)}; "
        f"net = {_money(effect.net)}"
    )


def _share_example(segments: Any) -> str:
    if not isinstance(segments, pd.DataFrame) or segments.empty:
        return "нет данных по сегментам"
    parts = []
    for _, row in segments.iterrows():
        seg = str(row.get("segment", ""))
        if seg.startswith("lift_"):
            continue
        n = row.get("n", "—")
        parts.append(
            f"<code>{escape(seg)}</code>: n={n}, "
            f"agreement={row.get('agreement_share', '—')}%, "
            f"pretension={row.get('pretension_share', '—')}%, "
            f"fu={row.get('fu_incident_share', '—')}%, "
            f"court={row.get('court_incident_share', '—')}%"
        )
    return "<br/>".join(parts) if parts else "нет данных"


def _base_glossary() -> str:
    return _block_filters(
        "Термины",
        [
            ("Bpilot", "пилот OISUU без Архангельского и Марийского (~50% ручеёк / ~50% контроль)"),
            ("Bam", "только Архангельский и Марийский (~100% модель)"),
            ("ручеёк / model", "<code>РезультатПроверки ∈ {0, 1}</code> — модель работала"),
            ("контроль / control", "<code>РезультатПроверки = −100</code> — модель не работала"),
            ("ВызовМодельСутяжность", "неинформативен для сегментации; не используем как критерий «модель работала»"),
        ],
    )


def _base_filters_block() -> str:
    return _block_filters(
        "Базовые фильтры (для всех слоёв, если не сказано иное)",
        [
            ("форма", "ФормаВозмещения ∈ {денежная, ремонт, соглашение}"),
            ("статус", "УбытокСтатус = первичный"),
            ("объект", "ТипОбъектаАвтотранспорт = 1"),
            ("филиал", "задаётся слоем: Bpilot или Bam"),
        ],
    )


def _filters_overview_table(
    variant: int,
    effect: MonitoringEffectResult,
) -> str:
    """Сводная карта фильтров и показателей по слоям отчёта."""
    if variant == 1:
        fin_effect_filter = (
            "Benefit: result=1 и выплата по модели=1; "
            "cost: result∈{0,1} и выплата по модели=1"
        )
        comparison_filter = "model: result∈{0,1}; control: result=−100"
        segment_filter = "model / control"
    else:
        fin_effect_filter = "вызов=1, result=1, выплата по модели=1"
        comparison_filter = (
            "4 кейса: applied_one_paid, ignored_zero_paid, "
            "out_of_model_paid, recommended_unpaid"
        )
        segment_filter = "четыре кейса call/result/payout"

    rows = [
        {
            "Слой расчёта": "Финэффект",
            "Контур": "Bpilot",
            "Базовые фильтры": "форма; первичный убыток; автотранспорт=1",
            "Фильтр сегмента": fin_effect_filter,
            "Показатели": (
                f"n_I; expected_psr; cost ({effect.cost_column}); net; "
                "экстраполяция на 365 дней"
            ),
        },
        {
            "Слой расчёта": "Соглашения / претензии / ФУ / суд",
            "Контур": "Bpilot",
            "Базовые фильтры": "форма; первичный убыток; автотранспорт=1",
            "Фильтр сегмента": comparison_filter,
            "Показатели": (
                "agreement_share; pretension_share; "
                "fu_incident_share; court_incident_share; lift_pp; lift_rel"
            ),
        },
        {
            "Слой расчёта": "Использование модели по филиалам",
            "Контур": "все филиалы",
            "Базовые фильтры": "форма; первичный убыток; автотранспорт=1",
            "Фильтр сегмента": "result=0 / result=1 / result=−100",
            "Показатели": (
                "число и доля ручейка; число и доля контроля; "
                "доли result 0/1/−100"
            ),
        },
        {
            "Слой расчёта": "Доли путей по филиалам",
            "Контур": "Bpilot, отдельно каждый филиал",
            "Базовые фильтры": "форма; первичный убыток; автотранспорт=1",
            "Фильтр сегмента": segment_filter,
            "Показатели": (
                "agreement_share; pretension_share; "
                "fu_incident_share; court_incident_share"
            ),
        },
        {
            "Слой расчёта": "Распределение убытков и выплаты",
            "Контур": "Bpilot",
            "Базовые фильтры": "форма; первичный убыток; автотранспорт=1",
            "Фильтр сегмента": "result=0 / result=1 / result∈{0,1} / result=−100",
            "Показатели": (
                "n; share_of_base; sum; mean; median; p25; p75; min; max"
            ),
        },
        {
            "Слой расчёта": "Выплаты по сегментам",
            "Контур": "Bpilot",
            "Базовые фильтры": "форма; первичный убыток; автотранспорт=1",
            "Фильтр сегмента": segment_filter,
            "Показатели": "n; n_amount; sum; mean; median; p25; p75; min; max",
        },
        {
            "Слой расчёта": "Диагностика применения модели",
            "Контур": "Bpilot",
            "Базовые фильтры": "форма; первичный убыток; автотранспорт=1",
            "Фильтр сегмента": "4 кейса call/result/payout",
            "Показатели": "число каждого кейса всего и по филиалам",
        },
        {
            "Слой расчёта": "Архангельский / Марийский",
            "Контур": "Bam",
            "Базовые фильтры": "форма; первичный убыток; автотранспорт=1",
            "Фильтр сегмента": segment_filter,
            "Показатели": (
                "те же доли, выплаты и кейсы; контроль ожидается почти пустым"
            ),
        },
    ]
    return _df_to_html_table(pd.DataFrame(rows))


def build_monitoring_html(
    effect: MonitoringEffectResult,
    *,
    variant: int = 1,
    annual: dict[str, Any] | None = None,
    analytics: dict[str, Any] | None = None,
    source_label: str = "[OISUU_report].[dbo].[ВитринаСутяжность]",
    title: str | None = None,
) -> str:
    """Собрать HTML для варианта 1 или 2."""
    analytics = analytics or {}
    p = effect.priors
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    window = (
        f"{p.window_start or '?'} … {p.window_end or '?'}"
        if p.window_start or p.window_end
        else "окно не задано"
    )
    if title is None:
        title = (
            "Querulus — вариант 1: ручеёк vs контроль"
            if variant == 1
            else "Querulus — вариант 2: кейсы применения модели"
        )

    annual_line = ""
    if annual:
        annual_line = (
            f"<p class='muted'>Экстраполяция: "
            f"net/день={escape(str(annual.get('net_per_day', '—')))}, "
            f"net_365={escape(str(annual.get('net_annual_365', '—')))}, "
            f"sample_days={escape(str(annual.get('sample_days', '—')))}."
            f"</p>"
        )

    if variant == 1:
        fin_filters = _block_filters(
            "Фильтры слоя финэффекта (Bpilot)",
            [
                ("база", "базовые фильтры + филиал ∉ {Архангельский, Марийский}"),
                ("benefit / I", "РезультатПроверки = 1 ∧ Выплата по модели = 1"),
                ("cost", "РезультатПроверки ∈ {0, 1} ∧ Выплата по модели = 1"),
                ("смысл нулей", "result=0 ∧ выплата=1 не даёт expected_psr, но входит в cost"),
            ],
        )
        share_filters = _block_filters(
            "Фильтры слоя долей путей (Bpilot)",
            [
                ("база", "базовые фильтры + Bpilot"),
                ("model", "РезультатПроверки ∈ {0, 1}"),
                ("control", "РезультатПроверки = −100"),
                ("события", "agreement / pretension / FU_incident / court_incident"),
            ],
        )
        fin_formula = _block_formula(
            "Формулы финэффекта",
            [
                "<strong>e<sub>fee</sub></strong> <span class='op'>=</span> "
                "p<sub>fu</sub> × 100 000 <span class='op'>+</span> p<sub>court</sub> × 15 000",
                "<strong>expected_psr</strong> <span class='op'>=</span> "
                "precision × (Σ<sub>i∈I</sub> OD<sub>i</sub> × k <span class='op'>+</span> "
                "n<sub>I</sub> × e<sub>fee</sub>)",
                "<strong>cost</strong> <span class='op'>=</span> "
                f"Σ<sub>j∈C</sub> {escape(effect.cost_column)}<sub>j</sub>",
                "<strong>net</strong> <span class='op'>=</span> expected_psr <span class='op'>−</span> cost",
            ],
            note="I = benefit-маска; C = cost-маска варианта 1. OD = "
            f"<code>{escape(effect.od_column)}</code>.",
        )
        share_formula = _block_formula(
            "Формулы долей путей",
            [
                "<strong>share<sub>event</sub>(S)</strong> <span class='op'>=</span> "
                "100% × n(event=1 ∩ S) / n(S)",
                "<strong>lift<sub>pp</sub></strong> <span class='op'>=</span> "
                "share(model) <span class='op'>−</span> share(control)  (п.п.)",
                "<strong>lift<sub>rel</sub></strong> <span class='op'>=</span> "
                "share(model)/share(control) <span class='op'>−</span> 1  (%)",
            ],
            note="S ∈ {model, control}. Доли показывают, как часто после выплаты "
            "возникают соглашение / претензия / ФУ / суд.",
        )
    else:
        fin_filters = _block_filters(
            "Фильтры слоя финэффекта (Bpilot, вариант 2)",
            [
                ("база", "базовые фильтры + Bpilot"),
                ("I = cost = benefit", "вызов=1 ∧ РезультатПроверки=1 ∧ Выплата по модели=1"),
                ("остальные кейсы", "в финэффект не входят; только в аналитику долей"),
            ],
        )
        share_filters = _block_filters(
            "Фильтры слоя долей (Bpilot, вариант 2)",
            [
                ("база", "базовые фильтры + Bpilot"),
                ("applied_one_paid", "вызов=1, result=1, выплата=1 — модель работала"),
                ("ignored_zero_paid", "вызов=1, result=0, выплата=1 — результат проигнорирован"),
                ("out_of_model_paid", "вызов=1, result=−100, выплата=1 — вне модели"),
                ("recommended_unpaid", "вызов=1, result=1, выплата=0 — рекомендация без доплаты"),
            ],
        )
        fin_formula = _block_formula(
            "Формулы финэффекта",
            [
                "<strong>e<sub>fee</sub></strong> <span class='op'>=</span> "
                "p<sub>fu</sub> × 100 000 <span class='op'>+</span> p<sub>court</sub> × 15 000",
                "<strong>expected_psr</strong> <span class='op'>=</span> "
                "precision × (Σ<sub>i∈I</sub> OD<sub>i</sub> × k <span class='op'>+</span> "
                "n<sub>I</sub> × e<sub>fee</sub>)",
                "<strong>cost</strong> <span class='op'>=</span> "
                f"Σ<sub>i∈I</sub> {escape(effect.cost_column)}<sub>i</sub>",
                "<strong>net</strong> <span class='op'>=</span> expected_psr <span class='op'>−</span> cost",
            ],
            note="I — только ожидаемый кейс applied_one_paid.",
        )
        case_lines = "".join(
            f"<div class='equation'><strong>{escape(k)}</strong> "
            f"<span class='op'>—</span> {escape(v)}</div>"
            for k, v in VARIANT2_LABELS.items()
        )
        share_formula = f"""
        <div class="formula">
          <div class="formula-title">Кейсы и доли</div>
          {case_lines}
          <div class="equation"><strong>share<sub>event</sub>(S)</strong>
          <span class="op">=</span> 100% × n(event=1 ∩ S) / n(S)</div>
          <div class="equation-note">model_worked = applied_one_paid;
          model_not_worked = остальные три кейса.</div>
        </div>
        """

    fin_section = f"""
  <h2>2. Финэффект</h2>
  <div class="card">
    {fin_filters}
    {fin_formula}
    {_block_example("Численный пример этого прогона", _fin_example(effect))}
    <div class="grid">
      <div class="stat"><span>n_I</span><b>{effect.n_intervention}</b></div>
      <div class="stat"><span>expected_psr</span><b>{_money(effect.expected_psr)}</b></div>
      <div class="stat"><span>cost</span><b>{_money(effect.cost)}</b></div>
      <div class="stat"><span>net</span><b>{_money(effect.net)}</b></div>
    </div>
    {annual_line}
  </div>
"""

    seg = analytics.get("segments_bpilot")
    share_section = _table_section(
        "3. Доли соглашений / претензий / ФУ / суда (Bpilot)",
        share_filters,
        share_formula,
        _block_example("Численный пример сегментов", _share_example(seg)),
        _block_legend(
            "<li><code>model</code> / <code>control</code> или кейсы варианта 2</li>"
            "<li><code>lift_pp</code> — разница долей в процентных пунктах</li>"
            "<li><code>lift_rel</code> — относительный рост доли, %</li>",
            "<li><code>n</code> — число строк сегмента</li>"
            "<li><code>*_share</code> — доля события в сегменте, %</li>",
        ),
        seg,
    )

    usage = analytics.get("filial_usage_all")
    usage_section = _table_section(
        "4. Филиалы: ручеёк / контроль",
        _block_filters(
            "Фильтры",
            [
                ("база", "базовые фильтры"),
                ("филиал", "все, включая Bam"),
                ("сегментация", "только по РезультатПроверки"),
            ],
        ),
        _block_formula(
            "Доли",
            [
                "<strong>model_rucheek_share</strong> = 100% × n(result∈{0,1}) / n",
                "<strong>control_share</strong> = 100% × n(result=−100) / n",
                "<strong>applied_share</strong> = 100% × n(result=1 ∧ выплата=1) / n",
            ],
        ),
        _block_example(
            "Как читать",
            "Высокий <code>model_rucheek_share</code> ≈ попадание в 50% ручеёк; "
            "в Bam ожидаем ~100%.",
        ),
        _block_legend(
            "<li>строка = филиал</li>",
            "<li><code>share_result_0/1</code> — доли решений модели, %</li>"
            "<li><code>share_out_of_model</code> — доля −100, %</li>",
        ),
        usage,
    )

    filial_shares = analytics.get("filial_shares_bpilot")
    filial_share_section = _table_section(
        "5. Филиалы Bpilot: доли путей",
        _block_filters(
            "Фильтры",
            [
                ("база", "базовые фильтры + Bpilot"),
                ("сегменты", "как в разделе 3, но по каждому филиалу"),
            ],
        ),
        _block_formula(
            "Смысл",
            [
                "Сравниваем частоты agreement / pretension / FU / court "
                "внутри model vs control (или кейсов) <em>по филиалу</em>.",
            ],
        ),
        _block_example("Чтение", "Смотрите пары строк одного филиала с разными segment."),
        _block_legend(
            "<li><code>filial</code> + <code>segment</code></li>",
            "<li><code>*_share</code> в %</li>",
        ),
        filial_shares,
    )

    dist = analytics.get("result_distribution_bpilot")
    dist_section = _table_section(
        "6. Распределение РезультатПроверки + выплаты (Bpilot)",
        _block_filters(
            "Фильтры",
            [("база", "базовые фильтры + Bpilot")],
        ),
        _block_formula(
            "Корзины",
            [
                "<strong>model_0 / model_1</strong> — решения модели",
                "<strong>model_rucheek</strong> — 0 ∪ 1",
                "<strong>control</strong> — −100",
                "<strong>other / missing</strong> — прочие / пустые значения",
            ],
        ),
        _block_example(
            "Чтение",
            "<code>share_of_base</code> — доля корзины от Bpilot; "
            "money-колонки — по выбранной amount_col.",
        ),
        _block_legend(
            "<li><code>other</code> — число ≠ {−100,0,1}</li>"
            "<li><code>missing</code> — пусто / не число</li>",
            "<li><code>n</code> — строк</li>"
            "<li><code>share_of_base</code> — % от базы</li>"
            "<li><code>n_amount</code> — непустых денежных значений</li>"
            "<li><code>sum/mean/median/…</code> — статистики выплаты</li>",
        ),
        dist,
    )

    pay = analytics.get("payments_bpilot")
    pay_section = _table_section(
        "7. Выплаты по сегментам (Bpilot)",
        _block_filters(
            "Фильтры",
            [
                ("база", "базовые фильтры + Bpilot"),
                ("сегменты", "вариант 1: model/control; вариант 2: 4 кейса"),
            ],
        ),
        _block_formula(
            "Метрики",
            ["sum, mean, median, p25, p75, min, max по amount_col сегмента"],
        ),
        _block_example("Чтение", "Сравнивайте типичный размер выплаты model vs control."),
        _block_legend(
            "<li>строка = сегмент</li>",
            "<li><code>n</code> — строк сегмента</li>"
            "<li><code>n_amount</code> — непустых сумм</li>",
        ),
        pay,
    )

    cases = analytics.get("cases_bpilot")
    cases_section = _table_section(
        "8. Диагностика кейсов call/result/payout (Bpilot)",
        _block_filters(
            "Фильтры",
            [("база", "базовые фильтры + Bpilot")],
        ),
        _block_formula(
            "Счётчики",
            [
                "n_applied_one_paid / n_ignored_zero_paid / "
                "n_out_of_model_paid / n_recommended_unpaid"
            ],
        ),
        _block_example(
            "Зачем",
            "Показывает, насколько часто рекомендация модели игнорируется "
            "или выплата возникает вне ручейка.",
        ),
        _block_legend(
            "<li><code>total</code> и строки по филиалам</li>",
            "<li>колонки — счётчики четырёх кейсов</li>",
        ),
        cases,
    )

    bam_seg = analytics.get("segments_bam")
    bam_section = _table_section(
        "9. Контур Bam (Архангельский / Марийский)",
        _block_filters(
            "Фильтры Bam",
            [
                ("база", "базовые фильтры + филиал ∈ {Архангельский, Марийский}"),
                ("ожидание", "~100% РезультатПроверки ∈ {0,1}, контроль почти пуст"),
            ],
        ),
        _block_formula(
            "Аналитика",
            ["Те же доли / кейсы, что и для Bpilot, но на Bam."],
        ),
        _block_example("Чтение", _share_example(bam_seg)),
        _block_legend(
            "<li>сегменты как в разделе 3</li>",
            "<li>доли в %</li>",
        ),
        bam_seg,
    )

    extra_bam = ""
    for key, title_ in (
        ("filial_usage_bam", "9.1 Bam: usage по филиалам"),
        ("filial_shares_bam", "9.2 Bam: доли путей по филиалам"),
        ("result_distribution_bam", "9.3 Bam: распределение РезультатПроверки"),
        ("payments_bam", "9.4 Bam: выплаты"),
        ("cases_bam", "9.5 Bam: кейсы"),
    ):
        frame = analytics.get(key)
        if frame is None:
            continue
        extra_bam += f"""
  <h3>{escape(title_)}</h3>
  <div class="card">
    {_df_to_html_table(frame)}
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
  p_fu={_pct(p.p_fu)}, p_court={_pct(p.p_court)}, e_fee={_money(effect.e_fee)}</p>

  <h2>1. База и термины</h2>
  <div class="card">
    {_base_glossary()}
    {_base_filters_block()}
    <h3>Карта расчётов: фильтры и показатели</h3>
    <p class="muted">Каждая строка показывает, на какой выборке считается слой
    и какие метрики попадают в таблицу.</p>
    {_filters_overview_table(variant, effect)}
  </div>

  {fin_section}
  {share_section}
  {usage_section}
  {filial_share_section}
  {dist_section}
  {pay_section}
  {cases_section}
  {bam_section}
  {extra_bam}
</div>
</body>
</html>
"""


def write_monitoring_html(
    effect: MonitoringEffectResult,
    path: str | Path,
    *,
    variant: int = 1,
    annual: dict[str, Any] | None = None,
    analytics: dict[str, Any] | None = None,
    source_label: str = "[OISUU_report].[dbo].[ВитринаСутяжность]",
    title: str | None = None,
    **_ignored: Any,
) -> Path:
    """Записать HTML-отчёт варианта 1 или 2."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html = build_monitoring_html(
        effect,
        variant=variant,
        annual=annual,
        analytics=analytics,
        source_label=source_label,
        title=title,
    )
    html = "\n".join(line.rstrip() for line in html.splitlines()) + "\n"
    path.write_text(html, encoding="utf-8")
    return path


def write_both_monitoring_htmls(
    effect_v1: MonitoringEffectResult,
    effect_v2: MonitoringEffectResult,
    data_dir: str | Path,
    *,
    annual_v1: dict[str, Any] | None = None,
    annual_v2: dict[str, Any] | None = None,
    analytics_v1: dict[str, Any] | None = None,
    analytics_v2: dict[str, Any] | None = None,
    source_label: str = "[OISUU_report].[dbo].[ВитринаСутяжность]",
) -> tuple[Path, Path]:
    """Записать оба HTML: variant1 и variant2."""
    data_dir = Path(data_dir)
    p1 = write_monitoring_html(
        effect_v1,
        data_dir / "fin_effect_report_v1_rucheek.html",
        variant=1,
        annual=annual_v1,
        analytics=analytics_v1,
        source_label=source_label,
    )
    p2 = write_monitoring_html(
        effect_v2,
        data_dir / "fin_effect_report_v2_cases.html",
        variant=2,
        annual=annual_v2,
        analytics=analytics_v2,
        source_label=source_label,
    )
    return p1, p2
