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
.page { max-width: 980px; margin: 0 auto; padding: 1.2rem 1.1rem 2rem; }
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
  font-size: 0.88rem;
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
  background: var(--soft);
  border-left: 4px solid var(--accent);
  padding: 0.55rem 0.75rem;
  margin: 0.4rem 0;
  font-family: Consolas, "Courier New", monospace;
  white-space: pre-wrap;
  font-size: 0.84rem;
}
ul { margin: 0.3rem 0 0.45rem 1.15rem; padding: 0; }
li { margin: 0.15rem 0; }
.muted { color: var(--muted); font-size: 0.9rem; }
"""

# Семантика только ключевых колонок (без повтора фильтров/формул).
COLUMN_SEMANTICS: list[dict[str, str]] = [
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
        return df.to_html(index=False, border=0, escape=True)
    except Exception:
        return f"<pre>{escape(str(df))}</pre>"


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
        <h2>5. Результат долей (этот прогон)</h2>
        <div class="card">
          {_df_to_html_table(segments)}
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
    <div class="formula">B (база аналитики) =
  Филиал ∉ {{Архангельский, Марийский}}
  ∧ ФормаВозмещения ∈ {{денежная, ремонт, соглашение}}
  ∧ УбытокСтатус = первичный
  ∧ ТипОбъектаАвтотранспорт = 1

I (финэффект) = B ∧ ВызовМодельСутяжность=1 ∧ Выплата по модели в Инциденте=1

W (with_model, доли) = B ∧ ВызовМодельСутяжность=1 ∧ Выплата по модели в Инциденте=1
  (= I по флагам модели; без требования «соглашение=1»)
U (without_model) = B \\ W</div>
    <p class="muted">ФУ/суд для долей: сначала
    <code>ЕстьОбращениеКФУ/СудуВИнциденте = max(флаг) по НомерИнцидент</code>,
    иначе на первичных строках loss-флаги почти всегда 0.
    <code>p_fu</code>/<code>p_court</code> для e_fee — только с ретро, не с витрины.</p>
  </div>

  <h2>2. Финэффект</h2>
  <div class="card">
    <div class="formula">e_fee = p_fu×100_000 + p_court×15_000
expected_psr = precision × ( Σ_I ОД_заявлено×k + n_I×e_fee )
cost = Σ_I {escape(effect.cost_column)}
net = expected_psr − cost

ОД = {escape(effect.od_column)}</div>
    <p><b>Пример этого прогона:</b> {escape(_fin_effect_example(effect))}</p>
  </div>

  {cost_block}

  <h2>4. Доли соглашений / претензий / ФУ / суда</h2>
  <div class="card">
    <div class="formula">на сегменте S ∈ {{W, U}}:
  agreement_share(S)     = mean(Заключено соглашение | S)
  pretension_share(S)    = mean(ЕстьПретензияВИнциденте | S)
  fu_incident_share(S)   = mean(ЕстьОбращениеКФУВИнциденте | S)
  court_incident_share(S)= mean(ЕстьОбращениеКСудуВИнциденте | S)

lift_pp  = share(W) − share(U)
lift_rel = share(W)/share(U) − 1</div>
    <p><b>Пример этого прогона:</b> {escape(_segment_example(segments))}</p>
  </div>

  {seg_block}

  <h2>6. Колонки витрины (справка)</h2>
  <div class="card">
    <table>
      <thead>
        <tr><th>Колонка</th><th>Источник ETL</th><th>Зачем</th><th>Пример</th></tr>
      </thead>
      <tbody>{semantics_rows}</tbody>
    </table>
    <p class="muted">В витрине нет отдельной <code>СуммаВыплаты</code>: она уже лежит
    под именем <code>СуммаКВыплате</code>.</p>
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
        source_label=source_label,
        title=title,
    )
    path.write_text(html, encoding="utf-8")
    return path
