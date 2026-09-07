"""HTML-отчёт по мониторингу финэффекта (Excel + ретро-приоры)."""
from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

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
h2 { margin: 1.4rem 0 0.45rem; font-size: 1.15rem; color: var(--accent); }
h3 { margin: 1rem 0 0.35rem; font-size: 1rem; }
.sub { color: var(--muted); margin: 0 0 1rem; }
.card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 0.85rem 1rem;
  margin: 0.6rem 0;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 0.5rem;
  margin: 0.6rem 0;
}
.stat {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 0.55rem 0.65rem;
}
.stat span { display: block; color: var(--muted); font-size: 0.78rem; }
.stat b { font-variant-numeric: tabular-nums; }
table {
  width: 100%;
  border-collapse: collapse;
  background: var(--surface);
  border: 1px solid var(--line);
  font-size: 0.9rem;
  margin: 0.4rem 0 0.7rem;
}
th, td {
  border: 1px solid var(--line);
  padding: 0.4rem 0.5rem;
  text-align: left;
  vertical-align: top;
}
th { background: var(--soft); color: var(--accent); }
code, pre {
  font-family: Consolas, "Courier New", monospace;
  font-size: 0.84rem;
}
pre {
  background: var(--code);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 0.75rem 0.85rem;
  overflow-x: auto;
  white-space: pre-wrap;
}
.formula {
  background: var(--soft);
  border-left: 4px solid var(--accent);
  padding: 0.65rem 0.85rem;
  margin: 0.5rem 0;
  font-family: Consolas, "Courier New", monospace;
  white-space: pre-wrap;
}
ul { margin: 0.35rem 0 0.6rem 1.2rem; }
.muted { color: var(--muted); }
"""


def _pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"


def _money(x: float) -> str:
    return format_money(float(x))


def _example_walkthrough(effect: MonitoringEffectResult) -> str:
    """Числовой разбор на фактических итогах прогона."""
    p = effect.priors
    n = effect.n_intervention
    e_fee = effect.e_fee
    sum_od = effect.sum_od
    fees_total = n * e_fee
    od_term = sum_od * p.k
    raw = od_term + fees_total
    return f"""n_I = {n}
e_fee (на 1 кейс) = p_fu×{ _money(p.fu_fee) } + p_court×{ _money(p.court_fee) }
                 = {_pct(p.p_fu)}×{_money(p.fu_fee)} + {_pct(p.p_court)}×{_money(p.court_fee)}
                 = {_money(e_fee)}

Σ_I ОД = {_money(sum_od)}
Σ_I (ОД×k) = {_money(sum_od)} × {p.k:.4f} = {_money(od_term)}
Σ_I e_fee  = {n} × {_money(e_fee)} = {_money(fees_total)}

сырой ПСР+взносы = {_money(od_term)} + {_money(fees_total)} = {_money(raw)}
expected_psr = precision × … = {p.precision:.4f} × {_money(raw)} = {_money(effect.expected_psr)}

cost = Σ_I СуммаПлатежа = {_money(effect.cost)}
net  = expected_psr − cost = {_money(effect.expected_psr)} − {_money(effect.cost)} = {_money(effect.net)}
"""


def build_monitoring_html(
    effect: MonitoringEffectResult,
    *,
    annual: dict[str, Any] | None = None,
    segments: Any | None = None,
    title: str = "Мониторинг фин. эффекта Querulus",
) -> str:
    """Собрать HTML-строку отчёта."""
    p = effect.priors
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    window = (
        f"{p.window_start or '?'} … {p.window_end or '?'}"
        if p.window_start or p.window_end
        else "окно не применено (нет даты / lookback=0)"
    )

    annual_block = ""
    if annual:
        rows = "".join(
            f"<tr><td>{escape(str(k))}</td><td>{escape(str(v))}</td></tr>"
            for k, v in annual.items()
        )
        annual_block = f"""
        <h2>Экстраполяция на год</h2>
        <div class="card">
          <table><thead><tr><th>Поле</th><th>Значение</th></tr></thead>
          <tbody>{rows}</tbody></table>
        </div>
        """

    seg_block = ""
    if segments is not None:
        try:
            seg_html = segments.to_html(index=False, border=0)
        except Exception:
            seg_html = f"<pre>{escape(str(segments))}</pre>"
        seg_block = f"""
        <h2>Доли соглашений и претензий</h2>
        <div class="card">
          <p class="muted">Сегмент <b>with_model</b> =
          общие фильтры ∧ ВызовМодельСутяжность=1 ∧ Выплата по модели=1
          (без отдельного фильтра «Заключено соглашение»).</p>
          {seg_html}
        </div>
        """

    code_block = escape(
        """# ретро-приоры за последние 2 года
priors = compute_retro_priors(retro, precision=PRECISION, lookback_years=2.0)

# финэффект на Excel
effect = estimate_monitoring_effect(df, priors,
                                    od_col="СуммаОсновногоДолгаЗаявлено",
                                    cost_col="СуммаКВыплате")
# внутри:
#   база = филиал/форма/статус/автотранспорт
#   I = база & ВызовМодельСутяжность=1 & Выплата по модели=1
#   e_fee = p_fu*100_000 + p_court*15_000
#   expected_psr = precision * (sum(ОД_I * k) + n_I * e_fee)
#   cost = sum(СуммаКВыплате_I)
#   net = expected_psr - cost
"""
    )

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
  <p class="sub">Сгенерировано {escape(generated)}. Это оценка без зрелого факта ПСР на выгрузке.</p>

  <div class="grid">
    <div class="stat"><span>Интервенции I</span><b>{effect.n_intervention}</b></div>
    <div class="stat"><span>expected_psr</span><b>{_money(effect.expected_psr)}</b></div>
    <div class="stat"><span>cost (СуммаПлатежа)</span><b>{_money(effect.cost)}</b></div>
    <div class="stat"><span>net</span><b>{_money(effect.net)}</b></div>
  </div>

  <h2>1. Что считаем</h2>
  <div class="card">
    <p><b>Интервенция I</b> — общие фильтры аналитики плюс вызов модели и выплата по модели:</p>
    <div class="formula">общие:
  Филиал ∉ {{Архангельский, Марийский}}
  ∧ ФормаВозмещения ∈ {{денежная, ремонт, соглашение}}
  ∧ УбытокСтатус = первичный
  ∧ ТипОбъектаАвтотранспорт = 1

I = общие
  ∧ ВызовМодельСутяжность = 1
  ∧ Выплата по модели в Инциденте = 1</div>
    <p>На этих кейсах оцениваем, сколько ПСР+взносов мы <i>ожидаемо избежали</i>,
    и вычитаем фактическую сумму к выплате.</p>
  </div>

  <h2>2. Формулы</h2>
  <div class="card">
    <div class="formula">e_fee = p_fu × 100_000 + p_court × 15_000

expected_psr = precision × Σ_{{i ∈ I}} (ОД_заявлено_i × k + e_fee)
             = precision × ( Σ_I ОД × k  +  n_I × e_fee )

cost = Σ_{{i ∈ I}} СуммаКВыплате_i

net  = expected_psr − cost</div>
    <ul>
      <li><b>ОД</b> (<code>{escape(effect.od_column)}</code>) — только чтобы оценить ПСР через коэффициент <code>k</code>.</li>
      <li><b>cost</b> — факт к выплате (<code>{escape(effect.cost_column)}</code>).</li>
      <li><b>e_fee</b> в summary — средний взнос <i>на один кейс</i>, не сумма по всем I.
          В expected_psr входит как <code>n_I × e_fee</code>.</li>
      <li><b>precision</b> — доля «правильных» срабатываний модели (задаётся вручную или с preds).</li>
      <li>Флаги Excel: претензия <code>ЕстьПретензияВИнциденте</code>, ФУ <code>Обращение к ФУ</code>,
          суд <code>Обращение к суду</code> (для долей/аналитики; e_fee по-прежнему с ретро-долей путей).</li>
    </ul>
  </div>

  <h2>3. Что такое p_fu и p_court</h2>
  <div class="card">
    <p>Среди ретро-кейсов с фактом ПСР (<code>TARGET_FREQ = 1</code>) делим пути
    (взаимоисключающе, приоритет суд → ФУ → претензия):</p>
    <table>
      <thead><tr><th>Доля</th><th>Смысл</th><th>Взнос в e_fee</th><th>Сейчас</th></tr></thead>
      <tbody>
        <tr>
          <td><code>p_pret</code></td>
          <td>ПСР остановился на претензии (без ФУ/иска)</td>
          <td>0 ₽</td>
          <td>{_pct(p.p_pret)}</td>
        </tr>
        <tr>
          <td><code>p_fu</code></td>
          <td>Дошли до финансового уполномоченного, без суда</td>
          <td>100 000 ₽</td>
          <td>{_pct(p.p_fu)}</td>
        </tr>
        <tr>
          <td><code>p_court</code></td>
          <td>Дошли до суда (иск)</td>
          <td>15 000 ₽</td>
          <td>{_pct(p.p_court)}</td>
        </tr>
      </tbody>
    </table>
    <p class="muted">Это не «доля всех убытков», а структура путей <b>внутри уже случившегося ПСР</b>.
    Отдельно <code>psr_share</code> = доля кейсов с ПСР во всём окне ретро =
    {_pct(p.psr_share)} ({p.n_pos} / {p.n_rows}).</p>
  </div>

  <h2>4. Период ретро для долей и k</h2>
  <div class="card">
    <p>Раньше priors брались со <b>всего</b> parquet. Сейчас по умолчанию —
    <b>последние 2 года</b> по дате T0 выплаты (<code>PAYMENT_ORDER_DATE_TIME</code>
    или запасная дата).</p>
    <table>
      <thead><tr><th>Параметр</th><th>Значение</th></tr></thead>
      <tbody>
        <tr><td>lookback_years</td><td>{p.lookback_years}</td></tr>
        <tr><td>date_column</td><td>{escape(str(p.date_column))}</td></tr>
        <tr><td>окно</td><td>{escape(window)}</td></tr>
        <tr><td>строк в окне</td><td>{p.n_rows}</td></tr>
        <tr><td>из них ПСР (TARGET_FREQ=1)</td><td>{p.n_pos}</td></tr>
        <tr><td>psr_share</td><td>{_pct(p.psr_share)}</td></tr>
        <tr><td>k</td><td>{p.k:.6f}</td></tr>
        <tr><td>precision</td><td>{p.precision:.4f}</td></tr>
        <tr><td>e_fee (на кейс)</td><td>{_money(effect.e_fee)}</td></tr>
      </tbody>
    </table>
  </div>

  <h2>5. Пример расчёта на этом прогоне</h2>
  <div class="card">
    <pre>{escape(_example_walkthrough(effect))}</pre>
  </div>

  <h2>6. Код</h2>
  <div class="card">
    <pre>{code_block}</pre>
    <p class="muted">Модули: <code>querulus.fin_effect.excel_monitoring</code>,
    отчёт — <code>querulus.fin_effect.monitoring_report</code>.</p>
  </div>

  {annual_block}
  {seg_block}

  <h2>Итог прогона</h2>
  <div class="card">
    <table>
      <thead><tr><th>Поле</th><th>Значение</th></tr></thead>
      <tbody>
        <tr><td>od_column</td><td>{escape(effect.od_column)}</td></tr>
        <tr><td>cost_column</td><td>{escape(effect.cost_column)}</td></tr>
        <tr><td>n_intervention</td><td>{effect.n_intervention}</td></tr>
        <tr><td>sum_od</td><td>{_money(effect.sum_od)}</td></tr>
        <tr><td>sum_paid / cost</td><td>{_money(effect.cost)}</td></tr>
        <tr><td>e_fee</td><td>{_money(effect.e_fee)}</td></tr>
        <tr><td>expected_psr</td><td>{_money(effect.expected_psr)}</td></tr>
        <tr><td>net</td><td>{_money(effect.net)}</td></tr>
      </tbody>
    </table>
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
    title: str = "Мониторинг фин. эффекта Querulus",
) -> Path:
    """Записать HTML-отчёт на диск."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html = build_monitoring_html(
        effect, annual=annual, segments=segments, title=title
    )
    path.write_text(html, encoding="utf-8")
    return path
