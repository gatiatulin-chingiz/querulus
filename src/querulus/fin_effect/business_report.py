"""Одностраничный HTML: формулы фин. эффекта, старый vs новый, цифры прогона."""
from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from querulus import PROJECT_ROOT
from querulus.fin_effect.calculator import FinEffectResult
from querulus.fin_effect.config import FinEffectConfig
from querulus.fin_effect.summary import create_summary_table

DEFAULT_HTML_PATH = PROJECT_ROOT / "notebooks" / "fin_effect_detailed.html"

_CSS = """
:root {
  --bg: #f7f5f1; --surface: #fff; --ink: #1a1a1a; --muted: #5c5c5c;
  --line: #ddd8ce; --accent: #0d5c63; --accent-soft: #e6f2f3;
  --old: #6b4c7a; --old-soft: #f3edf5; --new: #1a6b4a; --new-soft: #e8f5ef;
  --save: #1a6b4a; --loss: #8b3a3a;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 14px;
  line-height: 1.4;
  color: var(--ink);
  background: var(--bg);
}
.page {
  max-width: 1100px;
  margin: 0 auto;
  padding: 1rem 1.1rem 1.5rem;
}
h1 {
  margin: 0 0 0.25rem;
  font-size: 1.35rem;
  color: var(--accent);
}
.sub { margin: 0 0 0.75rem; color: var(--muted); font-size: 0.9rem; }
.hero {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.45rem;
  margin: 0.6rem 0 0.85rem;
}
@media (max-width: 720px) { .hero { grid-template-columns: 1fr 1fr; } }
.stat {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 0.55rem 0.65rem;
}
.stat span { display: block; color: var(--muted); font-size: 0.78rem; }
.stat b { font-size: 1.05rem; font-variant-numeric: tabular-nums; }
.stat.save b { color: var(--save); }
.stat.loss b { color: var(--loss); }
table {
  width: 100%;
  border-collapse: collapse;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 6px;
  overflow: hidden;
  font-size: 0.86rem;
  margin: 0.4rem 0 0.7rem;
}
th, td {
  border: 1px solid var(--line);
  padding: 0.4rem 0.5rem;
  text-align: left;
  vertical-align: top;
}
th { background: var(--accent-soft); color: var(--accent); font-weight: 600; }
th.old, td.old { background: var(--old-soft); }
th.new, td.new { background: var(--new-soft); }
td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.scroll { overflow-x: auto; margin: 0.4rem 0 0.7rem; }
.formula {
  font-family: Consolas, "Courier New", monospace;
  font-size: 0.82rem;
  white-space: nowrap;
}
.changed { font-weight: 600; color: var(--new); }
.same { color: var(--muted); }
.note {
  background: #fff8e6;
  border: 1px solid #ead9a8;
  border-radius: 6px;
  padding: 0.5rem 0.7rem;
  margin: 0.5rem 0;
  font-size: 0.86rem;
}
.legend { margin: 0.35rem 0 0.55rem; color: var(--muted); font-size: 0.82rem; }
.legend code {
  font-family: Consolas, "Courier New", monospace;
  background: #eeece7;
  padding: 0 0.25rem;
  border-radius: 3px;
}
h2 {
  margin: 0.85rem 0 0.35rem;
  font-size: 1.02rem;
  color: var(--accent);
}
p { margin: 0.3rem 0; }
@media print {
  body { background: #fff; }
  .page { max-width: none; padding: 0; }
  .stat, table { break-inside: avoid; }
}
"""


def _rub(value: float | None) -> str:
    if value is None:
        return "—"
    number = int(round(float(value)))
    text = f"{abs(number):,}".replace(",", " ")
    return f"−{text} ₽" if number < 0 else f"{text} ₽"


def _int(value: int | None) -> str:
    if value is None:
        return "—"
    return f"{int(value):,}".replace(",", " ")


def _cell_num(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return escape(str(value))
    if abs(number - round(number)) < 1e-9:
        text = f"{int(round(number)):,}".replace(",", " ")
    else:
        text = f"{number:,.2f}".replace(",", " ")
    return text


def _period(frame: pd.DataFrame | None, config: FinEffectConfig) -> str:
    if frame is None or config.date_column not in getattr(frame, "columns", []):
        return ""
    dates = pd.to_datetime(frame[config.date_column], errors="coerce")
    if dates.notna().sum() == 0:
        return ""
    return f"{dates.min():%d.%m.%Y}–{dates.max():%d.%m.%Y}"


def _hero(
    *,
    n: int | None,
    threshold: float | None,
    fact_total: float | None,
    model_total: float | None,
    net_effect: float | None,
) -> str:
    if n is None:
        return (
            '<p class="sub">Цифры появятся после прогона блока финансового эффекта '
            "в тетради.</p>"
        )
    net_cls = "stat"
    if net_effect is not None and net_effect > 0:
        net_cls = "stat save"
    elif net_effect is not None and net_effect < 0:
        net_cls = "stat loss"
    thr = "—" if threshold is None else f"{threshold:.2f}"
    return f"""
<div class="hero">
  <div class="stat"><span>Дел</span><b>{_int(n)}</b></div>
  <div class="stat"><span>Порог</span><b>{thr}</b></div>
  <div class="stat"><span>Расход без модели</span><b>{_rub(fact_total)}</b></div>
  <div class="{net_cls}"><span>Чистый эффект (новый)</span><b>{_rub(net_effect)}</b></div>
</div>
<p class="sub">Расход по модели: {_rub(model_total)}. Чистый эффект = модель − факт.</p>
"""


def _summary_table_html(summary: pd.DataFrame | None) -> str:
    if summary is None or summary.empty:
        return (
            '<p class="sub">Сводная таблица появится после прогона блока '
            "[C] Fin-effect на holdout Test.</p>"
        )
    headers = "".join(f"<th>{escape(str(col))}</th>" for col in summary.columns)
    body_rows: list[str] = []
    for row in summary.itertuples(index=False):
        cells: list[str] = []
        for idx, value in enumerate(row):
            col = str(summary.columns[idx])
            if col in ("Факт", "Классификация"):
                cells.append(f"<td class='num'>{int(value)}</td>")
            elif col == "Количество инцидентов с иными взысканиями":
                cells.append(f"<td class='num'>{_int(int(value))}</td>")
            else:
                cells.append(f"<td class='num'>{_cell_num(value)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"""
<div class="scroll">
<table>
  <thead><tr>{headers}</tr></thead>
  <tbody>
    {''.join(body_rows)}
  </tbody>
</table>
</div>
<p class="legend">
  Экономия = ФИН. ЭФФЕКТ ФАКТ − ФИН. ЭФФЕКТ МОДЕЛЬ.
  Регрессия заполняется только при классификации = 1. Сводка только агрегирует колонки calculator.
</p>
"""


def render_business_html(
    result: FinEffectResult | None = None,
    config: FinEffectConfig | None = None,
    *,
    subtitle: str = "",
    summary: pd.DataFrame | None = None,
) -> str:
    """Одна страница: легенда, старый vs новый, сводка прогона, пример."""
    config = config or FinEffectConfig()
    frame = None if result is None else result.frame
    if summary is None and frame is not None:
        summary = create_summary_table(frame, config)
    n = None if frame is None else int(len(frame))
    threshold = None if result is None else float(result.best_threshold)
    fact_total = None if result is None else float(result.fact_effect_total)
    model_total = None if result is None else float(result.model_effect_total)
    net_effect = None if result is None else float(result.net_effect)
    period = _period(frame, config)
    saved_at = datetime.now().strftime("%d.%m.%Y %H:%M")
    sub_bits = [escape(subtitle)] if subtitle else []
    if period:
        sub_bits.append(escape(period))
    sub_bits.append(escape(saved_at))
    sub_line = " · ".join(bit for bit in sub_bits if bit)

    fu = f"{int(config.fu_fee_amount):,}".replace(",", " ")
    court_fee = f"{int(config.court_fee_amount):,}".replace(",", " ")
    court_note = (
        f"судебный взнос {court_fee} ₽ включён"
        if config.apply_court_fee
        else "судебный взнос выключен"
    )

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Финансовый эффект: старый и новый расчёт</title>
  <style>{_CSS}</style>
</head>
<body>
<div class="page">
  <h1>Финансовый эффект модели: старый и новый расчёт</h1>
  <p class="sub">{sub_line}</p>

  {_hero(
      n=n,
      threshold=threshold,
      fact_total=fact_total,
      model_total=model_total,
      net_effect=net_effect,
  )}

  <h2>Сводка на holdout Test (порог с Val)</h2>
  {_summary_table_html(summary)}

  <p class="legend">
    <strong>Обозначения.</strong>
    Факт — было ли взыскание.
    Модель — решение «будет / не будет» по порогу.
    <code>pred_sev</code> — сумма, предложенная моделью.
    <code>T</code> — фактическая тяжесть (основной долг + износ + утрата товарной стоимости).
    <code>ПСР</code> — фактическая сумма взыскания (иски с госпошлиной + доплаты по претензиям).
    Взносы — взнос финансовому уполномоченному {fu} ₽ (если был), {court_note}.
    Расход пишется со знаком минус.
  </p>

  <h2>1. База факта</h2>
  <table>
    <thead>
      <tr>
        <th></th>
        <th class="old">Старый</th>
        <th class="new">Новый</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Из чего складывается факт</td>
        <td class="old">претензии + ФУ + иск + взносы</td>
        <td class="new">ПСР + взносы</td>
      </tr>
    </tbody>
  </table>

  <h2>2. Формулы по ситуациям (расход модели)</h2>
  <table>
    <thead>
      <tr>
        <th>Ситуация (факт → модель)</th>
        <th class="old">Старый</th>
        <th class="new">Новый</th>
        <th>Что изменилось</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Нет → нет</td>
        <td class="old formula">−(ПСР + взносы)</td>
        <td class="new formula">0</td>
        <td class="changed">Раньше штрафовали «тишину», теперь ноль</td>
      </tr>
      <tr>
        <td>Нет → да (ложная тревога)</td>
        <td class="old formula">−pred_sev − (ПСР + взносы)</td>
        <td class="new formula">−pred_sev</td>
        <td class="changed">Убрали двойной штраф: раньше модель + факт, теперь только предложенная сумма</td>
      </tr>
      <tr>
        <td>Да → нет (пропуск)</td>
        <td class="old formula">−(ПСР + взносы)</td>
        <td class="new formula">−(ПСР + взносы)</td>
        <td class="same">Без изменений</td>
      </tr>
      <tr>
        <td>Да → да, хватило (pred_sev ≥ T)</td>
        <td class="old formula">−pred_sev</td>
        <td class="new formula">−pred_sev</td>
        <td class="same">Без изменений</td>
      </tr>
      <tr>
        <td>Да → да, не хватило (pred_sev &lt; T)</td>
        <td class="old formula">−(ПСР + взносы)</td>
        <td class="new formula">−(ПСР × (1 − pred_sev / T) + взносы)</td>
        <td class="changed">Засчитывается доля покрытия</td>
      </tr>
    </tbody>
  </table>

  <h2>3. Пример: «хватило / не хватило»</h2>
  <p>
    T = 80&nbsp;000, pred_sev = 40&nbsp;000, ПСР = 100&nbsp;000, взносы = 100&nbsp;000
    → доля покрытия = 40&nbsp;000 / 80&nbsp;000 = 0,5.
  </p>
  <table>
    <thead>
      <tr>
        <th></th>
        <th class="old">Старый</th>
        <th class="new">Новый</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Формула</td>
        <td class="old formula">−(100 000 + 100 000)</td>
        <td class="new formula">−(100 000 × 0,5 + 100 000)</td>
      </tr>
      <tr>
        <td>Результат</td>
        <td class="old"><strong>−200 000 ₽</strong></td>
        <td class="new"><strong>−150 000 ₽</strong></td>
      </tr>
    </tbody>
  </table>
  <div class="note">
    Разница +50&nbsp;000 ₽: новый расчёт отдаёт модели кредит за половину покрытой планки.
    Взносы при недоборе не сокращаются.
  </div>

  <h2>4. Чистый эффект</h2>
  <p class="formula">чистый эффект = расход модели − расход без модели</p>
  <p>
    Плюс — экономия относительно факта. Минус — модель дороже факта.
    Порог подбирают так, чтобы чистый эффект был наибольшим.
  </p>
</div>
</body>
</html>
"""


def export_business_html(
    result: FinEffectResult | None = None,
    config: FinEffectConfig | None = None,
    *,
    path: str | Path | None = None,
    subtitle: str = "",
    summary: pd.DataFrame | None = None,
) -> Path:
    """Записать HTML. Без ``result`` — формулы без цифр прогона."""
    output = Path(path) if path is not None else DEFAULT_HTML_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_business_html(
            result, config, subtitle=subtitle, summary=summary
        ),
        encoding="utf-8",
    )
    return output
