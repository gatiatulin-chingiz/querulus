"""HTML-отчёт о финансовом эффекте модели — для бизнеса, с цифрами прогона."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd

from querulus import PROJECT_ROOT
from querulus.fin_effect.calculator import FinEffectResult
from querulus.fin_effect.config import FinEffectConfig

DEFAULT_HTML_PATH = PROJECT_ROOT / "notebooks" / "fin_effect_detailed.html"

_CSS = """
:root {
  --bg: #f7f5f1;
  --surface: #fff;
  --ink: #1a1a1a;
  --muted: #5c5c5c;
  --line: #ddd8ce;
  --accent: #0d5c63;
  --accent-soft: #e6f2f3;
  --save: #1a6b4a;
  --save-soft: #e8f5ef;
  --loss: #8b3a3a;
  --loss-soft: #f8ecec;
  --warn: #9a6b00;
  --warn-soft: #fff8e6;
  --neutral: #5c5c5c;
  --neutral-soft: #f3f1ec;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 16px;
  line-height: 1.55;
  color: var(--ink);
  background:
    radial-gradient(ellipse at 8% 0%, #e8f0ef 0%, transparent 42%),
    radial-gradient(ellipse at 92% 8%, #f0e9df 0%, transparent 38%),
    var(--bg);
}
header, main, nav, footer {
  max-width: 980px;
  margin-left: auto;
  margin-right: auto;
  padding-left: 1.25rem;
  padding-right: 1.25rem;
}
header { padding-top: 2rem; padding-bottom: 0.75rem; }
header h1 {
  margin: 0 0 0.4rem;
  font-size: 1.7rem;
  color: var(--accent);
  letter-spacing: -0.02em;
}
header p { margin: 0.35rem 0; color: var(--muted); }
nav {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem 1rem;
  margin-bottom: 1rem;
  position: sticky;
  top: 0;
  z-index: 10;
  background: rgba(247, 245, 241, 0.94);
  padding-top: 0.5rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--line);
}
nav a { color: var(--accent); text-decoration: none; font-size: 0.92rem; }
nav a:hover { text-decoration: underline; }
main { padding-bottom: 2.5rem; }
section {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 1.05rem 1.2rem 1.2rem;
  margin-bottom: 1rem;
}
h2 {
  margin: 0 0 0.7rem;
  font-size: 1.2rem;
  color: var(--accent);
  border-bottom: 2px solid var(--accent-soft);
  padding-bottom: 0.3rem;
}
h3 { margin: 1.05rem 0 0.4rem; font-size: 1.02rem; }
p { margin: 0.45rem 0; }
ul { margin: 0.4rem 0 0.7rem; padding-left: 1.2rem; }
li { margin: 0.35rem 0; }
.lead {
  background: linear-gradient(135deg, var(--accent-soft) 0%, #f8faf9 100%);
  border: 1px solid #b8d8db;
  border-radius: 8px;
  padding: 1rem 1.15rem;
  margin-bottom: 1rem;
}
.hero {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.65rem;
  margin: 0.8rem 0 0.3rem;
}
@media (max-width: 820px) { .hero { grid-template-columns: 1fr 1fr; } }
.stat {
  background: var(--neutral-soft);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 0.85rem 0.95rem;
}
.stat b {
  display: block;
  font-size: 1.25rem;
  margin-top: 0.2rem;
  font-variant-numeric: tabular-nums;
}
.stat span { color: var(--muted); font-size: 0.86rem; }
.stat.save { background: var(--save-soft); border-color: #b8dcc8; }
.stat.save b { color: var(--save); }
.stat.loss { background: var(--loss-soft); border-color: #e2c4c4; }
.stat.loss b { color: var(--loss); }
.note {
  background: var(--warn-soft);
  border: 1px solid #ead9a8;
  border-radius: 8px;
  padding: 0.75rem 0.95rem;
  margin: 0.7rem 0;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
  margin: 0.55rem 0;
}
th, td {
  border: 1px solid var(--line);
  padding: 0.5rem 0.6rem;
  text-align: left;
  vertical-align: top;
}
th {
  background: var(--accent-soft);
  color: var(--accent);
  font-weight: 600;
}
td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
tr:nth-child(even) td { background: #fafaf8; }
.case {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 0.85rem 1rem;
  margin: 0.7rem 0;
  background: #fafaf8;
}
.case h3 { margin: 0 0 0.4rem; }
.case.save { background: var(--save-soft); border-color: #b8dcc8; }
.case.loss { background: var(--loss-soft); border-color: #e2c4c4; }
.case.warn { background: var(--warn-soft); border-color: #ead9a8; }
.muted { color: var(--muted); }
footer { padding-bottom: 2rem; color: var(--muted); font-size: 0.86rem; }
"""


def _rub(value: float | None, *, empty: str = "будет заполнено после прогона") -> str:
    """Сумма в таблице: пробел как разделитель тысяч."""
    if value is None:
        return empty
    number = int(round(float(value)))
    formatted = f"{abs(number):,}".replace(",", " ")
    if number < 0:
        return f"−{formatted} ₽"
    return f"{formatted} ₽"


def _rub_text(value: float | None) -> str:
    """Сумма в предложении, без сокращений."""
    if value is None:
        return "сумма появится после прогона тетради"
    number = int(round(float(value)))
    formatted = f"{abs(number):,}".replace(",", " ")
    if number < 0:
        return f"минус {formatted} рублей"
    return f"{formatted} рублей"


def _int(value: int | None, *, empty: str = "—") -> str:
    if value is None:
        return empty
    return f"{int(value):,}".replace(",", " ")


def _pct(part: float, whole: float) -> str:
    if whole <= 0:
        return "—"
    return f"{100.0 * part / whole:.1f} %"


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


@dataclass(frozen=True)
class SituationRow:
    """Одна ситуация «факт × решение модели»."""

    title: str
    kind: str
    n: int | None
    model_sum: float | None
    fact_sum: float | None
    contribution: float | None
    formula_text: str
    meaning: str
    pred_sev_sum: float | None = None
    true_sev_sum: float | None = None
    recovery_sum: float | None = None
    premiums_sum: float | None = None


def _sum_mask(series: pd.Series, mask: pd.Series) -> float:
    return float(series.loc[mask].sum())


def _collect_situations(
    frame: pd.DataFrame | None,
    config: FinEffectConfig,
) -> list[SituationRow]:
    """Пять ситуаций текущей формулы покрытия."""
    specs = [
        (
            "Модель не ждала взыскания, и взыскания не было",
            "neutral",
            "00",
            "В расход модели ставится ноль.",
            (
                "Оба согласны: после основной выплаты дополнительных денег "
                "с компании не взыщут. Ни факт, ни модель здесь ничего не тратят."
            ),
        ),
        (
            "Модель ждала взыскания, а его не было",
            "loss",
            "01",
            "В расход модели ставится предложенная моделью сумма со знаком минус.",
            (
                "Это ложная тревога. Компания по модели как будто заранее "
                "направляет деньги, которых по факту не потребовалось. "
                "Вся предложенная сумма становится штрафом модели."
            ),
        ),
        (
            "Модель не ждала взыскания, а оно случилось",
            "warn",
            "10",
            (
                "В расход модели ставится фактическая сумма взыскания "
                "плюс взносы, целиком, со знаком минус."
            ),
            (
                "Это пропуск. Модель не предложила заранее закрыть риск, "
                "поэтому компания несёт тот же расход, что и без модели. "
                "На чистый эффект эта ситуация не влияет: модель не лучше "
                "и не хуже факта."
            ),
        ),
        (
            "Модель ждала взыскания, оно случилось, и предложенной суммы хватило",
            "save",
            "11over",
            "В расход модели ставится предложенная моделью сумма со знаком минус.",
            (
                "Предложенной суммы оказалось не меньше фактической тяжести "
                "(основной долг, износ и утрата товарной стоимости). "
                "Компания «закрывает» дело предложенной суммой вместо полного "
                "фактического взыскания и взносов. Если предложенная сумма "
                "меньше факта, появляется экономия. Если больше — модель "
                "переплатила относительно факта."
            ),
        ),
        (
            "Модель ждала взыскания, оно случилось, но предложенной суммы не хватило",
            "warn",
            "11under",
            (
                "Считается доля покрытия: предложенная сумма делится на фактическую "
                "тяжесть. В расход модели ставится незакрытая доля фактической "
                "суммы взыскания плюс взносы, со знаком минус."
            ),
            (
                "Модель угадала, что взыскание будет, но занизила сумму. "
                "Закрытой считается только доля, равная отношению предложенной "
                "суммы к фактической тяжести. Незакрытая доля фактического "
                "взыскания и взносы всё равно идут в расход. Взносы при недоборе "
                "не сокращаются."
            ),
        ),
    ]
    if frame is None or frame.empty:
        return [
            SituationRow(
                title=title,
                kind=kind,
                n=None,
                model_sum=None,
                fact_sum=None,
                contribution=None,
                formula_text=formula,
                meaning=meaning,
            )
            for title, kind, _code, formula, meaning in specs
        ]

    pred = pd.to_numeric(frame["pred_freq"], errors="coerce").fillna(0).astype(int)
    fact = pd.to_numeric(
        frame[config.frequency_target_column], errors="coerce"
    ).fillna(0).astype(int)
    pred_sev = _numeric(frame, "pred_sev")
    true_sev = _numeric(frame, config.severity_target_column)
    recovery = _numeric(frame, config.fact_amount_column)
    premiums = _numeric(frame, config.premiums_column)
    model = _numeric(frame, "fin_effect_model")
    fact_effect = _numeric(frame, "fin_effect_fact")

    masks = {
        "00": (fact == 0) & (pred == 0),
        "01": (fact == 0) & (pred == 1),
        "10": (fact == 1) & (pred == 0),
        "11over": (fact == 1) & (pred == 1) & (pred_sev >= true_sev),
        "11under": (fact == 1) & (pred == 1) & (pred_sev < true_sev),
    }
    rows: list[SituationRow] = []
    for title, kind, code, formula, meaning in specs:
        mask = masks[code]
        rows.append(
            SituationRow(
                title=title,
                kind=kind,
                n=int(mask.sum()),
                model_sum=_sum_mask(model, mask),
                fact_sum=_sum_mask(fact_effect, mask),
                contribution=_sum_mask(model, mask) - _sum_mask(fact_effect, mask),
                formula_text=formula,
                meaning=meaning,
                pred_sev_sum=_sum_mask(pred_sev, mask),
                true_sev_sum=_sum_mask(true_sev, mask),
                recovery_sum=_sum_mask(recovery, mask),
                premiums_sum=_sum_mask(premiums, mask),
            )
        )
    return rows


def _period_text(frame: pd.DataFrame | None, config: FinEffectConfig) -> str:
    if frame is None or config.date_column not in getattr(frame, "columns", []):
        return "период появится после прогона тетради"
    dates = pd.to_datetime(frame[config.date_column], errors="coerce")
    if dates.notna().sum() == 0:
        return "даты в выборке не заполнены"
    start = dates.min().strftime("%d.%m.%Y")
    end = dates.max().strftime("%d.%m.%Y")
    return f"с {start} по {end}"


def _worked_example_html(under: SituationRow) -> str:
    """Числовой разбор недобора: живые средние или учебный пример."""
    if under.n:
        n = max(int(under.n), 1)
        pred = (under.pred_sev_sum or 0.0) / n
        true = (under.true_sev_sum or 0.0) / n
        recovery = (under.recovery_sum or 0.0) / n
        premiums = (under.premiums_sum or 0.0) / n
        source = (
            "Ниже средний разбор по делам текущего прогона, где модель "
            f"правильно ждала взыскание, но предложенной суммы не хватило. "
            f"Таких дел { _int(under.n) }."
        )
    else:
        pred, true, recovery, premiums = 40_000.0, 80_000.0, 100_000.0, 100_000.0
        source = (
            "Тетрадь ещё не подставила живые цифры, поэтому ниже учебный пример "
            "с круглыми суммами. После прогона блока финансового эффекта на этом "
            "месте появятся средние по реальным делам."
        )
    share = 0.0 if true <= 0 else min(max(pred / true, 0.0), 1.0)
    model_cost = -(recovery * (1.0 - share) + premiums)
    fact_cost = -(recovery + premiums)
    saving = model_cost - fact_cost
    return f"""
<p>{escape(source)}</p>
<p>
Допустим, смотрим одно такое дело. Фактическая тяжесть — основной долг, износ
и утрата товарной стоимости — составила {_rub_text(true)}.
Модель предложила {_rub_text(pred)}.
Фактическая сумма взыскания (по искам с госпошлиной и доплатам по претензиям)
составила {_rub_text(recovery)}.
Взносы по этому делу составили {_rub_text(premiums)}.
</p>
<p>
Доля покрытия считается так: предложенная сумма делится на фактическую тяжесть.
Получается {share:.2f}, то есть модель закрыла {100 * share:.0f} процентов
планки тяжести.
</p>
<p>
Расход по модели: незакрытая доля фактического взыскания плюс взносы.
Это { _rub_text(recovery) } умножить на {1.0 - share:.2f}, затем прибавить
{ _rub_text(premiums) }. В знаке расхода получается {_rub_text(model_cost)}.
</p>
<p>
Расход без модели: фактическая сумма взыскания плюс взносы целиком.
В знаке расхода получается {_rub_text(fact_cost)}.
</p>
<p>
Разница на таком деле — {_rub_text(saving)}.
Это ровно та доля фактического взыскания, которую модель успела закрыть.
Взносы при недоборе не экономятся: они входят и в факт, и в модель.
</p>
"""


def _hero_html(
    *,
    n: int | None,
    threshold: float | None,
    fact_total: float | None,
    model_total: float | None,
    net_effect: float | None,
) -> str:
    net_class = "stat"
    if net_effect is not None and net_effect > 0:
        net_class = "stat save"
    elif net_effect is not None and net_effect < 0:
        net_class = "stat loss"
    return f"""
<div class="hero">
  <div class="stat">
    <span>Дел в расчёте</span>
    <b>{_int(n, empty="—")}</b>
  </div>
  <div class="stat">
    <span>Порог вероятности</span>
    <b>{"—" if threshold is None else f"{threshold:.2f}"}</b>
  </div>
  <div class="stat">
    <span>Расход без модели</span>
    <b>{_rub(fact_total)}</b>
  </div>
  <div class="{net_class}">
    <span>Чистый эффект модели</span>
    <b>{_rub(net_effect)}</b>
  </div>
</div>
<p class="muted">
Расход по модели на этой выборке: {_rub_text(model_total)}.
Чистый эффект — это расход по модели минус расход без модели.
Если результат со знаком плюс, модель снизила расход. Если со знаком минус,
модель обошлась дороже факта.
</p>
"""


def _situations_html(rows: list[SituationRow]) -> str:
    parts = [
        """
<p>
Каждое дело попадает ровно в одну ситуацию. Сначала модель отвечает,
будет ли взыскание. Затем, если она сказала «будет», сравнивается
предложенная сумма с фактической тяжестью.
</p>
"""
    ]
    for row in rows:
        cls = {"save": "save", "loss": "loss", "warn": "warn"}.get(row.kind, "")
        n_text = "число дел появится после прогона" if row.n is None else f"{_int(row.n)} дел"
        contrib = ""
        if row.contribution is not None:
            if row.contribution > 0.5:
                contrib = (
                    f"Вклад в чистый эффект: экономия {_rub_text(row.contribution)}."
                )
            elif row.contribution < -0.5:
                contrib = (
                    f"Вклад в чистый эффект: ухудшение на "
                    f"{_rub_text(abs(row.contribution))}."
                )
            else:
                contrib = "Вклад в чистый эффект: ноль."
        parts.append(
            f"""
<article class="case {cls}">
  <h3>{escape(row.title)}</h3>
  <p><strong>{escape(n_text)}.</strong> {escape(row.meaning)}</p>
  <p>{escape(row.formula_text)}</p>
  <p>
    Расход модели по этой группе: {_rub_text(row.model_sum)}.
    Расход без модели по этой группе: {_rub_text(row.fact_sum)}.
    {escape(contrib)}
  </p>
</article>
"""
        )
    return "".join(parts)


def _table_html(rows: list[SituationRow]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{escape(row.title)}</td>"
            f"<td class='num'>{_int(row.n)}</td>"
            f"<td class='num'>{_rub(row.model_sum)}</td>"
            f"<td class='num'>{_rub(row.fact_sum)}</td>"
            f"<td class='num'>{_rub(row.contribution)}</td>"
            "</tr>"
        )
    totals_n = None if any(r.n is None for r in rows) else sum(int(r.n or 0) for r in rows)
    totals_model = None if any(r.model_sum is None for r in rows) else sum(r.model_sum or 0 for r in rows)
    totals_fact = None if any(r.fact_sum is None for r in rows) else sum(r.fact_sum or 0 for r in rows)
    totals_c = None if any(r.contribution is None for r in rows) else sum(r.contribution or 0 for r in rows)
    body.append(
        "<tr>"
        "<td><strong>Всего</strong></td>"
        f"<td class='num'><strong>{_int(totals_n)}</strong></td>"
        f"<td class='num'><strong>{_rub(totals_model)}</strong></td>"
        f"<td class='num'><strong>{_rub(totals_fact)}</strong></td>"
        f"<td class='num'><strong>{_rub(totals_c)}</strong></td>"
        "</tr>"
    )
    return f"""
<table>
  <thead>
    <tr>
      <th>Ситуация</th>
      <th>Число дел</th>
      <th>Расход модели</th>
      <th>Расход без модели</th>
      <th>Вклад в чистый эффект</th>
    </tr>
  </thead>
  <tbody>
    {''.join(body)}
  </tbody>
</table>
<p class="muted">
В таблице расходы показаны со знаком минус. Вклад в чистый эффект со знаком
плюс означает экономию, со знаком минус — ухудшение относительно факта.
</p>
"""


def _penalties_html(rows: list[SituationRow], n_total: int | None) -> str:
    by_title = {row.title: row for row in rows}
    miss = by_title.get("Модель не ждала взыскания, а оно случилось")
    false_pos = by_title.get("Модель ждала взыскания, а его не было")
    under = by_title.get(
        "Модель ждала взыскания, оно случилось, но предложенной суммы не хватило"
    )
    over = by_title.get(
        "Модель ждала взыскания, оно случилось, и предложенной суммы хватило"
    )
    items = [
        (
            "Пропуск взыскания",
            miss,
            "Модель не включила доплату, а по факту деньги взыскали. "
            "Штраф здесь не отдельная сумма: компания просто платит факт целиком.",
        ),
        (
            "Ложная тревога",
            false_pos,
            "Модель включила доплату, а взыскания не было. "
            "Штраф равен всей предложенной сумме.",
        ),
        (
            "Недобор суммы",
            under,
            "Модель включила доплату, но предложенной суммы не хватило "
            "на фактическую тяжесть. Штраф — незакрытая доля фактического "
            "взыскания плюс взносы, которые не сокращаются.",
        ),
        (
            "Суммы хватило, в том числе с запасом",
            over,
            "Это не штраф в узком смысле. Если предложенная сумма больше "
            "фактической тяжести, модель может переплатить относительно "
            "тяжести, но всё равно сравнима с полным фактическим взысканием "
            "и взносами. Переплата относительно факта видна во вкладе "
            "со знаком минус.",
        ),
    ]
    blocks = []
    for title, row, text in items:
        n = "—" if row is None or row.n is None else _int(row.n)
        share = ""
        if row is not None and row.n is not None and n_total:
            share = f" Это {_pct(float(row.n), float(n_total))} выборки."
        money = ""
        if row is not None and row.contribution is not None:
            money = f" Вклад в чистый эффект: {_rub_text(row.contribution)}."
        blocks.append(
            f"<li><strong>{escape(title)}</strong> ({n} дел). {escape(text)}{share}{money}</li>"
        )
    return "<ul>" + "".join(blocks) + "</ul>"


def render_business_html(
    result: FinEffectResult | None = None,
    config: FinEffectConfig | None = None,
    *,
    subtitle: str = "",
) -> str:
    """Собрать HTML. Без результата — та же логика, без живых цифр."""
    config = config or FinEffectConfig()
    frame = None if result is None else result.frame
    rows = _collect_situations(frame, config)
    n = None if frame is None else int(len(frame))
    threshold = None if result is None else float(result.best_threshold)
    fact_total = None if result is None else float(result.fact_effect_total)
    model_total = None if result is None else float(result.model_effect_total)
    net_effect = None if result is None else float(result.net_effect)
    period = _period_text(frame, config)
    saved_at = datetime.now().strftime("%d.%m.%Y %H:%M")
    under = next(
        (
            row
            for row in rows
            if "не хватило" in row.title
        ),
        rows[-1],
    )
    numbers_note = (
        f"Цифры взяты из тетради: {escape(subtitle)}. Период дел по дате поручения на выплату: {escape(period)}."
        if result is not None
        else (
            "Живых цифр прогона пока нет. Откройте тетрадь collect, дойдите до блока "
            "финансового эффекта на отложенной контрольной выборке и выполните его. "
            "Этот файл перезапишется с теми же разделами и с суммами прогона."
        )
    )
    fu_fee = _rub_text(config.fu_fee_amount)
    court_fee_text = (
        f"Судебный взнос { _rub_text(config.court_fee_amount) } сейчас включён, "
        "если по делу есть исковая сумма."
        if config.apply_court_fee
        else (
            f"Судебный взнос { _rub_text(config.court_fee_amount) } в текущем "
            "расчёте выключен и в суммы не входит."
        )
    )
    subtitle_html = f"<p>{escape(subtitle)}</p>" if subtitle else ""
    net_sentence = (
        "После прогона здесь появится вывод, сэкономила модель или нет."
        if net_effect is None
        else (
            f"На этой выборке модель {'снизила' if net_effect >= 0 else 'увеличила'} "
            f"расход на {_rub_text(abs(net_effect))} относительно факта."
        )
    )

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Как считается финансовый эффект модели</title>
  <style>{_CSS}</style>
</head>
<body>
  <header>
    <h1>Как считается финансовый эффект модели</h1>
    <p>
      Документ для бизнеса. Здесь обычным языком разобрано, с чем сравнивается
      модель, из каких частей складывается расход, где модель получает штраф
      и как из этого получается экономия или переплата.
    </p>
    {subtitle_html}
    <p class="muted">Файл собран {escape(saved_at)}.</p>
  </header>
  <nav>
    <a href="#idea">Зачем сравнение</a>
    <a href="#parts">Из чего складывается факт</a>
    <a href="#model">Что отвечает модель</a>
    <a href="#threshold">Порог</a>
    <a href="#cases">Пять ситуаций</a>
    <a href="#penalties">Штрафы</a>
    <a href="#example">Числовой разбор</a>
    <a href="#numbers">Цифры прогона</a>
  </nav>
  <main>
    <div class="lead">
      <p>
        <strong>Коротко.</strong> Без модели компания по каждому делу платит то,
        что с неё фактически взыскали после основной выплаты, плюс взносы.
        Модель заранее говорит, будет ли такое взыскание и какую сумму имеет
        смысл заложить. Финансовый эффект — это не «насколько модель точно
        угадала рубли», а насколько её решения дешевле или дороже этого факта.
      </p>
      <p>{escape(net_sentence)}</p>
    </div>

    <section id="idea">
      <h2>С чем сравниваем модель</h2>
      <p>
        Точка отсчёта — уже состоявшаяся основная выплата по убытку. Дальше
        по части дел с компании дополнительно взыскивают деньги через иск
        или претензию: основной долг, износ, утрату товарной стоимости,
        связанные суммы с госпошлиной и доплаты.
      </p>
      <p>
        <strong>Расход без модели</strong> — это то, что компания уже заплатила
        по факту таких взысканий, плюс взносы. Этот расход в отчёте пишется
        со знаком минус, потому что это деньги, которые ушли.
      </p>
      <p>
        <strong>Расход по модели</strong> — это сколько компания потратила бы,
        если бы следовала решениям модели: ждать взыскание или нет, и какую
        сумму заранее направить. Этот расход тоже пишется со знаком минус.
      </p>
      <p>
        <strong>Чистый финансовый эффект</strong> считается так: расход по модели
        минус расход без модели. Если модель потратила меньше, разница
        получается со знаком плюс — это экономия. Если модель потратила больше,
        разница со знаком минус — это ухудшение.
      </p>
      <div class="note">
        Формула считает покрытие риска, а не двойную кассу. Когда предложенной
        суммы не хватает, в расход модели не добавляют и предложенную сумму,
        и весь суд. В расход идёт только незакрытая доля фактического взыскания
        и взносы. Когда суммы хватает, в расход идёт сама предложенная сумма
        вместо полного фактического взыскания.
      </div>
    </section>

    <section id="parts">
      <h2>Из каких частей складывается фактический расход</h2>
      <p>
        По каждому делу фактический расход — это сумма двух слагаемых.
      </p>
      <h3>Первое слагаемое. Фактическая сумма взыскания</h3>
      <p>
        Сюда входит взысканное по искам с учётом госпошлины на последней
        принятой судебной инстанции и доплаты по претензиям. Если этой суммы
        нет, считается, что дополнительного взыскания по делу не было.
      </p>
      <h3>Второе слагаемое. Взносы</h3>
      <p>
        Взнос финансовому уполномоченному равен {fu_fee} и начисляется только
        если по делу есть и обращение к финансовому уполномоченному, и ненулевая
        фактическая сумма взыскания. {court_fee_text}
      </p>
      <p>
        Отдельно в формуле живёт <strong>фактическая тяжесть</strong>: основной
        долг, износ и утрата товарной стоимости плюс те же доплаты по претензиям.
        По ней модель судят, хватило ли предложенной суммы. Это не то же самое,
        что фактическая сумма взыскания: тяжесть — более узкая планка,
        а в расход без модели идёт более широкая сумма взыскания вместе со взносами.
      </p>
    </section>

    <section id="model">
      <h2>Что отвечает модель</h2>
      <p>
        Модель даёт два ответа по каждому делу.
      </p>
      <ul>
        <li>
          Будет ли после основной выплаты дополнительное взыскание.
          Ответ «да» или «нет» получается из вероятности: если вероятность
          не ниже порога, модель считает, что взыскание будет.
        </li>
        <li>
          Какую сумму основного долга, износа и утраты товарной стоимости
          имеет смысл заложить, если взыскание будет. Это предложенная
          моделью сумма.
        </li>
      </ul>
      <p>
        Классификация решает, включать ли дело в контур доплаты.
        Оценка суммы решает, чем это дело закрывать и насколько оно покрыто.
      </p>
    </section>

    <section id="threshold">
      <h2>Как выбирается порог</h2>
      <p>
        Модель сначала выдаёт вероятность взыскания от нуля до единицы.
        Порог — это граница, после которой вероятность превращается в решение
        «взыскание будет».
      </p>
      <p>
        Порог подбирают не «на глаз» и не по самой красивой доле угаданных дел.
        На проверочной выборке перебирают значения от нуля до единицы с шагом
        одна сотая и оставляют то, при котором чистый финансовый эффект
        наибольший. Затем этот порог применяют к отложенной контрольной
        выборке — к делам, которые модель не видела при подборе порога.
      </p>
      <p>
        Низкий порог включает доплату чаще: меньше пропусков, больше ложных
        тревог. Высокий порог включает доплату реже: меньше зря направленных
        денег, больше пропусков, где компания платит факт целиком.
      </p>
    </section>

    <section id="cases">
      <h2>Пять ситуаций и как в каждой считается расход</h2>
      {_situations_html(rows)}
      {_table_html(rows)}
    </section>

    <section id="penalties">
      <h2>Где модель получает штраф</h2>
      <p>
        Штраф — это не отдельная бухгалтерская проводка, а то, как формула
        наказывает ошибку. Ниже четыре места, где решение модели расходится
        с выгодным исходом.
      </p>
      {_penalties_html(rows, n)}
      <p>
        Правильные отказы (модель не ждала взыскания, и его не было) штрафа
        не дают: и факт, и модель там равны нулю.
      </p>
    </section>

    <section id="example">
      <h2>Числовой разбор одной ситуации: суммы не хватило</h2>
      {_worked_example_html(under)}
    </section>

    <section id="numbers">
      <h2>Цифры текущего прогона</h2>
      <p>{numbers_note}</p>
      {_hero_html(
          n=n,
          threshold=threshold,
          fact_total=fact_total,
          model_total=model_total,
          net_effect=net_effect,
      )}
      <p>
        Итог по таблице ситуаций должен сходиться с этими тремя суммами:
        расход модели, расход без модели и чистый эффект.
      </p>
    </section>
  </main>
  <footer>
    Файл собирается функцией выгрузки финансового эффекта из тетради.
    Логика формул совпадает с расчётом покрытия в модуле финансового эффекта.
  </footer>
</body>
</html>
"""


def export_business_html(
    result: FinEffectResult | None = None,
    config: FinEffectConfig | None = None,
    *,
    path: str | Path | None = None,
    subtitle: str = "",
) -> Path:
    """Записать HTML. Без ``result`` — та же логика, без сумм прогона."""
    output = Path(path) if path is not None else DEFAULT_HTML_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_business_html(result, config, subtitle=subtitle),
        encoding="utf-8",
    )
    return output
