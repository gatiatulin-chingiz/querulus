"""Три HTML по финансовому эффекту: план, расчёт, заключение."""
from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from querulus.fin_effect.excel_monitoring import MonitoringEffectResult, format_money

PLAN_FILENAME = "fin_effect_plan.html"
REPORT_FILENAME = "fin_effect_report.html"
CONCLUSION_FILENAME = "fin_effect_conclusion.html"
FORMULA_VERSION = "ITT-U-2026-09-10-v2"

_CSS = """
:root {
  --bg:#f5f3ee; --surface:#fff; --ink:#1d2526; --muted:#5f696a;
  --line:#d9d4c9; --accent:#075f66; --soft:#e7f3f4; --warn:#fff4da;
  --model:#0b5f66; --control:#6b5b4b; --save:#2f6b3a;
}
* { box-sizing:border-box; }
body {
  margin:0; color:var(--ink); background:var(--bg);
  font:15px/1.48 "Segoe UI", Arial, sans-serif;
}
.page { max-width:1240px; margin:auto; padding:1.2rem 1rem 2.5rem; }
h1 { color:var(--accent); margin:0 0 .25rem; font-size:1.55rem; }
h2 { color:var(--accent); margin:1.4rem 0 .45rem; font-size:1.18rem; }
h3 { color:#244f53; margin:1rem 0 .35rem; font-size:1rem; }
p { margin:.35rem 0; }
.sub,.muted { color:var(--muted); }
.card {
  background:var(--surface); border:1px solid var(--line);
  border-radius:9px; padding:.85rem 1rem; margin:.45rem 0;
}
.grid {
  display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
  gap:.5rem; margin:.55rem 0;
}
.stat { background:#fbfbf9; border:1px solid var(--line); padding:.6rem; border-radius:7px; }
.stat span { display:block; color:var(--muted); font-size:.78rem; }
.stat b { font-size:1.05rem; font-variant-numeric:tabular-nums; }
.stat small { display:block; color:var(--muted); margin-top:.2rem; }
.formula {
  background:linear-gradient(135deg,var(--soft),#f7fbfb);
  border:1px solid #c7dfe1; border-left:4px solid var(--accent);
  border-radius:0 8px 8px 0; padding:.7rem .85rem; margin:.5rem 0;
  overflow-x:auto;
}
.formula b { color:var(--accent); }
.eq { font:1.02rem/1.65 Cambria, "Times New Roman", serif; white-space:nowrap; }
.note { color:var(--muted); border-top:1px solid #d4e6e7; margin-top:.35rem; padding-top:.35rem; }
.warning {
  background:var(--warn); border:1px solid #e5c66e; border-left:4px solid #9b7100;
  border-radius:0 8px 8px 0; padding:.65rem .8rem; margin:.4rem 0;
}
.scroll { overflow-x:auto; }
table { width:100%; border-collapse:collapse; font-size:.82rem; background:#fff; }
th,td { border:1px solid var(--line); padding:.35rem .45rem; text-align:left; vertical-align:top; }
th { background:var(--soft); color:var(--accent); position:sticky; top:0; }
code { background:#efede7; padding:.05rem .2rem; border-radius:3px; font-family:Consolas,monospace; }
ul,ol { margin:.3rem 0 .3rem 1.25rem; padding:0; }
li { margin:.18rem 0; }
.good { color:#20612d; } .bad { color:#8a2d21; }
.flow {
  display:grid; gap:.55rem;
  grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  margin:.6rem 0;
}
.flow-step {
  background:#fff; border:1px solid var(--line); border-top:4px solid var(--accent);
  border-radius:8px; padding:.65rem .7rem; min-height:7.2rem;
}
.flow-step .num {
  color:var(--accent); font-size:.72rem; font-weight:700; letter-spacing:.04em;
  text-transform:uppercase;
}
.flow-step h4 { margin:.15rem 0 .25rem; font-size:.95rem; }
.flow-step p { margin:0; color:var(--muted); font-size:.84rem; }
.compare {
  display:grid; grid-template-columns:1fr auto 1fr; gap:.7rem; align-items:stretch;
  margin:.7rem 0;
}
.compare .box {
  border:1px solid var(--line); border-radius:8px; padding:.7rem .8rem; background:#fff;
}
.compare .box.model { border-top:4px solid var(--model); }
.compare .box.control { border-top:4px solid var(--control); }
.compare .mid {
  display:flex; align-items:center; justify-content:center;
  font-weight:700; color:var(--save); min-width:3rem;
}
.nav { display:flex; flex-wrap:wrap; gap:.45rem; margin:.5rem 0 1rem; }
.nav a {
  text-decoration:none; color:var(--accent); border:1px solid #b7d4d7;
  background:var(--soft); border-radius:999px; padding:.25rem .7rem; font-size:.86rem;
}
@media (max-width:820px) {
  .compare { grid-template-columns:1fr; }
  .compare .mid { min-height:auto; }
  .eq { white-space:normal; }
}
"""


def _table(frame: pd.DataFrame | None) -> str:
    if frame is None or frame.empty:
        return "<p class='muted'>(нет данных)</p>"
    shown = frame.copy()
    numeric = shown.select_dtypes(include="number").columns
    shown[numeric] = shown[numeric].round(4)
    return (
        '<div class="scroll">'
        + shown.to_html(index=False, border=0, escape=True)
        + "</div>"
    )


def _formula(title: str, equations: list[str], note: str = "") -> str:
    body = "".join(f'<div class="eq">{equation}</div>' for equation in equations)
    note_html = f'<div class="note">{note}</div>' if note else ""
    return f'<div class="formula"><b>{title}</b>{body}{note_html}</div>'


def _warnings(items: list[str]) -> str:
    if not items:
        return "<p class='good'>Расчёт завершён без методологических предупреждений.</p>"
    return "".join(
        f"<div class='warning'>{escape(item)}</div>" for item in items
    )


def _nav(active: str) -> str:
    items = (
        (PLAN_FILENAME, "1. План"),
        (REPORT_FILENAME, "2. Расчёт"),
        (CONCLUSION_FILENAME, "3. Заключение"),
    )
    links = []
    for filename, label in items:
        if filename == active:
            links.append(
                f'<a href="{escape(filename)}" style="font-weight:700">{escape(label)}</a>'
            )
        else:
            links.append(f'<a href="{escape(filename)}">{escape(label)}</a>')
    return f'<div class="nav">{"".join(links)}</div>'


def _business_schema() -> str:
    return """
<div class="card">
  <h3>Как читается финансовый эффект</h3>
  <p>Сравниваем два случайных потока убытков внутри филиала:
  <b>model</b> (модель работала) и <b>control</b> (модель не работала).</p>
  <div class="compare">
    <div class="box control">
      <h4>Control</h4>
      <p>Обычный процесс урегулирования без рекомендаций модели.
      Считаем средний полный расход на один убыток.</p>
    </div>
    <div class="mid">−</div>
    <div class="box model">
      <h4>Model</h4>
      <p>Тот же тип убытков, но с назначением в модель.
      Считаем средний полный расход на один убыток, включая случаи,
      где рекомендацию не выполнили.</p>
    </div>
  </div>
  <p><b>Эффект на один убыток</b> = средний расход control − средний расход model.
  Плюс означает экономию.</p>

  <div class="flow">
    <div class="flow-step">
      <div class="num">Шаг 1</div>
      <h4>Кого сравниваем</h4>
      <p>Только убытки model и control в пилотных филиалах. Соглашения
      и исполнение рекомендаций не выкидывают строки из сравнения.</p>
    </div>
    <div class="flow-step">
      <div class="num">Шаг 2</div>
      <h4>Что уже заплатили</h4>
      <p><code>Yфакт</code> = фактическая касса
      <code>СуммаПлатежа</code> на дату отчёта.</p>
    </div>
    <div class="flow-step">
      <div class="num">Шаг 3</div>
      <h4>Что ещё может прийти</h4>
      <p>Хвост претензий / ФУ / суда оцениваем по ретро-коэффициентам.
      После соглашения оставляем экспертно 7%.</p>
    </div>
    <div class="flow-step">
      <div class="num">Шаг 4</div>
      <h4>Три горизонта</h4>
      <p><code>Yфакт</code> — сейчас; <code>Y365</code> и <code>Y1095</code> —
      тот же хвост, приведённый к сегодняшним деньгам на 1 и 3 года.</p>
    </div>
    <div class="flow-step">
      <div class="num">Шаг 5</div>
      <h4>Неопределённость</h4>
      <p>95% доверительный интервал показывает, насколько оценка устойчива.
      Если интервал проходит через 0, эффект пока не подтверждён.</p>
    </div>
    <div class="flow-step">
      <div class="num">Шаг 6</div>
      <h4>На год и сеть</h4>
      <p>Эффект на убыток × ожидаемый годовой поток пилота × поправка
      на объём и риск остальных филиалов. Это сценарий, не факт.</p>
    </div>
  </div>
</div>
"""


def _contract_table(result: MonitoringEffectResult) -> pd.DataFrame:
    meaning = {
        "loss": "Единица анализа; строки не схлопываются и не dedupe",
        "incident": "Кластер bootstrap; не единица итогового ITT",
        "result": "model={0,1}; control=−100",
        "filial": "Страта рандомизации и ITT-взвешивания",
        "payment": "paid_to_date; единственный источник фактических расходов",
        "to_pay_diagnostic_only": "Только сверка качества; в Y не прибавляется",
        "od": "OD для k_U×OD; при пропуске используется m_U",
        "recommended_extra": "Доплата в сценарии 100% compliance",
        "payout_by_model": "Признак исполнения рекомендации",
        "agreement": "При соглашении остаток ПСР равен 7%",
        "t0_primary": "Основная дата старта возраста убытка",
        "t0_fallback": "Fallback для t0",
        "psr_pretension": "Наблюдаемый ПСР; только вычитается из хвоста",
        "psr_fu": "Наблюдаемый ПСР; только вычитается из хвоста",
        "psr_court": "Наблюдаемый ПСР; только вычитается из хвоста",
    }
    return pd.DataFrame(
        [
            {
                "entity": key,
                "source_column": value,
                "meaning": meaning.get(key, ""),
            }
            for key, value in result.contract.items()
        ]
    )


def _glossary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "колонка": "effect_per_case",
                "смысл": "Экономия на один убыток: mean(control) − mean(model), "
                "взвешенно по филиалам",
                "единица": "₽ / убыток",
            },
            {
                "колонка": "unstratified_effect",
                "смысл": "Та же разность без взвешивания по филиалам",
                "единица": "₽ / убыток",
            },
            {
                "колонка": "ci_low / ci_high",
                "смысл": "Границы 95% доверительного интервала bootstrap",
                "единица": "₽ / убыток",
            },
            {
                "колонка": "n_bootstrap",
                "смысл": "Число успешных bootstrap-повторений",
                "единица": "шт.",
            },
            {
                "колонка": "agreement_share",
                "смысл": "Доля убытков с заключённым соглашением",
                "единица": "%",
            },
            {
                "колонка": "pretension_share",
                "смысл": "Доля убытков с претензией",
                "единица": "%",
            },
            {
                "колонка": "fu_incident_share",
                "смысл": "Доля убытков с обращением к ФУ на уровне инцидента",
                "единица": "%",
            },
            {
                "колонка": "court_incident_share",
                "смысл": "Доля убытков с обращением к суду на уровне инцидента",
                "единица": "%",
            },
            {
                "колонка": "seasonal_exposure",
                "смысл": "Какую долю типичного года покрывает текущее окно",
                "единица": "доля 0–1",
            },
            {
                "колонка": "N_pilot_eligible_year",
                "смысл": "Ожидаемый годовой поток eligible-убытков пилота",
                "единица": "убытки / год",
            },
            {
                "колонка": "volume_ratio_nonpilot",
                "смысл": "Во сколько раз годовой поток непилота больше пилота",
                "единица": "кратно",
            },
            {
                "колонка": "risk_ratio_nonpilot",
                "смысл": "Отношение среднего ПСР непилота к пилоту",
                "единица": "кратно",
            },
            {
                "колонка": "network_multiplier",
                "смысл": "1 + volume_ratio × risk_ratio; переход от пилота к сети",
                "единица": "кратно",
            },
            {
                "колонка": "annual_network_full",
                "смысл": "Сценарный годовой эффект сети при 100% rollout пилота "
                "и переносе на сеть",
                "единица": "₽ / год",
            },
        ]
    )


def _headline(result: MonitoringEffectResult) -> str:
    effects = result.effect_summary.set_index("horizon")
    annual = result.annual_summary.set_index("horizon")
    cells = []
    for horizon in ("fact", "365", "1095"):
        effect = float(effects.loc[horizon, "effect_per_case"])
        low = float(effects.loc[horizon, "ci_low"])
        high = float(effects.loc[horizon, "ci_high"])
        cells.append(
            "<div class='stat'>"
            f"<span>ITT эффект / убыток, Y{escape(horizon)}</span>"
            f"<b>{format_money(effect)}</b>"
            f"<small>95% CI: {format_money(low)} … {format_money(high)}</small>"
            "</div>"
        )
    cells.append(
        "<div class='stat'>"
        "<span>Годовой эффект сети, full rollout, Y1095</span>"
        f"<b>{format_money(float(annual.loc['1095', 'annual_network_full']))}</b>"
        "<small>сценарная оценка без CI</small>"
        "</div>"
    )
    return '<div class="grid">' + "".join(cells) + "</div>"


def build_plan_html(
    *,
    source_label: str = "[OISUU_report].[dbo].[ВитринаСутяжность]",
) -> str:
    """HTML №1: план методики для бизнеса."""
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>План расчёта финансового эффекта</title>
  <style>{_CSS}</style>
</head>
<body>
<div class="page">
  <h1>План расчёта финансового эффекта</h1>
  <p class="sub">Сформировано {escape(generated)} · версия {FORMULA_VERSION} ·
  источник <code>{escape(source_label)}</code></p>
  {_nav(PLAN_FILENAME)}
  {_business_schema()}

  <h2>Что фиксируем до расчёта</h2>
  <div class="card">
    <ul>
      <li>Сравниваем <b>model</b> и <b>control</b> внутри филиала
      (примерно 50/50).</li>
      <li>Единица — один убыток. Дубли не удаляем, но показываем их влияние.</li>
      <li>Фактический расход — только <code>СуммаПлатежа</code>.</li>
      <li>Будущий ПСР оцениваем по ретро-данным пилотных филиалов.</li>
      <li>Если заключено соглашение, экспертно оставляем 7% возможного ПСР.</li>
      <li>Главный результат — эффект назначения модели (ITT), включая случаи
      неисполнения рекомендаций.</li>
      <li>Сценарии «если бы всегда исполняли» и «на всю сеть» показываем отдельно.</li>
    </ul>
  </div>

  <h2>Какие файлы получаем</h2>
  <div class="card">
    <ol>
      <li><code>{PLAN_FILENAME}</code> — этот план и схема.</li>
      <li><code>{REPORT_FILENAME}</code> — цифры расчёта, доли путей,
      чувствительность и ограничения.</li>
      <li><code>{CONCLUSION_FILENAME}</code> — краткое заключение для бизнеса
      после проверки расчёта.</li>
    </ol>
  </div>

  <h2>Что нельзя обещать без оговорок</h2>
  <div class="card">
    <ul>
      <li>Если 95% CI проходит через 0, экономия пока статистически не подтверждена.</li>
      <li><code>Y365</code>/<code>Y1095</code> сейчас — прогноз хвоста, а не полностью
      дозревшие фактические расходы.</li>
      <li>Годовой эффект сети — сценарий масштабирования, не измеренный факт.</li>
    </ul>
  </div>
</div>
</body>
</html>
"""


def build_monitoring_html(
    result: MonitoringEffectResult,
    *,
    source_label: str = "[OISUU_report].[dbo].[ВитринаСутяжность]",
    title: str = "Querulus — расчёт финансового эффекта",
) -> str:
    """HTML №2: подробный расчёт."""
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    pilot = result.priors.pilot
    retro_window = (
        f"{result.priors.window_start or '?'} … {result.priors.window_end or '?'}"
    )
    quality = result.data_quality.set_index("metric")["value"].to_dict()
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
  <p class="sub">Сформировано {escape(generated)} · версия {FORMULA_VERSION} ·
  t_calc={result.t_calc.date()} · источник <code>{escape(source_label)}</code> ·
  наблюдения {quality.get("observation_start", "?")} …
  {quality.get("observation_end", "?")} · ретро-окно {escape(retro_window)}</p>
  {_nav(REPORT_FILENAME)}

  <h2>1. Главный результат</h2>
  <div class="card">
    {_headline(result)}
    {_formula(
        "Ключевой ITT estimand",
        [
            "<b>effect(H)</b> = Σ<sub>f</sub> w<sub>f</sub> × "
            "[mean(Y<sub>H</sub>|control,f) − mean(Y<sub>H</sub>|model,f)]",
            "w<sub>f</sub> = N<sub>eligible,f</sub> / Σ<sub>f</sub>N<sub>eligible,f</sub>",
        ],
        "Положительное значение означает меньшие ожидаемые расходы в model. "
        "Филиал — страта исходной рандомизации.",
    )}
    {_warnings(result.warnings)}
  </div>

  <h2>2. Схема для бизнеса</h2>
  {_business_schema()}

  <h2>3. Контракт данных и выборка</h2>
  <div class="card">
    <p><b>Единица:</b> одна строка витрины/убыток. Инциденты не схлопываются.
    Дубли не удаляются: их влияние явно показано в диагностике.</p>
    <p><b>ITT:</b> model = <code>РезультатПроверки ∈ {{0,1}}</code>;
    control = <code>РезультатПроверки = −100</code>. Соглашение и исполнение
    рекомендации не фильтруют ITT-популяцию, потому что это post-treatment.</p>
    <p><b>Базовые фильтры:</b> пилотные филиалы без Архангельского/Марийского;
    форма возмещения содержит «денежная», «ремонт» или «соглашение»;
    первичный убыток; объект «автотранспорт».</p>
    {_table(_contract_table(result))}
    <h3>Диагностика качества</h3>
    {_table(result.data_quality)}
    <p class="muted">Выбрано строк без dedupe: {quality.get("n_rows", "—")};
    пропусков OD: {quality.get("missing_od", "—")};
    возможное повторное суммирование СуммаПлатежа:
    {quality.get("payment_possible_inflation", "—")}.</p>
  </div>

  <h2>4. Доли соглашений / претензий / ФУ / суда</h2>
  <div class="card">
    <p>Доли считаются на той же ITT-выборке model/control.
    ФУ и суд подняты на уровень инцидента: если хотя бы один убыток инцидента
    имел ФУ/суд, флаг ставится всем строкам инцидента.</p>
    <h3>Model vs control</h3>
    {_table(result.path_shares)}
    <p class="muted">Строки <code>lift_pp</code> — разница долей в процентных пунктах
    (model − control). <code>lift_rel</code> — относительный рост доли.</p>
    <h3>По филиалам</h3>
    {_table(result.filial_path_shares)}
  </div>

  <h2>5. Терминальные ретро-коэффициенты</h2>
  <div class="card">
    {_formula(
        "Коэффициенты из финального incident-level df",
        [
            "<b>p<sub>U,g</sub></b> = count(TARGET_FREQ_AMOUNT &gt; 0) / count(rows)",
            "<b>k<sub>U,g</sub></b> = Σ TARGET_FREQ_AMOUNT / Σ RECOVEREDMAINDEBT_LAST_INST_SUM "
            "на положительных строках с OD &gt; 0",
            "<b>m<sub>U,g</sub></b> = mean(TARGET_FREQ_AMOUNT | TARGET_FREQ_AMOUNT &gt; 0)",
            "<b>e<sub>U,g</sub></b> = p(FU|PSR)×100 000 + p(court|PSR)×15 000",
        ],
        "U = ultimate/терминальный итог. g ∈ {pilot, nonpilot}. "
        "Основной расчёт использует pilot.",
    )}
    {_table(result.priors.table())}
    <p>Основной расчёт использует pilot:
    p_U={pilot.p_ultimate:.4f}, k_U={pilot.k_ultimate:.4f},
    m_U={format_money(pilot.mean_positive_psr)},
    e_U={format_money(pilot.expected_fee)}.</p>
  </div>

  <h2>6. Yfact, Y365 и Y1095</h2>
  <div class="card">
    {_formula(
        "Построение исхода каждого убытка i",
        [
            "<b>paid_to_date<sub>i</sub></b> = numeric(СуммаПлатежа<sub>i</sub>, missing=0)",
            "<b>base<sub>i</sub></b> = k<sub>U,pilot</sub>×OD<sub>i</sub>, если OD&gt;0; "
            "иначе m<sub>U,pilot</sub>",
            "<b>expected_open_PSR<sub>i</sub></b> = p<sub>U,pilot</sub>×"
            "(base<sub>i</sub> + e<sub>U,pilot</sub>)",
            "<b>observed_PSR<sub>i</sub></b> = претензия<sub>i</sub> + "
            "ФУ<sub>i</sub> + суд<sub>i</sub>",
            "<b>remaining<sub>i</sub></b> = max(q<sub>i</sub>×expected_open_PSR<sub>i</sub> "
            "− observed_PSR<sub>i</sub>, 0)",
            "<b>Yfact<sub>i</sub></b> = paid_to_date<sub>i</sub>",
            "<b>YH<sub>i</sub></b> = paid_to_date<sub>i</sub> + "
            "remaining<sub>i</sub>/(1+r)<sup>d<sub>i,H</sub>/365</sup>",
        ],
        f"OD = СуммаОсновногоДолгаЗаявлено; q=1 без соглашения и "
        f"q={result.residual_share:.0%} при соглашении; r={result.discount_rate:.0%}; "
        "age=(t_calc−t0) в днях; d_i,H=max(H−age_i,0)/2.",
    )}
    <h3>Итоги model/control</h3>
    {_table(result.group_summary)}
  </div>

  <h2>7. ITT по филиалам и неопределённость</h2>
  <div class="card">
    <p>Веса фиксируются по всей eligible-популяции филиала, а не по размеру
    model/control.</p>
    {_table(result.effect_summary)}
    <h3>Вклад филиалов</h3>
    {_table(result.filial_effects)}
    {_formula(
        "95% CI: двухчастный bootstrap",
        [
            "1) ретро incident-level строки resample → пересчёт p_U, k_U, m_U, e_U;",
            "2) текущая витрина resample кластерами по НомерИнцидент → пересчёт Y и ITT;",
            "CI = 2.5% и 97.5% квантили bootstrap effect(H).",
        ],
        f"Запрошено итераций: {result.bootstrap_iterations}. "
        "Фактическое число успешных итераций указано в n_bootstrap.",
    )}
  </div>

  <h2>8. Non-compliance</h2>
  <div class="card">
    <h3>A. As-complied — только описательная диагностика</h3>
    <p>Сравнение внутри model и <code>РезультатПроверки=1</code>.
    Не является причинным эффектом.</p>
    {_table(result.compliance_a)}
    <h3>B. Механический сценарий 100% исполнения</h3>
    {_formula(
        "Контракт сценария",
        [
            "<b>forced_extra<sub>i</sub></b> = рекомендованная доплата, если "
            "model, result=1 и Выплата по модели≠1; иначе 0",
            "<b>Yfact_100<sub>i</sub></b> = СуммаПлатежа<sub>i</sub> + forced_extra<sub>i</sub>",
            "для всех model/result=1: q<sub>i</sub>=7%; затем Y365_100 и Y1095_100",
            "<b>effect_100(H)</b> = stratified mean(control actual) − mean(model scenario)",
        ],
        "Сценарная механика, не LATE/IV.",
    )}
    {_table(result.compliance_b)}
  </div>

  <h2>9. Чувствительность</h2>
  <div class="card">
    <p>Сетка: r ∈ {{8%,12%,16%}} и остаток после соглашения q ∈ {{0%,7%,15%}}.</p>
    {_table(result.sensitivity)}
  </div>

  <h2>10. Сезонный годовой эффект и сеть</h2>
  <div class="card">
    {_formula(
        "Экстраполяция потока",
        [
            "<b>s<sub>m,P</sub></b> = средняя доля месяца m в годовом pilot-потоке ретро",
            "<b>seasonal_exposure</b> = Σ<sub>m</sub> coverage<sub>m</sub>×s<sub>m,P</sub>",
            "<b>N_pilot_eligible_year</b> = N_observed_pilot_eligible / seasonal_exposure",
            "<b>network_multiplier</b> = 1 + volume_ratio_NP×risk_ratio_NP",
        ],
        "Горизонт Y365/Y1095 не умножается на 365/30: масштабируется число новых убытков.",
    )}
    <h3>Сезонные доли и покрытие текущего окна</h3>
    {_table(result.seasonality)}
    <h3>Сценарии годового эффекта</h3>
    {_table(result.annual_summary)}
  </div>

  <h2>11. Словарь колонок</h2>
  <div class="card">
    {_table(_glossary())}
  </div>

  <h2>12. Ограничения</h2>
  <div class="card">
    <ul>
      <li><b>ITT</b> — эффект назначения в model-поток, включая фактический non-compliance.</li>
      <li><b>Y365/Y1095</b> — Yfact плюс модельный NPV остатка; это не полностью
      дозревшие фактические горизонты.</li>
      <li>Терминальные priors не содержат раздельных 365/1095 таргетов.</li>
      <li>В мониторинге OD — <code>СуммаОсновногоДолгаЗаявлено</code>, а k_U
      калибруется на <code>RECOVEREDMAINDEBT_LAST_INST_SUM</code>.</li>
      <li>7% — экспертное допущение; midpoint — упрощение времени будущего хвоста.</li>
      <li>Сетевой масштаб — сценарий, не рандомизированная оценка для nonpilot.</li>
      <li>Годовой эффект сети в текущей версии публикуется без собственного CI.</li>
    </ul>
  </div>
</div>
</body>
</html>
"""


def build_conclusion_html(
    *,
    source_label: str = "[OISUU_report].[dbo].[ВитринаСутяжность]",
    ready: bool = False,
    body_html: str | None = None,
) -> str:
    """HTML №3: заключение. Пока заготовка, пока не прислан файл №2."""
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    if ready and body_html:
        content = body_html
    else:
        content = """
<div class="card">
  <div class="warning">Заключение ещё не заполнено.</div>
  <p>После проверки файла <code>fin_effect_report.html</code> пришлите его
  как «файл №2» — по нему будет подготовлено краткое бизнес-заключение:</p>
  <ul>
    <li>есть ли подтверждённая экономия;</li>
    <li>что означают Yфакт / Y365 / Y1095;</li>
    <li>какой годовой сценарий можно и нельзя обещать;</li>
    <li>какие риски и следующие шаги.</li>
  </ul>
</div>
"""
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Заключение по финансовому эффекту</title>
  <style>{_CSS}</style>
</head>
<body>
<div class="page">
  <h1>Заключение по финансовому эффекту</h1>
  <p class="sub">Сформировано {escape(generated)} · версия {FORMULA_VERSION} ·
  источник <code>{escape(source_label)}</code></p>
  {_nav(CONCLUSION_FILENAME)}
  {content}
</div>
</body>
</html>
"""


def _write(path: Path, html: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(line.rstrip() for line in html.splitlines()) + "\n",
        encoding="utf-8",
    )
    return path


def write_monitoring_html(
    result: MonitoringEffectResult,
    path: str | Path,
    *,
    source_label: str = "[OISUU_report].[dbo].[ВитринаСутяжность]",
) -> Path:
    """Записать HTML №2 расчёта."""
    return _write(
        Path(path),
        build_monitoring_html(result, source_label=source_label),
    )


def write_plan_html(
    path: str | Path,
    *,
    source_label: str = "[OISUU_report].[dbo].[ВитринаСутяжность]",
) -> Path:
    """Записать HTML №1 плана."""
    return _write(Path(path), build_plan_html(source_label=source_label))


def write_conclusion_html(
    path: str | Path,
    *,
    source_label: str = "[OISUU_report].[dbo].[ВитринаСутяжность]",
    ready: bool = False,
    body_html: str | None = None,
) -> Path:
    """Записать HTML №3 заключения."""
    return _write(
        Path(path),
        build_conclusion_html(
            source_label=source_label,
            ready=ready,
            body_html=body_html,
        ),
    )


def write_all_monitoring_htmls(
    result: MonitoringEffectResult,
    data_dir: str | Path,
    *,
    source_label: str = "[OISUU_report].[dbo].[ВитринаСутяжность]",
) -> tuple[Path, Path, Path]:
    """Записать все три HTML в каталог data."""
    data_dir = Path(data_dir)
    plan = write_plan_html(data_dir / PLAN_FILENAME, source_label=source_label)
    report = write_monitoring_html(
        result,
        data_dir / REPORT_FILENAME,
        source_label=source_label,
    )
    conclusion = write_conclusion_html(
        data_dir / CONCLUSION_FILENAME,
        source_label=source_label,
    )
    return plan, report, conclusion


def write_error_html(
    error: Exception,
    path: str | Path,
    *,
    source_label: str,
) -> Path:
    """Записать диагностический HTML, если расчёт остановлен guard-проверкой."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/><title>Ошибка расчёта</title>
<style>{_CSS}</style></head><body><div class="page">
<h1>Финансовый эффект не рассчитан</h1>
{_nav(REPORT_FILENAME)}
<p class="sub">Источник <code>{escape(source_label)}</code></p>
<div class="warning"><b>{escape(type(error).__name__)}</b>: {escape(str(error))}</div>
<p>Guard-проверка остановила расчёт, чтобы HTML не содержал недостоверный эффект.
Исправьте контракт/данные и сформируйте отчёт повторно.</p>
</div></body></html>"""
    return _write(destination, html)


__all__ = [
    "CONCLUSION_FILENAME",
    "FORMULA_VERSION",
    "PLAN_FILENAME",
    "REPORT_FILENAME",
    "build_conclusion_html",
    "build_monitoring_html",
    "build_plan_html",
    "write_all_monitoring_htmls",
    "write_conclusion_html",
    "write_error_html",
    "write_monitoring_html",
    "write_plan_html",
]
