"""Единый подробный HTML-отчёт по ITT-оценке финансового эффекта."""
from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from querulus.fin_effect.excel_monitoring import MonitoringEffectResult, format_money

REPORT_FILENAME = "fin_effect_report.html"
FORMULA_VERSION = "ITT-U-2026-09-09-v1"

_CSS = """
:root {
  --bg:#f5f3ee; --surface:#fff; --ink:#1d2526; --muted:#5f696a;
  --line:#d9d4c9; --accent:#075f66; --soft:#e7f3f4; --warn:#fff4da;
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
@media (max-width:650px) { .eq { white-space:normal; } }
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
        "</div>"
    )
    return '<div class="grid">' + "".join(cells) + "</div>"


def _plan_section() -> str:
    return """
<h2>2. План, по которому реализован расчёт</h2>
<div class="card">
  <h3>Зафиксированные решения</h3>
  <ul>
    <li>Единица расчёта — строка убытка; incident-collapse и dedupe запрещены.</li>
    <li>ITT-популяция — только result∈{0,1,−100}; соглашение и compliance
    не являются фильтрами.</li>
    <li>Фактический outcome состоит только из <code>СуммаПлатежа</code>;
    OD — <code>СуммаОсновногоДолгаЗаявлено</code>; остаток после соглашения — 7%.</li>
    <li>Один terminal-набор priors применяется к 365/1095; сеть переносится
    по фактическому объёму и риску, без множителя 84/10.</li>
  </ul>
  <h3>Этапы build</h3>
  <ol>
    <li>Зафиксировать единицу анализа «убыток», ITT-группы model/control,
    денежные колонки и запрет dedupe.</li>
    <li>На финальном incident-level ретро-датасете рассчитать терминальные
    коэффициенты <code>p_U</code>, <code>k_U</code>, <code>m_U</code>,
    <code>e_U</code> отдельно для pilot/nonpilot.</li>
    <li>Построить <code>Yfact</code>, <code>Y365</code>, <code>Y1095</code>:
    фактическая выплата плюс дисконтированный непрореализованный хвост ПСР.</li>
    <li>Оценить ITT как стратифицированную по филиалу разность
    <code>mean(control) − mean(model)</code> и двухчастный bootstrap CI.</li>
    <li>Показать non-compliance отдельно: описательное as-complied сравнение
    и механический сценарий 100% исполнения рекомендаций.</li>
    <li>Годовой поток восстановить через сезонные доли ретро, затем отдельно
    показать pilot current share, pilot full rollout и full network.</li>
    <li>Собрать расчёт, формулы, легенду, допущения, качество данных и
    ограничения в одном HTML; notebook остаётся только генератором.</li>
  </ol>
  <h3>Критерии готовности</h3>
  <ul>
    <li>Yfact основан только на кассе; observed ПСР не прибавляется повторно.</li>
    <li>ITT включает non-compliance; compliance A/B показаны отдельно.</li>
    <li>NPV приведён к t_calc; сезонность масштабирует поток новых убытков,
    а не горизонт дозревания.</li>
    <li>Один HTML содержит цифры, формулы, легенды, предупреждения,
    ограничения и шпаргалку; одна notebook только формирует его.</li>
  </ul>
  <p class="muted">Полные формулы и легенды этапов находятся непосредственно
  в разделах 4–9 этого самодостаточного отчёта.</p>
</div>
"""


def build_monitoring_html(
    result: MonitoringEffectResult,
    *,
    source_label: str = "[OISUU_report].[dbo].[ВитринаСутяжность]",
    title: str = "Querulus — единая методика финансового эффекта",
) -> str:
    """Собрать самодостаточный HTML со всеми расчётами и допущениями."""
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

  {_plan_section()}

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

  <h2>4. Терминальные ретро-коэффициенты</h2>
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
        "U = ultimate/терминальный итог: сумма ПСР накоплена до доступного "
        "финального состояния и не является отдельным Y365 или Y1095. "
        "g ∈ {pilot, nonpilot}. Суд имеет приоритет; ФУ считается только "
        "среди положительных ПСР без суда, поэтому пути взаимоисключающие.",
    )}
    {_table(result.priors.table())}
    <p><b>Точные ретро-поля:</b>
    <code>TARGET_FREQ_AMOUNT</code> (итог claims + pretensions),
    <code>TARGET_FREQ</code>, <code>RECOVEREDMAINDEBT_LAST_INST_SUM</code>,
    <code>FILIAL</code>, <code>PAYMENT_ORDER_DATE_TIME</code>,
    <code>TARGET_FREQ_PRET_AMOUNT</code>,
    <code>TARGET_FREQ_CLAIMS_AMOUNT</code>,
    <code>Сумма_взыскано_по_ФУ</code>,
    <code>Суммы_взыскано_по_иску</code>.</p>
    <p>Основной расчёт использует pilot:
    p_U={pilot.p_ultimate:.4f}, k_U={pilot.k_ultimate:.4f},
    m_U={format_money(pilot.mean_positive_psr)},
    e_U={format_money(pilot.expected_fee)}.</p>
  </div>

  <h2>5. Yfact, Y365 и Y1095</h2>
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
        "age=(t_calc−t0) в днях; d_i,H=max(H−age_i,0)/2. "
        "Yfact не дисконтируется. Выплаты ПСР не прибавляются повторно: "
        "они уже входят в СуммаПлатежа и только уменьшают хвост.",
    )}
    <h3>Итоги model/control</h3>
    {_table(result.group_summary)}
  </div>

  <h2>6. ITT по филиалам и неопределённость</h2>
  <div class="card">
    <p>Веса фиксируются по всей eligible-популяции филиала, а не по размеру
    model/control. Это сохраняет смысл рандомизированного сравнения.</p>
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

  <h2>7. Non-compliance</h2>
  <div class="card">
    <h3>A. As-complied — только описательная диагностика</h3>
    <p>Сравнение выполняется внутри model и <code>РезультатПроверки=1</code>:
    исполнено, если <code>Выплата по модели=1</code>. Оно не является причинным:
    решение сотрудника после рандомизации создаёт selection/post-treatment bias.</p>
    {_table(result.compliance_a)}
    <h3>B. Механический сценарий 100% исполнения</h3>
    {_formula(
        "Контракт сценария",
        [
            "<b>forced_extra<sub>i</sub></b> = рекомендованная доплата, если "
            "model, result=1 и Выплата по модели≠1; иначе 0",
            "<b>Yfact_100<sub>i</sub></b> = СуммаПлатежа<sub>i</sub> + forced_extra<sub>i</sub>",
            "для всех model/result=1: q<sub>i</sub>=7%; затем заново считаются Y365_100 и Y1095_100",
            "<b>effect_100(H)</b> = stratified mean(control actual) − mean(model scenario)",
        ],
        "Это сценарная механика, а не LATE/IV-оценка. Доплата не подменяет "
        "СуммаПлатежа: она прибавляется только в сценарии.",
    )}
    {_table(result.compliance_b)}
  </div>

  <h2>8. Чувствительность</h2>
  <div class="card">
    <p>Полная сетка: ставка дисконтирования r ∈ {{8%,12%,16%}} и остаток
    после соглашения q ∈ {{0%,7%,15%}}. Ретро-параметры не подгоняются.</p>
    {_table(result.sensitivity)}
  </div>

  <h2>9. Сезонный годовой эффект и сеть</h2>
  <div class="card">
    {_formula(
        "Экстраполяция потока",
        [
            "<b>s<sub>m,P</sub></b> = средняя доля месяца m в годовом pilot-потоке ретро",
            "<b>seasonal_exposure</b> = Σ<sub>m</sub> coverage<sub>m</sub>×s<sub>m,P</sub>",
            "<b>N_pilot_eligible_year</b> = N_observed_pilot_eligible / seasonal_exposure",
            "<b>N_pilot_model_year_current</b> = share_model×N_pilot_eligible_year",
            "<b>N_pilot_model_year_full</b> = N_pilot_eligible_year",
            "<b>network_multiplier</b> = 1 + volume_ratio_NP×risk_ratio_NP",
        ],
        "volume_ratio_NP — отношение среднегодового eligible-потока nonpilot/pilot; "
        "risk_ratio_NP — отношение mean(TARGET_FREQ_AMOUNT) nonpilot/pilot. "
        "Горизонт Y365/Y1095 не умножается на 365/30: масштабируется число новых убытков.",
    )}
    <h3>Сезонные доли и покрытие текущего окна</h3>
    {_table(result.seasonality)}
    <h3>Сценарии годового эффекта</h3>
    {_table(result.annual_summary)}
  </div>

  <h2>10. Обозначения и ограничения</h2>
  <div class="card">
    <ul>
      <li><b>i</b> — строка/убыток текущей витрины; <b>q</b> — incident-level строка ретро;
      <b>f</b> — филиал; <b>g</b> — pilot/nonpilot; <b>H</b> — fact/365/1095.</li>
      <li><b>ITT</b> — эффект назначения в model-поток, включая фактический non-compliance.</li>
      <li><b>Yfact</b> — наблюдаемые расходы на t_calc; <b>Y365/Y1095</b> —
      Yfact плюс модельный NPV остатка до горизонта.</li>
      <li>Рандомизация считается корректной внутри филиала. Balance/covariate adjustment
      не выполняются из-за отсутствия согласованного pre-treatment набора признаков.</li>
      <li>Терминальные priors не содержат раздельных 365/1095 таргетов; различие
      горизонтов возникает только через доступное время и дисконтирование хвоста.</li>
      <li>В мониторинге OD — <code>СуммаОсновногоДолгаЗаявлено</code>, а k_U
      калибруется на <code>RECOVEREDMAINDEBT_LAST_INST_SUM</code>. Переносимость
      между разными бизнес-величинами является допущением.</li>
      <li>Priors pooled внутри pilot/nonpilot; 7% — экспертное, не оценённое
      по данным допущение; midpoint означает равномерное распределение будущего
      хвоста и не является моделью времени наступления ПСР.</li>
      <li>Terminal-вероятность ПСР не пересчитывается условно на текущий возраст
      убытка и отсутствие ПСР к t_calc: это упрощённая оценка без модели дозревания.</li>
      <li>Сценарий сети переносит pilot-эффект через отдельные коэффициенты объёма
      и риска. Это экстраполяция, а не рандомизированная оценка для nonpilot.</li>
      <li>Если age≥365, код останавливает расчёт: для зрелых строк нужен
      наблюдаемый Y365, а его текущий контракт данных не предоставляет.</li>
    </ul>
  </div>
</div>
</body>
</html>
"""


def write_monitoring_html(
    result: MonitoringEffectResult,
    path: str | Path,
    *,
    source_label: str = "[OISUU_report].[dbo].[ВитринаСутяжность]",
) -> Path:
    """Записать единый HTML-отчёт."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    html = build_monitoring_html(result, source_label=source_label)
    destination.write_text(
        "\n".join(line.rstrip() for line in html.splitlines()) + "\n",
        encoding="utf-8",
    )
    return destination


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
<p class="sub">Источник <code>{escape(source_label)}</code></p>
<div class="warning"><b>{escape(type(error).__name__)}</b>: {escape(str(error))}</div>
<p>Guard-проверка остановила расчёт, чтобы HTML не содержал недостоверный эффект.
Исправьте контракт/данные и сформируйте отчёт повторно.</p>
</div></body></html>"""
    destination.write_text(html, encoding="utf-8")
    return destination


__all__ = [
    "FORMULA_VERSION",
    "REPORT_FILENAME",
    "build_monitoring_html",
    "write_error_html",
    "write_monitoring_html",
]
