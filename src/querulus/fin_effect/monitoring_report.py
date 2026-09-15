"""Три HTML по финансовому эффекту: план, расчёт, заключение."""
from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from querulus.fin_effect.excel_monitoring import MonitoringEffectResult, format_money

PLAN_FILENAME = "fin_effect_plan.html"
REPORT_FILENAME = "fin_effect_report.html"
CONCLUSION_FILENAME = "fin_effect_conclusion.html"
FORMULA_VERSION = "ITT-U-2026-09-15-v8"


def report_export_stamp(
    when: datetime | pd.Timestamp | str | None = None,
) -> str:
    """Метка выгрузки для имён файлов: YYYY-MM-DD_HHMM."""
    ts = pd.Timestamp(when) if when is not None else pd.Timestamp.now()
    return ts.strftime("%Y-%m-%d_%H%M")


def dated_artifact_names(stamp: str) -> dict[str, str]:
    """Имена plan/report/conclusion/audit/weekly с датой выгрузки."""
    return {
        "plan": f"fin_effect_plan_{stamp}.html",
        "report": f"fin_effect_report_{stamp}.html",
        "conclusion": f"fin_effect_conclusion_{stamp}.html",
        "audit": f"fin_effect_audit_{stamp}.xlsx",
        "weekly_html": f"fin_effect_weekly_{stamp}.html",
        "weekly_csv": f"fin_effect_weekly_{stamp}.csv",
        "weekly_filial_csv": f"fin_effect_weekly_filial_{stamp}.csv",
        "error": f"fin_effect_error_{stamp}.html",
    }

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
.charts {
  display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
  gap:.75rem; margin:.55rem 0;
}
.chart {
  background:#fff; border:1px solid var(--line); border-radius:8px;
  padding:.55rem .65rem;
}
.chart img { width:100%; height:auto; display:block; }
.chart .cap { color:var(--muted); font-size:.8rem; margin-top:.35rem; }
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
.legend {
  background:#fbfbf9; border:1px solid var(--line); border-radius:7px;
  padding:.55rem .7rem; margin:.35rem 0 .7rem; font-size:.84rem;
}
.legend h4 { margin:0 0 .3rem; color:#244f53; font-size:.9rem; }
.legend ul { margin:.15rem 0 .15rem 1.1rem; }
.legend li { margin:.12rem 0; color:var(--muted); }
.legend code { font-size:.82rem; }
@media (max-width:820px) {
  .compare { grid-template-columns:1fr; }
  .compare .mid { min-height:auto; }
  .eq { white-space:normal; }
}
"""


def _fmt_number(value: Any) -> str:
    """Число без научной нотации."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return f"{int(value):,}".replace(",", " ")
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if abs(number) >= 1 or number == 0:
            return f"{number:,.2f}".replace(",", " ")
        return f"{number:.8f}".rstrip("0").rstrip(".").replace(",", " ")
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _fig_to_img(fig: Any, *, alt: str, caption: str = "") -> str:
    import base64
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    cap = f'<div class="cap">{escape(caption)}</div>' if caption else ""
    return (
        f'<div class="chart"><img alt="{escape(alt)}" '
        f'src="data:image/png;base64,{encoded}"/>{cap}</div>'
    )


def _style_axes(ax: Any) -> None:
    ax.set_facecolor("#fbfbf9")
    ax.grid(True, axis="y", color="#d9d4c9", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#5f696a", labelsize=8)


def _chart_bootstrap_hist(
    samples: pd.DataFrame,
    effects: pd.DataFrame,
    *,
    title: str,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if samples is None or samples.empty:
        return ""
    horizons = [h for h in ("fact", "365") if h in set(samples["horizon"].astype(str))]
    if not horizons:
        return ""
    effect_idx = effects.set_index("horizon") if not effects.empty else None
    fig, axes = plt.subplots(1, len(horizons), figsize=(5.2 * len(horizons), 3.4))
    if len(horizons) == 1:
        axes = [axes]
    for ax, horizon in zip(axes, horizons):
        part = samples.loc[samples["horizon"].astype(str).eq(horizon), "effect_per_case"]
        values = pd.to_numeric(part, errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            ax.set_visible(False)
            continue
        ax.hist(
            values,
            bins=min(40, max(10, values.size // 20)),
            color="#075f66",
            alpha=0.72,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.axvline(0.0, color="#8a2d21", linestyle="--", linewidth=1.2, label="0")
        if effect_idx is not None and horizon in effect_idx.index:
            point = float(effect_idx.loc[horizon, "effect_per_case"])
            low = float(effect_idx.loc[horizon, "ci_low"])
            high = float(effect_idx.loc[horizon, "ci_high"])
            ax.axvline(point, color="#2f6b3a", linewidth=1.6, label="point")
            if pd.notna(low):
                ax.axvline(low, color="#6b5b4b", linestyle=":", linewidth=1.2, label="CI")
            if pd.notna(high):
                ax.axvline(high, color="#6b5b4b", linestyle=":", linewidth=1.2)
        ax.set_title(f"Y{horizon}", color="#075f66", fontsize=10)
        ax.set_xlabel("effect_per_case, ₽")
        ax.set_ylabel("число bootstrap")
        _style_axes(ax)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle(title, color="#075f66", fontsize=11, y=1.02)
    fig.tight_layout()
    return _fig_to_img(
        fig,
        alt=title,
        caption="Гистограмма bootstrap effect_per_case; пунктир — 0, "
        "зелёная линия — точечная оценка, коричневые — границы 95% CI.",
    )


def _chart_filial_effects(filial_effects: pd.DataFrame) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if filial_effects is None or filial_effects.empty:
        return ""
    part = filial_effects.loc[filial_effects["horizon"].astype(str).eq("365")].copy()
    if part.empty:
        part = filial_effects.loc[filial_effects["horizon"].astype(str).eq("fact")].copy()
    if part.empty:
        return ""
    part = part.sort_values("effect_control_minus_model")
    fig, ax = plt.subplots(figsize=(7.2, max(3.2, 0.32 * len(part) + 1.2)))
    colors = [
        "#2f6b3a" if float(v) >= 0 else "#8a2d21"
        for v in part["effect_control_minus_model"]
    ]
    ax.barh(
        part["filial"].astype(str),
        part["effect_control_minus_model"].astype(float),
        color=colors,
        alpha=0.85,
    )
    ax.axvline(0.0, color="#5f696a", linewidth=1.0)
    horizon = str(part["horizon"].iloc[0])
    ax.set_title(f"ITT по филиалам, Y{horizon}", color="#075f66", fontsize=11)
    ax.set_xlabel("effect control − model, ₽")
    _style_axes(ax)
    fig.tight_layout()
    return _fig_to_img(
        fig,
        alt="ITT по филиалам",
        caption="Зелёный — экономия (control дороже model), красный — model дороже.",
    )


def _chart_path_shares(path_shares: pd.DataFrame) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path_shares is None or path_shares.empty:
        return ""
    metrics = [
        ("agreement_share", "Соглашения"),
        ("pretension_share", "Претензии"),
        ("fu_incident_share", "ФУ"),
        ("court_incident_share", "Суд"),
    ]
    base = path_shares.loc[
        path_shares["segment"].astype(str).isin(["control", "model"])
    ].copy()
    if base.empty:
        return ""
    labels = [label for _, label in metrics]
    control_vals = []
    model_vals = []
    for col, _ in metrics:
        control_vals.append(
            float(
                pd.to_numeric(
                    base.loc[base["segment"].eq("control"), col], errors="coerce"
                ).iloc[0]
            )
            if base["segment"].eq("control").any() and col in base.columns
            else 0.0
        )
        model_vals.append(
            float(
                pd.to_numeric(
                    base.loc[base["segment"].eq("model"), col], errors="coerce"
                ).iloc[0]
            )
            if base["segment"].eq("model").any() and col in base.columns
            else 0.0
        )
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(x - width / 2, control_vals, width, label="control", color="#6b5b4b")
    ax.bar(x + width / 2, model_vals, width, label="model", color="#0b5f66")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("%")
    ax.set_title("Доли путей: control vs model", color="#075f66", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)
    fig.tight_layout()
    return _fig_to_img(
        fig,
        alt="Доли путей",
        caption="Доли соглашений / претензий / ФУ / суда в ITT-выборке.",
    )


def _chart_group_means(group_summary: pd.DataFrame) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if group_summary is None or group_summary.empty:
        return ""
    horizons = [
        h for h in ("fact", "365") if h in set(group_summary["horizon"].astype(str))
    ]
    if not horizons:
        return ""
    control = []
    model = []
    for horizon in horizons:
        part = group_summary.loc[group_summary["horizon"].astype(str).eq(horizon)]
        control.append(
            float(part.loc[part["group"].eq("control"), "mean_cost"].iloc[0])
            if part["group"].eq("control").any()
            else np.nan
        )
        model.append(
            float(part.loc[part["group"].eq("model"), "mean_cost"].iloc[0])
            if part["group"].eq("model").any()
            else np.nan
        )
    x = np.arange(len(horizons))
    width = 0.36
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.bar(x - width / 2, control, width, label="control", color="#6b5b4b")
    ax.bar(x + width / 2, model, width, label="model", color="#0b5f66")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Y{h}" for h in horizons])
    ax.set_ylabel("mean cost, ₽")
    ax.set_title("Средний расход control / model", color="#075f66", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)
    fig.tight_layout()
    return _fig_to_img(
        fig,
        alt="Средний расход",
        caption="Сравнение среднего YH: сначала control, затем model.",
    )


def _chart_annual_ci(annual_summary: pd.DataFrame) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if annual_summary is None or annual_summary.empty:
        return ""
    part = annual_summary.loc[annual_summary["horizon"].astype(str).eq("365")]
    if part.empty:
        part = annual_summary.iloc[[0]]
    row = part.iloc[0]
    point = float(row["annual_network_full"])
    low = float(row.get("annual_network_full_ci_low", np.nan))
    high = float(row.get("annual_network_full_ci_high", np.nan))
    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    if pd.notna(low) and pd.notna(high):
        ax.errorbar(
            [point],
            [0],
            xerr=[[max(0.0, point - low)], [max(0.0, high - point)]],
            fmt="o",
            color="#075f66",
            ecolor="#6b5b4b",
            elinewidth=2,
            capsize=6,
            markersize=8,
        )
    else:
        ax.plot([point], [0], "o", color="#075f66", markersize=8)
    ax.axvline(0.0, color="#8a2d21", linestyle="--", linewidth=1.1)
    ax.set_yticks([])
    ax.set_xlabel("₽ / год")
    ax.set_title(
        f"Годовой эффект сети, Y{row['horizon']}",
        color="#075f66",
        fontsize=11,
    )
    _style_axes(ax)
    fig.tight_layout()
    return _fig_to_img(
        fig,
        alt="Годовой эффект сети",
        caption="Точка — annual_network_full; усы — 95% CI (линейный перенос с /убыток).",
    )


def _charts_section(result: MonitoringEffectResult) -> str:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return (
            "<p class='muted'>Графики пропущены: пакет <code>matplotlib</code> "
            "не установлен в окружении.</p>"
        )
    try:
        blocks = [
            _chart_bootstrap_hist(
                result.bootstrap_samples,
                result.effect_summary,
                title="Распределение bootstrap ITT",
            ),
            _chart_bootstrap_hist(
                result.bootstrap_compliance_samples,
                result.compliance_b,
                title="Распределение bootstrap: 100% compliance",
            ),
            _chart_filial_effects(result.filial_effects),
            _chart_path_shares(result.path_shares),
            _chart_group_means(result.group_summary),
            _chart_annual_ci(result.annual_summary),
        ]
    except Exception as exc:  # noqa: BLE001
        return (
            f"<p class='muted'>Графики не построены: "
            f"{escape(type(exc).__name__)}: {escape(str(exc))}</p>"
        )
    blocks = [block for block in blocks if block]
    if not blocks:
        return "<p class='muted'>Графики недоступны: нет данных для отрисовки.</p>"
    return '<div class="charts">' + "".join(blocks) + "</div>"


def _table(
    frame: pd.DataFrame | None,
    *,
    columns: dict[str, str] | None = None,
    rows: list[str] | None = None,
) -> str:
    """Таблица + легенда колонок и строк под ней."""
    if frame is None or frame.empty:
        return "<p class='muted'>(нет данных)</p>"
    shown = frame.copy()
    for col in shown.columns:
        series = shown[col]

        def _cell(value: Any) -> Any:
            if pd.isna(value):
                return ""
            if isinstance(value, (bool, np.bool_)):
                return "true" if bool(value) else "false"
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
                value, (bool, np.bool_)
            ):
                return _fmt_number(value)
            return value

        shown[col] = series.map(_cell)
    html = (
        '<div class="scroll">'
        + shown.to_html(index=False, border=0, escape=True)
        + "</div>"
    )
    legend_parts: list[str] = []
    if columns:
        present = set(map(str, frame.columns))
        items = "".join(
            f"<li><code>{escape(name)}</code> — {escape(desc)}</li>"
            for name, desc in columns.items()
            if name in present
        )
        if items:
            legend_parts.append(f"<h4>Колонки</h4><ul>{items}</ul>")
    if rows:
        items = "".join(f"<li>{escape(item)}</li>" for item in rows)
        legend_parts.append(f"<h4>Строки</h4><ul>{items}</ul>")
    if legend_parts:
        html += '<div class="legend">' + "".join(legend_parts) + "</div>"
    return html


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


def _nav(
    active: str,
    *,
    plan_name: str = PLAN_FILENAME,
    report_name: str = REPORT_FILENAME,
    conclusion_name: str = CONCLUSION_FILENAME,
) -> str:
    items = (
        (plan_name, "1. План"),
        (report_name, "2. Расчёт"),
        (conclusion_name, "3. Заключение"),
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
      <p>Обычный процесс урегулирования <b>без рекомендаций модели</b>,
      с применением действующих бизнес-правил (в т.ч. доплата из лимита
      директора филиала, ЕМР +20% и пр.). Считаем средний расход на один
      первичный убыток (см. определение ниже).</p>
      </div>
    <div class="mid">−</div>
    <div class="box model">
      <h4>Model</h4>
      <p>Тот же тип убытков, но с <b>назначением в модель</b>
      (<code>РезультатПроверки ∈ {0,1}</code>). В среднее входят и случаи,
      где рекомендацию не выполнили (ITT).</p>
      </div>
      </div>

  <h3>Средний расход на один убыток</h3>
  <ul>
    <li><b>Единица</b> — первичный убыток (строка витрины): форма денежная /
    ремонт / соглашение; статус «первичный»; автотранспорт; пилотные филиалы
    <b>без</b> Марийского и Архангельского.</li>
    <li><b>Yfact</b> = только <code>СуммаПлатежа</code> этого первичного убытка
    (касса на дату отчёта). Это <b>не</b> сумма «первичка + претензия + ФУ +
    суд»: претензионные/судебные выплаты обычно на других LossID и в Yfact
    первичной строки не входят.</li>
    <li><b>observed_PSR</b> на строке =
    <code>Cумма выплаты по претензии</code> +
    <code>Cумма выплат по ФУ</code> +
    <code>Cумма выплаты по суду</code> (на первичных почти всегда 0).
    Используется только чтобы вычесть уже видимый ПСР из хвоста, не чтобы
    нарастить Yfact.</li>
    <li><b>Y365</b> = Yfact + NPV(ожидаемый ещё несозревший хвост ПСР по
    ретро-коэффициентам). Это оценка «полного» расхода в смысле методики,
    а не фактический lifecycle по всем убыткам инцидента.</li>
  </ul>

  <h3>Формула эффекта</h3>
  <div class="formula">
    <div class="eq">effect(H) = Σ<sub>f</sub> w<sub>f</sub> ·
    (Ȳ<sub>H</sub>(control, f) − Ȳ<sub>H</sub>(model, f))</div>
    <div class="eq">w<sub>f</sub> = N<sub>f</sub> / Σ<sub>g</sub> N<sub>g</sub>,
    N<sub>f</sub> = число eligible-убытков филиала f (control + model)</div>
    <div class="note">Плюс = control дороже model = экономия.
    H ∈ {fact, 365}. Непростая разность глобальных средних: стратификация
    по филиалу сохраняет дизайн рандомизации.</div>
      </div>
  <p><b>Число убытков</b> — сумма N<sub>f</sub> по пилотным филиалам
  (в отчёте control / model отдельно). Марийский и Архангельский в ITT не
  входят (там модель ~100%, нет честного control).</p>

  <h3>ITT (intention-to-treat)</h3>
  <p>Главный результат — эффект <b>назначения</b> в model-поток, а не
  «эффект идеального исполнения». В model остаются и complied, и
  not_complied; соглашения / non-compliance строки не выкидывают
  (это post-treatment). Сценарий «если бы всегда исполняли» — отдельно
  (compliance B), это уже не ITT.</p>

  <div class="flow">
    <div class="flow-step">
      <div class="num">Шаг 1</div>
      <h4>Кого сравниваем</h4>
      <p>Model и control в пилотных филиалах. Соглашения и исполнение
      рекомендаций не выкидывают строки из ITT.</p>
      </div>
    <div class="flow-step">
      <div class="num">Шаг 2</div>
      <h4>Что уже заплатили</h4>
      <p><code>Yfact</code> = <code>СуммаПлатежа</code> первичного убытка.</p>
      </div>
    <div class="flow-step">
      <div class="num">Шаг 3</div>
      <h4>Что ещё может прийти</h4>
      <p>Хвост ПСР по ретро (p_U, k_U, m_U, e_U). После соглашения — 7%.</p>
      </div>
    <div class="flow-step">
      <div class="num">Шаг 4</div>
      <h4>Два горизонта</h4>
      <p><code>Yfact</code> — сейчас; <code>Y365</code> — хвост с NPV на 1 год.</p>
    </div>
    <div class="flow-step">
      <div class="num">Шаг 5</div>
      <h4>Неопределённость</h4>
      <p>95% CI. Если интервал через 0 — эффект статистически не подтверждён.</p>
      </div>
    <div class="flow-step">
      <div class="num">Шаг 6</div>
      <h4>На год и сеть</h4>
      <p>Эффект/убыток × годовой поток пилота × множитель сети — сценарий.</p>
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


def _path_share_columns() -> dict[str, str]:
    return {
        "segment": "Сегмент: control, model или lift",
        "filial": "Филиал",
        "n": "Число убытков в сегменте",
        "agreement_share": "Доля убытков с соглашением, %",
        "pretension_share": "Доля убытков с претензией, %",
        "fu_incident_share": "Доля убытков с ФУ на уровне инцидента, %",
        "court_incident_share": "Доля убытков с судом на уровне инцидента, %",
    }


def _headline(result: MonitoringEffectResult) -> str:
    effects = result.effect_summary.set_index("horizon")
    annual = result.annual_summary.set_index("horizon")
    cells = []
    for horizon in ("fact", "365"):
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
    ann = float(annual.loc["365", "annual_network_full"])
    ann_low = float(annual.loc["365", "annual_network_full_ci_low"])
    ann_high = float(annual.loc["365", "annual_network_full_ci_high"])
    cells.append(
        "<div class='stat'>"
        "<span>Годовой эффект сети, full rollout, Y365</span>"
        f"<b>{format_money(ann)}</b>"
        f"<small>95% CI: {format_money(ann_low)} … {format_money(ann_high)}</small>"
        "</div>"
    )
    return '<div class="grid">' + "".join(cells) + "</div>"


def build_plan_html(
    *,
    source_label: str = "[OISUU_report].[dbo].[ВитринаСутяжность]",
    plan_name: str = PLAN_FILENAME,
    report_name: str = REPORT_FILENAME,
    conclusion_name: str = CONCLUSION_FILENAME,
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
  источник <code>{escape(source_label)}</code> · файл <code>{escape(plan_name)}</code></p>
  {_nav(
    plan_name,
    plan_name=plan_name,
    report_name=report_name,
    conclusion_name=conclusion_name,
  )}
  {_business_schema()}

  <h2>Что фиксируем до расчёта</h2>
  <div class="card">
    <ul>
      <li>Сравниваем <b>model</b> и <b>control</b> внутри филиала
      (примерно 50/50). Марийский и Архангельский в ITT не входят.</li>
      <li>Единица — один <b>первичный</b> убыток. Дубли не удаляем, но
      показываем их влияние.</li>
      <li><b>Yfact</b> — только <code>СуммаПлатежа</code> первичного убытка
      (не сумма претензия+ФУ+суд).</li>
      <li><b>Y365</b> — Yfact + NPV хвоста ПСР по ретро пилотных филиалов.</li>
      <li>Если заключено соглашение, экспертно оставляем 7% возможного ПСР.</li>
      <li>Главный результат — ITT (эффект назначения), включая non-compliance.
      Сценарий 100% исполнения — отдельно, не ITT.</li>
      <li>Для сверки цифр — Excel-аудит с формулами
      (<code>fin_effect_audit.xlsx</code>).</li>
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
      <li><code>Y365</code> сейчас — прогноз хвоста, а не полностью
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
    plan_name: str = PLAN_FILENAME,
    report_name: str = REPORT_FILENAME,
    conclusion_name: str = CONCLUSION_FILENAME,
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
  {quality.get("observation_end", "?")} · ретро-окно {escape(retro_window)} ·
  файл <code>{escape(report_name)}</code></p>
  {_nav(
    report_name,
    plan_name=plan_name,
    report_name=report_name,
    conclusion_name=conclusion_name,
  )}

  <h2>1. Главный результат</h2>
  <div class="card">
    {_headline(result)}
    <h3>Графики</h3>
    {_charts_section(result)}
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
    {_table(
      _contract_table(result),
      columns={
        "entity": "Логическая сущность контракта",
        "source_column": "Исходная колонка витрины/ретро",
        "meaning": "Как используется в расчёте",
      },
      rows=[
        "Каждая строка — одна сущность контракта данных.",
      ],
    )}
    <h3>Диагностика качества</h3>
    {_table(
      result.data_quality,
      columns={
        "metric": "Метрика качества данных",
        "value": "Значение метрики",
      },
      rows=[
        "Каждая строка — отдельная проверка качества выборки.",
      ],
    )}
    <p class="muted">Выбрано строк без dedupe: {quality.get("n_rows", "—")};
    пропусков OD: {quality.get("missing_od", "—")};
    возможное повторное суммирование СуммаПлатежа:
    {quality.get("payment_possible_inflation", "—")}.</p>
  </div>

  <h2>4. Доли соглашений / претензий / ФУ / суда</h2>
  <div class="card">
    <p>Доли считаются на той же ITT-выборке. Сначала control, затем model.
    ФУ и суд подняты на уровень инцидента: если хотя бы один убыток инцидента
    имел ФУ/суд, флаг ставится всем строкам инцидента.</p>
    <h3>Control vs model</h3>
    {_table(
      result.path_shares,
      columns=_path_share_columns(),
      rows=[
        "control — группа без модели",
        "model — группа с назначением в модель",
        "lift_pp (model - control) — разница долей в процентных пунктах",
        "lift_rel (model / control - 1) — относительный рост доли model к control",
      ],
    )}
    <h3>По филиалам</h3>
    {_table(
      result.filial_path_shares,
      columns=_path_share_columns(),
      rows=[
        "Для каждого филиала сначала строка control, затем model.",
      ],
    )}
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
    {_table(
      result.priors.table(),
      columns={
        "group": "Группа ретро: pilot или nonpilot",
        "n_filials": "Число филиалов в группе",
        "filials": "Список филиалов / all other",
        "n_rows": "Число строк ретро",
        "n_positive": "Число строк с положительным ПСР",
        "n_positive_with_od": "Положительный ПСР и OD > 0",
        "n_positive_without_retro_od": "Положительный ПСР без OD",
        "p_U": "Доля убытков с ПСР",
        "k_U": "Коэффициент ПСР к OD",
        "m_U": "Средний положительный ПСР (₽), fallback",
        "mean_PSR_all": "Средний ПСР по всем строкам, ₽",
        "p_FU_given_PSR": "Вероятность ФУ при наличии ПСР",
        "p_court_given_PSR": "Вероятность суда при наличии ПСР",
        "e_U": "Ожидаемые издержки ФУ/суда (₽)",
      },
      rows=[
        "pilot — ретро пилотных филиалов (основной расчёт)",
        "nonpilot — ретро остальных филиалов (для сетевого масштаба)",
      ],
    )}
    <p>Основной расчёт использует pilot:
    p_U={pilot.p_ultimate:.4f}, k_U={pilot.k_ultimate:.4f},
    m_U={format_money(pilot.mean_positive_psr)},
    e_U={format_money(pilot.expected_fee)}.</p>
        </div>

  <h2>6. Yfact и Y365</h2>
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
            "<b>remaining_days<sub>i</sub></b> = max(365 − age<sub>i</sub>, 0)",
            "<b>midpoint_days<sub>i</sub></b> = remaining_days<sub>i</sub> / 2",
            "<b>Yfact<sub>i</sub></b> = paid_to_date<sub>i</sub>",
            "<b>Y365<sub>i</sub></b> = paid_to_date<sub>i</sub> + "
            "remaining<sub>i</sub> / (1+r)<sup>midpoint_days<sub>i</sub>/365</sup>",
        ],
        f"OD = СуммаОсновногоДолгаЗаявлено; q=1 без соглашения и "
        f"q={result.residual_share:.0%} при соглашении; r={result.discount_rate:.0%}; "
        "age=(t_calc−t0) в днях. Горизонт расчёта — только 365 дней.",
    )}
    <h3>Что означают remaining и midpoint</h3>
    <p><b>remaining</b> — ещё не проявившийся хвост ПСР (номинал). Это не дисконт,
    а «сколько ещё может прийти» после вычитания уже видимых претензии/ФУ/суда.</p>
    <p><b>midpoint</b> — упрощение срока будущего платежа: хвост считается
    поступившим в середине оставшегося окна до горизонта 365 дней.
    Дисконтирование здесь — NPV (приведение будущих рублей к сегодняшним),
    а не рост ПСР со временем.</p>
    <div class="formula">
      <b>Числовые примеры remaining</b>
      <div class="eq">Без соглашения: q=1, expected=100 000, observed=20 000 →
      remaining = max(100 000−20 000, 0) = 80 000</div>
      <div class="eq">С соглашением: q=0.07, expected=100 000, observed=0 →
      remaining = max(7 000−0, 0) = 7 000</div>
      <div class="eq">Уже «перебрали»: expected=100 000, observed=150 000 →
      remaining = 0</div>
    </div>
    <div class="formula">
      <b>Пример midpoint / NPV для Y365</b>
      <div class="eq">age=184, remaining=80 000, paid=50 000, r=12%</div>
      <div class="eq">remaining_days=181, midpoint=90.5, DF≈1.028 →
      Y365 ≈ 50 000 + 77 821 = 127 821</div>
    </div>
    <h3>Итоги control / model</h3>
    {_table(
      result.group_summary,
      columns={
        "horizon": "Горизонт: fact / 365",
        "group": "Группа: control или model",
        "n": "Число убытков",
        "sum_cost": "Сумма исхода YH по группе, ₽",
        "mean_cost": "Средний исход YH на убыток, ₽",
      },
      rows=[
        "Для каждого горизонта сначала control, затем model.",
      ],
    )}
  </div>

  <h2>7. ITT по филиалам и неопределённость</h2>
  <div class="card">
    <p>Веса фиксируются по всей eligible-популяции филиала, а не по размеру
    control/model.</p>
    {_table(
      result.effect_summary,
      columns={
        "horizon": "Горизонт: fact / 365",
        "effect_per_case": "ITT: mean(control) − mean(model), взвешенно по филиалам, ₽/убыток",
        "unstratified_effect": "Та же разность без взвешивания по филиалам, ₽/убыток",
        "n_weighted": "Сумма весов филиалов (число eligible-убытков)",
        "n_filials": "Число филиалов с парой control и model",
        "ci_low": "Нижняя граница 95% CI bootstrap, ₽/убыток",
        "ci_high": "Верхняя граница 95% CI bootstrap, ₽/убыток",
        "n_bootstrap": "Число успешных bootstrap-повторений",
      },
      rows=[
        "Одна строка на горизонт fact / 365.",
      ],
    )}
    <h3>Вклад филиалов</h3>
    {_table(
      result.filial_effects,
      columns={
        "horizon": "Горизонт",
        "filial": "Филиал",
        "n": "Всего убытков в филиале",
        "n_control": "Число убытков control",
        "n_model": "Число убытков model",
        "mean_control": "Средний YH в control, ₽",
        "mean_model": "Средний YH в model, ₽",
        "effect_control_minus_model": "Эффект филиала: mean(control) − mean(model), ₽",
      },
      rows=[
        "Одна строка на пару горизонт × филиал; в колонках сначала control, затем model.",
      ],
    )}
    <h3>Филиалы на внимании</h3>
    <p>Показываем филиалы, где доля соглашений в control выше, чем в model,
    и/или ITT-эффект control−model отрицательный (model дороже).</p>
    {_table(
      result.attention_filials,
      columns={
        "filial": "Филиал",
        "agreement_share_control": "Доля соглашений в control, %",
        "agreement_share_model": "Доля соглашений в model, %",
        "agreement_gap_pp": "control − model по соглашениям, п.п.",
        "negative_effect_horizons": "Горизонты с отрицательным ITT",
        "min_effect_control_minus_model": "Минимальный эффект по горизонтам, ₽",
        "flags": "Почему филиал в списке",
      },
      rows=[
        "Пустая таблица означает, что таких филиалов нет.",
        "flags: agreement_control_gt_model и/или negative_itt.",
      ],
    )}
    {_formula(
        "95% CI: двухчастный bootstrap",
        [
            "1) ретро incident-level строки resample → пересчёт p_U, k_U, m_U, e_U;",
            "2) текущая витрина resample кластерами по НомерИнцидент → пересчёт Y и ITT;",
            "CI = 2.5% и 97.5% квантили bootstrap effect(H).",
        ],
        f"ITT: запрошено {result.bootstrap_iterations} итераций. "
        f"Сценарий 100% compliance: "
        f"{result.bootstrap_compliance_iterations} итераций. "
        "Bootstrap параллелится (bootstrap_n_jobs); прогресс — в tqdm.",
    )}
  </div>

  <h2>8. Non-compliance</h2>
  <div class="card">
    <h3>A. As-complied — только описательная диагностика (Yfact)</h3>
    <p>Сравнение внутри model и <code>РезультатПроверки=1</code> только по факту.
    Не является причинным эффектом; горизонт 365 здесь не считается.</p>
    {_table(
      result.compliance_a,
      columns={
        "horizon": "Только fact",
        "compliance": "complied или not_complied",
        "n": "Число убытков",
        "agreement_share": "Доля соглашений внутри группы, %",
        "mean_cost": "Средний Yfact, ₽",
        "descriptive_only": "Признак: только описание, не causal effect",
      },
      rows=[
        "complied — рекомендация исполнена (Выплата по модели=1)",
        "not_complied — рекомендация не исполнена",
        "База: model и РезультатПроверки=1.",
      ],
    )}
    <h3>B. Механический сценарий 100% исполнения</h3>
    {_formula(
        "Контракт сценария",
        [
            "<b>forced_extra<sub>i</sub></b> = рекомендованная доплата, если "
            "model, result=1 и Выплата по модели≠1; иначе 0",
            "<b>Yfact_100<sub>i</sub></b> = СуммаПлатежа<sub>i</sub> + forced_extra<sub>i</sub>",
            "для всех model/result=1: q<sub>i</sub>=7%; затем Y365_100",
            "<b>effect_100(H)</b> = stratified mean(control actual) − mean(model scenario)",
        ],
        "Сценарная механика, не LATE/IV. CI — отдельный bootstrap на Y*_100.",
    )}
    {_table(
      result.compliance_b,
      columns={
        "horizon": "Горизонт сценария 100% compliance",
        "effect_per_case": "Сценарный ITT: control actual − model scenario, ₽/убыток",
        "unstratified_effect": "Та же разность без стратификации, ₽/убыток",
        "n_weighted": "Сумма весов филиалов",
        "n_filials": "Число филиалов в расчёте",
        "ci_low": "Нижняя граница 95% CI bootstrap сценария, ₽/убыток",
        "ci_high": "Верхняя граница 95% CI bootstrap сценария, ₽/убыток",
        "n_bootstrap": "Число успешных bootstrap-повторений сценария",
        "scenario": "Описание сценария",
      },
      rows=[
        "Одна строка на горизонт сценария 100% исполнения.",
      ],
    )}
  </div>

  <h2>9. Чувствительность</h2>
  <div class="card">
    <p>Сетка: r ∈ {{8%,12%,16%}} и остаток после соглашения q ∈ {{0%,7%,15%}}.</p>
    {_table(
      result.sensitivity,
      columns={
        "discount_rate": "Ставка дисконтирования r",
        "residual_share": "Остаток ПСР после соглашения q",
        "horizon": "Горизонт",
        "effect_per_case": "ITT эффект на убыток при этих параметрах, ₽",
      },
      rows=[
        "Каждая строка — одна комбинация (r, q, горизонт).",
      ],
    )}
  </div>

  <h2>10. Сезонный годовой эффект и сеть</h2>
  <div class="card">
    {_formula(
        "Экстраполяция потока",
        [
            "<b>s<sub>m,P</sub></b> = средняя доля месяца m в годовом pilot-потоке ретро",
            "<b>coverage<sub>m</sub></b> = доля дней месяца m, попавших в окно мониторинга",
            "<b>seasonal_exposure</b> = Σ<sub>m</sub> coverage<sub>m</sub>×s<sub>m,P</sub>",
            "<b>N_obs</b> = число eligible-убытков current (model+control)",
            "<b>N_pilot_eligible_year</b> = N_obs / seasonal_exposure",
            "<b>network_multiplier</b> = 1 + volume_ratio_NP×risk_ratio_NP",
            "<b>annual_network_full</b> = effect_per_case × N_pilot_eligible_year × network_multiplier",
            "<b>annual CI</b> = [ci_low, ci_high] × N_pilot_eligible_year × network_multiplier",
        ],
        "Горизонт Y365 не умножается на 365/30: масштабируется число новых убытков. "
        "CI годового эффекта — линейный перенос bootstrap CI с уровня убытка.",
    )}
    <h3>Зачем сезонность и откуда берутся числа</h3>
    <p>Нужно ответить: «в окне набралось N_obs убытков за неполный год;
    сколько ждать за полный типичный год?» Это <b>число убытков</b>, не сумма выплат.</p>
    <ul>
      <li><b>s<sub>m</sub></b> — из ретро-пилота по дате
      (<code>PAYMENT_ORDER_DATE_TIME</code> / fallback), доля строк месяца в годе.</li>
      <li><b>coverage<sub>m</sub></b> — по мониторингу <code>ДатаЗаявления</code>
      (<code>_application_date</code>): какая часть месяца попала в окно.</li>
      <li><b>seasonal_exposure</b> — какую долю типичного года мы уже увидели.</li>
      <li><b>N_pilot_eligible_year</b> — годовой поток eligible = N_obs / exposure.</li>
      <li><b>risk_ratio</b> — уже по деньгам <code>TARGET_FREQ_AMOUNT</code> (для сети).</li>
    </ul>
        <div class="formula">
      <b>Пример</b>
      <div class="eq">N_obs = 2 500, seasonal_exposure = 0.25 →
      N_pilot_eligible_year = 2 500 / 0.25 = 10 000</div>
      <div class="eq">effect_per_case = 4 694, network_multiplier = 2.5 →
      annual_network_full ≈ 4 694 × 10 000 × 2.5 = 117 350 000</div>
      <div class="note">Если окно попало в «лёгкие» месяцы, exposure меньше и
      годовой поток выше при том же N_obs — поэтому и нужна сезонная поправка.</div>
        </div>
    <h3>Сезонные доли и покрытие текущего окна</h3>
    {_table(
      result.seasonality,
      columns={
        "month": "Месяц (1–12)",
        "retro_share": "Средняя доля месяца в годовом pilot-потоке ретро",
        "observed_coverage": "Покрытие месяца текущим окном наблюдения (0–1+)",
        "exposure_contribution": "Вклад месяца в seasonal_exposure",
      },
      rows=[
        "Одна строка на календарный месяц.",
      ],
    )}
    <h3>Сценарии годового эффекта</h3>
    {_table(
      result.annual_summary,
      columns={
        "horizon": "Горизонт",
        "effect_per_case": "ITT эффект на убыток, ₽",
        "effect_100_compliance": "Сценарный эффект 100% compliance на убыток, ₽",
        "seasonal_exposure": "Доля года, покрытая текущим окном",
        "N_pilot_eligible_year": "Ожидаемый годовой поток eligible-убытков пилота",
        "N_pilot_model_year_current": "Годовой поток model при текущей доле model",
        "N_pilot_model_year_full": "Годовой поток при full rollout (= N_pilot_eligible_year)",
        "volume_ratio_nonpilot": "Отношение годового потока nonpilot/pilot",
        "risk_ratio_nonpilot": "Отношение среднего ПСР nonpilot/pilot",
        "network_multiplier": "1 + volume_ratio × risk_ratio",
        "annual_pilot_current": "Годовой эффект пилота при текущей доле model, ₽/год",
        "annual_pilot_full": "Годовой эффект пилота при full rollout, ₽/год",
        "annual_pilot_full_ci_low": "Нижняя граница 95% CI годового эффекта пилота, ₽/год",
        "annual_pilot_full_ci_high": "Верхняя граница 95% CI годового эффекта пилота, ₽/год",
        "effect_per_case_nonpilot": "Эффект на убыток × risk_ratio, ₽",
        "annual_network_full": "Годовой эффект сети full rollout, ₽/год",
        "annual_network_full_ci_low": "Нижняя граница 95% CI годового эффекта сети, ₽/год",
        "annual_network_full_ci_high": "Верхняя граница 95% CI годового эффекта сети, ₽/год",
        "annual_network_full_compliance": "Годовой эффект сети при 100% compliance, ₽/год",
      },
      rows=[
        "Одна строка на горизонт fact / 365.",
        "CI годового эффекта = CI эффекта на убыток × N_pilot_eligible_year × network_multiplier.",
      ],
    )}
  </div>

  <h2>11. Ограничения</h2>
  <div class="card">
    <ul>
      <li><b>ITT</b> — эффект назначения в model-поток, включая фактический non-compliance.</li>
      <li><b>Y365</b> — Yfact плюс модельный NPV остатка; это не полностью
      дозревший фактический горизонт.</li>
      <li>Терминальные priors не содержат отдельного 365-таргета вызревания.</li>
      <li>В мониторинге OD — <code>СуммаОсновногоДолгаЗаявлено</code>, а k_U
      калибруется на <code>RECOVEREDMAINDEBT_LAST_INST_SUM</code>.</li>
      <li>7% — экспертное допущение; midpoint — упрощение времени будущего хвоста.</li>
      <li>Сетевой масштаб — сценарий, не рандомизированная оценка для nonpilot.</li>
      <li>CI годового эффекта — линейный перенос CI с уровня убытка; отдельный
      bootstrap годового масштаба не считается.</li>
    </ul>
  </div>
    </div>
</body>
</html>
"""



def _fmt_days(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{float(value):,.0f}".replace(",", " ")


def build_conclusion_body_from_result(
    result: MonitoringEffectResult,
    *,
    development_lags: pd.DataFrame | None = None,
) -> str:
    """Тело заключения по фактическому расчёту."""
    effects = result.effect_summary.set_index("horizon")
    annual = result.annual_summary.set_index("horizon")
    quality = result.data_quality.set_index("metric")["value"].to_dict()
    paths = (
        result.path_shares.set_index("segment")
        if not result.path_shares.empty
        else None
    )

    def _eff(horizon: str) -> tuple[float, float, float]:
        return (
            float(effects.loc[horizon, "effect_per_case"]),
            float(effects.loc[horizon, "ci_low"]),
            float(effects.loc[horizon, "ci_high"]),
        )

    e_f, lo_f, hi_f = _eff("fact")
    e_y, lo_y, hi_y = _eff("365")
    ann = float(annual.loc["365", "annual_network_full"])
    ann_lo = float(annual.loc["365", "annual_network_full_ci_low"])
    ann_hi = float(annual.loc["365", "annual_network_full_ci_high"])
    confirmed = lo_f > 0 and lo_y > 0

    n_rows = int(quality.get("n_rows", len(result.frame)))
    n_c = int(quality.get("n_control", 0))
    n_m = int(quality.get("n_model", 0))
    n_fil = int(quality.get("n_filials", 0))
    max_age = int(quality.get("max_age_days", 0))
    obs_start = quality.get("observation_start", "?")
    obs_end = quality.get("observation_end", "?")

    path_lines: list[str] = []
    if paths is not None:
        for seg in ("control", "model"):
            if seg not in paths.index:
                continue
            row = paths.loc[seg]
            path_lines.append(
                f"<li>{escape(seg)}: соглашения "
                f"{float(row.get('agreement_share', np.nan)):.1f}%, "
                f"претензии {float(row.get('pretension_share', np.nan)):.1f}%, "
                f"ФУ {float(row.get('fu_incident_share', np.nan)):.1f}%, "
                f"суд {float(row.get('court_incident_share', np.nan)):.1f}%.</li>"
            )

    if result.attention_filials.empty:
        attention_html = "<p class='muted'>Нет филиалов с флагами внимания.</p>"
    else:
        items = []
        for row in result.attention_filials.to_dict("records"):
            items.append(
                f"<li><b>{escape(str(row['filial']))}</b> — "
                f"{escape(str(row.get('flags', '')))}; "
                f"горизонты: "
                f"{escape(str(row.get('negative_effect_horizons', '') or '—'))}</li>"
            )
        attention_html = "<ul>" + "".join(items) + "</ul>"

    lag_html = """
  <p>Эмпирические перцентили лагов T0→претензия/ФУ/суд считаются по
  ретро pretensions/claims (см. notebook). Без этих файлов — ориентир
  политики maturity:</p>
  <ul>
    <li>тишина по претензиям ≈ <b>6 месяцев</b> (~183 дня) после T0;</li>
    <li>тишина по ФУ/суду ≈ <b>24 месяца</b> (~730 дней) после T0;</li>
    <li>охлаждение после события ПСР ≈ <b>24 месяца</b>.</li>
  </ul>
  <p>Без претензии ФУ/суд редки: сначала смотрим сроки претензий, затем ФУ/суда.
  При max age пилота существенно меньше этих горизонтов хвост ещё короткий.</p>
"""
    if development_lags is not None and not development_lags.empty:
        cols = ["stage", "n", "definition"] + [
            c for c in development_lags.columns if str(c).startswith("p")
        ]
        show = development_lags[
            [c for c in cols if c in development_lags.columns]
        ].copy()
        for c in show.columns:
            if str(c).startswith("p"):
                show[c] = show[c].map(_fmt_days)
        lag_html = (
            "<p>Перцентили лагов (дни) на ретро. Без претензии ФУ/суд обычно "
            "не начинаются — стадии идут цепочкой.</p>"
            + _table(
                show,
                columns={
                    "stage": "Стадия",
                    "n": "N событий",
                    "definition": "Определение лага",
                    "p50": "p50, дни",
                    "p70": "p70, дни",
                    "p80": "p80, дни",
                    "p90": "p90, дни",
                    "p95": "p95, дни",
                },
                rows=["Каждая строка — одна стадия развития ПСР."],
            )
            + """
  <p class="muted">Политика maturity дополнительно: тишина претензий ~6 мес.,
  ФУ/суд и cooloff ~24 мес.</p>
"""
        )

    half_y = (
        (hi_y - lo_y) / 2.0
        if np.isfinite(hi_y) and np.isfinite(lo_y)
        else np.nan
    )
    target = abs(e_y) if abs(e_y) > 1 else 5000.0
    scale = (
        (half_y / target) ** 2
        if half_y and half_y > 0 and target > 0
        else np.nan
    )
    n_need = int(round(n_rows * scale)) if np.isfinite(scale) else None
    scale_txt = f"{scale:.0f}" if np.isfinite(scale) else "—"
    half_txt = format_money(half_y) if np.isfinite(half_y) else "—"
    ratio_txt = f"{(half_y / target):.1f}" if np.isfinite(half_y) else "—"
    need_txt = str(n_need) if n_need is not None else "—"

    verdict = (
        "<b>Статистически подтверждённая экономия</b> (оба CI выше 0)."
        if confirmed
        else (
            "<b>Точечная оценка показывает экономию</b>, но "
            "<b>95% CI проходит через 0</b> — статистически подтверждённой "
            "экономии пока нет."
        )
    )

    return f"""
  <div class="card">
  <h3>Вердикт</h3>
  <p>{verdict}
  Годовой эффект сети — сценарная точка; его CI тоже нужно читать вместе
  с точечной оценкой.</p>
</div>

<div class="grid">
  <div class="stat">
    <span>ITT / убыток, Yfact</span>
    <b>{format_money(e_f)}</b>
    <small>95% CI: {format_money(lo_f)} … {format_money(hi_f)}</small>
  </div>
  <div class="stat">
    <span>ITT / убыток, Y365</span>
    <b>{format_money(e_y)}</b>
    <small>95% CI: {format_money(lo_y)} … {format_money(hi_y)}</small>
  </div>
  <div class="stat">
    <span>Сеть full rollout, Y365</span>
    <b>{format_money(ann)}</b>
    <small>95% CI: {format_money(ann_lo)} … {format_money(ann_hi)}</small>
  </div>
</div>

<div class="card">
  <h3>Что означают горизонты</h3>
  <ul>
    <li><b>Yfact</b> — только <code>СуммаПлатежа</code> первичного убытка.</li>
    <li><b>Y365</b> — Yfact + NPV ultimate-хвоста ПСР на 1 год (не lifecycle
    по всем LossID инцидента).</li>
    <li>Плюс = средний расход control выше, чем у model (экономия).</li>
    <li>ITT включает non-compliance; сценарий 100% исполнения — отдельно.</li>
  </ul>
</div>

<div class="card">
  <h3>Выборка и пути</h3>
  <ul>
    <li>Окно: {escape(str(obs_start))} … {escape(str(obs_end))};
    <b>{n_rows}</b> строк (control {n_c} / model {n_m}),
    {n_fil} пилотных филиалов (без Марийского/Архангельского).</li>
    {"".join(path_lines)}
    <li>Максимальный возраст убытка <b>{max_age}</b> дней — хвост ещё короткий
    относительно типичного развития претензий/ФУ/суда.</li>
    <li>Bootstrap ITT: {result.bootstrap_iterations}; compliance:
    {result.bootstrap_compliance_iterations}.</li>
  </ul>
  <h4>Сроки развития ПСР (ретро)</h4>
  {lag_html}
</div>

<div class="card">
  <h3>Сколько ещё статистики нужно</h3>
  <ul>
    <li>Сейчас полуширина CI по Y365 ≈ {half_txt}
    при точечном эффекте ≈ {format_money(e_y)}.</li>
    <li>Чтобы при том же эффекте CI перестал включать 0, шум нужно сжать
    примерно в {ratio_txt}× → объём порядка
    <b>×{scale_txt}</b> (~{need_txt} убытков)
    при том же составе — грубая оценка.</li>
    <li>До декабря на текущих 10 филиалах одного ожидания, скорее всего,
    не хватит. Имеет смысл добавить крупные филиалы с честной рандомизацией
    ~50/50 (не Марийка/Архангельск в режиме 100% model) и параллельно
    дозревать хвост.</li>
  </ul>
</div>

<div class="card">
  <h3>Филиалы на внимании</h3>
  {attention_html}
</div>

<div class="card">
  <h3>Что можно и нельзя обещать</h3>
  <ul>
    <li><b>Можно:</b> направление точечной оценки и сценарий сети как гипотезу.</li>
    <li><b>Нельзя:</b> утверждать статистически подтверждённую экономию,
    пока CI включают 0.</li>
    <li><b>Нельзя:</b> трактовать годовой сетевой эффект как измеренный факт.</li>
  </ul>
</div>

<div class="card">
  <h3>Риски и следующие шаги</h3>
  <ul>
    <li>Набрать дозревание и объём; пересчитать с Excel-аудитом.</li>
    <li>Разобрать филиалы с отрицательным ITT / расхождением соглашений.</li>
    <li>Проверить «Выплата по модели» и расхождения СуммаПлатежа / СуммаКВыплате.</li>
  </ul>
  </div>
"""


def build_conclusion_html(
    *,
    source_label: str = "[OISUU_report].[dbo].[ВитринаСутяжность]",
    ready: bool = False,
    body_html: str | None = None,
    plan_name: str = PLAN_FILENAME,
    report_name: str = REPORT_FILENAME,
    conclusion_name: str = CONCLUSION_FILENAME,
) -> str:
    """HTML №3: заключение. Пока заготовка, пока не прислан файл №2."""
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    if ready and body_html:
        content = body_html
    else:
        content = f"""
<div class="card">
  <div class="warning">Заключение ещё не заполнено.</div>
  <p>После проверки файла <code>{escape(report_name)}</code> пришлите его
  как «файл №2» — по нему будет подготовлено краткое бизнес-заключение:</p>
  <ul>
    <li>есть ли подтверждённая экономия;</li>
    <li>что означают Yфакт / Y365;</li>
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
  источник <code>{escape(source_label)}</code> ·
  файл <code>{escape(conclusion_name)}</code></p>
  {_nav(
    conclusion_name,
    plan_name=plan_name,
    report_name=report_name,
    conclusion_name=conclusion_name,
  )}
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
    plan_name: str = PLAN_FILENAME,
    report_name: str = REPORT_FILENAME,
    conclusion_name: str = CONCLUSION_FILENAME,
) -> Path:
    """Записать HTML №2 расчёта."""
    return _write(
        Path(path),
        build_monitoring_html(
            result,
            source_label=source_label,
            plan_name=plan_name,
            report_name=report_name,
            conclusion_name=conclusion_name,
        ),
    )


def write_plan_html(
    path: str | Path,
    *,
    source_label: str = "[OISUU_report].[dbo].[ВитринаСутяжность]",
    plan_name: str = PLAN_FILENAME,
    report_name: str = REPORT_FILENAME,
    conclusion_name: str = CONCLUSION_FILENAME,
) -> Path:
    """Записать HTML №1 плана."""
    return _write(
        Path(path),
        build_plan_html(
            source_label=source_label,
            plan_name=plan_name,
            report_name=report_name,
            conclusion_name=conclusion_name,
        ),
    )


def write_conclusion_html(
    path: str | Path,
    *,
    source_label: str = "[OISUU_report].[dbo].[ВитринаСутяжность]",
    ready: bool = False,
    body_html: str | None = None,
    plan_name: str = PLAN_FILENAME,
    report_name: str = REPORT_FILENAME,
    conclusion_name: str = CONCLUSION_FILENAME,
) -> Path:
    """Записать HTML №3 заключения."""
    return _write(
        Path(path),
        build_conclusion_html(
            source_label=source_label,
            ready=ready,
            body_html=body_html,
            plan_name=plan_name,
            report_name=report_name,
            conclusion_name=conclusion_name,
        ),
    )


def write_all_monitoring_htmls(
    result: MonitoringEffectResult,
    data_dir: str | Path,
    *,
    source_label: str = "[OISUU_report].[dbo].[ВитринаСутяжность]",
    development_lags: pd.DataFrame | None = None,
    stamp: str | None = None,
) -> tuple[Path, Path, Path, dict[str, str]]:
    """Записать три HTML с датой выгрузки в имени (не затирают прошлые).

    Возвращает (plan, report, conclusion, names).
    """
    data_dir = Path(data_dir)
    export_stamp = stamp or report_export_stamp()
    names = dated_artifact_names(export_stamp)
    plan = write_plan_html(
        data_dir / names["plan"],
        source_label=source_label,
        plan_name=names["plan"],
        report_name=names["report"],
        conclusion_name=names["conclusion"],
    )
    report = write_monitoring_html(
        result,
        data_dir / names["report"],
        source_label=source_label,
        plan_name=names["plan"],
        report_name=names["report"],
        conclusion_name=names["conclusion"],
    )
    conclusion = write_conclusion_html(
        data_dir / names["conclusion"],
        source_label=source_label,
        ready=True,
        body_html=build_conclusion_body_from_result(
            result,
            development_lags=development_lags,
        ),
        plan_name=names["plan"],
        report_name=names["report"],
        conclusion_name=names["conclusion"],
    )
    return plan, report, conclusion, names


def write_error_html(
    error: Exception,
    path: str | Path,
    *,
    source_label: str,
    plan_name: str = PLAN_FILENAME,
    report_name: str = REPORT_FILENAME,
    conclusion_name: str = CONCLUSION_FILENAME,
) -> Path:
    """Записать диагностический HTML, если расчёт остановлен guard-проверкой."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"/><title>Ошибка расчёта</title>
<style>{_CSS}</style></head><body><div class="page">
<h1>Финансовый эффект не рассчитан</h1>
{_nav(
    report_name,
    plan_name=plan_name,
    report_name=report_name,
    conclusion_name=conclusion_name,
)}
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
    "build_conclusion_body_from_result",
    "build_conclusion_html",
    "build_monitoring_html",
    "build_plan_html",
    "dated_artifact_names",
    "report_export_stamp",
    "write_all_monitoring_htmls",
    "write_conclusion_html",
    "write_error_html",
    "write_monitoring_html",
    "write_plan_html",
]
