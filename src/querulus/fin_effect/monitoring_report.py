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
FORMULA_VERSION = "ITT-U-2026-09-18-v14"


def report_export_stamp(
    when: datetime | pd.Timestamp | str | None = None,
) -> str:
    """Метка выгрузки для папки прогона: YYYY-MM-DD_HHMM."""
    ts = pd.Timestamp(when) if when is not None else pd.Timestamp.now()
    return ts.strftime("%Y-%m-%d_%H%M")


def export_run_dir(
    data_dir: str | Path,
    stamp: str | None = None,
) -> Path:
    """Папка одного прогона: ``data_dir/<stamp>/`` (создаётся при необходимости)."""
    export_stamp = stamp or report_export_stamp()
    path = Path(data_dir) / export_stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def dated_artifact_names(stamp: str | None = None) -> dict[str, str]:
    """Имена артефактов одного прогона внутри ``data/<stamp>/``.

    Общие файлы (snapshots, weekly HTML) лежат в корне ``data/`` и сюда не входят.
    Аргумент ``stamp`` сохранён для совместимости и на имена не влияет.
    """
    _ = stamp
    return {
        "plan": PLAN_FILENAME,
        "report": REPORT_FILENAME,
        "conclusion": CONCLUSION_FILENAME,
        "audit": "fin_effect_audit.xlsx",
        "error": "fin_effect_error.html",
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
.fx-tree { margin:.4rem 0; }
.fx-node {
  border:1px solid var(--line); border-radius:8px; background:#fff;
  padding:.65rem .75rem; margin:.45rem 0;
}
.fx-node.main {
  border-left:4px solid var(--accent);
  background:linear-gradient(180deg,#f7fbfb 0%,#fff 48%);
}
.fx-node.sub {
  border-left:3px solid #9bb9bc;
  background:#fbfbf9;
}
.fx-head {
  display:flex; flex-wrap:wrap; align-items:baseline; gap:.45rem;
  margin:0 0 .35rem; font-weight:600; color:#244f53;
}
.fx-tag {
  display:inline-block; font-size:.68rem; font-weight:700;
  letter-spacing:.03em; text-transform:uppercase;
  border-radius:999px; padding:.12rem .45rem;
}
.fx-tag.main {
  color:#fff; background:var(--accent);
}
.fx-tag.sub {
  color:#35565a; background:#e7f0f1; border:1px solid #c7d9db;
}
.fx-need {
  margin:.45rem 0 .2rem; color:var(--muted); font-size:.86rem;
}
.fx-kids {
  margin:.35rem 0 0 .35rem; padding-left:.55rem;
  border-left:2px dashed #c9d8d9;
}
.kv { width:100%; border-collapse:collapse; margin:.35rem 0 .7rem; font-size:.92rem; }
.kv th, .kv td { border:none; border-bottom:1px solid var(--line); padding:.4rem .35rem; vertical-align:top; }
.kv th { width:34%; color:var(--accent); background:transparent; font-weight:600; position:static; }
.kv td { color:var(--ink); }
.million { font-size:1.35rem; font-weight:700; color:var(--save); }
@media (max-width:820px) {
  .compare { grid-template-columns:1fr; }
  .compare .mid { min-height:auto; }
  .eq { white-space:normal; }
  .kv th { width:40%; }
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


def _c(title: str, desc: str, formula: str = "") -> tuple[str, str, str]:
    """Краткий заголовок, текстовое описание и формула колонки для ``_table``."""
    return (title, desc, formula)


def _parse_col_spec(name: str, spec: Any) -> tuple[str, str, str]:
    """Нормализовать спецификацию колонки → (title, desc, formula).

    - ``str`` → заголовок = код колонки, описание = строка
    - ``(title, desc)`` → без формулы
    - ``(title, desc, formula)`` → полный вариант (также результат ``_c``)
    """
    if isinstance(spec, tuple):
        if len(spec) >= 3:
            return str(spec[0]), str(spec[1]), str(spec[2])
        if len(spec) == 2:
            return str(spec[0]), str(spec[1]), ""
        if len(spec) == 1:
            return str(name), str(spec[0]), ""
    return str(name), str(spec), ""


def _table(
    frame: pd.DataFrame | None,
    *,
    columns: dict[str, Any] | None = None,
    rows: list[str] | None = None,
    formulas: list[str] | None = None,
    notes: list[str] | None = None,
) -> str:
    """Таблица с краткими заголовками + легенда только по реальным колонкам.

    В шапке — короткий ``title`` (или код колонки). Внизу для каждой колонки
    таблицы: ``код`` — описание и формула, если задана.
    Ключи из ``columns``, которых нет в DataFrame, в легенду не попадают.
    Колонки DataFrame без записи в ``columns`` остаются в таблице и в легенде
    помечаются как без описания.
    ``formulas`` / ``notes`` — общие пояснения ко всей таблице.
    """
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

    present = [str(c) for c in shown.columns]
    present_set = set(present)
    specs: dict[str, tuple[str, str, str]] = {}
    if columns:
        for name, spec in columns.items():
            key = str(name)
            if key in present_set:
                specs[key] = _parse_col_spec(key, spec)
    for key in present:
        if key not in specs:
            specs[key] = (key, "(нет описания в легенде)", "")

    ordered = [name for name in (columns or {}) if str(name) in present_set]
    ordered_set = set(ordered)
    rest = [c for c in present if c not in ordered_set]
    display_order = ordered + rest
    shown = shown.loc[:, display_order]
    rename = {name: specs[name][0] for name in display_order}
    shown = shown.rename(columns=rename)

    legend_parts: list[str] = []
    items: list[str] = []
    for name in display_order:
        title, desc, formula = specs[name]
        head = f"<code>{escape(name)}</code>"
        if title and title != name:
            head += f" → <b>{escape(title)}</b>"
        block = f"<li>{head} — {escape(desc)}"
        if formula:
            block += f'<div class="eq">{escape(formula)}</div>'
        block += "</li>"
        items.append(block)
    legend_parts.append(f"<h4>Колонки</h4><ul>{''.join(items)}</ul>")

    html = (
        '<div class="scroll">'
        + shown.to_html(index=False, border=0, escape=True)
        + "</div>"
    )
    if rows:
        row_items = "".join(f"<li>{escape(item)}</li>" for item in rows)
        legend_parts.append(f"<h4>Строки</h4><ul>{row_items}</ul>")
    if formulas:
        eqs = "".join(f'<div class="eq">{equation}</div>' for equation in formulas)
        legend_parts.append(f"<h4>Как считается</h4>{eqs}")
    if notes:
        note_items = "".join(f"<li>{escape(item)}</li>" for item in notes)
        legend_parts.append(f"<h4>Пояснения</h4><ul>{note_items}</ul>")
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
      директора филиала, ЕМР +20% и пр.).       Считаем средний расход на один
      инцидент (см. определение ниже).</p>
      </div>
    <div class="mid">−</div>
    <div class="box model">
      <h4>Model</h4>
      <p>Тот же тип убытков, но с <b>назначением в модель</b>
      (<code>РезультатПроверки ∈ {0,1}</code>). В среднее входят и случаи,
      где рекомендацию не выполнили (ITT).</p>
      </div>
      </div>

  <h3>Средний расход на один инцидент</h3>
  <ul>
    <li><b>Единица</b> — инцидент после дедупа по номеру убытка и схлопывания
    убытков: форма денежная / ремонт / соглашение; статус «первичный»;
    автотранспорт; пилотные филиалы <b>без</b> Марийского и Архангельского.
    Списанные по форме возмещения в схлопывание не входят.</li>
    <li><b>Result</b>: если у части убытков инцидента Result пустой, а у одного
    −100 (или 0/1) — на весь инцидент берём это значение.</li>
    <li><b>Yfact</b> = сумма <code>СуммаПлатежа</code> по убыткам инцидента
    (касса на дату отчёта). Observed ПСР вычитается из хвоста, не наращивает
    Yfact.</li>
    <li><b>observed_PSR</b> на инциденте = сумма выплат по претензии / ФУ /
    суду по схлопнутым убыткам.</li>
    <li><b>Y365</b> = Yfact + NPV(ожидаемый ещё несозревший хвост ПСР по
    ретро-коэффициентам).</li>
  </ul>

  <h3>Формула эффекта</h3>
  <div class="formula">
    <div class="eq">effect(H) = Σ<sub>f</sub> w<sub>f</sub> ·
    (Ȳ<sub>H</sub>(control, f) − Ȳ<sub>H</sub>(model, f))</div>
    <div class="eq">w<sub>f</sub> = N<sub>f</sub> / Σ<sub>g</sub> N<sub>g</sub>,
    N<sub>f</sub> = число eligible-инцидентов филиала f (control + model)</div>
    <div class="note">Плюс = control дороже model = экономия.
    H ∈ {fact, 365}. Непростая разность глобальных средних: стратификация
    по филиалу сохраняет дизайн рандомизации.</div>
      </div>
  <p><b>Число инцидентов</b> — сумма N<sub>f</sub> по пилотным филиалам
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
      <p><code>Yfact</code> = сумма <code>СуммаПлатежа</code> по убыткам инцидента.</p>
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


def _plan_glossary() -> str:
    """Словарь терминов в начале plan (по-русски, без full rollout)."""
    return """
  <h2>Словарь: что есть что</h2>
  <div class="card">
    <table class="kv">
      <tr>
        <th>Пилот</th>
        <td>Филиалы, где идёт эксперимент с моделью (примерно 50/50 ручеек и модель).
        Без Марийского и Архангельского — там нет честного ручейка.</td>
      </tr>
      <tr>
        <th>Ручеек</th>
        <td>Control: <code>РезультатПроверки = −100</code>. Обычный процесс без модели.</td>
      </tr>
      <tr>
        <th>Модель</th>
        <td>Поток с назначением в модель: <code>РезультатПроверки ∈ {0,1}</code>.
        В среднее входят и случаи, где рекомендацию не исполнили.</td>
      </tr>
      <tr>
        <th>Инцидент</th>
        <td>Единица расчёта: после удаления дублей убытка и схлопывания убытков
        одного номера инцидента в одну строку.</td>
      </tr>
      <tr>
        <th>Отобранные (eligible)</th>
        <td>Инциденты, которые прошли фильтры и имеют Result ∈ {0,1,−100} —
        входят в сравнение ручеек vs модель.</td>
      </tr>
      <tr>
        <th>Yfact</th>
        <td>Уже выплаченное: сумма <code>СуммаПлатежа</code> по убыткам инцидента.</td>
      </tr>
      <tr>
        <th>Y365</th>
        <td>Yfact плюс оценка ещё не дозревшего хвоста ПСР (с дисконтом на год).</td>
      </tr>
      <tr>
        <th>ITT</th>
        <td>Эффект назначения: насколько ручеек в среднем дороже модели
        (со стратификацией по филиалам). Плюс = экономия.</td>
      </tr>
      <tr>
        <th>Окно пилота</th>
        <td>Фактический период наблюдений в текущей выгрузке (не полный год).</td>
      </tr>
      <tr>
        <th>Год на тех же филиалах</th>
        <td>Экстраполяция эффекта пилота на типичный год (сезонная поправка),
        филиалы те же.</td>
      </tr>
      <tr>
        <th>Шарим на все филиалы</th>
        <td>Сценарий: тот же эффект на инцидент переносим на nonpilot через
        множитель сети (объём × относительная тяжесть ПСР).</td>
      </tr>
      <tr>
        <th>Весь поток (не 50/50)</th>
        <td>Сценарий: все eligible идут в поток модели, а не только текущая доля
        ~половины. Это не факт пилота, а «если модель на весь поток».</td>
      </tr>
      <tr>
        <th>ПСР</th>
        <td>Постстраховые расходы / взыскания (претензии, ФУ, суд) — в хвосте Y365
        и в ретро-коэффициентах.</td>
      </tr>
    </table>
  </div>
"""


def _plan_calculation_steps() -> str:
    """Дерево формул: главные сверху, под ними — что нужно посчитать."""
    return f"""
  <h2>Формулы финансового эффекта</h2>
  <div class="card fx-tree">
    <p>Сверху — <b>главные</b> формулы финреза (то, что уходит в заключение).
    Под каждой — блок <b>«чтобы посчитать»</b>: второстепенные формулы,
    и так дальше по вложенности. <b>Ручеек</b> = Result=−100,
    <b>модель</b> = Result∈{{0,1}}.</p>

    <!-- ========== 1. Окно ========== -->
    <div class="fx-node main">
      <div class="fx-head">
        <span class="fx-tag main">главная</span>
        1. Факт на окне пилота (Y365)
      </div>
      <div class="formula">
        <div class="eq">Финрез₁ = ITT(365) × N<sub>obs</sub></div>
        <div class="note">N<sub>obs</sub> — число eligible-инцидентов в окне
        (ручеек + модель). 95% CI = [ci_low, ci_high] × N<sub>obs</sub>.</div>
      </div>
      <p class="fx-need">Чтобы посчитать Финрез₁:</p>
      <div class="fx-kids">
        {_fx_itt_branch()}
        <div class="fx-node sub">
          <div class="fx-head">
            <span class="fx-tag sub">второстепенная</span>
            N<sub>obs</sub> — размер окна
          </div>
          <div class="formula">
            <div class="eq">N<sub>obs</sub> = count(eligible в окне пилота)</div>
          </div>
          <p class="fx-need">Чтобы получить eligible:</p>
          <div class="fx-kids">
            {_fx_eligible_branch()}
          </div>
        </div>
      </div>
    </div>

    <!-- ========== 2. Год, те же филиалы ========== -->
    <div class="fx-node main">
      <div class="fx-head">
        <span class="fx-tag main">главная</span>
        2. Факт за год (те же филиалы пилота, доля модели как сейчас)
      </div>
      <div class="formula">
        <div class="eq">Финрез₂ = ITT(365) × N<sub>model,year,cur</sub></div>
        <div class="note">Экстраполяция окна на типичный год при текущей
        доле модели (~50/50). Не измеренный факт.</div>
      </div>
      <p class="fx-need">Чтобы посчитать Финрез₂:</p>
      <div class="fx-kids">
        {_fx_itt_branch(compact=True)}
        <div class="fx-node sub">
          <div class="fx-head">
            <span class="fx-tag sub">второстепенная</span>
            N<sub>model,year,cur</sub>
          </div>
          <div class="formula">
            <div class="eq">N<sub>model,year,cur</sub> =
            share<sub>model</sub> × N<sub>year</sub></div>
            <div class="eq">share<sub>model</sub> =
            N<sub>model,obs</sub> / N<sub>obs</sub></div>
          </div>
          <p class="fx-need">Чтобы посчитать N<sub>year</sub>:</p>
          <div class="fx-kids">
            {_fx_n_year_branch()}
          </div>
        </div>
      </div>
    </div>

    <!-- ========== 3. Все филиалы, доля как сейчас ========== -->
    <div class="fx-node main">
      <div class="fx-head">
        <span class="fx-tag main">главная</span>
        3. Шарим на все филиалы (год, доля модели как сейчас)
      </div>
      <div class="formula">
        <div class="eq">Финрез₃ = Финрез₂ × множитель<sub>сети</sub></div>
        <div class="note">Тот же эффект на инцидент переносим на nonpilot
        через объём и тяжесть ПСР.</div>
      </div>
      <p class="fx-need">Чтобы посчитать Финрез₃:</p>
      <div class="fx-kids">
        <div class="fx-node sub">
          <div class="fx-head">
            <span class="fx-tag sub">второстепенная</span>
            Финрез₂ — см. пункт 2 выше
          </div>
        </div>
        {_fx_network_multiplier_branch()}
      </div>
    </div>

    <!-- ========== 4. Все филиалы + весь поток ========== -->
    <div class="fx-node main">
      <div class="fx-head">
        <span class="fx-tag main">главная</span>
        4. Все филиалы + год + весь поток в модель (не 50/50)
      </div>
      <div class="formula">
        <div class="eq">Финрез₄ = ITT(365) × N<sub>year</sub> ×
        множитель<sub>сети</sub></div>
        <div class="note">= (ITT × N<sub>year</sub>) × множитель<sub>сети</sub>
        — сценарий «вся сеть, все eligible в модель».</div>
      </div>
      <p class="fx-need">Чтобы посчитать Финрез₄:</p>
      <div class="fx-kids">
        {_fx_itt_branch(compact=True)}
        {_fx_n_year_branch()}
        {_fx_network_multiplier_branch()}
      </div>
    </div>

    <div class="fx-node sub">
      <div class="fx-head">
        <span class="fx-tag sub">не финрез</span>
        Доплаты (описание, не входят в Финрез₁…₄)
      </div>
      <p class="muted" style="margin:0">Объём колонки «Сумма рекомендованная
      к доплате по модели» (Σ рекомендации). Это не ITT и не экономия.</p>
    </div>
  </div>
"""


def _fx_itt_branch(*, compact: bool = False) -> str:
    """Ветка ITT (+ вложенные Y / ретро / группы)."""
    if compact:
        return """
        <div class="fx-node sub">
          <div class="fx-head">
            <span class="fx-tag sub">второстепенная</span>
            ITT(365) — см. развёртку в пункте 1
          </div>
        </div>
"""
    return f"""
        <div class="fx-node sub">
          <div class="fx-head">
            <span class="fx-tag sub">второстепенная</span>
            ITT(H) — эффект на инцидент
          </div>
          <div class="formula">
            <div class="eq">ITT(H) = Σ<sub>f</sub> w<sub>f</sub> · effect<sub>f</sub>(H)</div>
            <div class="eq">effect<sub>f</sub>(H) =
            mean(Y<sub>H</sub>|ручеек,f) − mean(Y<sub>H</sub>|модель,f)</div>
            <div class="eq">w<sub>f</sub> = N<sub>f</sub> / Σ<sub>g</sub> N<sub>g</sub></div>
            <div class="note">H ∈ {{fact, 365}}. Плюс = ручеек дороже модели =
            экономия. 95% CI — квантили bootstrap по пересчёту ITT.</div>
          </div>
          <p class="fx-need">Чтобы посчитать ITT:</p>
          <div class="fx-kids">
            {_fx_y_branch()}
            <div class="fx-node sub">
              <div class="fx-head">
                <span class="fx-tag sub">второстепенная</span>
                Группы: ручеек и модель
              </div>
              <div class="formula">
                <div class="eq">ручеек ⇔ Result = −100</div>
                <div class="eq">модель ⇔ Result ∈ {{0, 1}}</div>
              </div>
              <div class="legend">
                <ul>
                  <li>Соглашение и исполнение рекомендации строки не выкидывают
                  (post-treatment). ITT = эффект назначения.</li>
                </ul>
              </div>
              <p class="fx-need">Чтобы получить Result на инциденте:</p>
              <div class="fx-kids">
                {_fx_eligible_branch()}
              </div>
            </div>
          </div>
        </div>
"""


def _fx_y_branch() -> str:
    """Yfact / Y365 и всё, что под ними."""
    return f"""
            <div class="fx-node sub">
              <div class="fx-head">
                <span class="fx-tag sub">второстепенная</span>
                Y<sub>H</sub> — расход на инцидент
              </div>
              <div class="formula">
                <div class="eq">Yfact<sub>i</sub> = Σ СуммаПлатежа по убыткам
                инцидента i</div>
                <div class="eq">Y365<sub>i</sub> = Yfact<sub>i</sub> +
                remaining<sub>i</sub> / (1+r)<sup>midpoint<sub>i</sub>/365</sup></div>
              </div>
              <p class="fx-need">Чтобы посчитать Y365:</p>
              <div class="fx-kids">
                <div class="fx-node sub">
                  <div class="fx-head">
                    <span class="fx-tag sub">второстепенная</span>
                    remaining — NPV-хвост ПСР
                  </div>
                  <div class="formula">
                    <div class="eq">remaining<sub>i</sub> =
                    max(q<sub>i</sub>×expected_open<sub>i</sub> −
                    observed_PSR<sub>i</sub>, 0)</div>
                    <div class="eq">expected_open<sub>i</sub> =
                    p<sub>U</sub> × (ultimate_base<sub>i</sub> + e<sub>U</sub>)</div>
                    <div class="eq">ultimate_base<sub>i</sub> =
                    OD<sub>i</sub>×k<sub>U</sub> если OD&gt;0, иначе m<sub>U</sub></div>
                    <div class="eq">q<sub>i</sub> = 0.07 при соглашении, иначе 1</div>
                    <div class="eq">age<sub>i</sub> = t_calc − t0<sub>i</sub> (дни)</div>
                    <div class="eq">midpoint<sub>i</sub> =
                    max(365 − age<sub>i</sub>, 0) / 2</div>
                  </div>
                  <div class="legend">
                    <ul>
                      <li><code>observed_PSR</code> — претензия+ФУ+суд на инциденте
                      (на первичных часто 0).</li>
                      <li><code>r</code> — ставка дисконта (обычно 12%); деление —
                      обычный NPV, не «обратная ставка».</li>
                    </ul>
                  </div>
                  <p class="fx-need">Чтобы посчитать p<sub>U</sub>, k<sub>U</sub>,
                  m<sub>U</sub>, e<sub>U</sub>:</p>
                  <div class="fx-kids">
                    {_fx_retro_branch()}
                  </div>
                </div>
              </div>
            </div>
"""


def _fx_retro_branch() -> str:
    return """
                    <div class="fx-node sub">
                      <div class="fx-head">
                        <span class="fx-tag sub">второстепенная</span>
                        Ретро-коэффициенты ПСР (пилот)
                      </div>
                      <div class="formula">
                        <div class="eq">p<sub>U</sub> = доля строк с ПСР &gt; 0</div>
                        <div class="eq">k<sub>U</sub> = Σ ПСР / Σ OD
                        на делах с ПСР и OD &gt; 0</div>
                        <div class="eq">m<sub>U</sub> = mean(ПСР | ПСР &gt; 0)</div>
                        <div class="eq">e<sub>U</sub> =
                        p(FU|PSR)×100 000 + p(court|PSR)×15 000</div>
                      </div>
                      <div class="legend">
                        <ul>
                          <li>Один пакет на весь пилот, не пофилиально.</li>
                          <li>Источник — зрелый ретро-кадр (не текущее окно пилота).</li>
                        </ul>
                      </div>
                    </div>
"""


def _fx_eligible_branch() -> str:
    return """
                <div class="fx-node sub">
                  <div class="fx-head">
                    <span class="fx-tag sub">второстепенная</span>
                    Eligible и схлоп на инцидент
                  </div>
                  <div class="formula">
                    <div class="eq">eligible = фильтры ∧ Result ∈ {0, 1, −100}</div>
                  </div>
                  <div class="legend">
                    <ul>
                      <li>Фильтры: пилот без Марийского/Архангельского;
                      форма денежная / ремонт / соглашение; первичный;
                      автотранспорт.</li>
                      <li>Дедуп: одна строка на номер убытка.</li>
                      <li>Схлоп: числа — сумма, текст — мода, даты — min;
                      Result — размазка (Null игнорируем; конфликт — мода,
                      ничья: −100 → 1 → 0).</li>
                      <li>Списанные по форме возмещения в схлоп не входят.</li>
                    </ul>
                  </div>
                </div>
"""


def _fx_n_year_branch() -> str:
    return """
            <div class="fx-node sub">
              <div class="fx-head">
                <span class="fx-tag sub">второстепенная</span>
                N<sub>year</sub> — годовой поток eligible пилота
              </div>
              <div class="formula">
                <div class="eq">N<sub>year</sub> =
                N<sub>obs</sub> / seasonal_exposure</div>
                <div class="eq">seasonal_exposure =
                Σ<sub>m</sub> coverage<sub>m</sub> × s<sub>m,P</sub></div>
              </div>
              <div class="legend">
                <ul>
                  <li><code>s<sub>m,P</sub></code> — средняя за годы доля
                  месяца m в годовом потоке пилота (ретро).</li>
                  <li><code>coverage<sub>m</sub></code> — доля месяца m,
                  покрытая окном наблюдений.</li>
                </ul>
              </div>
            </div>
"""


def _fx_network_multiplier_branch() -> str:
    return """
        <div class="fx-node sub">
          <div class="fx-head">
            <span class="fx-tag sub">второстепенная</span>
            множитель<sub>сети</sub>
          </div>
          <div class="formula">
            <div class="eq">множитель<sub>сети</sub> =
            1 + (объём<sub>NP</sub>/объём<sub>P</sub>) ×
            (mean ПСР<sub>NP</sub> / mean ПСР<sub>P</sub>)</div>
          </div>
          <div class="legend">
            <ul>
              <li>NP = nonpilot, P = pilot (ретро, не окно пилота).</li>
              <li>Учитывает и объём потока, и относительную тяжесть ПСР.</li>
            </ul>
          </div>
        </div>
"""


def _contract_table(result: MonitoringEffectResult) -> pd.DataFrame:
    meaning = {
        "unit": "Единица ITT — инцидент после дедупа убытков и схлопывания",
        "loss": "Список LossID инцидента через «; » (не единица ITT)",
        "incident": "Единица анализа (= ключ группы; не сумма и не мода)",
        "result": (
            "model={0,1}; control=−100; Null на убытке размазывается "
            "с непустого Result инцидента"
        ),
        "filial": "Страта рандомизации и ITT-взвешивания",
        "payment": "paid_to_date; сумма СуммаПлатежа по убыткам инцидента",
        "to_pay_diagnostic_only": "Только сверка качества; в Y не прибавляется",
        "od": "OD для k_U×OD; сумма по убыткам; при пропуске используется m_U",
        "recommended_extra": "Доплата в сценарии 100% compliance (сумма)",
        "payout_by_model": "Признак исполнения рекомендации (any/sum>0)",
        "agreement": "При соглашении остаток ПСР равен 7%",
        "t0_primary": "Основная дата старта возраста (min по убыткам)",
        "t0_fallback": "Fallback для t0",
        "psr_pretension": "Наблюдаемый ПСР; сумма; вычитается из хвоста",
        "psr_fu": "Наблюдаемый ПСР; сумма; вычитается из хвоста",
        "psr_court": "Наблюдаемый ПСР; сумма; вычитается из хвоста",
        "n_rows_after_loss_dedupe_removed": "Сколько лишних строк убытка убрано",
        "n_writeoff_excluded": "Списанные по форме возмещения (не схлопывались)",
        "n_multi_loss_incidents": "Инцидентов с >1 убытком до схлопывания",
        "n_incidents_after_collapse": "Число строк после схлопывания на инцидент",
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


def _path_share_columns(*, include_filial: bool = False) -> dict[str, Any]:
    """Легенда долей путей. ``filial`` только для таблицы по филиалам."""
    cols: dict[str, Any] = {
        "segment": _c("segment", "Сегмент строки: control, model или lift"),
        "n": _c("n", "Число строк сегмента в ITT-выборке", "count(rows|segment)"),
        "agreement_share": _c(
            "agreement %",
            "Доля строк с соглашением, %",
            "100 × mean(is_agreement)",
        ),
        "pretension_share": _c(
            "pretension %",
            "Доля строк с претензией, %",
            "100 × mean(is_pretension)",
        ),
        "fu_incident_share": _c(
            "FU %",
            "Доля строк с флагом ФУ на уровне инцидента, %",
            "100 × mean(fu_incident)",
        ),
        "court_incident_share": _c(
            "court %",
            "Доля строк с флагом суда на уровне инцидента, %",
            "100 × mean(court_incident)",
        ),
    }
    if include_filial:
        out: dict[str, Any] = {
            "segment": cols["segment"],
            "filial": _c("filial", "Филиал"),
        }
        out.update({k: v for k, v in cols.items() if k != "segment"})
        return out
    return cols


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
        "<span>Год, все филиалы, весь поток, Y365</span>"
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
  {_plan_glossary()}
  {_business_schema()}

  {_plan_calculation_steps()}

  <h2>Что фиксируем до расчёта</h2>
  <div class="card">
    <ul>
      <li>Сравниваем <b>model</b> и <b>control</b> внутри филиала
      (примерно 50/50). Марийский и Архангельский в ITT не входят.</li>
      <li>Единица — один <b>инцидент</b> (после дедупа убытков и схлопывания;
      списанные по форме возмещения не схлопываем).</li>
      <li><b>Yfact</b> — сумма <code>СуммаПлатежа</code> по убыткам инцидента
      (не претензия+ФУ+суд как отдельный lifecycle).</li>
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
    <p><b>Единица:</b> инцидент после дедупликации по номеру убытка и
    схлопывания убытков (числа — сумма, текст — мода). Списанные по форме
    возмещения в схлопывание не входят. Если у убытков инцидента Result=Null,
    а у одного −100 (или 0/1) — на весь инцидент берём это значение.</p>
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
    <p class="muted">Строк (инцидентов) после схлопывания:
    {quality.get("n_rows", "—")};
    пропусков OD: {quality.get("missing_od", "—")};
    среднее убытков на инцидент:
    {quality.get("mean_losses_per_incident", "—")}.</p>
  </div>

  <h2>4. Доли соглашений / претензий / ФУ / суда</h2>
  <div class="card">
    <p>Доли считаются на той же ITT-выборке. Сначала control, затем model.
    ФУ и суд подняты на уровень инцидента: если хотя бы один убыток инцидента
    имел ФУ/суд, флаг ставится всем строкам инцидента.</p>
    <h3>Control vs model</h3>
    {_table(
      result.path_shares,
      columns=_path_share_columns(include_filial=False),
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
      columns=_path_share_columns(include_filial=True),
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
        "horizon": _c("H", "Горизонт: fact (=Yfact) или 365 (=Y365)"),
        "group": _c(
            "group",
            "control (Result=−100) или model (Result∈{{0,1}})",
        ),
        "n": _c("n", "Число инцидентов в группе"),
        "sum_cost": _c(
            "Σ YH",
            "Сумма исхода YH по группе, ₽",
            "Σ_i∈group Y_H,i",
        ),
        "mean_cost": _c(
            "mean YH",
            "Средний исход на инцидент, ₽",
            "sum_cost / n",
        ),
      },
      rows=[
        "Для каждого горизонта сначала control, затем model.",
        "Единица строки после дедупа/схлопывания — инцидент.",
      ],
      notes=[
        "Yfact = Σ СуммаПлатежа по убыткам инцидента; "
        "Y365 = Yfact + NPV(remaining ПСР).",
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
        "horizon": _c("H", "Горизонт: fact / 365"),
        "effect_per_case": _c(
            "ITT страт",
            "Стратифицированный ITT, ₽/инцидент",
            "Σ_f w_f · (mean_c,f − mean_m,f), w_f = N_f/Σ N_g",
        ),
        "unstratified_effect": _c(
            "ITT без страт",
            "Разница средних без весов филиалов, ₽",
            "mean(YH|control) − mean(YH|model)",
        ),
        "n_weighted": _c(
            "N",
            "Число eligible-инцидентов (сумма весов филиалов)",
            "Σ_f N_f",
        ),
        "n_filials": _c("филиалы", "Число филиалов с парой control и model"),
        "ci_low": _c("CI low", "2.5% квантиль bootstrap ITT, ₽"),
        "ci_high": _c("CI high", "97.5% квантиль bootstrap ITT, ₽"),
        "n_bootstrap": _c("n boot", "Число успешных bootstrap-повторений"),
      },
      rows=[
        "Одна строка на горизонт fact / 365.",
      ],
      notes=[
        "Плюс = control дороже model = экономия модели (ITT).",
      ],
    )}
    <h3>Вклад филиалов</h3>
    {_table(
      result.filial_effects,
      columns={
        "horizon": _c("H", "Горизонт"),
        "filial": _c("filial", "Филиал"),
        "n": _c("n", "Инциденты филиала (control+model)"),
        "n_control": _c("n_c", "Число инцидентов control"),
        "n_model": _c("n_m", "Число инцидентов model"),
        "mean_control": _c(
            "mean_c",
            "Средний YH в control, ₽",
            "mean(YH | control, filial)",
        ),
        "mean_model": _c(
            "mean_m",
            "Средний YH в model, ₽",
            "mean(YH | model, filial)",
        ),
        "effect_control_minus_model": _c(
            "effect_f",
            "Локальный ITT филиала, ₽",
            "mean_control − mean_model",
        ),
      },
      rows=[
        "Одна строка на пару горизонт × филиал.",
        "Филиал без control или без model в стратифицированный ITT не входит.",
      ],
      notes=[
        "Вклад филиала в ITT страт: contribution_f = w_f × effect_f "
        "(колонки contribution в этой таблице нет).",
      ],
    )}
    <h3>Филиалы на внимании</h3>
    <p>Показываем филиалы, где доля соглашений в control выше, чем в model,
    и/или ITT-эффект control−model отрицательный (model дороже).</p>
    {_table(
      result.attention_filials,
      columns={
        "filial": _c("filial", "Филиал"),
        "agreement_share_control": _c(
            "agr_c %",
            "Доля соглашений в control, %",
        ),
        "agreement_share_model": _c(
            "agr_m %",
            "Доля соглашений в model, %",
        ),
        "agreement_gap_pp": _c(
            "gap п.п.",
            "Разрыв долей соглашений, п.п.",
            "agreement_share_control − agreement_share_model",
        ),
        "negative_effect_horizons": _c(
            "H<0",
            "Горизонты с effect_f < 0",
        ),
        "min_effect_control_minus_model": _c(
            "min effect",
            "Минимальный effect_f по горизонтам, ₽",
            "min_H effect_f",
        ),
        "flags": _c(
            "flags",
            "agreement_control_gt_model и/или negative_itt",
            "gap_pp&gt;0 → agreement_control_gt_model; "
            "effect_f&lt;0 → negative_itt",
        ),
      },
      rows=[
        "Пустая таблица означает, что таких филиалов нет.",
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

  <h2>8. Рекомендованные доплаты и факт оплаты</h2>
  <div class="card">
    <p>Объём колонки <code>Сумма рекомендованная к доплате по модели</code>
    и факт оплаты: <b>исполнено = СуммаПлатежа &gt; 0</b>.
    Это описание объёмов, не ITT и не оценка экономии от доплат.</p>
    {_table(
      result.recommended_extra_summary,
      columns={
        "segment": _c(
            "segment",
            "control / model / model_result_1 (риск) / model_result_0",
        ),
        "n": _c("n", "Число инцидентов в сегменте"),
        "n_recommended_gt0": _c(
            "n рек.>0",
            "Инциденты с рекомендованной доплатой &gt; 0",
        ),
        "sum_recommended_extra": _c(
            "Σ рек.",
            "Сумма рекомендованных доплат, ₽",
            "Σ Сумма рекомендованная к доплате по модели",
        ),
        "n_paid_gt0": _c(
            "n paid>0",
            "Инциденты с СуммаПлатежа &gt; 0 (факт оплаты)",
        ),
        "share_paid_gt0": _c(
            "paid %",
            "Доля инцидентов с СуммаПлатежа &gt; 0, %",
            "100 × mean(СуммаПлатежа &gt; 0)",
        ),
        "n_recommended_and_paid": _c(
            "n рек.∩paid",
            "Рекомендация &gt; 0 и СуммаПлатежа &gt; 0",
        ),
        "sum_recommended_paid": _c(
            "Σ рек. paid",
            "Σ рекомендации на инцидентах с СуммаПлатежа &gt; 0, ₽",
        ),
        "sum_recommended_unpaid": _c(
            "Σ рек. unpaid",
            "Σ рекомендации на инцидентах с СуммаПлатежа = 0, ₽",
        ),
        "descriptive_only": _c(
            "desc only",
            "true: только описание, не causal effect",
        ),
      },
      rows=[
        "control — Result=−100; model — Result∈{0,1}.",
        "model_result_1 — model с Result=1 (флаг риска / рекомендация).",
        "model_result_0 — model с Result=0.",
        "Исполнение здесь = СуммаПлатежа &gt; 0, не флаг «Выплата по модели».",
      ],
      notes=[
        "У control сумма рекомендации обычно ≈ 0 (модели не было).",
        "Не интерпретировать Σ рек. как экономию ITT.",
      ],
    )}
  </div>

  <h2>9. Non-compliance</h2>
  <div class="card">
    <h3>A. As-complied — только описательная диагностика (Yfact)</h3>
    <p>Сравнение внутри model и <code>РезультатПроверки=1</code> только по факту.
    Не является причинным эффектом; горизонт 365 здесь не считается.</p>
    {_table(
      result.compliance_a,
      columns={
        "horizon": _c("H", "Только fact"),
        "compliance": _c("compliance", "complied или not_complied"),
        "n": _c("n", "Число инцидентов"),
        "agreement_share": _c(
            "agr %",
            "Доля соглашений внутри группы, %",
        ),
        "mean_cost": _c(
            "mean Yfact",
            "Средний Yfact, ₽/инцидент",
            "mean(Yfact | compliance)",
        ),
        "descriptive_only": _c(
            "desc only",
            "true: только описание, не causal effect",
        ),
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
        "horizon": _c("H", "Горизонт сценария 100% compliance"),
        "effect_per_case": _c(
            "ITT_100",
            "Сценарный ITT: control actual − model scenario, ₽/инцидент",
            "Σ_f w_f · (mean_c actual − mean_m scenario)",
        ),
        "unstratified_effect": _c(
            "ITT_100 без страт",
            "Та же разность без стратификации, ₽/инцидент",
            "mean(control actual) − mean(model scenario)",
        ),
        "n_weighted": _c("N", "Сумма весов филиалов (eligible-инциденты)"),
        "n_filials": _c("филиалы", "Число филиалов в расчёте"),
        "ci_low": _c("CI low", "2.5% квантиль bootstrap сценария, ₽"),
        "ci_high": _c("CI high", "97.5% квантиль bootstrap сценария, ₽"),
        "n_bootstrap": _c("n boot", "Число успешных bootstrap-повторений"),
        "scenario": _c("scenario", "Описание сценария"),
      },
      rows=[
        "Одна строка на горизонт сценария 100% исполнения.",
      ],
    )}
  </div>

  <h2>10. Чувствительность</h2>
  <div class="card">
    <p>Сетка: r ∈ {{8%,12%,16%}} и остаток после соглашения q ∈ {{0%,7%,15%}}.</p>
    {_table(
      result.sensitivity,
      columns={
        "discount_rate": _c("r", "Ставка дисконтирования"),
        "residual_share": _c("q", "Остаток ПСР после соглашения"),
        "horizon": _c("H", "Горизонт"),
        "effect_per_case": _c(
            "ITT",
            "ITT на инцидент при этих (r, q), ₽",
            "stratified ITT(YH | r, q)",
        ),
      },
      rows=[
        "Каждая строка — одна комбинация (r, q, горизонт).",
      ],
    )}
  </div>

  <h2>11. Сезонный годовой эффект и сеть</h2>
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
        "month": _c("месяц", "Календарный месяц 1–12"),
        "retro_share": _c(
            "s_m",
            "Средняя доля месяца в годовом pilot-потоке ретро",
            "доля строк месяца m в ретро-пилоте",
        ),
        "observed_coverage": _c(
            "coverage",
            "Покрытие месяца текущим окном (0–1+)",
            "доля дней месяца m в окне мониторинга",
        ),
        "exposure_contribution": _c(
            "вклад",
            "Вклад месяца в seasonal_exposure",
            "coverage_m × s_m",
        ),
      },
      rows=[
        "Одна строка на календарный месяц.",
      ],
    )}
    <h3>Сценарии годового эффекта</h3>
    {_table(
      result.annual_summary,
      columns={
        "horizon": _c("H", "Горизонт"),
        "effect_per_case": _c("ITT", "ITT на инцидент, ₽"),
        "effect_100_compliance": _c(
            "ITT_100",
            "Сценарный ITT 100% compliance на инцидент, ₽",
        ),
        "seasonal_exposure": _c(
            "exposure",
            "Доля типичного года, уже увиденная в окне",
            "Σ_m coverage_m × s_m,P",
        ),
        "N_pilot_eligible_year": _c(
            "N_year",
            "Ожидаемый годовой поток eligible пилота",
            "N_obs / seasonal_exposure",
        ),
        "N_pilot_model_year_current": _c(
            "N_model cur",
            "Годовой поток model при текущей доле model",
        ),
        "N_pilot_model_year_full": _c(
            "N_model весь",
            "Годовой поток model, если весь eligible идёт в модель",
            "= N_pilot_eligible_year",
        ),
        "volume_ratio_nonpilot": _c(
            "vol_NP",
            "Годовой поток nonpilot / pilot (ретро)",
        ),
        "risk_ratio_nonpilot": _c(
            "risk_NP",
            "Средний ПСР nonpilot / pilot (ретро)",
        ),
        "network_multiplier": _c(
            "net mult",
            "Множитель сети",
            "1 + volume_ratio_nonpilot × risk_ratio_nonpilot",
        ),
        "annual_pilot_current": _c(
            "год пилот cur",
            "Годовой эффект пилота при текущей доле model, ₽/год",
            "effect_per_case × N_pilot_model_year_current",
        ),
        "annual_pilot_full": _c(
            "год пилот весь",
            "Годовой эффект пилота при всём потоке в модель, ₽/год",
            "effect_per_case × N_pilot_eligible_year",
        ),
        "annual_pilot_full_ci_low": _c(
            "год пилот CI↓",
            "Нижняя граница CI годового эффекта пилота, ₽/год",
            "ci_low × N_pilot_eligible_year",
        ),
        "annual_pilot_full_ci_high": _c(
            "год пилот CI↑",
            "Верхняя граница CI годового эффекта пилота, ₽/год",
            "ci_high × N_pilot_eligible_year",
        ),
        "effect_per_case_nonpilot": _c(
            "ITT_NP",
            "Эффект на инцидент × risk_ratio, ₽",
            "effect_per_case × risk_ratio_nonpilot",
        ),
        "annual_network_full": _c(
            "год сеть",
            "Годовой эффект сети при всём потоке в модель, ₽/год",
            "annual_pilot_full × network_multiplier",
        ),
        "annual_network_full_ci_low": _c(
            "год сеть CI↓",
            "Нижняя граница CI годового эффекта сети, ₽/год",
            "ci_low × N_pilot_eligible_year × network_multiplier",
        ),
        "annual_network_full_ci_high": _c(
            "год сеть CI↑",
            "Верхняя граница CI годового эффекта сети, ₽/год",
            "ci_high × N_pilot_eligible_year × network_multiplier",
        ),
        "annual_network_full_compliance": _c(
            "год сеть 100%",
            "Годовой эффект сети при 100% compliance, ₽/год",
            "effect_100_compliance × N_pilot_eligible_year × network_multiplier",
        ),
      },
      rows=[
        "Одна строка на горизонт fact / 365.",
      ],
      notes=[
        "Весь поток в модель = все eligible идут в model-поток; это сценарий, не факт.",
      ],
    )}
  </div>

  <h2>12. Ограничения</h2>
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


def _fmt_million(value: Any) -> str:
    """Сумма в млн ₽ для бизнес-заключения."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    mln = float(value) / 1_000_000.0
    return f"{mln:,.2f}".replace(",", " ")


def _path_pct(paths: pd.DataFrame | None, segment: str, column: str) -> str:
    if paths is None or segment not in paths.index or column not in paths.columns:
        return "—"
    value = paths.loc[segment, column]
    if pd.isna(value):
        return "—"
    return f"{float(value):.1f}%"


def _extra_row(summary: pd.DataFrame | None, segment: str) -> dict[str, Any]:
    empty = {
        "n": 0,
        "sum_recommended_extra": 0.0,
        "n_recommended_gt0": 0,
        "n_paid_gt0": 0,
        "sum_recommended_paid": 0.0,
        "sum_recommended_unpaid": 0.0,
    }
    if summary is None or summary.empty or "segment" not in summary.columns:
        return empty
    part = summary.loc[summary["segment"].eq(segment)]
    if part.empty:
        return empty
    return part.iloc[0].to_dict()


def build_conclusion_body_from_result(
    result: MonitoringEffectResult,
    *,
    development_lags: pd.DataFrame | None = None,
) -> str:
    """Финальное бизнес-заключение: ручеек vs модель, доплаты, пути, год."""
    _ = development_lags  # лаги остаются в полном report; conclusion — кратко
    effects = result.effect_summary.set_index("horizon")
    annual = result.annual_summary.set_index("horizon")
    quality = result.data_quality.set_index("metric")["value"].to_dict()
    paths = (
        result.path_shares.set_index("segment")
        if not result.path_shares.empty
        else None
    )
    extras = getattr(result, "recommended_extra_summary", None)

    def _eff(horizon: str) -> tuple[float, float, float]:
        return (
            float(effects.loc[horizon, "effect_per_case"]),
            float(effects.loc[horizon, "ci_low"]),
            float(effects.loc[horizon, "ci_high"]),
        )

    e_f, lo_f, hi_f = _eff("fact")
    e_y, lo_y, hi_y = _eff("365")
    ann_pilot_cur = float(annual.loc["365", "annual_pilot_current"])
    ann_pilot_full = float(annual.loc["365", "annual_pilot_full"])
    ann_net = float(annual.loc["365", "annual_network_full"])
    ann_net_lo = float(annual.loc["365", "annual_network_full_ci_low"])
    ann_net_hi = float(annual.loc["365", "annual_network_full_ci_high"])
    mult = float(annual.loc["365", "network_multiplier"])
    n_year = float(annual.loc["365", "N_pilot_eligible_year"])
    n_model_year_cur = float(annual.loc["365", "N_pilot_model_year_current"])

    n_rows = int(quality.get("n_rows", len(result.frame)))
    n_c = int(quality.get("n_control", 0))
    n_m = int(quality.get("n_model", 0))
    n_fil = int(quality.get("n_filials", 0))
    obs_start = quality.get("observation_start", "?")
    obs_end = quality.get("observation_end", "?")

    # 1) факт на окне пилота
    save_window_y = e_y * n_rows
    save_window_lo = lo_y * n_rows
    save_window_hi = hi_y * n_rows
    # 2) год, те же филиалы, текущая доля модели
    save_pilot_year = ann_pilot_cur
    save_pilot_year_lo = lo_y * n_model_year_cur
    save_pilot_year_hi = hi_y * n_model_year_cur
    # 3) все филиалы, год, доля модели как сейчас (~50/50)
    save_all_cur = ann_pilot_cur * mult
    save_all_cur_lo = save_pilot_year_lo * mult
    save_all_cur_hi = save_pilot_year_hi * mult
    # 4) все филиалы, год, весь поток в модель (не 50/50)
    save_all_full = ann_net
    # ann_pilot_full = ITT × N_year (весь eligible пилота) — база для пункта 4 до × сети
    _ = ann_pilot_full

    ctrl_x = _extra_row(extras, "control")
    model_x = _extra_row(extras, "model")
    model1_x = _extra_row(extras, "model_result_1")

    agr_c = _path_pct(paths, "control", "agreement_share")
    agr_m = _path_pct(paths, "model", "agreement_share")
    pret_c = _path_pct(paths, "control", "pretension_share")
    pret_m = _path_pct(paths, "model", "pretension_share")
    lift_agr = _path_pct(paths, "lift_pp (model - control)", "agreement_share")
    lift_pret = _path_pct(paths, "lift_pp (model - control)", "pretension_share")
    if paths is not None:
        lift_idx = [
            i for i in paths.index.astype(str) if "lift_pp" in i and "model" in i
        ]
        if lift_idx:
            lift_agr = _path_pct(paths, lift_idx[0], "agreement_share")
            lift_pret = _path_pct(paths, lift_idx[0], "pretension_share")

    confirmed = lo_f > 0 and lo_y > 0
    ci_note = (
        "оба 95% CI выше 0 — направление подтверждено"
        if confirmed
        else "95% CI проходит через 0 — точечная экономия есть, статистически не подтверждена"
    )

    return f"""
  <div class="card">
    <p class="sub" style="margin:0">Для филиалов пилота, за период пилота
    (<b>{escape(str(obs_start))}</b> … <b>{escape(str(obs_end))}</b>;
    {n_fil} филиалов; {n_rows} инцидентов:
    ручеек {n_c} / модель {n_m})</p>
  </div>

  <div class="card">
    <h3>Цель</h3>
    <p>Снижение ПСР-расходов относительно <b>ручейка</b>;
    увеличение доли соглашений относительно ручейка;
    снижение конверсии в претензии относительно ручейка.</p>
  </div>

  <div class="card">
    <h3>Результаты</h3>

    <h4>Сколько доплатили</h4>
    <p class="muted">Колонка «Сумма рекомендованная к доплате по модели».
    Отдельно — сумма рекомендаций при Result=1 (красная рекомендация).</p>
    <table class="kv">
      <tr>
        <th>Ручеек</th>
        <td>Σ рекомендации: <b>{format_money(ctrl_x.get("sum_recommended_extra", 0))}</b> ₽</td>
      </tr>
      <tr>
        <th>Модель</th>
        <td>Σ рекомендации: <b>{format_money(model_x.get("sum_recommended_extra", 0))}</b> ₽</td>
      </tr>
      <tr>
        <th>Из них Result=1</th>
        <td>Σ рекомендации: <b>{format_money(model1_x.get("sum_recommended_extra", 0))}</b> ₽</td>
      </tr>
    </table>

    <h4>Сколько потенциально экономим (ручеек − модель)</h4>
    <p class="muted">ITT на инцидент = mean(ручеек) − mean(модель), со стратификацией.
    Плюс = экономия. {ci_note}.</p>
    <table class="kv">
      <tr>
        <th>ITT Yfact / инцидент</th>
        <td><b>{format_money(e_f)}</b> ₽
        <small>(95% CI: {format_money(lo_f)} … {format_money(hi_f)})</small></td>
      </tr>
      <tr>
        <th>ITT Y365 / инцидент</th>
        <td><b>{format_money(e_y)}</b> ₽
        <small>(95% CI: {format_money(lo_y)} … {format_money(hi_y)})</small></td>
      </tr>
    </table>

    <h4>Доля соглашений по потокам</h4>
    <table class="kv">
      <tr><th>Ручеек</th><td>{agr_c}</td></tr>
      <tr><th>Модель</th><td>{agr_m}</td></tr>
      <tr><th>Δ модель − ручеек</th><td>{lift_agr} п.п.</td></tr>
    </table>

    <h4>Конверсия в претензии по потокам</h4>
    <table class="kv">
      <tr><th>Ручеек</th><td>{pret_c}</td></tr>
      <tr><th>Модель</th><td>{pret_m}</td></tr>
      <tr><th>Δ модель − ручеек</th><td>{lift_pret} п.п.</td></tr>
    </table>
  </div>

  <div class="card">
    <h3>Финрез (Y365)</h3>
    <p class="muted">Четыре уровня масштаба. Сценарии 2–4 — экстраполяция, не измеренный факт.</p>
    <table class="kv">
      <tr>
        <th>1. Факт (пилот, окно)</th>
        <td><span class="million">{_fmt_million(save_window_y)} млн ₽</span>
        <small>= ITT × {n_rows} инцидентов окна;
        95% CI: {_fmt_million(save_window_lo)} … {_fmt_million(save_window_hi)} млн</small></td>
      </tr>
      <tr>
        <th>2. Факт за год (те же филиалы пилота)</th>
        <td><span class="million">{_fmt_million(save_pilot_year)} млн ₽</span>
        <small>= ITT × годовой поток model при <b>текущей</b> доле модели
        (~{100.0 * n_m / n_rows if n_rows else 0:.0f}% сейчас);
        N_year eligible ≈ {n_year:,.0f}; 95% CI:
        {_fmt_million(save_pilot_year_lo)} … {_fmt_million(save_pilot_year_hi)} млн</small></td>
      </tr>
      <tr>
        <th>3. Шарим на все филиалы (год, доля как сейчас)</th>
        <td><span class="million">{_fmt_million(save_all_cur)} млн ₽</span>
        <small>= пункт 2 × множитель сети {mult:.2f}
        (объём и тяжесть ПСР nonpilot / pilot);
        95% CI: {_fmt_million(save_all_cur_lo)} … {_fmt_million(save_all_cur_hi)} млн</small></td>
      </tr>
      <tr>
        <th>4. Все филиалы + год + весь поток (не 50/50)</th>
        <td><span class="million">{_fmt_million(save_all_full)} млн ₽</span>
        <small>= ITT × весь годовой eligible пилота
        (~{n_year:,.0f}) × множитель сети {mult:.2f}
        (все идут в поток модели); 95% CI:
        {_fmt_million(ann_net_lo)} … {_fmt_million(ann_net_hi)} млн</small></td>
      </tr>
    </table>
  </div>

  <div class="card">
    <h3>Как читать</h3>
    <ul>
      <li><b>Ручеек</b> = Result=−100; <b>модель</b> = Result∈{{0,1}}.</li>
      <li><b>Yfact</b> — уже выплаченное; <b>Y365</b> — плюс NPV хвоста ПСР.</li>
      <li>Пункты финреза 2–4 — сценарии; обещать как факт нельзя, пока CI через 0.</li>
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
  <title>Заключение: ручеек vs модель</title>
  <style>{_CSS}</style>
</head>
<body>
<div class="page">
  <h1>Заключение: ручеек vs модель</h1>
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
    """Записать три HTML в ``data_dir/<stamp>/`` (прошлые прогоны не затираются).

    Общие файлы (например лог снимков) остаются в ``data_dir``.
    Возвращает (plan, report, conclusion, names).
    """
    export_stamp = stamp or report_export_stamp()
    run_dir = export_run_dir(data_dir, export_stamp)
    names = dated_artifact_names(export_stamp)
    plan = write_plan_html(
        run_dir / names["plan"],
        source_label=source_label,
        plan_name=names["plan"],
        report_name=names["report"],
        conclusion_name=names["conclusion"],
    )
    report = write_monitoring_html(
        result,
        run_dir / names["report"],
        source_label=source_label,
        plan_name=names["plan"],
        report_name=names["report"],
        conclusion_name=names["conclusion"],
    )
    conclusion = write_conclusion_html(
        run_dir / names["conclusion"],
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
    "export_run_dir",
    "report_export_stamp",
    "write_all_monitoring_htmls",
    "write_conclusion_html",
    "write_error_html",
    "write_monitoring_html",
    "write_plan_html",
]
