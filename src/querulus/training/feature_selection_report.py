"""HTML-отчёт по отобранным фичам (для бизнеса, после feature selection)."""
from __future__ import annotations

import base64
import html
import io
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from querulus.training.feature_labels import feature_ru_name
from querulus.training.feature_selection_io import DEFAULT_FEATURE_SELECTION_DIR

_EXPOSURE = "expos"


def _ensure_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить колонку экспозиции для EDA."""
    result = df.copy()
    if _EXPOSURE not in result.columns:
        result[_EXPOSURE] = 1
    return result


def _qcut_feature(series: pd.Series, quantiles: int) -> pd.Series:
    """Безопасный qcut (как в research_eda)."""
    values = pd.to_numeric(series, errors="coerce")
    arr = values.to_numpy(dtype=float, copy=False)
    finite = np.isfinite(arr)
    if finite.sum() < 2 or pd.Series(arr[finite]).nunique() < 2:
        raise ValueError("insufficient unique finite values for binning")
    n_bins = max(2, min(int(quantiles), int(pd.Series(arr[finite]).nunique())))
    try:
        return pd.qcut(values, n_bins, duplicates="drop")
    except (ValueError, TypeError):
        return pd.cut(values, bins=n_bins, duplicates="drop")


def _group_for_plot(
    df: pd.DataFrame,
    feature: str,
    *,
    model_type: str,
    is_categorical: bool,
    frequency_target: str,
    severity_target: str,
    numeric_bins: int,
) -> pd.DataFrame:
    """Агрегация экспозиции и ratio (frequency / severity)."""
    data = df.copy()
    if not is_categorical:
        data[feature] = _qcut_feature(data[feature], numeric_bins)

    if model_type == "frequency":
        cols = [feature, _EXPOSURE, frequency_target]
        grouped = data[cols].dropna(subset=[feature]).groupby(feature, dropna=False, observed=True).sum()
        grouped["ratio"] = grouped[frequency_target] / grouped[_EXPOSURE]
    elif model_type == "severity":
        cols = [feature, _EXPOSURE, frequency_target, severity_target]
        grouped = data[cols].dropna(subset=[feature]).groupby(feature, dropna=False, observed=True).sum()
        grouped["ratio"] = grouped[severity_target] / grouped[frequency_target]
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    if is_categorical:
        grouped = grouped.sort_values(by="ratio", ascending=False)
    return grouped


def _plot_to_base64(
    grouped: pd.DataFrame,
    feature: str,
    model_type: str,
    *,
    figsize: tuple[float, float] = (12, 4),
    rotation: int = 45,
) -> str:
    """Нарисовать EDA-график и вернуть PNG как base64."""
    n = grouped.shape[0]
    ind = np.arange(n)
    fig, ax = plt.subplots(dpi=110, figsize=figsize)
    ax.bar(ind, grouped[_EXPOSURE], color="#4C78A8", alpha=0.85)
    ax.set_ylabel("Экспозиция", fontsize=11)
    labels = [str(x) for x in grouped.index.tolist()]
    ax.set_xticks(ind)
    ax.set_xticklabels(labels, fontsize=9, rotation=rotation, ha="right")
    ax.tick_params(axis="y", labelsize=9)

    ax2 = ax.twinx()
    ax2.plot(ind, grouped["ratio"], color="#E45756", marker="o", linewidth=2)
    ax2.set_ylabel("Частота" if model_type == "frequency" else "Severity", fontsize=11)
    ax2.tick_params(axis="y", labelsize=9)

    title_suffix = "FREQUENCY" if model_type == "frequency" else "SEVERITY"
    ax.set_title(f"{feature} — {title_suffix}", fontsize=13)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _render_feature_plot(
    df: pd.DataFrame,
    feature: str,
    *,
    model_type: str,
    is_categorical: bool,
    frequency_target: str,
    severity_target: str,
    numeric_bins: int,
) -> str | None:
    """Собрать base64 PNG или None при ошибке."""
    if feature not in df.columns:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grouped = _group_for_plot(
                df,
                feature,
                model_type=model_type,
                is_categorical=is_categorical,
                frequency_target=frequency_target,
                severity_target=severity_target,
                numeric_bins=numeric_bins,
            )
        if grouped.empty:
            return None
        return _plot_to_base64(grouped, feature, model_type)
    except (ValueError, TypeError, ZeroDivisionError, KeyError) as exc:
        print(f"[FS-report] skip plot {feature!r}/{model_type}: {exc}")
        return None


def _badge(text: str, kind: str) -> str:
    return f'<span class="badge badge-{html.escape(kind)}">{html.escape(text)}</span>'


def _build_html(
    *,
    stack: str,
    frequency_features: list[str],
    severity_features: list[str],
    severity_only: list[str],
    frequency_only: list[str],
    shared: list[str],
    rows_html: str,
    cards_html: str,
    saved_at: str,
) -> str:
    """Собрать HTML-презентацию."""
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Отобранные признаки — {html.escape(stack)}</title>
<style>
  :root {{
    --bg: #f7f5f1;
    --card: #ffffff;
    --ink: #1c1c1c;
    --muted: #5c5c5c;
    --line: #ddd6c8;
    --freq: #2f6f4e;
    --sev: #8a4b1f;
    --only: #7a1f2b;
    --shared: #1f4b7a;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0;
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    background: linear-gradient(180deg, #efe9df 0%, var(--bg) 220px);
    color: var(--ink); line-height: 1.45;
  }}
  header {{
    max-width: 1100px; margin: 0 auto; padding: 40px 24px 16px;
  }}
  header h1 {{ margin: 0 0 8px; font-size: 1.9rem; letter-spacing: -0.02em; }}
  header p {{ margin: 4px 0; color: var(--muted); }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 8px 24px 48px; }}
  section {{
    background: var(--card); border: 1px solid var(--line);
    border-radius: 10px; padding: 20px 22px; margin: 18px 0;
  }}
  h2 {{ margin: 0 0 12px; font-size: 1.25rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
  th, td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: left; vertical-align: top; }}
  th {{ background: #f3eee4; font-weight: 600; }}
  tr.sev-only {{ background: #f9ecec; }}
  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 999px;
    font-size: 0.75rem; font-weight: 600; margin-right: 4px;
  }}
  .badge-freq {{ background: #e4f2ea; color: var(--freq); }}
  .badge-sev {{ background: #f6ece2; color: var(--sev); }}
  .badge-only {{ background: #f3d9dd; color: var(--only); }}
  .badge-shared {{ background: #dfeaf6; color: var(--shared); }}
  .stats {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0 0; }}
  .stat {{
    background: #f3eee4; border-radius: 8px; padding: 10px 14px; min-width: 120px;
  }}
  .stat b {{ display: block; font-size: 1.3rem; }}
  .stat span {{ color: var(--muted); font-size: 0.85rem; }}
  .card {{ margin: 22px 0; padding-top: 8px; border-top: 1px solid var(--line); }}
  .card:first-child {{ border-top: none; padding-top: 0; }}
  .card h3 {{ margin: 0 0 4px; font-size: 1.05rem; }}
  .card .code {{ color: var(--muted); font-family: Consolas, monospace; font-size: 0.85rem; }}
  .plots {{ display: grid; grid-template-columns: 1fr; gap: 12px; margin-top: 10px; }}
  @media (min-width: 900px) {{
    .plots.two {{ grid-template-columns: 1fr 1fr; }}
  }}
  .plots figure {{ margin: 0; background: #faf8f4; border: 1px solid var(--line); border-radius: 8px; padding: 8px; }}
  .plots img {{ width: 100%; height: auto; display: block; }}
  .plots figcaption {{ font-size: 0.8rem; color: var(--muted); margin-top: 4px; }}
  .muted {{ color: var(--muted); }}
  nav a {{ color: var(--shared); margin-right: 14px; }}
</style>
</head>
<body>
<header>
  <h1>Отобранные признаки модели</h1>
  <p>Стек: <b>{html.escape(stack)}</b> · сохранено UTC: {html.escape(saved_at)}</p>
  <p>Графики: экспозиция (столбцы) и частота / severity (линия) — как в EDA.</p>
  <div class="stats">
    <div class="stat"><b>{len(frequency_features)}</b><span>классификация (freq)</span></div>
    <div class="stat"><b>{len(severity_features)}</b><span>регрессия (sev)</span></div>
    <div class="stat"><b>{len(shared)}</b><span>общие</span></div>
    <div class="stat"><b>{len(severity_only)}</b><span>только регрессия</span></div>
    <div class="stat"><b>{len(frequency_only)}</b><span>только классификация</span></div>
  </div>
</header>
<main>
  <section>
    <h2>Навигация</h2>
    <nav>
      <a href="#summary">Сводная таблица</a>
      <a href="#sev-only">Только регрессия</a>
      <a href="#plots">Графики по признакам</a>
    </nav>
  </section>

  <section id="summary">
    <h2>Сводная таблица</h2>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Признак</th>
          <th>Название (RU)</th>
          <th>Модели</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </section>

  <section id="sev-only">
    <h2>Фичи регрессии, которых нет в классификации</h2>
    {"<p class='muted'>Таких фич нет — наборы frequency и severity совпадают.</p>" if not severity_only else ""}
    {"<ul>" + "".join(f"<li><code>{html.escape(f)}</code> — {html.escape(feature_ru_name(f))}</li>" for f in severity_only) + "</ul>" if severity_only else ""}
  </section>

  <section id="plots">
    <h2>Графики зависимостей</h2>
    {cards_html}
  </section>
</main>
</body>
</html>
"""


def save_feature_selection_report(
    df: pd.DataFrame,
    *,
    frequency_features: Iterable[str],
    severity_features: Iterable[str],
    categorical_features: Iterable[str] | None = None,
    frequency_target: str = "TARGET_FREQ",
    severity_target: str = "TARGET_SEV",
    stack: str = "new",
    directory: Path | str | None = None,
    numeric_bins: int = 10,
) -> Path:
    """Построить HTML-отчёт по отобранным фичам и сохранить рядом с JSON FS.

    Путь: ``{directory}/{stack}_selected_features_report_{ts}.html``
    + ``{stack}_selected_features_report_latest.html``.
    """
    freq = list(dict.fromkeys(frequency_features))
    sev = list(dict.fromkeys(severity_features))
    freq_set, sev_set = set(freq), set(sev)
    union = list(dict.fromkeys([*freq, *sev]))
    shared = [f for f in union if f in freq_set and f in sev_set]
    severity_only = [f for f in sev if f not in freq_set]
    frequency_only = [f for f in freq if f not in sev_set]
    cat_set = set(categorical_features or ())

    plot_df = _ensure_exposure(df)
    saved_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    rows: list[str] = []
    cards: list[str] = []
    for idx, feature in enumerate(union, start=1):
        ru = feature_ru_name(feature)
        in_freq = feature in freq_set
        in_sev = feature in sev_set
        sev_only = in_sev and not in_freq
        badges = []
        if in_freq and in_sev:
            badges.append(_badge("общая", "shared"))
        if in_freq:
            badges.append(_badge("классификация", "freq"))
        if in_sev:
            badges.append(_badge("регрессия", "sev"))
        if sev_only:
            badges.append(_badge("только регрессия", "only"))
        row_cls = ' class="sev-only"' if sev_only else ""
        rows.append(
            f"<tr{row_cls}><td>{idx}</td>"
            f"<td><code>{html.escape(feature)}</code></td>"
            f"<td>{html.escape(ru)}</td>"
            f"<td>{''.join(badges)}</td></tr>"
        )

        is_cat = feature in cat_set
        figures: list[str] = []
        for model_type, enabled in (("frequency", in_freq), ("severity", in_sev)):
            if not enabled:
                continue
            b64 = _render_feature_plot(
                plot_df,
                feature,
                model_type=model_type,
                is_categorical=is_cat,
                frequency_target=frequency_target,
                severity_target=severity_target,
                numeric_bins=numeric_bins,
            )
            if b64 is None:
                figures.append(
                    f"<figure><figcaption class='muted'>"
                    f"Нет графика ({model_type}) для {html.escape(feature)}"
                    f"</figcaption></figure>"
                )
            else:
                label = "Частота (классификация)" if model_type == "frequency" else "Severity (регрессия)"
                figures.append(
                    f"<figure><img alt='{html.escape(feature)} {model_type}' "
                    f"src='data:image/png;base64,{b64}'/>"
                    f"<figcaption>{label}</figcaption></figure>"
                )

        plot_cls = "plots two" if len(figures) == 2 else "plots"
        cards.append(
            f"<article class='card' id='f-{html.escape(feature)}'>"
            f"<h3>{html.escape(ru)}</h3>"
            f"<div class='code'>{html.escape(feature)}</div>"
            f"<div style='margin-top:6px'>{''.join(badges)}</div>"
            f"<div class='{plot_cls}'>{''.join(figures)}</div>"
            f"</article>"
        )

    content = _build_html(
        stack=stack,
        frequency_features=freq,
        severity_features=sev,
        severity_only=severity_only,
        frequency_only=frequency_only,
        shared=shared,
        rows_html="\n".join(rows),
        cards_html="\n".join(cards) if cards else "<p class='muted'>Нет отобранных фич.</p>",
        saved_at=saved_at,
    )

    out_dir = Path(directory) if directory is not None else DEFAULT_FEATURE_SELECTION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamped = out_dir / f"{stack}_selected_features_report_{saved_at}.html"
    latest = out_dir / f"{stack}_selected_features_report_latest.html"
    stamped.write_text(content, encoding="utf-8")
    latest.write_text(content, encoding="utf-8")
    return latest
