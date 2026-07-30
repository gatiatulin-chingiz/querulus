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
from querulus.training.selected_features import (
    PROD_FREQUENCY_FEATURES,
    PROD_SEVERITY_FEATURES,
)

_EXPOSURE = "expos"
_MAX_CAT_LEVELS = 25


def _importance_map(frame: pd.DataFrame | None) -> dict[str, float]:
    """feature → importance из DataFrame (колонки feature / importance)."""
    if frame is None or getattr(frame, "empty", True):
        return {}
    if "feature" not in frame.columns or "importance" not in frame.columns:
        return {}
    out: dict[str, float] = {}
    for row in frame.itertuples(index=False):
        out[str(row.feature)] = float(row.importance)
    return out


def _fmt_weight(value: float | None) -> str:
    """Формат веса для таблицы."""
    if value is None:
        return "—"
    return f"{value:.2f}"


def _ensure_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить колонку экспозиции для EDA."""
    result = df.copy()
    if _EXPOSURE not in result.columns:
        result[_EXPOSURE] = 1
    return result


def _is_binary_series(series: pd.Series) -> bool:
    """Ненулевые значения ⊆ {0, 1} (с учётом float)."""
    values = pd.to_numeric(series, errors="coerce")
    uniq = {float(x) for x in values.dropna().unique().tolist()}
    return bool(uniq) and uniq.issubset({0.0, 1.0})


def _pretty_edges(lo: float, hi: float, n_bins: int) -> np.ndarray:
    """Границы бинов с «круглыми» шагами (0, 50, 100, …)."""
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        return np.array([lo - 0.5, hi + 0.5], dtype=float)
    rough = span / max(n_bins, 1)
    exp = np.floor(np.log10(rough)) if rough > 0 else 0.0
    base = 10.0 ** exp
    frac = rough / base
    if frac <= 1.5:
        nice = 1.0
    elif frac <= 3.0:
        nice = 2.0
    elif frac <= 7.0:
        nice = 5.0
    else:
        nice = 10.0
    step = nice * base
    start = np.floor(lo / step) * step
    stop = np.ceil(hi / step) * step
    edges = np.arange(start, stop + step * 0.5, step, dtype=float)
    if len(edges) < 2:
        return np.array([lo, hi], dtype=float)
    edges[0] = min(edges[0], lo)
    edges[-1] = max(edges[-1], hi)
    return np.unique(edges)


def _bin_continuous(series: pd.Series, numeric_bins: int) -> pd.Series:
    """Равные «круглые» бины вместо qcut с произвольными квантилями."""
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values.to_numpy(dtype=float))]
    if finite.size < 2 or finite.nunique() < 2:
        raise ValueError("insufficient unique finite values for binning")
    n_bins = max(2, min(int(numeric_bins), int(finite.nunique())))
    edges = _pretty_edges(float(finite.min()), float(finite.max()), n_bins)
    return pd.cut(values, bins=edges, include_lowest=True, duplicates="drop")


def _prepare_feature_column(
    series: pd.Series,
    *,
    is_categorical: bool,
    numeric_bins: int,
) -> tuple[pd.Series, bool]:
    """Подготовить колонку для группировки; вернуть (series, as_categorical)."""
    if _is_binary_series(series):
        # 0 / 1 / NaN — три категории, не один float-бин
        mapped = pd.to_numeric(series, errors="coerce").round()
        cats = pd.Categorical(
            mapped.map({0.0: "0", 1.0: "1"}).fillna("NaN"),
            categories=["0", "1", "NaN"],
            ordered=True,
        )
        return pd.Series(cats, index=series.index), True

    if is_categorical or series.dtype == object or str(series.dtype) in {
        "string",
        "category",
        "boolean",
    }:
        return series.astype("string"), True

    # Int64 с небольшим числом уровней — как категории
    if pd.api.types.is_integer_dtype(series) and series.nunique(dropna=True) <= 15:
        return series.astype("string"), True

    return _bin_continuous(series, numeric_bins), False


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
    prepared, as_cat = _prepare_feature_column(
        data[feature],
        is_categorical=is_categorical,
        numeric_bins=numeric_bins,
    )
    data[feature] = prepared

    if model_type == "frequency":
        cols = [feature, _EXPOSURE, frequency_target]
        grouped = (
            data[cols]
            .dropna(subset=[feature])
            .groupby(feature, dropna=False, observed=True)
            .sum()
        )
        grouped["ratio"] = grouped[frequency_target] / grouped[_EXPOSURE]
    elif model_type == "severity":
        cols = [feature, _EXPOSURE, frequency_target, severity_target]
        grouped = (
            data[cols]
            .dropna(subset=[feature])
            .groupby(feature, dropna=False, observed=True)
            .sum()
        )
        grouped["ratio"] = grouped[severity_target] / grouped[frequency_target]
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    if as_cat and not _is_binary_series(df[feature]):
        # Высокая кардинальность: топ по экспозиции + «прочие»
        if len(grouped) > _MAX_CAT_LEVELS:
            top = grouped.nlargest(_MAX_CAT_LEVELS - 1, _EXPOSURE)
            other = grouped.drop(index=top.index).sum(numeric_only=True)
            other_row = other.to_frame().T
            other_row.index = pd.Index(["прочие"])
            if model_type == "frequency":
                other_row["ratio"] = other_row[frequency_target] / other_row[_EXPOSURE]
            else:
                other_row["ratio"] = other_row[severity_target] / other_row[frequency_target]
            grouped = pd.concat([top, other_row])
        grouped = grouped.sort_values(by="ratio", ascending=False)
    return grouped


def _format_tick_label(value: object) -> str:
    """Короткий текст для оси X."""
    if isinstance(value, pd.Interval):
        left, right = float(value.left), float(value.right)

        def _fmt(x: float) -> str:
            return str(int(x)) if abs(x - round(x)) < 1e-9 else f"{x:g}"

        text = f"({_fmt(left)}, {_fmt(right)}]"
    else:
        text = str(value)
    if len(text) > 28:
        return text[:25] + "…"
    return text


def _plot_to_base64(
    grouped: pd.DataFrame,
    feature: str,
    model_type: str,
    *,
    figsize: tuple[float, float] | None = None,
) -> str:
    """Нарисовать EDA-график и вернуть PNG как base64."""
    n = grouped.shape[0]
    # Больше категорий → шире фигура, сильнее поворот подписей
    if figsize is None:
        width = min(16.0, max(10.0, 0.45 * n + 4.0))
        figsize = (width, 4.5)
    rotation = 90 if n > 12 else (55 if n > 6 else 30)
    fontsize = 7 if n > 18 else (8 if n > 10 else 9)

    ind = np.arange(n)
    fig, ax = plt.subplots(dpi=110, figsize=figsize)
    ax.bar(ind, grouped[_EXPOSURE], color="#4C78A8", alpha=0.85)
    ax.set_ylabel("Экспозиция", fontsize=11)
    labels = [_format_tick_label(x) for x in grouped.index.tolist()]
    ax.set_xticks(ind)
    ax.set_xticklabels(labels, fontsize=fontsize, rotation=rotation, ha="right")
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
    n_from_old: int,
    n_new: int,
    old_rows_html: str,
    freq_rows_html: str,
    sev_rows_html: str,
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
    --shared: #1f4b7a;
    --old-bg: #eef2f7;
    --new-bg: #eef8ea;
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
    overflow-x: auto;
  }}
  h2 {{ margin: 0 0 12px; font-size: 1.25rem; }}
  h3.sub {{ margin: 18px 0 10px; font-size: 1.05rem; color: var(--ink); }}
  h3.sub:first-of-type {{ margin-top: 4px; }}
  table {{
    width: 100%; border-collapse: collapse; font-size: 0.88rem;
    table-layout: fixed;
  }}
  th, td {{
    border-bottom: 1px solid var(--line); padding: 8px 10px;
    text-align: left; vertical-align: top;
    word-break: break-word; overflow-wrap: anywhere;
  }}
  th {{ background: #f3eee4; font-weight: 600; }}
  th.col-idx, td.col-idx {{ width: 3.2rem; }}
  th.col-code, td.col-code {{ width: 32%; font-family: Consolas, monospace; font-size: 0.8rem; }}
  th.col-ru, td.col-ru {{ width: 42%; }}
  th.col-w, td.col-w {{ width: 7rem; }}
  th.col-model, td.col-model {{ width: 8.5rem; }}
  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 999px;
    font-size: 0.75rem; font-weight: 600; margin-right: 4px;
  }}
  .badge-freq {{ background: #e4f2ea; color: var(--freq); }}
  .badge-sev {{ background: #f6ece2; color: var(--sev); }}
  tr.is-old {{ background: var(--old-bg); }}
  tr.is-new {{ background: var(--new-bg); }}
  .legend span {{
    display: inline-block; padding: 2px 10px; border-radius: 6px;
    margin-right: 10px; font-size: 0.85rem;
  }}
  .legend .lg-old {{ background: var(--old-bg); }}
  .legend .lg-new {{ background: var(--new-bg); }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
  .stats {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0 0; }}
  .stat {{
    background: #f3eee4; border-radius: 8px; padding: 10px 14px; min-width: 120px;
  }}
  .stat b {{ display: block; font-size: 1.3rem; }}
  .stat span {{ color: var(--muted); font-size: 0.85rem; }}
  .card {{ margin: 22px 0; padding-top: 8px; border-top: 1px solid var(--line); }}
  .card:first-child {{ border-top: none; padding-top: 0; }}
  .card h3 {{ margin: 0 0 4px; font-size: 1.05rem; }}
  .card .code {{ color: var(--muted); font-family: Consolas, monospace; font-size: 0.85rem; word-break: break-all; }}
  .plots {{ display: grid; grid-template-columns: 1fr; gap: 12px; margin-top: 10px; }}
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
  <div class="stats">
    <div class="stat"><b>{len(frequency_features)}</b><span>классификация</span></div>
    <div class="stat"><b>{len(severity_features)}</b><span>регрессия</span></div>
    <div class="stat"><b>{n_from_old}</b><span>из старой модели</span></div>
    <div class="stat"><b>{n_new}</b><span>новые</span></div>
  </div>
</header>
<main>
  <section>
    <h2>Навигация</h2>
    <nav>
      <a href="#old-model">Признаки старой модели</a>
      <a href="#summary-freq">Сводка: классификация</a>
      <a href="#summary-sev">Сводка: регрессия</a>
      <a href="#plots">Графики по признакам</a>
    </nav>
  </section>

  <section id="old-model">
    <h2>Признаки старой (прод) модели</h2>
    <p class="muted">Источник: <code>PROD_*_FEATURES</code> в <code>selected_features.py</code>
      (= фичи из <code>config_cf_3.json</code> / <code>config_rg_3.json</code>).</p>
    <table>
      <thead>
        <tr>
          <th class="col-idx">#</th>
          <th class="col-code">Признак</th>
          <th class="col-ru">Название (RU)</th>
          <th class="col-model">Модель</th>
        </tr>
      </thead>
      <tbody>
        {old_rows_html}
      </tbody>
    </table>
  </section>

  <section id="summary-freq">
    <h2>Сводная таблица — классификация</h2>
    <p class="legend muted">
      Подсветка:
      <span class="lg-old">была в старой модели</span>
      <span class="lg-new">новая</span>
    </p>
    <table>
      <thead>
        <tr>
          <th class="col-idx">#</th>
          <th class="col-code">Признак</th>
          <th class="col-ru">Название (RU)</th>
          <th class="col-w num">Значимость</th>
        </tr>
      </thead>
      <tbody>
        {freq_rows_html}
      </tbody>
    </table>
  </section>

  <section id="summary-sev">
    <h2>Сводная таблица — регрессия</h2>
    <p class="legend muted">
      Подсветка:
      <span class="lg-old">была в старой модели</span>
      <span class="lg-new">новая</span>
    </p>
    <table>
      <thead>
        <tr>
          <th class="col-idx">#</th>
          <th class="col-code">Признак</th>
          <th class="col-ru">Название (RU)</th>
          <th class="col-w num">Значимость</th>
        </tr>
      </thead>
      <tbody>
        {sev_rows_html}
      </tbody>
    </table>
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
    frequency_importance: pd.DataFrame | None = None,
    severity_importance: pd.DataFrame | None = None,
    old_frequency_features: Iterable[str] | None = None,
    old_severity_features: Iterable[str] | None = None,
    frequency_target: str = "TARGET_FREQ",
    severity_target: str = "TARGET_SEV",
    stack: str = "new",
    directory: Path | str | None = None,
    numeric_bins: int = 10,
) -> Path:
    """Построить HTML-отчёт по отобранным фичам и сохранить рядом с JSON FS.

    Две сводные таблицы: классификация и регрессия (каждая по importance ↓).
    Старые прод-фичи — отдельная таблица выше.
    """
    freq = list(dict.fromkeys(frequency_features))
    sev = list(dict.fromkeys(severity_features))
    cat_set = set(categorical_features or ())

    old_freq = list(
        old_frequency_features
        if old_frequency_features is not None
        else PROD_FREQUENCY_FEATURES
    )
    old_sev = list(
        old_severity_features
        if old_severity_features is not None
        else PROD_SEVERITY_FEATURES
    )
    old_set = set(old_freq) | set(old_sev)

    freq_w = _importance_map(frequency_importance)
    sev_w = _importance_map(severity_importance)

    freq_ordered = sorted(freq, key=lambda n: (-freq_w.get(n, -1.0), n))
    sev_ordered = sorted(sev, key=lambda n: (-sev_w.get(n, -1.0), n))
    entries: list[tuple[str, str]] = [
        *((name, "frequency") for name in freq_ordered),
        *((name, "severity") for name in sev_ordered),
    ]

    selected_names = list(dict.fromkeys([*freq, *sev]))
    from_old = [f for f in selected_names if f in old_set]
    new_feats = [f for f in selected_names if f not in old_set]

    # Таблица старой модели (freq, затем sev; дубли если в обеих)
    old_entries: list[tuple[str, str]] = []
    for name in old_freq:
        old_entries.append((name, "frequency"))
    for name in old_sev:
        old_entries.append((name, "severity"))

    old_rows: list[str] = []
    for idx, (feature, model_type) in enumerate(old_entries, start=1):
        ru = feature_ru_name(feature)
        model_badge = (
            _badge("классификация", "freq")
            if model_type == "frequency"
            else _badge("регрессия", "sev")
        )
        old_rows.append(
            f"<tr><td class='col-idx'>{idx}</td>"
            f"<td class='col-code'>{html.escape(feature)}</td>"
            f"<td class='col-ru'>{html.escape(ru)}</td>"
            f"<td class='col-model'>{model_badge}</td></tr>"
        )

    plot_df = _ensure_exposure(df)
    saved_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def _summary_rows(
        names: list[str],
        weights: dict[str, float],
    ) -> str:
        if not names:
            return "<tr><td colspan='4'>Нет отобранных признаков</td></tr>"
        lines: list[str] = []
        for idx, feature in enumerate(names, start=1):
            ru = feature_ru_name(feature)
            is_old = feature in old_set
            weight = weights.get(feature)
            row_cls = ' class="is-old"' if is_old else ' class="is-new"'
            lines.append(
                f"<tr{row_cls}><td class='col-idx'>{idx}</td>"
                f"<td class='col-code'>{html.escape(feature)}</td>"
                f"<td class='col-ru'>{html.escape(ru)}</td>"
                f"<td class='col-w num'>{_fmt_weight(weight)}</td></tr>"
            )
        return "\n".join(lines)

    cards: list[str] = []
    for feature, model_type in entries:
        ru = feature_ru_name(feature)
        weight = freq_w.get(feature) if model_type == "frequency" else sev_w.get(feature)
        model_badge = (
            _badge("классификация", "freq")
            if model_type == "frequency"
            else _badge("регрессия", "sev")
        )
        is_cat = feature in cat_set
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
            figure = (
                f"<figure><figcaption class='muted'>"
                f"Нет графика ({model_type}) для {html.escape(feature)}"
                f"</figcaption></figure>"
            )
        else:
            label = (
                "Частота (классификация)"
                if model_type == "frequency"
                else "Severity (регрессия)"
            )
            figure = (
                f"<figure><img alt='{html.escape(feature)} {model_type}' "
                f"src='data:image/png;base64,{b64}'/>"
                f"<figcaption>{label}</figcaption></figure>"
            )
        cards.append(
            f"<article class='card' id='f-{html.escape(feature)}-{model_type}'>"
            f"<h3>{html.escape(ru)}</h3>"
            f"<div class='code'>{html.escape(feature)}</div>"
            f"<div style='margin-top:6px'>{model_badge}</div>"
            f"<p class='muted' style='margin:6px 0 0'>Значимость: {_fmt_weight(weight)}</p>"
            f"<div class='plots'>{figure}</div>"
            f"</article>"
        )

    content = _build_html(
        stack=stack,
        frequency_features=freq,
        severity_features=sev,
        n_from_old=len(from_old),
        n_new=len(new_feats),
        old_rows_html="\n".join(old_rows) if old_rows else "<tr><td colspan='4'>Нет данных</td></tr>",
        freq_rows_html=_summary_rows(freq_ordered, freq_w),
        sev_rows_html=_summary_rows(sev_ordered, sev_w),
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


def _importance_frame_from_payload(payload: dict) -> pd.DataFrame | None:
    """DataFrame importance из JSON FS (или None)."""
    rows = payload.get("importance") or []
    if not rows:
        return None
    return pd.DataFrame(rows)


def rebuild_feature_selection_report(
    df: pd.DataFrame,
    *,
    stack: str = "new",
    directory: Path | str | None = None,
    frequency_target: str = "TARGET_FREQ",
    severity_target: str = "TARGET_SEV",
    numeric_bins: int = 10,
) -> Path:
    """Пересобрать HTML из ``{stack}_{frequency|severity}_latest.json`` без повторного FS.

    Нужны ранее сохранённые списки фич (+ опционально importance / categorical).
    """
    from querulus.training.feature_selection_io import load_feature_selection_latest

    out_dir = Path(directory) if directory is not None else DEFAULT_FEATURE_SELECTION_DIR
    freq_payload = load_feature_selection_latest(stack, "frequency", directory=out_dir)
    sev_payload = load_feature_selection_latest(stack, "severity", directory=out_dir)
    if freq_payload is None or sev_payload is None:
        raise FileNotFoundError(
            f"Нет FS JSON в {out_dir}: ожидаются "
            f"{stack}_frequency_latest.json и {stack}_severity_latest.json. "
            "Сначала прогоните feature select."
        )
    freq_feats = list(freq_payload.get("selected_features") or [])
    sev_feats = list(sev_payload.get("selected_features") or [])
    freq_imp = _importance_frame_from_payload(freq_payload)
    sev_imp = _importance_frame_from_payload(sev_payload)
    from querulus.training.feature_selection_io import (
        drop_zero_importance_features,
        save_feature_selection,
    )

    freq_feats, freq_zero, freq_imp = drop_zero_importance_features(freq_feats, freq_imp)
    sev_feats, sev_zero, sev_imp = drop_zero_importance_features(sev_feats, sev_imp)
    if freq_zero or sev_zero:
        # Обновить latest JSON, чтобы набор без нулей сохранился
        save_feature_selection(
            stack=stack,
            task="frequency",
            selected_features=freq_feats,
            summary={"zero_importance_dropped": freq_zero},
            directory=out_dir,
            importance=freq_imp,
            categorical_features=list(freq_payload.get("categorical_features") or []),
        )
        save_feature_selection(
            stack=stack,
            task="severity",
            selected_features=sev_feats,
            summary={"zero_importance_dropped": sev_zero},
            directory=out_dir,
            importance=sev_imp,
            categorical_features=list(sev_payload.get("categorical_features") or []),
        )
    cat_feats = list(
        dict.fromkeys(
            [
                c
                for c in list(freq_payload.get("categorical_features") or [])
                + list(sev_payload.get("categorical_features") or [])
                if c in freq_feats or c in sev_feats
            ]
        )
    )
    return save_feature_selection_report(
        df,
        frequency_features=freq_feats,
        severity_features=sev_feats,
        categorical_features=cat_feats,
        frequency_importance=freq_imp,
        severity_importance=sev_imp,
        frequency_target=frequency_target,
        severity_target=severity_target,
        stack=stack,
        directory=out_dir,
        numeric_bins=numeric_bins,
    )
