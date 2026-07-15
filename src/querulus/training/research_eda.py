"""Оригинальные функции EDA из Litigant (research_continous / research_feature)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

expsoure = "expos"
damage_count = "TARGET_2"
damage_sum = "TARGET_SEV"


def plot_cat_vs_target(data, x_min, x_max, figsize, feature, save, model_type, rotation):
    """Бары экспозиции и линия таргета (ratio) на второй оси."""
    if x_min:
        data = data[data["ratio"] > x_min]
    if x_max:
        data = data[data["ratio"] < x_max]
    n = data.shape[0]
    ind = np.arange(n)

    fig, ax = plt.subplots(dpi=100, figsize=figsize)

    ax.bar(ind, data[expsoure])
    ax.set_ylabel(expsoure, fontsize=20)
    ax.set_xticks(ind, data.index.tolist(), fontsize=20, rotation=rotation)

    ax.tick_params(axis="both", labelsize=20)

    axes2 = ax.twinx()
    axes2.plot(ind, data["ratio"], color="r", marker="o")

    if model_type == "frequency":
        axes2.set_ylabel("Частота", fontsize=20)
    elif model_type == "severity":
        axes2.set_ylabel("Severity", fontsize=20)

    axes2.tick_params(axis="both", labelsize=20)
    plt.grid(False)
    plt.title(f"""{feature}_{str(model_type).upper()}""", fontsize=25)

    if save:
        plt.savefig(f"""plots/{feature}_{str(model_type).upper()}.png""", bbox_inches="tight", dpi=1200)
    plt.show()


def _qcut_feature(series: pd.Series, quantiles: int) -> pd.Series:
    """Безопасный qcut: numeric + dropna; fallback на cut при сбое pandas IntervalIndex."""
    values = pd.to_numeric(series, errors="coerce")
    arr = values.to_numpy(dtype=float, copy=False)
    finite = np.isfinite(arr)
    if finite.sum() < 2 or pd.Series(arr[finite]).nunique() < 2:
        raise ValueError("insufficient unique finite values for binning")

    n_bins = max(2, min(int(quantiles), int(pd.Series(arr[finite]).nunique())))
    try:
        return pd.qcut(values, n_bins, duplicates="drop")
    except (ValueError, TypeError):
        # Pandas IntervalIndex / совпадающие quantile edges (часто на float+NaN).
        return pd.cut(values, bins=n_bins, duplicates="drop")


def research_continous(
    data,
    feature,
    quantiles,
    model_type="frequency",
    figsize: tuple = (55, 10),
    save=False,
    rotation=90,
    *,
    frequency_target: str | None = None,
    severity_target: str | None = None,
):
    """Числовой признак по квантильным бинам: экспозиция + линия частоты/severity."""
    freq_col = frequency_target or damage_count
    sev_col = severity_target or damage_sum
    binned = _qcut_feature(data[feature], quantiles)
    data = data.copy()
    data[feature] = binned

    if model_type == "frequency":
        data = data[[feature, expsoure, freq_col]].dropna(subset=[feature])
        grouped = data.groupby(feature, observed=True).agg("sum")
        grouped["ratio"] = grouped[freq_col] / grouped[expsoure]
        count_col = freq_col
    elif model_type == "severity":
        data = data[[feature, sev_col, expsoure, freq_col]].dropna(subset=[feature])
        grouped = data.groupby(feature, observed=True).agg("sum")
        grouped["ratio"] = grouped[sev_col] / grouped[freq_col]
        count_col = freq_col
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    grouped[count_col] = grouped[count_col] / sum(grouped[count_col])

    plot_cat_vs_target(grouped, None, None, figsize, feature, save, model_type, rotation)
    return grouped


def research_feature(
    data,
    feature,
    bounds=None,
    sort_by=None,
    x_min: float | None = None,
    x_max: float | None = None,
    figsize: tuple = (55, 10),
    max_limit=None,
    min_limit=None,
    model_type="frequency",
    save=False,
    rotation=90,
    *,
    frequency_target: str | None = None,
    severity_target: str | None = None,
):
    """Категориальный признак: экспозиция + линия частоты/severity по категориям."""
    freq_col = frequency_target or damage_count
    sev_col = severity_target or damage_sum

    if model_type == "frequency":
        data = data[[feature, expsoure, freq_col]]
        grouped = data.groupby(feature, dropna=False).agg("sum")
        grouped["ratio"] = grouped[freq_col] / grouped[expsoure]
    elif model_type == "severity":
        data = data[[feature, expsoure, freq_col, sev_col]]
        grouped = data.groupby(feature, dropna=False).agg("sum")
        grouped["ratio"] = grouped[sev_col] / grouped[freq_col]
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    if sort_by == "index":
        grouped = grouped.sort_index()
    elif sort_by == "index_d":

        def quarter(q):
            res = []
            for quarter_val in q:
                quar, year = str(quarter_val).split()[1:]
                quar = quar[0]
                res.append(int(year + quar))
            return res

        grouped = grouped.sort_index(key=quarter)

    elif sort_by is None:
        grouped = grouped.sort_values(by="ratio", ascending=False)
    else:
        grouped = grouped.sort_values(by=sort_by, ascending=False)

    if max_limit is not None:
        grouped = grouped[grouped[expsoure] < max_limit]

    if min_limit is not None:
        grouped = grouped[grouped[expsoure] > min_limit]

    if bounds is not None:
        raise NotImplementedError("concat_group не включён в querulus; передайте bounds=None")

    plot_cat_vs_target(grouped, x_min, x_max, figsize, feature, save, model_type, rotation)
    return grouped
