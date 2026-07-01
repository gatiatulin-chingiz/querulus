"""Оригинальные функции EDA из Litigant (research_continous / research_feature)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

expsoure = "expos"
damage_count = "TARGET_2"
damage_sum = "TARGET_3_SEV"


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


def research_continous(
    data,
    feature,
    quantiles,
    model_type="frequency",
    figsize: tuple = (55, 10),
    save=False,
    rotation=90,
):
    """Числовой признак по квантильным бинам: экспозиция + линия частоты/severity."""
    if model_type == "frequency":
        data = data[[feature, expsoure, damage_count]]
        quantiles, bins = pd.qcut(data[feature], quantiles, duplicates="drop", retbins=True)
        data.drop(feature, axis=1, inplace=True)
        data = pd.concat([data, quantiles], axis=1, join="outer")
        grouped = data.groupby(feature).agg(sum)
        grouped["ratio"] = grouped[damage_count] / grouped[expsoure]
    elif model_type == "severity":
        data = data[[feature, damage_sum, expsoure, damage_count]]
        quantiles, bins = pd.qcut(data[feature], quantiles, duplicates="drop", retbins=True)
        data.drop(feature, axis=1, inplace=True)
        data = pd.concat([data, quantiles], axis=1, join="outer")
        grouped = data.groupby(feature).agg(sum)
        grouped["ratio"] = grouped[damage_sum] / grouped[damage_count]

    grouped[damage_count] = grouped[damage_count] / sum(grouped[damage_count])

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
):
    """Категориальный признак: экспозиция + линия частоты/severity по категориям."""
    if model_type == "frequency":
        data = data[[feature, expsoure, damage_count]]
        grouped = data.groupby(feature, dropna=False).agg(sum)

        grouped["ratio"] = grouped[damage_count] / grouped[expsoure]
    elif model_type == "severity":
        data = data[[feature, expsoure, damage_count, damage_sum]]
        grouped = data.groupby(feature, dropna=False).agg(sum)

        grouped["ratio"] = grouped[damage_sum] / grouped[damage_count]

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
