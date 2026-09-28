"""Smoke: bootstrap-финэффект Test_prod (5 фолдов → медиана) без Hive/OutBoxML.

Запуск: ``python scripts/smoke_bootstrap_fin_effect.py``.
Проверяет: фолды = выборки размера n с возвращением, медиана = медиана фолдов,
детерминизм по seed, разброс min/max/std, сводную таблицу и prod-only/
parity+prod ветки ``run_test_prod_fin_effect`` на синтетическом датасете.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT.parent.parent, PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from querulus.fin_effect import DEFAULT_BOOTSTRAP_FOLDS, bootstrap_fin_effect  # noqa: E402
from querulus.fin_effect.resolve import resolve_fin_effect_config  # noqa: E402
from querulus.training import example_pipeline as ep  # noqa: E402
from querulus.training.example_pipeline import (  # noqa: E402
    bootstrap_fin_effect_table,
    fin_effect_table,
    run_test_prod_fin_effect,
)

THRESHOLD = 0.5
N_FOLDS = 5
SEED = 42

dataset_path = PROJECT_ROOT / "data" / "processed" / "df_final_3_synthetic.parquet"
df = pd.read_parquet(dataset_path)
idx = pd.Index(df.index[:150])
rng = np.random.default_rng(0)
proba = pd.Series(rng.random(len(idx)), index=idx)
sev = pd.Series(rng.random(len(idx)) * 50_000, index=idx)

config = resolve_fin_effect_config(
    df, frequency_target="TARGET_FREQ", severity_target="TARGET_SEV"
)
common = ep._fin_effect_common_index(df, idx, proba, sev)
aligned = df.loc[common]
n_rows = len(aligned)
proba_arr = proba.reindex(common)
sev_arr = sev.reindex(common)
print(f"датасет {df.shape}; Test_prod n = {n_rows}")

# 1) Точечный расчёт на всей выборке (регресс после рефакторинга общего align).
point = fin_effect_table(df, idx, proba, sev, threshold=THRESHOLD, render=False)
print(f"[1] fin_effect_table (вся выборка): net = {point.fin_effect.net_effect:,.0f} ₽")

# 2) Bootstrap: метрики фолдов, медиана, разброс, сводная таблица.
result = bootstrap_fin_effect_table(
    df, idx, proba, sev, threshold=THRESHOLD, n_folds=N_FOLDS, seed=SEED, render=True
)
assert result.n_folds == N_FOLDS and result.n_rows == n_rows
assert result.threshold == THRESHOLD and len(result.folds) == N_FOLDS
assert set(result.folds.columns) == {
    "fold", "n", "n_fact_1", "n_pred_1", "thr", "net_effect", "model_effect", "fact_effect"
}
assert (result.folds["n"] == n_rows).all()
assert np.isclose(result.median_net_effect, np.median(result.folds["net_effect"]))
assert np.isclose(result.median_model_effect, np.median(result.folds["model_effect"]))
assert np.isclose(result.median_fact_effect, np.median(result.folds["fact_effect"]))
assert np.isclose(result.min_net_effect, result.folds["net_effect"].min())
assert np.isclose(result.max_net_effect, result.folds["net_effect"].max())
assert np.isclose(result.std_net_effect, result.folds["net_effect"].std(ddof=1))
summary = result.summary_table()
assert summary.shape[0] == N_FOLDS + 1 and summary["fold"].iloc[-1] == "median"
print(
    f"[2] медиана {N_FOLDS} фолдов: net = {result.median_net_effect:,.0f} ₽ "
    f"[min {result.min_net_effect:,.0f}; max {result.max_net_effect:,.0f}; "
    f"std {result.std_net_effect:,.0f}]"
)
print(summary.to_string(index=False))

# 3) Фолды воспроизводимы вручную: n позиций с возвращением тем же rng.
rng_check = np.random.default_rng(SEED)
for fold, expected_net in enumerate(result.folds["net_effect"], start=1):
    pos = rng_check.integers(0, n_rows, size=n_rows)
    manual = aligned.iloc[pos].copy()
    manual.index = pd.RangeIndex(n_rows)
    manual_fin = ep.run_fin_effect_pipeline(
        manual,
        pd.Series(proba_arr.to_numpy()[pos], index=manual.index),
        pd.Series(sev_arr.to_numpy()[pos], index=manual.index),
        pd.Series(manual["TARGET_FREQ"].to_numpy(), index=manual.index),
        threshold=THRESHOLD,
        config=config,
    )
    assert np.isclose(manual_fin.net_effect, expected_net), fold
print("[3] фолды совпадают с ручным bootstrap-ресэмплом (n строк с возвращением)")

# 4) Детерминизм по seed.
same = bootstrap_fin_effect_table(
    df, idx, proba, sev, threshold=THRESHOLD, n_folds=N_FOLDS, seed=SEED, render=False
)
assert same.folds.equals(result.folds)
other = bootstrap_fin_effect_table(
    df, idx, proba, sev, threshold=THRESHOLD, n_folds=N_FOLDS, seed=SEED + 1, render=False
)
assert not np.allclose(other.folds["net_effect"], result.folds["net_effect"])
print(
    f"[4] seed {SEED} воспроизводим; seed {SEED + 1} → "
    f"медиана net = {other.median_net_effect:,.0f} ₽"
)

# 5) Прямой вызов (без df/index-обвязки) и поиск порога внутри фолдов.
direct = bootstrap_fin_effect(
    aligned, proba_arr, sev_arr, aligned["TARGET_FREQ"], threshold=THRESHOLD, config=config
)
assert np.isclose(direct.median_net_effect, result.median_net_effect)
searched = bootstrap_fin_effect(
    aligned, proba_arr, sev_arr, aligned["TARGET_FREQ"], config=config, seed=SEED
)
assert searched.threshold is None
print(f"[5] прямой вызов ок; τ по фолдам при threshold=None: {list(searched.folds['thr'])}")

# 6) Покрытие < 95% → ошибка.
try:
    bootstrap_fin_effect_table(
        df, idx, proba.iloc[:50], sev, threshold=THRESHOLD, n_folds=N_FOLDS, render=False
    )
except ValueError as exc:
    print(f"[6] покрытие проверено: {exc}")
else:
    raise AssertionError("ожидали ValueError по покрытию")

# 7) run_test_prod_fin_effect: prod-only (example_final) и parity+prod (example).
test_prod_idx = pd.Index(df.index[300:440])
parity_proba = pd.Series(
    np.random.default_rng(1).random(len(test_prod_idx)), index=test_prod_idx
)
parity_sev = pd.Series(
    np.random.default_rng(2).random(len(test_prod_idx)) * 40_000, index=test_prod_idx
)
prod_proba = pd.Series(
    np.random.default_rng(3).random(len(test_prod_idx)), index=test_prod_idx
)
prod_sev = pd.Series(
    np.random.default_rng(4).random(len(test_prod_idx)) * 40_000, index=test_prod_idx
)


class FakeBundle:
    """Мини-замена ExampleDatasetBundle для проверки сводки."""

    df = df
    cf_name = "cf"
    rg_name = "rg"
    periods = {
        "prod_holdout_idx": test_prod_idx,
        "prod_test_period": ("2025-03-01", "2025-06-01"),
        "prod_train_period": ("2022-01-01", "2025-01-01"),
        "parity_train_period": ("2022-01-01", "2024-12-01"),
    }


parity_dsm = object()
prod_dsm = object()


class FakeModels:
    def __init__(self, with_parity: bool) -> None:
        self.dsm_cf = parity_dsm if with_parity else None
        self.dsm_rg = parity_dsm if with_parity else None
        self.dsm_cf_prod = prod_dsm
        self.dsm_rg_prod = prod_dsm


orig_predict_cf, orig_predict_rg = ep.predict_cf, ep.predict_rg
ep.predict_cf = lambda dsm, name, data: (
    parity_proba if dsm is parity_dsm else prod_proba
).reindex(data.index)
ep.predict_rg = lambda dsm, name, data: (
    parity_sev if dsm is parity_dsm else prod_sev
).reindex(data.index)

thresholds = ep.ExampleThresholds(parity=0.3, prod=0.45)
expected_columns = {
    "train", "train_period", "n_test_prod", "n_folds", "seed", "thr",
    "net_effect", "net_effect_min", "net_effect_max", "net_effect_std",
    "model_effect", "fact_effect",
}
try:
    only_prod = run_test_prod_fin_effect(
        FakeModels(with_parity=False), FakeBundle(), thresholds=thresholds
    )
    assert only_prod.parity is None
    assert list(only_prod.compare_table["train"]) == ["prod"]
    assert set(only_prod.compare_table.columns) == expected_columns
    prod_row = only_prod.compare_table.iloc[0]
    assert prod_row["n_folds"] == DEFAULT_BOOTSTRAP_FOLDS and prod_row["seed"] == SEED
    assert np.isclose(
        prod_row["net_effect"], np.median(only_prod.prod.folds["net_effect"])
    )
    print("[7] prod-only (example_final):")
    print(only_prod.compare_table.to_string(index=False))

    both = run_test_prod_fin_effect(
        FakeModels(with_parity=True), FakeBundle(), thresholds=thresholds
    )
    assert both.parity is not None and len(both.compare_table) == 2
    assert np.isclose(
        both.compare_table["net_effect"].iloc[0], np.median(both.parity.folds["net_effect"])
    )
    assert np.isclose(
        both.compare_table["net_effect"].iloc[1], np.median(both.prod.folds["net_effect"])
    )
    print("[8] parity+prod (example):")
    print(both.compare_table.to_string(index=False))
finally:
    ep.predict_cf, ep.predict_rg = orig_predict_cf, orig_predict_rg

print("OK: bootstrap-финэффект проверен")
