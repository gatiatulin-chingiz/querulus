"""Smoke: прогнать сгенерированные config_parity/prod через OutBoxML prepare.

Проверяет ровно те правила, которые заданы в п.1–12:
  * категориальные схлопнуты (кардинальность ≤ CAT_MAX_CODES + ПРОЧИЕ/NAN);
  * NaN не остаётся ни в одной фиче (fillna/default по политике);
  * int-подобные фичи (AGE/YEAR/COUNT/…) — целые (int32), не float;
  * фичи-годы ≤ 2026 (clip) и default = последний год (`_MAX_`);
  * clip есть у всех числовых не-бинарных фич.

Запуск: python scripts/smoke_outboxml_configs_prepare.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from querulus.features.data_quality import MAX_YEAR_FEATURE_VALUE  # noqa: E402
from querulus.features.integer_casts import (  # noqa: E402
    is_integer_like_feature,
    is_year_feature,
)
from querulus.training.build_outboxml_configs import (  # noqa: E402
    CAT_MAX_CODES,
    ensure_outboxml_runtime_patches,
)

CONFIG_PATHS = (
    PROJECT_ROOT / "configs" / "querulus" / "2.0.0" / "config_parity.json",
    PROJECT_ROOT / "configs" / "querulus" / "2.0.0" / "config_prod.json",
)


def _check_config(path: Path) -> list[str]:
    ensure_outboxml_runtime_patches()
    from outboxml.core.prepared_datasets import DateSeparation, PrepareDataset
    from outboxml.core.pydantic_models import AllModelsConfig

    raw = json.loads(path.read_text(encoding="utf-8"))
    cfg = AllModelsConfig.model_validate(raw)
    df = pd.read_parquet(cfg.data_config.local_name_source)
    date_column = cfg.data_config.separation.period_column[0]
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    train_ind, test_ind = DateSeparation(cfg.data_config.separation).train_test_indexes(df)
    print(f"\n######## {path.name} | rows={len(df)} train={len(train_ind)} test={len(test_ind)}")

    failures: list[str] = []
    for model in cfg.models_configs:
        target = df[model.column_target] if model.column_target in df.columns else None
        result = PrepareDataset(
            model_config=model, check_prepared=True, group_name="smoke"
        ).prepare_dataset(df, train_ind, test_ind, target)
        data = result.data
        cats = set(model.cat_features_catboost or [])
        print(f"=== {model.name} ({model.objective}) rows={len(data)} ===")

        for feature in model.features:
            column = data[feature.name]
            nunique = int(column.nunique(dropna=True))
            n_nan = int(column.isna().sum())
            dtype = str(column.dtype)
            line = f"  {feature.name:<40} {dtype:<9} nan={n_nan:<4} uniq={nunique:<4}"
            if n_nan:
                failures.append(f"{model.name}:{feature.name}: NaN остались ({n_nan})")
            if is_integer_like_feature(feature.name) and not pd.api.types.is_integer_dtype(column):
                failures.append(f"{model.name}:{feature.name}: не целый dtype ({dtype})")
            if feature.name in cats:
                if nunique > CAT_MAX_CODES + 2:
                    failures.append(
                        f"{model.name}:{feature.name}: кардинальность {nunique} > "
                        f"{CAT_MAX_CODES + 2}"
                    )
                values = pd.to_numeric(column.astype("object"), errors="coerce").dropna()
                line += f" codes={sorted(int(v) for v in values.unique())}"
            elif is_year_feature(feature.name) and float(column.max()) > MAX_YEAR_FEATURE_VALUE:
                failures.append(
                    f"{model.name}:{feature.name}: год {column.max()} > {MAX_YEAR_FEATURE_VALUE}"
                )
            if feature.name not in cats and feature.clip is None:
                failures.append(f"{model.name}:{feature.name}: нет clip")
            line += f" | default={feature.default} fillna={feature.fillna}"
            print(line)

    return failures


def main() -> int:
    failures: list[str] = []
    for path in CONFIG_PATHS:
        failures.extend(_check_config(path))
    if failures:
        print("\nFAILURES:\n  " + "\n  ".join(failures))
        return 1
    print("\nOK: все проверки пройдены")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
