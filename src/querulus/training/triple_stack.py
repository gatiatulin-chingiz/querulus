"""Обучение и сравнение трёх стеков таргетов: legacy / new / new_claims."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import pandas as pd

from querulus.fin_effect.calculator import FinEffectResult, run_fin_effect_from_training
from querulus.fin_effect.resolve import resolve_fin_effect_config
from querulus.training.config import TrainingConfig, resolve_features_config
from querulus.training.pipeline import TrainingArtifacts, format_metric_value, train_models

from querulus.training.splits import default_inner_periods_from_train

# Стеки new/new_claims: train_core + Val + holdout Test; legacy — train + Test (test-tuned).
_NEW_STACKS_WITH_VAL = frozenset({"new", "new_claims"})
TARGET_STACKS: tuple[tuple[str, str, str], ...] = (
    ("legacy", "TARGET_2", "TARGET_3_SEV"),
    ("new", "TARGET_FREQ", "TARGET_SEV"),
    ("new_claims", "TARGET_FREQ_CLAIMS", "TARGET_SEV_CLAIMS"),
)


@dataclass(frozen=True)
class TripleStackResult:
    """Три обученных стека + сводки метрик и фин. эффекта."""

    trainings: dict[str, TrainingArtifacts]
    metrics_summary: pd.DataFrame
    fin_effects: dict[str, FinEffectResult] | None = None
    fin_effect_summary: pd.DataFrame | None = None


def train_triple_stacks(
    df: pd.DataFrame,
    config: TrainingConfig | None = None,
    *,
    stacks: Iterable[tuple[str, str, str]] = TARGET_STACKS,
    select_stacks: frozenset[str] = frozenset({"new"}),
) -> dict[str, TrainingArtifacts]:
    """Обучить frequency+severity для каждого стека таргетов.

    Feature selection (freq/sev) — только для ``select_stacks`` (по умолчанию ``new``).
    Остальные стеки в режиме MVP+select получают **те же** отобранные фичи
    (иначе legacy/new_claims учились бы на полном MVP-пуле → завышенные метрики).
    """
    base = resolve_features_config(config or TrainingConfig())
    stack_list = list(stacks)
    select_on = bool(
        select_stacks
        and (base.frequency_select_features or base.severity_select_features)
        and base.features_source == "mvp"
    )

    # Сначала стеки с отбором (prefer new), затем остальные.
    if select_on:
        primary = "new" if "new" in select_stacks else next(iter(select_stacks))
        ordered = sorted(
            stack_list,
            key=lambda item: (
                0 if item[0] == primary else 1 if item[0] in select_stacks else 2,
                [name for name, *_ in stack_list].index(item[0]),
            ),
        )
    else:
        ordered = stack_list

    trainings: dict[str, TrainingArtifacts] = {}
    shared_freq: tuple[str, ...] | None = None
    shared_sev: tuple[str, ...] | None = None

    for stack_name, freq_target, sev_target in ordered:
        print(f"Обучение стека {stack_name}: {freq_target} + {sev_target} ...")
        stack_cfg = replace(
            base,
            frequency_target=freq_target,
            severity_target=sev_target,
        )
        if stack_name in _NEW_STACKS_WITH_VAL:
            train_core, val_period, _cal = default_inner_periods_from_train(
                base.train_period
            )
            stack_cfg = replace(
                stack_cfg,
                train_period=train_core,
                val_period=val_period,
                use_train_val_test_split=True,
            )
            print(
                f"  split train/val/test: core={train_core} val={val_period} "
                f"test={base.test_period}"
            )
        elif stack_name == "legacy":
            print("  legacy: train/test (test-tuned baseline, без Val)")
        if select_on and stack_name in select_stacks:
            trainings[stack_name] = train_models(df, stack_cfg)
            # Фиксируем отобранные фичи для остальных стеков.
            if shared_freq is None:
                shared_freq = tuple(trainings[stack_name].frequency_features)
                shared_sev = tuple(trainings[stack_name].severity_features)
                print(
                    f"  select@{stack_name}: "
                    f"freq={len(shared_freq)} sev={len(shared_sev)} → reuse on other stacks"
                )
                from querulus.training.feature_selection_io import save_feature_selection
                from querulus.training.feature_selection_report import (
                    save_feature_selection_report,
                )

                freq_path = save_feature_selection(
                    stack=stack_name,
                    task="frequency",
                    selected_features=list(shared_freq),
                    summary=trainings[stack_name].frequency_feature_selection_summary,
                    importance=trainings[stack_name].frequency_importance,
                    categorical_features=list(
                        trainings[stack_name].frequency_categorical_features
                    ),
                )
                sev_path = save_feature_selection(
                    stack=stack_name,
                    task="severity",
                    selected_features=list(shared_sev),
                    summary=trainings[stack_name].severity_feature_selection_summary,
                    importance=trainings[stack_name].severity_importance,
                    categorical_features=list(
                        trainings[stack_name].severity_categorical_features
                    ),
                )
                print(f"  feature selection saved: {freq_path}")
                print(f"  feature selection saved: {sev_path}")
                art = trainings[stack_name]
                cat_feats = list(
                    dict.fromkeys(
                        [
                            *art.frequency_categorical_features,
                            *art.severity_categorical_features,
                        ]
                    )
                )
                report_path = save_feature_selection_report(
                    df,
                    frequency_features=list(shared_freq),
                    severity_features=list(shared_sev),
                    categorical_features=cat_feats,
                    frequency_importance=art.frequency_importance,
                    severity_importance=art.severity_importance,
                    frequency_target=freq_target,
                    severity_target=sev_target,
                    stack=stack_name,
                    directory=freq_path.parent,
                )
                print(f"  feature selection report: {report_path}")
            continue

        if select_on and shared_freq is not None and shared_sev is not None:
            stack_cfg = replace(
                stack_cfg,
                frequency_features=shared_freq,
                severity_features=shared_sev,
                frequency_select_features=False,
                severity_select_features=False,
            )
        elif stack_name not in select_stacks:
            stack_cfg = replace(
                stack_cfg,
                frequency_select_features=False,
                severity_select_features=False,
            )
        trainings[stack_name] = train_models(df, stack_cfg)
    return trainings


def build_metrics_summary(
    trainings: dict[str, TrainingArtifacts],
    *,
    stack_order: tuple[str, ...] = ("legacy", "new", "new_claims"),
) -> pd.DataFrame:
    """Сводная таблица метрик: task × metric × split → колонки стеков."""
    rows: list[dict[str, object]] = []
    for stack_name, artifacts in trainings.items():
        for task, table in (
            ("frequency", artifacts.frequency_metrics_table),
            ("severity", artifacts.severity_metrics_table),
        ):
            if table is None or table.empty:
                continue
            for record in table.to_dict(orient="records"):
                metric = record.get("metric")
                for split_name in ("train", "val", "test"):
                    value = record.get(split_name)
                    if value is None and split_name == "val":
                        continue
                    rows.append(
                        {
                            "task": task,
                            "metric": metric,
                            "split": split_name,
                            "stack": stack_name,
                            "value": value,
                        }
                    )
    if not rows:
        return pd.DataFrame(columns=["task", "metric", "split", *stack_order])

    long = pd.DataFrame(rows)
    wide = long.pivot_table(
        index=["task", "metric", "split"],
        columns="stack",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    cols = ["task", "metric", "split"] + [c for c in stack_order if c in wide.columns]
    cols += [c for c in wide.columns if c not in cols]
    wide = wide[cols].sort_values(["task", "split", "metric"]).reset_index(drop=True)
    for col in wide.columns:
        if col not in ("task", "metric", "split"):
            wide[col] = wide[col].map(format_metric_value)
    return wide


def run_triple_fin_effects(
    df: pd.DataFrame,
    trainings: dict[str, TrainingArtifacts],
    *,
    split: str = "test",
    loaded_from_checkpoint: bool = True,
    legacy_dataset: bool | None = None,
    stacks: Iterable[tuple[str, str, str]] = TARGET_STACKS,
) -> tuple[dict[str, FinEffectResult], pd.DataFrame]:
    """Фин. эффект по каждому стеку + одна сводная таблица."""
    from querulus.fin_effect.threshold_policy import (
        resolve_or_pick_val_threshold,
        val_index_from_trainings,
    )

    shared_val_index = val_index_from_trainings(trainings)
    results: dict[str, FinEffectResult] = {}
    rows: list[dict[str, object]] = []
    for stack_name, freq_target, sev_target in stacks:
        if stack_name not in trainings:
            continue
        training = trainings[stack_name]
        cfg = resolve_fin_effect_config(
            df,
            frequency_target=freq_target,
            severity_target=sev_target,
            loaded_from_checkpoint=loaded_from_checkpoint,
            legacy_dataset=legacy_dataset,
        )
        thr = resolve_or_pick_val_threshold(
            df,
            training,
            val_index=shared_val_index,
            frequency_target_column=freq_target,
            config=cfg,
        )
        result = run_fin_effect_from_training(
            df,
            training,
            split=split,
            threshold=thr,
            config=cfg,
        )
        results[stack_name] = result
        rows.append(
            {
                "stack": stack_name,
                "frequency_target": freq_target,
                "severity_target": sev_target,
                "fact_mode": cfg.fact_mode,
                "fact_amount_column": (
                    "legacy_psr"
                    if cfg.uses_legacy_psr_fact
                    else cfg.fact_amount_column
                ),
                "best_threshold": thr,
                "fact_effect": result.fact_effect_total,
                "model_effect": result.model_effect_total,
                "net_effect": result.net_effect,
            }
        )
    return results, pd.DataFrame(rows)


def run_triple_stack(
    df: pd.DataFrame,
    config: TrainingConfig | None = None,
    *,
    split: str = "test",
    loaded_from_checkpoint: bool = True,
    legacy_dataset: bool | None = None,
    run_fin_effect: bool = True,
    select_stacks: frozenset[str] = frozenset({"new"}),
) -> TripleStackResult:
    """Обучить 3 стека, собрать сводки метрик и (опционально) фин. эффекта."""
    trainings = train_triple_stacks(df, config, select_stacks=select_stacks)
    metrics_summary = build_metrics_summary(trainings)
    fin_effects: dict[str, FinEffectResult] | None = None
    fin_effect_summary: pd.DataFrame | None = None
    if run_fin_effect:
        fin_effects, fin_effect_summary = run_triple_fin_effects(
            df,
            trainings,
            split=split,
            loaded_from_checkpoint=loaded_from_checkpoint,
            legacy_dataset=legacy_dataset,
        )
    return TripleStackResult(
        trainings=trainings,
        metrics_summary=metrics_summary,
        fin_effects=fin_effects,
        fin_effect_summary=fin_effect_summary,
    )
