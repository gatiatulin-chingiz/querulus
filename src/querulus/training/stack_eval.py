"""Сравнение legacy vs new: классификация на своих таргетах, покрытие severity."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from querulus.fin_effect.calculator import add_premiums_column
from querulus.fin_effect.resolve import resolve_fin_effect_config
from querulus.training.config import TrainingConfig
from querulus.training.pipeline import TrainingArtifacts, frequency_predict_proba, train_models
from querulus.training.severity_training import severity_predict
from querulus.training.target_compare import compare_old_vs_new_targets

_PSR_COL = "TARGET_FREQ_AMOUNT"
_SEV_COL = "TARGET_SEV"
_STACKS = (
    ("legacy", "TARGET_2"),
    ("new", "TARGET_FREQ"),
)


@dataclass(frozen=True)
class StackEvalReport:
    """Сверка меток, freq/sev-метрики, покрытие (новая классификация) и доля планки."""

    label_agreement: pd.DataFrame
    frequency_metrics: pd.DataFrame
    severity_metrics: pd.DataFrame
    coverage: pd.DataFrame
    pred_freq_disagree: pd.DataFrame
    coverage_share: pd.DataFrame


def _predict_features(
    training: TrainingArtifacts,
    df: pd.DataFrame,
    index: pd.Index,
) -> pd.DataFrame:
    """Строки признаков как при обучении (feature_frame / stringify cat)."""
    if training.feature_frame is not None:
        return training.feature_frame.loc[index]
    from querulus.training.pipeline import stringify_categorical_columns

    cats = list(
        dict.fromkeys(
            [
                *training.frequency_categorical_features,
                *training.severity_categorical_features,
            ]
        )
    )
    return stringify_categorical_columns(df.loc[index], cats)


def _holdout_index(
    legacy: TrainingArtifacts,
    new: TrainingArtifacts,
) -> pd.Index:
    """Пересечение test-индексов frequency-сплитов."""
    if legacy.frequency_split is None or new.frequency_split is None:
        raise ValueError("Нужен frequency_split у legacy и new")
    return legacy.frequency_split.x_test.index.intersection(
        new.frequency_split.x_test.index
    )


def _classification_threshold(
    training: TrainingArtifacts,
    *,
    df: pd.DataFrame,
    val_index: pd.Index | None = None,
    frequency_target: str | None = None,
) -> float:
    from querulus.fin_effect.threshold_policy import resolve_or_pick_val_threshold

    return resolve_or_pick_val_threshold(
        df,
        training,
        val_index=val_index,
        frequency_target_column=frequency_target,
    )


def stack_predictions(
    training: TrainingArtifacts,
    df: pd.DataFrame,
    index: pd.Index,
    *,
    val_index: pd.Index | None = None,
    frequency_target: str | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """proba, pred_freq (порог Val), pred_sev на index."""
    thr = _classification_threshold(
        training,
        df=df,
        val_index=val_index,
        frequency_target=frequency_target,
    )
    feats = _predict_features(training, df, index)
    proba = pd.Series(
        frequency_predict_proba(training, feats[training.frequency_features]),
        index=index,
        dtype=float,
    )
    pred_freq = (proba >= thr).astype(int)
    sev_raw = severity_predict(
        training.severity_model,
        feats[training.severity_features],
        training.severity_categorical_features,
        transform=getattr(training, "severity_target_transform", "raw"),
    )
    sev_calibrator = getattr(training, "severity_calibrator", None)
    if sev_calibrator is not None:
        from querulus.training.calibration import apply_severity_calibrator

        sev_raw = apply_severity_calibrator(sev_calibrator, sev_raw)
    pred_sev = pd.Series(np.asarray(sev_raw, dtype=float), index=index, dtype=float)
    return proba, pred_freq, pred_sev


def _gini(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Gini frequency: Lorenz по proba (``collect_metrics.classification_gini``)."""
    from querulus.training.collect_metrics import classification_gini

    return classification_gini(y_true, proba)


def _freq_metrics_row(
    stack: str,
    y_true: pd.Series,
    proba: pd.Series,
    pred: pd.Series,
    *,
    split_label: str = "test",
    threshold: float,
) -> dict[str, object]:
    """PR-AUC, ROC-AUC, Gini, shift (= pred+/fact+) при пороге с Val."""
    y = y_true.astype(int)
    y_np = y.to_numpy()
    proba_np = np.asarray(proba, dtype=float)
    p = pred.astype(int)
    fact_pos = float(y_np.sum())
    try:
        pr_auc = float(average_precision_score(y_np, proba_np))
    except ValueError:
        pr_auc = float("nan")
    try:
        roc_auc = float(roc_auc_score(y_np, proba_np))
    except ValueError:
        roc_auc = float("nan")
    return {
        "stack": stack,
        "split": split_label,
        "y_true": y.name,
        "n": int(len(y)),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "gini": _gini(y_np, proba_np),
        "shift": float(p.sum() / fact_pos) if fact_pos > 0 else float("nan"),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "threshold": threshold,
    }


def _sev_metrics_row(
    stack: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    split_label: str = "test",
) -> dict[str, object]:
    """MAE, RMSE, R², shift (= Σpred / Σfact) на том же holdout."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    n = int(len(yt))
    fact_sum = float(np.nansum(yt))
    try:
        r2 = float(r2_score(yt, yp)) if n > 1 else float("nan")
    except ValueError:
        r2 = float("nan")
    err = yp - yt
    return {
        "stack": stack,
        "split": split_label,
        "y_true": y_true.name,
        "n": n,
        "mae": float(mean_absolute_error(yt, yp)) if n else float("nan"),
        "rmse": float(np.sqrt(np.mean(err**2))) if n else float("nan"),
        "r2": r2,
        "shift": float(np.nansum(yp) / fact_sum) if fact_sum > 0 else float("nan"),
    }


def _coverage_table(
    stack: str,
    y_true: pd.Series,
    pred_freq: pd.Series,
    pred_sev: pd.Series,
    target_sev: pd.Series,
    psr: pd.Series,
    premiums: pd.Series,
) -> pd.DataFrame:
    """Агрегат n и ₽ по новой формуле (расход отрицательный)."""
    from querulus.fin_effect.calculator import compute_fin_effect_model_coverage

    fact = y_true.to_numpy(dtype=int)
    pred = pred_freq.to_numpy(dtype=int)
    sev_p = np.nan_to_num(pred_sev.to_numpy(dtype=float), nan=0.0)
    sev_t = np.nan_to_num(target_sev.to_numpy(dtype=float), nan=0.0)
    psr_v = np.nan_to_num(psr.to_numpy(dtype=float), nan=0.0)
    prem_v = np.nan_to_num(premiums.to_numpy(dtype=float), nan=0.0)
    amount = compute_fin_effect_model_coverage(pred, fact, sev_p, sev_t, psr_v, prem_v)

    m00 = (fact == 0) & (pred == 0)
    m01 = (fact == 0) & (pred == 1)
    m10 = (fact == 1) & (pred == 0)
    m11 = (fact == 1) & (pred == 1)
    covered = m11 & (sev_p >= sev_t)
    short = m11 & ~covered

    order = (
        ("0-0", m00, "ноль"),
        ("0-1", m01, "расход"),
        ("1-0", m10, "расход"),
        ("1-1 хватило", covered, "расход"),
        ("1-1 не хватило", short, "расход"),
    )
    rows = [
        {
            "stack": stack,
            "outcome": name,
            "n": int(mask.sum()),
            "amount": float(amount[mask].sum()),
            "kind": kind,
        }
        for name, mask, kind in order
    ]
    rows.append(
        {
            "stack": stack,
            "outcome": "модель всего",
            "n": int(len(fact)),
            "amount": float(amount.sum()),
            "kind": "расход",
        }
    )
    return pd.DataFrame(rows)


def _coverage_share_table(
    pred_sev_by_stack: dict[str, pd.Series],
    target_sev: pd.Series,
    mask: pd.Series,
) -> pd.DataFrame:
    """Доля покрытой планки TARGET_SEV на строках 1-1 (новая классификация)."""
    t = np.nan_to_num(target_sev.loc[mask].to_numpy(dtype=float), nan=0.0)
    sum_t = float(t.sum())
    rows: list[dict[str, object]] = []
    by_stack: dict[str, dict[str, object]] = {}
    for stack, pred_sev in pred_sev_by_stack.items():
        p = np.nan_to_num(pred_sev.loc[mask].to_numpy(dtype=float), nan=0.0)
        covered_amt = np.minimum(np.maximum(p, 0.0), np.maximum(t, 0.0))
        under = np.maximum(t - p, 0.0)
        row = {
            "stack": stack,
            "n": int(mask.sum()),
            "n_covered": int((p >= t).sum()),
            "share_rows_covered": float((p >= t).mean()) if len(t) else float("nan"),
            "share_amount_covered": float(covered_amt.sum() / sum_t) if sum_t else float("nan"),
            "share_under": float(under.sum() / sum_t) if sum_t else float("nan"),
        }
        by_stack[stack] = row
        rows.append(row)
    if "legacy" in by_stack and "new" in by_stack:
        leg = by_stack["legacy"]
        neu = by_stack["new"]

        def _delta(key: str) -> float:
            a, b = neu[key], leg[key]
            if a is None or b is None:
                return float("nan")
            try:
                return float(a) - float(b)
            except (TypeError, ValueError):
                return float("nan")

        rows.append(
            {
                "stack": "new − legacy",
                "n": int(mask.sum()),
                "n_covered": int(neu["n_covered"]) - int(leg["n_covered"]),
                "share_rows_covered": _delta("share_rows_covered"),
                "share_amount_covered": _delta("share_amount_covered"),
                "share_under": _delta("share_under"),
            }
        )
    return pd.DataFrame(rows)


def _pred_freq_disagree_table(
    pred_legacy: pd.Series,
    pred_new: pd.Series,
) -> pd.DataFrame:
    """Сводка расхождений pred_freq: счётчики и доли по направлениям."""
    n = int(len(pred_legacy))
    leg1_new0 = (pred_legacy == 1) & (pred_new == 0)
    leg0_new1 = (pred_legacy == 0) & (pred_new == 1)
    disagree = pred_legacy.ne(pred_new)
    n_disagree = int(disagree.sum())
    n_l1n0 = int(leg1_new0.sum())
    n_l0n1 = int(leg0_new1.sum())
    return pd.DataFrame(
        [
            {"показатель": "строк holdout", "значение": n},
            {"показатель": "pred_freq не совпали", "значение": n_disagree},
            {
                "показатель": "доля несовпадений",
                "значение": float(disagree.mean()) if n else float("nan"),
            },
            {"показатель": "legacy=1, new=0 (n)", "значение": n_l1n0},
            {"показатель": "legacy=0, new=1 (n)", "значение": n_l0n1},
            {
                "показатель": "legacy=1, new=0 (доля holdout)",
                "значение": float(n_l1n0 / n) if n else float("nan"),
            },
            {
                "показатель": "legacy=0, new=1 (доля holdout)",
                "значение": float(n_l0n1 / n) if n else float("nan"),
            },
            {
                "показатель": "legacy=1, new=0 (доля среди несовпадений)",
                "значение": float(n_l1n0 / n_disagree) if n_disagree else float("nan"),
            },
            {
                "показатель": "legacy=0, new=1 (доля среди несовпадений)",
                "значение": float(n_l0n1 / n_disagree) if n_disagree else float("nan"),
            },
        ]
    )


def _premiums_on_index(df: pd.DataFrame, index: pd.Index) -> pd.Series:
    """Взносы ФУ на holdout (колонка Взносы или расчёт payments_fee)."""
    cfg = resolve_fin_effect_config(df, frequency_target="TARGET_FREQ", severity_target="TARGET_SEV")
    if cfg.premiums_column in df.columns:
        return pd.to_numeric(df.loc[index, cfg.premiums_column], errors="coerce").fillna(0.0)
    work = df.loc[index]
    return add_premiums_column(work, cfg)


def _catboost_params_from_model(model: object) -> dict[str, object]:
    """Гиперпараметры обученной CatBoost-модели для повторного fit."""
    raw = dict(model.get_params())  # type: ignore[attr-defined]
    raw.pop("train_dir", None)
    raw["allow_writing_files"] = False
    raw["verbose"] = False
    return {key: value for key, value in raw.items() if value is not None}


def _retrain_freq_on_target(
    df: pd.DataFrame,
    training: TrainingArtifacts,
    *,
    target_column: str,
    eval_index: pd.Index,
    threshold: float,
) -> tuple[pd.Series, pd.Series]:
    """Переобучить freq clf (фичи + hparams training) на ``target_column``; score на eval.

    Окно обучения — ``frequency_split.x_train`` той же модели (без подглядывания в eval,
    если eval пересекается с train — строки eval из train исключаются из fit).
    """
    from querulus.training.pipeline import make_pool, require_catboost

    if training.frequency_split is None:
        raise ValueError("Для retrain нужен frequency_split у legacy")
    if target_column not in df.columns:
        raise KeyError(f"Нет колонки {target_column}")

    CatBoostClassifier, *_ = require_catboost()
    feats = list(training.frequency_features)
    cats = [c for c in training.frequency_categorical_features if c in feats]
    train_idx = training.frequency_split.x_train.index.difference(eval_index)
    if train_idx.empty:
        raise ValueError("Пустой train после исключения eval_index для retrain")

    if training.feature_frame is not None:
        x_tr = training.feature_frame.loc[train_idx, feats]
    else:
        x_tr = training.frequency_split.x_train.loc[train_idx, feats]
    y_tr = df.loc[train_idx, target_column].fillna(0).astype(int)

    eval_set = None
    test_idx = training.frequency_split.x_test.index.difference(eval_index)
    if len(test_idx) > 0:
        if training.feature_frame is not None:
            x_va = training.feature_frame.loc[test_idx, feats]
        else:
            x_va = training.frequency_split.x_test.loc[test_idx, feats]
        y_va = df.loc[test_idx, target_column].fillna(0).astype(int)
        eval_set = make_pool(
            x_va, y_va, cat_features=cats, feature_names=feats
        )

    model = CatBoostClassifier(**_catboost_params_from_model(training.frequency_model))
    train_pool = make_pool(x_tr, y_tr, cat_features=cats, feature_names=feats)
    fit_kwargs: dict[str, object] = {"plot": False}
    if eval_set is not None:
        fit_kwargs["eval_set"] = eval_set
    model.fit(train_pool, **fit_kwargs)

    x_ev = _predict_features(training, df, eval_index)[feats]
    eval_pool = make_pool(x_ev, None, cat_features=cats, feature_names=feats)
    proba = pd.Series(
        np.asarray(model.predict_proba(eval_pool)[:, 1], dtype=float),
        index=eval_index,
        dtype=float,
    )
    thr = float(threshold)
    pred = (proba >= thr).astype(int)
    return proba, pred


def score_stack_on_index(
    df: pd.DataFrame,
    training: TrainingArtifacts,
    index: pd.Index,
    *,
    stack: str,
    freq_target: str,
    sev_target: str = _SEV_COL,
    psr_col: str = _PSR_COL,
    val_index: pd.Index | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Freq-метрики и ₽-квадранты одного стека на фиксированном index."""
    if index.empty:
        raise ValueError("Пустой holdout index")
    holdout = df.loc[index]
    for col in (freq_target, sev_target, psr_col):
        if col not in holdout.columns:
            raise KeyError(f"Для оценки нужна колонка {col}")
    proba, pred_freq, pred_sev = stack_predictions(
        training,
        df,
        index,
        val_index=val_index,
        frequency_target=freq_target,
    )
    y_true = holdout[freq_target].fillna(0).astype(int)
    y_true.name = freq_target
    freq_row = _freq_metrics_row(
        stack,
        y_true,
        proba,
        pred_freq,
        threshold=_classification_threshold(
            training,
            df=df,
            val_index=val_index,
            frequency_target=freq_target,
        ),
    )
    coverage = _coverage_table(
        stack,
        y_true,
        pred_freq,
        pred_sev,
        holdout[sev_target].fillna(0),
        holdout[psr_col].fillna(0),
        _premiums_on_index(df, index),
    )
    return freq_row, coverage


def train_legacy_matching_new(
    df: pd.DataFrame,
    config: TrainingConfig,
    *,
    frequency_features: tuple[str, ...] | list[str],
    severity_features: tuple[str, ...] | list[str],
) -> TrainingArtifacts:
    """Обучить legacy (TARGET_2 / TARGET_3_SEV) на тех же фичах и окнах, что new."""
    cfg = replace(
        config,
        frequency_target="TARGET_2",
        severity_target="TARGET_3_SEV",
        frequency_features=tuple(frequency_features),
        severity_features=tuple(severity_features),
        frequency_select_features=False,
        severity_select_features=False,
    )
    return train_models(df, cfg)


def evaluate_legacy_vs_new(
    df: pd.DataFrame,
    legacy: TrainingArtifacts,
    new: TrainingArtifacts,
    *,
    index: pd.Index | None = None,
) -> StackEvalReport:
    """Holdout: freq-метрики (свои метки + legacy retrain на TARGET_FREQ); покрытие.

    Строки classification:
    - ``legacy`` — старые фичи/hparams, метка TARGET_2;
    - ``legacy feats/hparams @ TARGET_FREQ`` — те же фичи/hparams, переобучение на TARGET_FREQ;
    - ``new`` — новые фичи/hparams, метка TARGET_FREQ.

    Покрытие обеих регрессий — на ``pred_freq`` new. ``index`` — явный holdout
    (после B: ``splits.test``), иначе пересечение ``frequency_split.x_test``.
    """
    for col in ("TARGET_2", "TARGET_FREQ", "TARGET_3_SEV", _SEV_COL, _PSR_COL):
        if col not in df.columns:
            raise KeyError(f"Для сравнения нужна колонка {col}")

    index = _holdout_index(legacy, new) if index is None else index
    if index.empty:
        raise ValueError("Пустой holdout для сравнения legacy vs new")

    holdout = df.loc[index]
    agreement = compare_old_vs_new_targets(
        holdout,
        pairs=[("TARGET_2", "TARGET_FREQ")],
        quiet=True,
    ).report

    from querulus.fin_effect.threshold_policy import val_index_from_training

    val_index = val_index_from_training(new)

    freq_rows: list[dict[str, object]] = []
    trainings = {"legacy": legacy, "new": new}

    def _append_stack_rows(eval_index: pd.Index, split_label: str, stacks: tuple[str, ...]) -> None:
        part = df.loc[eval_index]
        for stack in stacks:
            y_col = _STACKS[0][1] if stack == "legacy" else _STACKS[1][1]
            proba, pred_freq, _ = stack_predictions(
                trainings[stack],
                df,
                eval_index,
                val_index=val_index,
                frequency_target=y_col,
            )
            y_true = part[y_col].fillna(0).astype(int)
            y_true.name = y_col
            thr = _classification_threshold(
                trainings[stack],
                df=df,
                val_index=val_index,
                frequency_target=y_col,
            )
            freq_rows.append(
                _freq_metrics_row(
                    stack,
                    y_true,
                    proba,
                    pred_freq,
                    split_label=split_label,
                    threshold=thr,
                )
            )

    _append_stack_rows(index, "test", ("legacy", "new"))

    if val_index is not None and len(val_index) > 0:
        _append_stack_rows(val_index, "val", ("legacy", "new"))

    preds: dict[str, tuple[pd.Series, pd.Series, pd.Series]] = {}
    for stack, y_col in _STACKS:
        proba, pred_freq, pred_sev = stack_predictions(
            trainings[stack],
            df,
            index,
            val_index=val_index,
            frequency_target=y_col,
        )
        preds[stack] = (proba, pred_freq, pred_sev)

    # legacy retrain on TARGET_FREQ — только test holdout (test-tuned baseline).
    y_freq = holdout["TARGET_FREQ"].fillna(0).astype(int)
    y_freq.name = "TARGET_FREQ"
    thr_freq = _classification_threshold(
        new,
        df=df,
        val_index=val_index,
        frequency_target="TARGET_FREQ",
    )
    proba_re, pred_re = _retrain_freq_on_target(
        df,
        legacy,
        target_column="TARGET_FREQ",
        eval_index=index,
        threshold=thr_freq,
    )
    freq_rows.insert(
        1,
        _freq_metrics_row(
            "legacy feats/hparams @ TARGET_FREQ",
            y_freq,
            proba_re,
            pred_re,
            split_label="test",
            threshold=thr_freq,
        ),
    )

    _, pred_freq_new, pred_sev_new = preds["new"]
    _, pred_l, pred_sev_legacy = preds["legacy"]
    premiums = _premiums_on_index(df, index)
    psr = holdout[_PSR_COL].fillna(0)
    target_sev = holdout[_SEV_COL].fillna(0)
    cov_parts = [
        _coverage_table(
            stack,
            y_freq,
            pred_freq_new,
            pred_sev,
            target_sev,
            psr,
            premiums,
        )
        for stack, pred_sev in (("legacy", pred_sev_legacy), ("new", pred_sev_new))
    ]
    one_one = y_freq.eq(1) & pred_freq_new.eq(1)
    coverage_share = _coverage_share_table(
        {"legacy": pred_sev_legacy, "new": pred_sev_new},
        target_sev,
        one_one,
    )
    pred_freq_disagree = _pred_freq_disagree_table(pred_l, pred_freq_new)

    y_3 = holdout["TARGET_3_SEV"].fillna(0) if "TARGET_3_SEV" in holdout.columns else target_sev
    y_3.name = "TARGET_3_SEV"
    y_sev = target_sev.copy()
    y_sev.name = "TARGET_SEV"
    sev_rows = [
        _sev_metrics_row("legacy", y_3, pred_sev_legacy, split_label="test"),
        _sev_metrics_row("legacy @ TARGET_SEV", y_sev, pred_sev_legacy, split_label="test"),
        _sev_metrics_row("new", y_sev, pred_sev_new, split_label="test"),
    ]

    return StackEvalReport(
        label_agreement=agreement,
        frequency_metrics=pd.DataFrame(freq_rows),
        severity_metrics=pd.DataFrame(sev_rows),
        coverage=pd.concat(cov_parts, ignore_index=True),
        pred_freq_disagree=pred_freq_disagree,
        coverage_share=coverage_share,
    )
