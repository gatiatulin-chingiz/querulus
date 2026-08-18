"""Сравнение legacy vs new: классификация на своих таргетах, покрытие severity."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from querulus.training.pipeline import TrainingArtifacts, frequency_predict_proba
from querulus.training.severity_training import severity_predict
from querulus.training.target_compare import compare_old_vs_new_targets

FREQ_THRESHOLD = 0.5
_PSR_COL = "TARGET_FREQ_AMOUNT"
_SEV_COL = "TARGET_SEV"
_STACKS = (
    ("legacy", "TARGET_2"),
    ("new", "TARGET_FREQ"),
)


@dataclass(frozen=True)
class StackEvalReport:
    """Три таблицы: сверка меток, freq-метрики, ₽-квадранты + расхождения pred_freq."""

    label_agreement: pd.DataFrame
    frequency_metrics: pd.DataFrame
    coverage: pd.DataFrame
    pred_freq_disagree: pd.DataFrame


def _predict_features(
    training: TrainingArtifacts,
    df: pd.DataFrame,
    index: pd.Index,
) -> pd.DataFrame:
    """Строки признаков как при обучении (feature_frame / stringify cat)."""
    if training.feature_frame is not None:
        return training.feature_frame.loc[index]
    from querulus.training.pipeline import _stringify_categorical_columns

    cats = list(
        dict.fromkeys(
            [
                *training.frequency_categorical_features,
                *training.severity_categorical_features,
            ]
        )
    )
    return _stringify_categorical_columns(df.loc[index], cats)


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


def _stack_predictions(
    training: TrainingArtifacts,
    df: pd.DataFrame,
    index: pd.Index,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """proba, pred_freq (порог 0.5), pred_sev на index."""
    feats = _predict_features(training, df, index)
    proba = pd.Series(
        frequency_predict_proba(training, feats[training.frequency_features]),
        index=index,
        dtype=float,
    )
    pred_freq = (proba >= FREQ_THRESHOLD).astype(int)
    pred_sev = pd.Series(
        severity_predict(
            training.severity_model,
            feats[training.severity_features],
            training.severity_categorical_features,
            transform=getattr(training, "severity_target_transform", "raw"),
        ),
        index=index,
        dtype=float,
    )
    return proba, pred_freq, pred_sev


def _freq_metrics_row(
    stack: str,
    y_true: pd.Series,
    proba: pd.Series,
    pred: pd.Series,
) -> dict[str, object]:
    """PR-AUC и метрики при пороге 0.5."""
    y = y_true.astype(int)
    p = pred.astype(int)
    try:
        pr_auc = float(average_precision_score(y, proba))
    except ValueError:
        pr_auc = float("nan")
    return {
        "stack": stack,
        "y_true": y.name,
        "n": int(len(y)),
        "pr_auc": pr_auc,
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "threshold": FREQ_THRESHOLD,
    }


def _coverage_table(
    stack: str,
    y_true: pd.Series,
    pred_freq: pd.Series,
    pred_sev: pd.Series,
    target_sev: pd.Series,
    psr: pd.Series,
) -> pd.DataFrame:
    """Агрегат n и ₽ по квадрантам одного стека."""
    fact = y_true.to_numpy(dtype=int)
    pred = pred_freq.to_numpy(dtype=int)
    sev_p = np.nan_to_num(pred_sev.to_numpy(dtype=float), nan=0.0)
    sev_t = np.nan_to_num(target_sev.to_numpy(dtype=float), nan=0.0)
    psr_v = np.nan_to_num(psr.to_numpy(dtype=float), nan=0.0)

    m00 = (fact == 0) & (pred == 0)
    m01 = (fact == 0) & (pred == 1)
    m10 = (fact == 1) & (pred == 0)
    m11 = (fact == 1) & (pred == 1)
    covered = m11 & (sev_p >= sev_t)
    short = m11 & ~covered

    amount = np.zeros(len(fact), dtype=float)
    amount[m01] = sev_p[m01]
    amount[m10] = psr_v[m10]
    amount[covered] = sev_t[covered]
    share = np.zeros(len(fact), dtype=float)
    need_share = short & (sev_t > 0)
    share[need_share] = np.clip(sev_p[need_share] / sev_t[need_share], 0.0, 1.0)
    amount[short] = psr_v[short] * (1.0 - share[short])

    order = (
        ("0-0", m00, "ноль"),
        ("0-1", m01, "штраф"),
        ("1-0", m10, "штраф"),
        ("1-1 хватило", covered, "покрытие"),
        ("1-1 не хватило", short, "штраф"),
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
    penalty_mask = m01 | m10 | short
    rows.append(
        {
            "stack": stack,
            "outcome": "штраф всего",
            "n": int(penalty_mask.sum()),
            "amount": float(amount[penalty_mask].sum()),
            "kind": "штраф",
        }
    )
    return pd.DataFrame(rows)


def evaluate_legacy_vs_new(
    df: pd.DataFrame,
    legacy: TrainingArtifacts,
    new: TrainingArtifacts,
) -> StackEvalReport:
    """Holdout: метки, freq-метрики на своих таргетах, ₽-квадранты vs TARGET_SEV / PSR."""
    for col in ("TARGET_2", "TARGET_FREQ", _SEV_COL, _PSR_COL):
        if col not in df.columns:
            raise KeyError(f"Для сравнения нужна колонка {col}")

    index = _holdout_index(legacy, new)
    if index.empty:
        raise ValueError("Пустое пересечение test-индексов legacy и new")

    holdout = df.loc[index]
    agreement = compare_old_vs_new_targets(
        holdout,
        pairs=[("TARGET_2", "TARGET_FREQ")],
        quiet=True,
    ).report

    preds: dict[str, tuple[pd.Series, pd.Series, pd.Series]] = {}
    freq_rows: list[dict[str, object]] = []
    cov_parts: list[pd.DataFrame] = []
    trainings = {"legacy": legacy, "new": new}
    for stack, y_col in _STACKS:
        proba, pred_freq, pred_sev = _stack_predictions(trainings[stack], df, index)
        preds[stack] = (proba, pred_freq, pred_sev)
        y_true = holdout[y_col].fillna(0).astype(int)
        y_true.name = y_col
        freq_rows.append(_freq_metrics_row(stack, y_true, proba, pred_freq))
        cov_parts.append(
            _coverage_table(
                stack,
                y_true,
                pred_freq,
                pred_sev,
                holdout[_SEV_COL].fillna(0),
                holdout[_PSR_COL].fillna(0),
            )
        )

    _, pred_l, _ = preds["legacy"]
    _, pred_n, _ = preds["new"]
    disagree = pred_l.ne(pred_n)
    pred_freq_disagree = pd.DataFrame(
        [
            {
                "n_holdout": int(len(index)),
                "n_pred_freq_disagree": int(disagree.sum()),
                "share_disagree": float(disagree.mean()),
                "legacy_pos_new_neg": int(((pred_l == 1) & (pred_n == 0)).sum()),
                "legacy_neg_new_pos": int(((pred_l == 0) & (pred_n == 1)).sum()),
            }
        ]
    )
    return StackEvalReport(
        label_agreement=agreement,
        frequency_metrics=pd.DataFrame(freq_rows),
        coverage=pd.concat(cov_parts, ignore_index=True),
        pred_freq_disagree=pred_freq_disagree,
    )
