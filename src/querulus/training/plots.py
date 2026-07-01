"""Визуализации EDA и диагностики моделей (как в model_learn.py)."""
from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    roc_auc_score,
)

from querulus.training.config import TrainingConfig
from querulus.training.pipeline import TrainingArtifacts
from querulus.training.research_eda import research_continous, research_feature

ModelType = Literal["frequency", "severity"]


def _ensure_exposure(df: pd.DataFrame, exposure_col: str = "expos") -> pd.DataFrame:
    """Добавить колонку экспозиции для EDA, как в Litigant."""
    result = df.copy()
    if exposure_col not in result.columns:
        result[exposure_col] = 1
    return result


def plot_frequency_probability_diagnostics(
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred_train: np.ndarray,
    y_pred_test: np.ndarray,
) -> None:
    """Распределение вероятностей, разделимость классов, ROC/PR — как model_learn.py."""
    plt.figure(figsize=(12, 5))
    sns.histplot(
        y_pred_train,
        bins=50,
        alpha=0.5,
        label=f"Train (n={len(y_pred_train):,})",
        color="blue",
        stat="density",
    )
    sns.histplot(
        y_pred_test,
        bins=50,
        alpha=0.5,
        label=f"Test (n={len(y_pred_test):,})",
        color="green",
        stat="density",
    )
    plt.axvline(x=0.5, color="red", linestyle="--", label="Порог 0.5")
    plt.xlabel("Predicted Probability (класс 1)")
    plt.ylabel("Плотность (density)")
    plt.title("Распределение предсказанных вероятностей: Train vs Test")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(
        y_pred_train[y_train == 0],
        bins=50,
        alpha=0.5,
        label="Train",
        color="blue",
        ax=axes[0],
        stat="density",
    )
    sns.histplot(
        y_pred_test[y_test == 0],
        bins=50,
        alpha=0.5,
        label="Test",
        color="green",
        ax=axes[0],
        stat="density",
    )
    axes[0].axvline(x=0.5, color="red", linestyle="--")
    axes[0].set_title("Класс 0 (негативные)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    sns.histplot(
        y_pred_train[y_train == 1],
        bins=50,
        alpha=0.5,
        label="Train",
        color="blue",
        ax=axes[1],
        stat="density",
    )
    sns.histplot(
        y_pred_test[y_test == 1],
        bins=50,
        alpha=0.5,
        label="Test",
        color="green",
        ax=axes[1],
        stat="density",
    )
    axes[1].axvline(x=0.5, color="red", linestyle="--")
    axes[1].set_title("Класс 1 (позитивные)")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    plt.suptitle("Распределение вероятностей по классам", y=1.02)
    plt.tight_layout()
    plt.show()

    def _calc_stats(probs: np.ndarray, labels: pd.Series, name: str) -> float:
        neg = probs[labels == 0]
        pos = probs[labels == 1]
        print(f"\n{name}:")
        print(f"  Класс 0: mean={neg.mean():.3f}, std={neg.std():.3f}, median={np.median(neg):.3f}")
        print(f"  Класс 1: mean={pos.mean():.3f}, std={pos.std():.3f}, median={np.median(pos):.3f}")
        print(f"  Разрыв mean: {pos.mean() - neg.mean():.3f}")
        return float(pos.mean() - neg.mean())

    print("=" * 50)
    print("СТАТИСТИКА РАЗДЕЛИМОСТИ КЛАССОВ")
    print("=" * 50)
    gap_train = _calc_stats(y_pred_train, y_train, "Train")
    gap_test = _calc_stats(y_pred_test, y_test, "Test")
    print(f"\nРазница в разрыве (test - train): {gap_test - gap_train:+.3f}")

    ap_train = average_precision_score(y_train, y_pred_train)
    ap_test = average_precision_score(y_test, y_pred_test)
    auc_train = roc_auc_score(y_train, y_pred_train)
    auc_test = roc_auc_score(y_test, y_pred_test)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    PrecisionRecallDisplay.from_predictions(y_train, y_pred_train, ax=axes[0], label=f"Train (AP={ap_train:.3f})")
    PrecisionRecallDisplay.from_predictions(y_test, y_pred_test, ax=axes[0], label=f"Test (AP={ap_test:.3f})")
    axes[0].set_title("Precision-Recall Curve")
    axes[0].grid(alpha=0.3)

    RocCurveDisplay.from_predictions(y_train, y_pred_train, ax=axes[1], label=f"Train (AUC={auc_train:.3f})")
    RocCurveDisplay.from_predictions(y_test, y_pred_test, ax=axes[1], label=f"Test (AUC={auc_test:.3f})")
    axes[1].set_title("ROC Curve")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 50)
    print("СВОДКА ПО МЕТРИКАМ")
    print("=" * 50)
    print(f"Average Precision: Train={ap_train:.3f}, Test={ap_test:.3f}, Δ={ap_test - ap_train:+.3f}")
    print(f"ROC-AUC:           Train={auc_train:.3f}, Test={auc_test:.3f}, Δ={auc_test - auc_train:+.3f}")


def _plot_features_eda(
    df: pd.DataFrame,
    features: list[str],
    cat_features: list[str],
    *,
    model_type: ModelType,
    numeric_bins: int = 10,
    figsize: tuple[float, float] = (25, 10),
    rotation: int = 90,
    max_categories: int = 2000,
    skip_geo: bool = True,
) -> None:
    """Построить research_* графики по списку признаков (оригинальный API Litigant)."""
    cat_set = set(cat_features)
    geo_columns = {"LONGITUDE", "LATITUDE"}
    for column in features:
        if skip_geo and column in geo_columns:
            continue
        if column not in df.columns:
            continue
        if column in cat_set:
            print(column, "nulls:", df[column].isnull().mean() * 100, "%")
            print(column, "nunique:", df[column].nunique())
            if df[column].nunique(dropna=False) >= max_categories:
                continue
            research_feature(
                df.copy(),
                column,
                model_type=model_type,
                figsize=figsize,
                rotation=rotation,
            )
        else:
            print(column, "nulls:", df[column].isnull().mean() * 100, "%")
            try:
                research_continous(
                    df.copy(),
                    column,
                    numeric_bins,
                    model_type=model_type,
                    figsize=figsize,
                    rotation=rotation,
                )
            except (ValueError, TypeError):
                df = df.copy()
                df[column] = pd.to_numeric(df[column], errors="coerce")
                research_continous(
                    df,
                    column,
                    numeric_bins,
                    model_type=model_type,
                    figsize=figsize,
                    rotation=rotation,
                )


def run_mvp_frequency_eda(
    df: pd.DataFrame,
    config: TrainingConfig | None = None,
    *,
    numeric_bins: int = 10,
    figsize: tuple[float, float] = (25, 10),
    rotation: int = 90,
    max_categories: int = 2000,
) -> None:
    """EDA по MVP-признакам до обучения (как model_learn.py:406-427)."""
    config = config or TrainingConfig()
    plot_df = _ensure_exposure(df)
    types = config.mvp_input_types
    geo_columns = {"LONGITUDE", "LATITUDE"}

    numeric_cols = [col for col in types["NUMERIC"] if col not in geo_columns and col in plot_df.columns]
    binary_cols = [col for col in types["BINARY"] if col in plot_df.columns]
    categorical_cols = [col for col in types["CATEGORIAL"] if col in plot_df.columns]

    print("\n=== Frequency MVP EDA (numeric) ===")
    _plot_features_eda(
        plot_df,
        numeric_cols,
        [],
        model_type="frequency",
        numeric_bins=numeric_bins,
        figsize=figsize,
        rotation=rotation,
        max_categories=max_categories,
        skip_geo=False,
    )

    print("\n=== Frequency MVP EDA (binary) ===")
    _plot_features_eda(
        plot_df,
        binary_cols,
        binary_cols,
        model_type="frequency",
        numeric_bins=numeric_bins,
        figsize=figsize,
        rotation=rotation,
        max_categories=max_categories,
        skip_geo=False,
    )

    print("\n=== Frequency MVP EDA (categorical) ===")
    _plot_features_eda(
        plot_df,
        categorical_cols,
        categorical_cols,
        model_type="frequency",
        numeric_bins=numeric_bins,
        figsize=figsize,
        rotation=rotation,
        max_categories=max_categories,
        skip_geo=False,
    )


def _severity_plot_frame(df: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
    """Оставить строки в диапазоне severity, как при обучении регрессии."""
    data = df.copy()
    data[config.date_column] = pd.to_datetime(data[config.date_column])
    return data[data[config.severity_target].between(*config.severity_range)]


def run_model_diagnostics_visualizations(
    df: pd.DataFrame,
    training: TrainingArtifacts,
    config: TrainingConfig | None = None,
    *,
    plot_frequency_probabilities: bool = True,
    plot_model_diagnostics: bool = True,
    plot_severity_eda: bool = True,
    severity_eda_features: list[str] | None = None,
) -> None:
    """Графики после обучения: вероятности, ModelDiagnostics, severity EDA по importance."""
    config = config or TrainingConfig()
    plot_df = _ensure_exposure(df)

    if plot_frequency_probabilities and training.frequency_split is not None:
        print("\n=== Frequency probability diagnostics ===")
        features = training.frequency_features
        x_train = training.frequency_split.x_train[features]
        x_test = training.frequency_split.x_test[features]
        y_train = training.frequency_split.y_train
        y_test = training.frequency_split.y_test
        y_pred_train = training.frequency_model.predict_proba(x_train)[:, 1]
        y_pred_test = training.frequency_model.predict_proba(x_test)[:, 1]
        plot_frequency_probability_diagnostics(y_train, y_test, y_pred_train, y_pred_test)

    if plot_model_diagnostics and training.frequency_split is not None and training.severity_split is not None:
        if training.frequency_diagnostics is not None:
            proba_train = training.frequency_model.predict_proba(
                training.frequency_split.x_train[training.frequency_features]
            )[:, 1]
            proba_test = training.frequency_model.predict_proba(
                training.frequency_split.x_test[training.frequency_features]
            )[:, 1]
            training.frequency_diagnostics.diagnostics_plots(
                training.frequency_split.y_train, proba_train, title_prefix="Train"
            )
            training.frequency_diagnostics.diagnostics_plots(
                training.frequency_split.y_test, proba_test, title_prefix="Test"
            )
        if training.severity_diagnostics is not None:
            pred_train = training.severity_model.predict(
                training.severity_split.x_train[training.severity_features]
            )
            pred_test = training.severity_model.predict(
                training.severity_split.x_test[training.severity_features]
            )
            training.severity_diagnostics.diagnostics_plots(
                training.severity_split.y_train, pred_train, title_prefix="Train"
            )
            training.severity_diagnostics.diagnostics_plots(
                training.severity_split.y_test, pred_test, title_prefix="Test"
            )

    if plot_severity_eda:
        print("\n=== Severity feature EDA (importance) ===")
        severity_df = _severity_plot_frame(plot_df, config)
        if severity_eda_features is None:
            severity_eda_features = training.severity_importance["feature"].tolist()
        severity_eda_features = [
            column for column in severity_eda_features if column in severity_df.columns
        ]
        severity_cat = set(training.severity_categorical_features)
        for column in severity_eda_features:
            if column in severity_cat:
                _plot_features_eda(
                    severity_df,
                    [column],
                    [column],
                    model_type="severity",
                )
            else:
                _plot_features_eda(
                    severity_df,
                    [column],
                    [],
                    model_type="severity",
                )


def run_training_visualizations(
    df: pd.DataFrame,
    training: TrainingArtifacts,
    config: TrainingConfig | None = None,
    **kwargs,
) -> None:
    """Совместимость: только post-training графики (EDA — через run_mvp_frequency_eda)."""
    run_model_diagnostics_visualizations(df, training, config, **kwargs)
