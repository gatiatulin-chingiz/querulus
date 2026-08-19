"""Год старта train: когорты по поручению и погодичный сдвиг на одном тесте."""
from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from querulus.training.config import TrainingConfig
from querulus.training.pipeline import train_models
from querulus.training.stack_eval import score_stack_on_index

_PSR_COL = "TARGET_FREQ_AMOUNT"
DEFAULT_START_YEARS = (2020, 2021, 2022, 2023)
_MIN_TRAIN_ROWS = 50


@dataclass(frozen=True)
class StartYearEvalReport:
    """Когорты по году поручения и метрики на общем тесте при сдвиге train_from."""

    cohorts: pd.DataFrame
    start_shift: pd.DataFrame
    coverage: pd.DataFrame


def payment_year_cohorts(
    df: pd.DataFrame,
    *,
    date_column: str,
    freq_target: str = "TARGET_FREQ",
    sev_target: str = "TARGET_SEV",
    psr_col: str = _PSR_COL,
) -> pd.DataFrame:
    """n / частота / суммы по календарному году ``date_column``."""
    for col in (date_column, freq_target):
        if col not in df.columns:
            raise KeyError(f"Для когорт нужна колонка {col}")
    work = df[[date_column, freq_target]].copy()
    work["_year"] = pd.to_datetime(work[date_column], errors="coerce").dt.year
    work["_freq"] = pd.to_numeric(work[freq_target], errors="coerce").fillna(0).astype(int)
    if sev_target in df.columns:
        work["_sev"] = pd.to_numeric(df[sev_target], errors="coerce")
    else:
        work["_sev"] = pd.NA
    if psr_col in df.columns:
        work["_psr"] = pd.to_numeric(df[psr_col], errors="coerce")
    else:
        work["_psr"] = pd.NA
    work = work.loc[work["_year"].notna()]
    if work.empty:
        return pd.DataFrame(
            columns=["year", "n", "freq_rate", "n_pos", "sev_p50", "sev_p90", "psr_p50"]
        )

    rows: list[dict[str, object]] = []
    for year, part in work.groupby("_year", sort=True):
        pos = part["_freq"].eq(1)
        sev_pos = part.loc[pos, "_sev"]
        psr_pos = part.loc[pos, "_psr"]
        rows.append(
            {
                "year": int(year),
                "n": int(len(part)),
                "freq_rate": float(part["_freq"].mean()),
                "n_pos": int(pos.sum()),
                "sev_p50": float(sev_pos.median()) if pos.any() else float("nan"),
                "sev_p90": float(sev_pos.quantile(0.90)) if pos.any() else float("nan"),
                "psr_p50": float(psr_pos.median()) if pos.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _holdout_index(df: pd.DataFrame, config: TrainingConfig) -> pd.Index:
    dates = pd.to_datetime(df[config.date_column], errors="coerce")
    mask = dates.between(*config.test_period)
    return df.index[mask.fillna(False)]


def _n_train(df: pd.DataFrame, config: TrainingConfig, train_from: str) -> int:
    dates = pd.to_datetime(df[config.date_column], errors="coerce")
    return int(dates.between(train_from, config.train_period[1]).fillna(False).sum())


def _eligible_start_years(
    df: pd.DataFrame,
    config: TrainingConfig,
    start_years: tuple[int, ...] | None,
    *,
    min_train_rows: int,
) -> list[int]:
    """Годы, у которых train_from=YYYY-01-01 даёт достаточно строк до train_end."""
    dates = pd.to_datetime(df[config.date_column], errors="coerce")
    present = tuple(sorted({int(y) for y in dates.dt.year.dropna().unique()}))
    train_end = pd.Timestamp(config.train_period[1])
    test_start = pd.Timestamp(config.test_period[0])
    candidates = start_years if start_years is not None else present
    out: list[int] = []
    for year in candidates:
        start = pd.Timestamp(f"{int(year)}-01-01")
        if start >= train_end or start >= test_start:
            continue
        if _n_train(df, config, start.strftime("%Y-%m-%d")) < min_train_rows:
            continue
        out.append(int(year))
    return out


def _quiet_config(config: TrainingConfig) -> TrainingConfig:
    freq = dict(config.frequency_classifier_params)
    sev = dict(config.severity_regressor_params)
    freq["verbose"] = 0
    sev["verbose"] = 0
    return replace(config, frequency_classifier_params=freq, severity_regressor_params=sev)


def _penalty_amount(coverage: pd.DataFrame) -> float:
    hit = coverage.loc[coverage["outcome"].eq("модель всего"), "amount"]
    if hit.empty:
        return float("nan")
    return float(hit.iloc[0])


def evaluate_train_start_years(
    df: pd.DataFrame,
    config: TrainingConfig,
    *,
    start_years: tuple[int, ...] | None = DEFAULT_START_YEARS,
    frequency_features: tuple[str, ...] | list[str] | None = None,
    severity_features: tuple[str, ...] | list[str] | None = None,
    min_train_rows: int = _MIN_TRAIN_ROWS,
) -> StartYearEvalReport:
    """Обучить с ``train_from`` по годам; test и фичи общие. Без повторного FS.

    Правило чтения: если PR-AUC и штраф ₽ близки — брать более поздний старт.
    """
    freq_target = config.frequency_target
    sev_target = config.severity_target
    cohorts = payment_year_cohorts(
        df,
        date_column=config.date_column,
        freq_target=freq_target,
        sev_target=sev_target,
    )
    years = _eligible_start_years(
        df, config, start_years, min_train_rows=min_train_rows
    )
    if not years:
        print("[start-year] нет подходящих лет (мало строк train или год правее train_end/test)")
        return StartYearEvalReport(
            cohorts=cohorts, start_shift=pd.DataFrame(), coverage=pd.DataFrame()
        )
    holdout = _holdout_index(df, config)
    if holdout.empty:
        raise ValueError(
            f"Пустой test {config.test_period[0]}…{config.test_period[1]} "
            f"по {config.date_column}"
        )

    frozen = _quiet_config(config)
    if frequency_features is not None:
        frozen = replace(frozen, frequency_features=tuple(frequency_features))
    if severity_features is not None:
        frozen = replace(frozen, severity_features=tuple(severity_features))
    frozen = replace(
        frozen,
        frequency_select_features=False,
        severity_select_features=False,
    )

    current_from = pd.Timestamp(config.train_period[0]).strftime("%Y-%m-%d")
    shift_rows: list[dict[str, object]] = []
    cov_parts: list[pd.DataFrame] = []
    train_end = config.train_period[1]

    for year in years:
        train_from = f"{year}-01-01"
        n_train = _n_train(df, frozen, train_from)
        print(
            f"[start-year] train {train_from}…{train_end}  n_train={n_train}  "
            f"test {frozen.test_period[0]}…{frozen.test_period[1]}  n_test={len(holdout)}"
        )
        year_cfg = replace(frozen, train_period=(train_from, train_end))
        artifacts = train_models(df, year_cfg)
        freq_row, coverage = score_stack_on_index(
            df,
            artifacts,
            holdout,
            stack=str(year),
            freq_target=freq_target,
            sev_target=sev_target,
        )
        coverage = coverage.copy()
        coverage.insert(0, "train_from", train_from)
        cov_parts.append(coverage)
        shift_rows.append(
            {
                "train_from": train_from,
                "train_to": train_end,
                "is_current": train_from == current_from,
                "n_train": n_train,
                "n_test": int(freq_row["n"]),
                "pr_auc": freq_row["pr_auc"],
                "precision": freq_row["precision"],
                "recall": freq_row["recall"],
                "f1": freq_row["f1"],
                "penalty": _penalty_amount(coverage),
            }
        )

    start_shift = pd.DataFrame(shift_rows)
    coverage = (
        pd.concat(cov_parts, ignore_index=True) if cov_parts else pd.DataFrame()
    )
    return StartYearEvalReport(
        cohorts=cohorts, start_shift=start_shift, coverage=coverage
    )
