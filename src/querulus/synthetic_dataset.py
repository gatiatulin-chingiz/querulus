"""Синтетический ``df_final_3`` для локального smoke-теста без remote-данных."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from querulus import PROJECT_ROOT

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "df_final_3_synthetic.parquet"

# Категории с несколькими уровнями (AutoMVP не должен их выкинуть).
_FILIALS = ("МОСКВА", "СПБ", "КАЗАНЬ", "ЕКАТЕРИНБУРГ")
_CATEGORIES = ("B", "C", "D")
_FORMS = ("ПОТЕРПЕВШИЙ", "ПРЕДСТАВИТЕЛЬ", "ЮРИСТ")
_METHODS = ("ОЧНО", "ЭЛЕКТРОННО", "ПОЧТА")
_ZONES = ("ЦЕНТР", "СЕВЕР", "ЮГ", "ВОСТОК")
_COUNTRIES = ("РОССИЯ", "БЕЛАРУСЬ", "КАЗАХСТАН")


def build_synthetic_final_dataset(
    n_rows: int = 400,
    *,
    seed: int = 42,
    positive_rate: float = 0.28,
) -> pd.DataFrame:
    """Собрать минимальный финальный датасет под ``selected`` + triple-stack + fin_effect + zoo.

    Даты покрывают train ``2022-01-01``–``2024-05-31`` и test ``2024-06-01``–``2025-06-01``.
    Среди положительной severity есть сегменты ``VALUE_BEFORE_WITH`` ≤/> 50k.
    """
    if n_rows < 80:
        raise ValueError("n_rows должен быть >= 80 (иначе пустые train/test / сегменты)")

    rng = np.random.default_rng(seed)
    n_train = int(round(n_rows * 0.7))
    n_test = n_rows - n_train

    # Даты: равномерно внутри периодов обучения.
    train_start = np.datetime64("2022-01-01")
    train_end = np.datetime64("2024-05-31")
    test_start = np.datetime64("2024-06-01")
    test_end = np.datetime64("2025-05-31")
    train_days = (train_end - train_start).astype(int)
    test_days = (test_end - test_start).astype(int)
    loss_dates = np.concatenate(
        [
            train_start + rng.integers(0, train_days + 1, size=n_train).astype("timedelta64[D]"),
            test_start + rng.integers(0, test_days + 1, size=n_test).astype("timedelta64[D]"),
        ]
    )

    # Частота: ~positive_rate единиц.
    target_freq = (rng.random(n_rows) < positive_rate).astype(int)
    # Severity > 0 только при positive freq (как в бою).
    sev_base = rng.lognormal(mean=10.5, sigma=0.85, size=n_rows)
    target_sev = np.where(target_freq == 1, sev_base, 0.0)
    # Claims-стек чуть реже / с другим масштабом.
    target_freq_claims = (
        (target_freq == 1) & (rng.random(n_rows) < 0.75)
    ).astype(int)
    target_sev_claims = np.where(
        target_freq_claims == 1,
        target_sev * rng.uniform(0.7, 1.1, size=n_rows),
        0.0,
    )
    # Legacy стек: похож на new с небольшим шумом.
    target_2 = (
        (target_freq == 1) | ((target_freq == 0) & (rng.random(n_rows) < 0.03))
    ).astype(int)
    target_3_sev = np.where(target_2 == 1, target_sev * rng.uniform(0.85, 1.15, size=n_rows), 0.0)

    pret_amount = np.where(target_freq == 1, target_sev * rng.uniform(0.2, 0.5, size=n_rows), 0.0)
    claims_amount = np.where(
        target_freq_claims == 1,
        target_sev_claims * rng.uniform(0.5, 0.9, size=n_rows),
        0.0,
    )
    freq_amount = pret_amount + claims_amount

    # VALUE_BEFORE: половина positive sev ниже 50k, половина выше.
    value_before = rng.uniform(15_000, 45_000, size=n_rows)
    high_mask = (target_freq == 1) & (np.arange(n_rows) % 2 == 0)
    value_before[high_mask] = rng.uniform(55_000, 180_000, size=int(high_mask.sum()))
    value_before_without = value_before * rng.uniform(0.85, 1.0, size=n_rows)

    event_year = pd.to_datetime(loss_dates).year.astype(int)

    return pd.DataFrame(
        {
            "INCIDENT_NUMBER": [f"SYN-{i:06d}" for i in range(n_rows)],
            "LOSS_DATE_TIME": pd.to_datetime(loss_dates),
            "TARGET_2": target_2,
            "TARGET_3_SEV": target_3_sev,
            "TARGET_FREQ": target_freq,
            "TARGET_SEV": target_sev,
            "TARGET_FREQ_CLAIMS": target_freq_claims,
            "TARGET_SEV_CLAIMS": target_sev_claims,
            "TARGET_FREQ_AMOUNT": freq_amount,
            "TARGET_FREQ_CLAIMS_AMOUNT": claims_amount,
            "TARGET_FREQ_PRET_AMOUNT": pret_amount,
            "EVENT_CREATED_BY_GIBDD_FLAG": rng.choice(["0", "1"], size=n_rows).astype(object),
            "FILIAL": rng.choice(_FILIALS, size=n_rows),
            "VICTIM_VEHICLE_CATEGORY": rng.choice(_CATEGORIES, size=n_rows),
            "APPLICANT_FORM": rng.choice(_FORMS, size=n_rows),
            "RECIEVE_METHOD": rng.choice(_METHODS, size=n_rows),
            "VICTIM_VEHICLE_AGE": rng.integers(0, 20, size=n_rows).astype(float),
            "VICTIM_MAX_WEIGHT": rng.uniform(1200, 3500, size=n_rows),
            "GUILTY_CAPACITY_ENGINE": rng.uniform(1000, 3000, size=n_rows),
            "APPLICANT_AGE": rng.integers(18, 75, size=n_rows).astype(float),
            "EVENT_YEAR": event_year,
            "LOSS_UNIT_ZONE": rng.choice(_ZONES, size=n_rows),
            "VICTIM_VEHICLE_COUNTRY": rng.choice(_COUNTRIES, size=n_rows),
            "APPLY_DELAY": rng.integers(0, 120, size=n_rows).astype(float),
            "VALUE_BEFORE_WITHOUT": value_before_without,
            "VALUE_BEFORE_WITH": value_before,
            "Выплата_по_основному_убытку": rng.uniform(5_000, 80_000, size=n_rows),
            "Сумма_выплат_по_претензиям": pret_amount * rng.uniform(0.3, 0.8, size=n_rows),
            "Сумма_взыскано_по_ФУ": np.where(
                (target_freq == 1) & (rng.random(n_rows) < 0.15),
                rng.uniform(50_000, 150_000, size=n_rows),
                0.0,
            ),
            "Суммы_взыскано_по_иску": np.where(
                target_freq == 1,
                freq_amount * rng.uniform(0.1, 0.4, size=n_rows),
                0.0,
            ),
        }
    )


def write_synthetic_final_dataset(
    path: Path | str | None = None,
    *,
    n_rows: int = 400,
    seed: int = 42,
) -> Path:
    """Записать parquet; каталог создаётся при необходимости."""
    out = Path(path) if path is not None else DEFAULT_OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    df = build_synthetic_final_dataset(n_rows=n_rows, seed=seed)
    df.to_parquet(out, index=False)
    return out


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m querulus.synthetic_dataset`` или ``make synthetic-data``."""
    parser = argparse.ArgumentParser(description="Синтетический df_final_3 для локального smoke.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Путь parquet (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("-n", "--n-rows", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    path = write_synthetic_final_dataset(args.output, n_rows=args.n_rows, seed=args.seed)
    df = pd.read_parquet(path)
    train = df["LOSS_DATE_TIME"] < "2024-06-01"
    print(f"Wrote {path}")
    print(f"shape={df.shape} train={int(train.sum())} test={int((~train).sum())}")
    print(f"TARGET_FREQ mean={df['TARGET_FREQ'].mean():.3f}")
    print(f"sev>0={int((df['TARGET_SEV'] > 0).sum())}")


if __name__ == "__main__":
    main()
