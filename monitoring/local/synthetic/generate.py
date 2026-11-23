"""Generate synthetic monitoring inputs for local Airflow/Grafana smoke.

Writes CSV under local/synthetic/data/ (gitignored):
  errors.csv
  train_features.csv
  prod_features.csv
  fin_effect_monitoring.csv + fin_effect_retro.csv
  (расчёт weekly — querulus.fin_effect в pipeline, не этот CSV)

Models mirror integration/results/models.json:
  slot=main   → querulus_ansamble_2026_04_18_v1
  slot=second → UU_querulus_2026_11_05_12_44_24

Only matched (stem, slot) pairs — no cross products.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

# (slot, stem) — как в нормальном /api/predict; model_kind = ВидМодели в 1С
MODELS = (
    {
        "slot": "main",
        "stem": "querulus_ansamble_2026_04_18_v1",
        "label": "1 · main · querulus_ansamble_2026_04_18_v1",
        "model_kind": "Сутяжничество main",
        "model_version": "1.4.0",
    },
    {
        "slot": "second",
        "stem": "UU_querulus_2026_11_05_12_44_24",
        "label": "2 · second · UU_querulus_2026_11_05_12_44_24",
        "model_kind": "Сутяжничество second",
        "model_version": "2.0.0",
    },
)
TRACKS = ("cf", "rg")
FILIALS = ("Архангельский", "Курский", "Омский", "Пермский")
FORMS = ("Физлицо", "Скрытый юрист", "Юрлицо")

# Имитация ОписаниеОшибки = traceback при РезультатПроверки = -1
_TRACEBACKS = (
    (
        "Traceback (most recent call last):\n"
        '  File "/app/main.py", line 312, in score_track\n'
        "    clf_out = await ensemble_predict(stem, clf_model, df_pre)\n"
        "KeyError: 'FILIAL'"
    ),
    (
        "Traceback (most recent call last):\n"
        '  File "/app/pipelines/second_pipeline.py", line 88, in preprocess\n'
        "    value = float(row['AMOUNT_REPAIR'])\n"
        "ValueError: could not convert string to float: 'N/A'"
    ),
    (
        "Traceback (most recent call last):\n"
        '  File "/home/airflow/.local/lib/catboost.py", line 120, in predict\n'
        "    return model.predict(pool)\n"
        "CatBoostError: cat_feature[2] has new value"
    ),
    (
        "Traceback (most recent call last):\n"
        '  File "/app/main.py", line 289, in _http_500\n'
        "    raise HTTPException(status_code=500, detail=str(exc))\n"
        "RuntimeError: registry not loaded"
    ),
)


def generate_errors(n_days: int = 7, seed: int = 42) -> pd.DataFrame:
    """Rows shaped like oisuu81_t_MatemModelsLogs (only РезультатПроверки = -1)."""
    rng = np.random.default_rng(seed)
    today = date.today()
    rows = []
    loss_id = 10_000_000
    for d in range(n_days):
        day = today - timedelta(days=d)
        for model in MODELS:
            n_err = int(rng.integers(1, 5))
            for i in range(n_err):
                tb = _TRACEBACKS[int(rng.integers(0, len(_TRACEBACKS)))]
                # Same fingerprint gets repeated sometimes (realistic)
                if rng.random() < 0.35:
                    tb = _TRACEBACKS[0]
                loss_id += 1
                hour = int(rng.integers(8, 20))
                minute = int(rng.integers(0, 60))
                checked = f"{day.isoformat()}T{hour:02d}:{minute:02d}:00"
                rows.append(
                    {
                        "ДатаПроверки": checked,
                        "LossNumber": loss_id,
                        "РезультатПроверки": -1,
                        "ОписаниеОшибки": tb,
                        "ВерсияМодели": model["model_version"],
                        "Комментарий": "",
                        "Filial": FILIALS[int(rng.integers(0, len(FILIALS)))],
                        "ВидМодели": model["model_kind"],
                        "Подразделение": "",
                        "LossProcess": "ОСАГО",
                        # helpers for Grafana filter (not in 1C table)
                        "stem": model["stem"],
                        "slot": model["slot"],
                        "model_label": model["label"],
                    }
                )
    return pd.DataFrame(rows)


def generate_features(
    n_train: int = 800, n_prod: int = 400, seed: int = 7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_parts = []
    prod_parts = []
    n_per = max(n_train // (len(MODELS) * len(TRACKS)), 50)
    n_prod_per = max(n_prod // (len(MODELS) * len(TRACKS)), 25)
    for model in MODELS:
        for track in TRACKS:
            train = pd.DataFrame(
                {
                    "stem": model["stem"],
                    "slot": model["slot"],
                    "model_label": model["label"],
                    "track": track,
                    "AMOUNT_REPAIR": rng.normal(45000, 12000, n_per).clip(1000),
                    "APPLICANT_AGE": rng.integers(18, 75, n_per),
                    "VICTIM_VEHICLE_AGE": rng.integers(0, 25, n_per),
                    "APPLY_DELAY": rng.integers(0, 60, n_per),
                    "FILIAL": rng.choice(FILIALS, n_per),
                    "APPLICANT_FORM": rng.choice(FORMS, n_per, p=[0.7, 0.2, 0.1]),
                }
            )
            prod = pd.DataFrame(
                {
                    "stem": model["stem"],
                    "slot": model["slot"],
                    "model_label": model["label"],
                    "track": track,
                    "AMOUNT_REPAIR": rng.normal(52000, 15000, n_prod_per).clip(1000),
                    "APPLICANT_AGE": rng.integers(18, 80, n_prod_per),
                    "VICTIM_VEHICLE_AGE": rng.integers(0, 30, n_prod_per),
                    "APPLY_DELAY": rng.integers(0, 90, n_prod_per),
                    "FILIAL": rng.choice(
                        FILIALS, n_prod_per, p=[0.15, 0.35, 0.25, 0.25]
                    ),
                    "APPLICANT_FORM": rng.choice(
                        FORMS, n_prod_per, p=[0.55, 0.35, 0.1]
                    ),
                }
            )
            train_parts.append(train)
            prod_parts.append(prod)
    return pd.concat(train_parts, ignore_index=True), pd.concat(
        prod_parts, ignore_index=True
    )


def generate_fin_effect_inputs(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Витрины для querulus.fin_effect (не готовый weekly)."""
    # import через pipeline, чтобы не дублировать builders
    import sys

    mon_root = Path(__file__).resolve().parents[2]
    src = Path(__file__).resolve().parents[3] / "src"
    for p in (mon_root, src):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from pipeline.fin_effect import build_synthetic_frames

    return build_synthetic_frames(seed=seed)


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    """Write via temp file to avoid empty/partial CSV on concurrent generators."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def main(out_dir: Path | None = None) -> Path:
    out = out_dir or DATA_DIR
    out.mkdir(parents=True, exist_ok=True)
    errors = generate_errors()
    train, prod = generate_features()
    mon, retro = generate_fin_effect_inputs()
    _atomic_csv(errors, out / "errors.csv")
    _atomic_csv(train, out / "train_features.csv")
    _atomic_csv(prod, out / "prod_features.csv")
    _atomic_csv(mon, out / "fin_effect_monitoring.csv")
    _atomic_csv(retro, out / "fin_effect_retro.csv")
    # старый фейковый weekly больше не пишем
    legacy = out / "fin_effect_weekly.csv"
    if legacy.is_file():
        legacy.unlink()
    print(f"Wrote synthetic data to {out}")
    print(
        f"  errors={len(errors)} train={len(train)} prod={len(prod)} "
        f"fin_mon={len(mon)} fin_retro={len(retro)}"
    )
    print("  models:")
    for m in MODELS:
        print(f"    slot={m['slot']} → {m['stem']}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Querulus monitoring synthetic data"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: local/synthetic/data)",
    )
    args = parser.parse_args()
    main(args.out)
