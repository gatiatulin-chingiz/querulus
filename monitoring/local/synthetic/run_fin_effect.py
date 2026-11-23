"""Local-only: synthetic monitoring/retro → pipeline fin_effect compute + write."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SYNTH = Path(__file__).resolve().parent
_MONITORING = _SYNTH.parents[1]
if str(_MONITORING) not in sys.path:
    sys.path.insert(0, str(_MONITORING))

from pipeline.settings import load_settings
from pipeline.fin_effect import (
    FILIAL_ALL,
    FinEffectJob,
    PILOT_FILIALS,
    RESULT_OUT_OF_MODEL,
    WIDE_COLS,
    _finalize_wide,
    build_wide_frame,
    compute_fin_effect_weekly,
)
from pipeline.write_sql import get_engine, write_append_guarded

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("local.fin_effect")

_MAIN_MODELS = (
    {
        "stem": "querulus_ansamble_2026_04_18_v1",
        "label": "querulus_ansamble_2026_04_18_v1",
        "seed": 42,
    },
    {
        "stem": "UU_querulus_2026_11_05_12_44_24",
        "label": "UU_querulus_2026_11_05_12_44_24",
        "seed": 99,
    },
)


def _build_monitoring_claims(*, n_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    result = rng.choice([RESULT_OUT_OF_MODEL, 0, 1], size=n_rows)
    payout = (result == 1) & (rng.random(n_rows) < 0.6)
    agreement = (result == 1) & (rng.random(n_rows) < 0.5)
    today = pd.Timestamp.today().normalize()
    psr = np.where(rng.random(n_rows) < 0.08, rng.uniform(1_000, 30_000, n_rows), 0.0)
    frame = pd.DataFrame(
        {
            "Убыток": [f"SYN-{index:06d}" for index in range(n_rows)],
            "НомерИнцидент": [f"INC-{index // 2:06d}" for index in range(n_rows)],
            "Филиал": rng.choice(PILOT_FILIALS, size=n_rows),
            "ФормаВозмещения": np.where(agreement, "Соглашение", "Денежная"),
            "УбытокСтатус": "Первичный",
            "ТипОбъектаАвтотранспорт": 1,
            "РезультатПроверки": result,
            "СуммаПлатежа": rng.uniform(30_000, 250_000, n_rows),
            "СуммаКВыплате": rng.uniform(30_000, 250_000, n_rows),
            "СуммаОсновногоДолгаЗаявлено": rng.uniform(20_000, 300_000, n_rows),
            "Сумма рекомендованная к доплате по модулю": np.where(
                result == 1, rng.uniform(10_000, 80_000, n_rows), 0.0
            ),
            "Иные затраты": np.where(payout, rng.uniform(5_000, 60_000, n_rows), 0.0),
            "Выплата по модели": payout.astype(int),
            "Заключено соглашение": agreement.astype(int),
            "ВызовМодельСутяжность": np.where(
                result != RESULT_OUT_OF_MODEL,
                1,
                (rng.random(n_rows) < 0.35).astype(int),
            ),
            "Дата вызова модели сутяжности": today
            - pd.to_timedelta(rng.integers(0, 120, n_rows), unit="D"),
            "ДатаЗаявления": today
            - pd.to_timedelta(rng.integers(0, 120, n_rows), unit="D"),
            "Cумма выплаты по претензии": psr,
            "Сумма выплат по ФУ": 0.0,
            "Сумма выплаты по суду": 0.0,
            "ЕМР+20": ((rng.random(n_rows) < 0.12) & agreement).astype(int),
            "ПереплатаЕМР+(безКВ)": 0.0,
            "Выплата из лимитов директоров филиалов": (
                (rng.random(n_rows) < 0.10) & agreement
            ).astype(int),
            "СогласованнаяСуммаКВ": 0.0,
        }
    )
    frame["СуммаКВыплате"] = frame["СуммаПлатежа"]
    frame["ПереплатаЕМР+(безКВ)"] = np.where(
        frame["ЕМР+20"].eq(1), rng.uniform(3_000, 40_000, n_rows), 0.0
    )
    frame["СогласованнаяСуммаКВ"] = np.where(
        frame["Выплата из лимитов директоров филиалов"].eq(1),
        rng.uniform(5_000, 50_000, n_rows),
        0.0,
    )
    return frame


def _build_synthetic_retro(*, n_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    nonpilot = ("Московский", "Санкт-Петербургский", "Казанский", "Новосибирский")
    filials = list(PILOT_FILIALS) + list(nonpilot)
    end = pd.Timestamp("2025-06-30")
    start = end - pd.DateOffset(years=2)
    days = (end - start).days
    dates = start + pd.to_timedelta(rng.integers(0, max(days, 1), n_rows), unit="D")
    amount = rng.uniform(20_000, 200_000, n_rows)
    return pd.DataFrame(
        {
            "FILIAL": rng.choice(filials, size=n_rows),
            "LOSS_DATE": dates,
            "TARGET_FREQ_AMOUNT": amount,
            "RECOVEREDMAINDEBT_LAST_INST_SUM": amount * rng.uniform(0.3, 0.9, n_rows),
            "Сумма_взыскано_по_ФУ": np.where(
                rng.random(n_rows) < 0.1, amount * 0.2, 0.0
            ),
            "Суммы_взыскано_по_иску": np.where(
                rng.random(n_rows) < 0.05, amount * 0.3, 0.0
            ),
        }
    )


def build_synthetic_frames(
    *, n_monitoring: int = 400, n_retro: int = 800, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monitoring = _build_monitoring_claims(n_rows=n_monitoring, seed=seed)
    today = pd.Timestamp.today().normalize()
    pilot_start = pd.Timestamp("2026-04-20")
    span = max((today - pilot_start).days, 30)
    rng = np.random.default_rng(seed)
    monitoring["ДатаЗаявления"] = pilot_start + pd.to_timedelta(
        rng.integers(0, span, len(monitoring)), unit="D"
    )
    monitoring["Дата вызова модели сутяжности"] = monitoring["ДатаЗаявления"]
    retro = _build_synthetic_retro(n_rows=n_retro, seed=seed + 1)
    return monitoring, retro


def main() -> int:
    data_dir = Path(
        os.environ.get("SYNTHETIC_DATA_DIR", str(_SYNTH / "data"))
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    mon_path = data_dir / "fin_effect_monitoring.csv"
    retro_path = data_dir / "fin_effect_retro.csv"
    settings = load_settings()

    jobs: list[FinEffectJob] = []
    for i, model in enumerate(_MAIN_MODELS):
        if i == 0 and mon_path.is_file() and retro_path.is_file():
            monitoring = pd.read_csv(mon_path)
            retro = pd.read_csv(retro_path)
        else:
            monitoring, retro = build_synthetic_frames(seed=int(model["seed"]))
            if i == 0:
                monitoring.to_csv(mon_path, index=False)
                retro.to_csv(retro_path, index=False)
        jobs.append(
            FinEffectJob(
                monitoring_df=monitoring.copy(),
                retro_df=retro.copy(),
                stem=model["stem"],
                slot="main",
                model_label=model["label"],
            )
        )

    results = [compute_fin_effect_weekly(job) for job in jobs]
    frames = [build_wide_frame(res) for res in results]
    wide = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=WIDE_COLS)
    )
    wide = _finalize_wide(wide)
    if wide.empty:
        logger.warning("Empty fin_effect wide report")
        return 0
    engine = get_engine(settings.grafana_sqlalchemy_url)
    n = write_append_guarded(
        wide,
        engine=engine,
        table=settings.table_fin_effect,
        schema=settings.schema,
        guard_cols=["week_start", "stem", "slot", "filial"],
    )
    logger.info(
        "local synthetic fin_effect done || rows=%s || (все)=%s",
        n,
        int((wide["filial"] == FILIAL_ALL).sum()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
