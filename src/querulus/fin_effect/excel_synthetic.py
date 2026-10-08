"""Синтетическая Excel-витрина для smoke-тестов мониторинга."""
from __future__ import annotations

import numpy as np
import pandas as pd

RESULT_OUT_OF_MODEL = -100
PILOT_FILIALS = (
    "Владимирский",
    "Кемеровский",
    "Курский",
    "Магнитогорский",
    "Мурманский",
    "Омский",
    "Пермский",
    "Петропавловск-Камчатский",
    "Уфимский",
    "Ярославский",
)


def build_synthetic_claims_excel(
    n_rows: int = 300,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Сохранить совместимость synthetic-источника загрузчика мониторинга."""
    if n_rows < 20:
        raise ValueError("n_rows должен быть не меньше 20")
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
            "СуммаОсновногоДолгаЗаявлено": rng.uniform(
                20_000,
                300_000,
                n_rows,
            ),
            "Сумма рекомендованная к доплате по модулю": np.where(
                result == 1,
                rng.uniform(10_000, 80_000, n_rows),
                0.0,
            ),
            "Иные затраты": np.where(
                payout,
                rng.uniform(5_000, 60_000, n_rows),
                0.0,
            ),
            "Выплата по модели": payout.astype(int),
            "Заключено соглашение": agreement.astype(int),
            # вызов чаще у model; часть control тоже «вызывали» (ответ −100)
            "ВызовМодельСутяжность": np.where(
                result != RESULT_OUT_OF_MODEL,
                1,
                (rng.random(n_rows) < 0.35).astype(int),
            ),
            "Дата вызова модели сутяжности": today - pd.to_timedelta(
                rng.integers(0, 120, n_rows),
                unit="D",
            ),
            "ДатаЗаявления": today - pd.to_timedelta(
                rng.integers(0, 120, n_rows),
                unit="D",
            ),
            "Cумма выплаты по претензии": psr,
            "Сумма выплат по ФУ": 0.0,
            "Сумма выплаты по суду": 0.0,
        }
    )
    frame["СуммаКВыплате"] = frame["СуммаПлатежа"]
    return frame
