"""Специфика second_: CPI, FE_*, THRESHOLD из вектора, калибраторы."""
from __future__ import annotations

import pickle
import sys
import unittest
from pathlib import Path

_INTEGRATION = Path(__file__).resolve().parents[1]
_QUERULUS = _INTEGRATION.parent
if str(_QUERULUS) not in sys.path:
    sys.path.insert(0, str(_QUERULUS))

from integration.src.shim import install_mldataworker_shim

install_mldataworker_shim()

import pandas as pd

from integration import config
from integration.src.calibration import (
    apply_cf_calibrator,
    apply_rg_calibrator,
)
from integration.src.preprocessing.common import resolve_threshold
from integration.src.preprocessing.common import coerce_and_enrich
from integration.src.preprocessing.cpi import (
    MONETARY_COLUMNS_FOR_REAL,
    RU_CPI_LEVEL_VS_BASE,
    deflate_to_base_year,
    resolve_cpi_year,
)
from integration.src.preprocessing.second import (
    add_second_model_features,
    validate_input_vector,
)
from integration.tests.artifacts import ART, require_file


# new
class TestCpi(unittest.TestCase):
    def test_exact_year(self):
        self.assertEqual(resolve_cpi_year(2022), 2022)

    def test_future_year_uses_max(self):
        max_y = max(RU_CPI_LEVEL_VS_BASE)
        self.assertEqual(resolve_cpi_year(max_y + 5), max_y)

    def test_past_year_uses_min(self):
        min_y = min(RU_CPI_LEVEL_VS_BASE)
        self.assertEqual(resolve_cpi_year(min_y - 3), min_y)

    def test_missing_value_before_raises_at_validate(self):
        """Без VALUE_BEFORE_* валидация падает; AMOUNT_REPAIR не подменяет."""
        df = pd.DataFrame(
            {
                "EVENT_DATE": ["2022-06-01"],
                "PAYMENT_ORDER_DATE_TIME": ["2022-06-01"],
                "AMOUNT_REPAIR": [1000.0],
                "EVENT_YEAR": [2022],
                "APPLY_DELAY": [0],
            }
        )
        with self.assertRaises(ValueError) as ctx:
            validate_input_vector(df)
        msg = str(ctx.exception)
        for col in MONETARY_COLUMNS_FOR_REAL:
            self.assertIn(col, msg)


# new
class TestDerivedFe(unittest.TestCase):
    def test_share_work_tier_from_works(self):
        df = coerce_and_enrich(
            pd.DataFrame(
                {
                    "EVENT_DATE": ["2022-01-15"],
                    "PAYMENT_ORDER_DATE_TIME": ["2022-02-01"],
                    "VALUE_BEFORE_WITH": [80000.0],
                    "VALUE_BEFORE_WITHOUT": [100000.0],
                    "WORK": [50000.0],
                    "VICTIM_CAPACITY_ENGINE": [100.0],
                    "VICTIM_MAX_WEIGHT": [1500.0],
                    "VICTIM_OBJECT_YEAR": [2018],
                    "LOSS_DATE_TIME": ["2022-01-20"],
                    "POLICYHOLDER_CONTRACT_END_DATE": ["2022-08-01"],
                }
            )
        )
        out = add_second_model_features(df)
        self.assertIn("FE_SHARE_WORK_TIER", out.columns)
        self.assertIn("FE_VALUE_BEFORE_RATIO", out.columns)
        self.assertIn("FE_VICTIM_POWER_PER_TON", out.columns)
        # 0.5 → mid (bins 0.3, 0.6)
        self.assertEqual(str(out["FE_SHARE_WORK_TIER"].iloc[0]), "mid")
        self.assertEqual(int(out["FE_DAYS_EVENT_TO_LOSS"].iloc[0]), 5)
        self.assertAlmostEqual(
            float(out["FE_VICTIM_POWER_PER_TON"].iloc[0]), 100.0 / 1500.0 * 1e6, places=3
        )

    def test_real_uses_payment_year_not_event_year(self):
        """CPI as-of = дата выплаты, как в обучении (t0), не EVENT_DATE."""
        df = coerce_and_enrich(
            pd.DataFrame(
                {
                    "EVENT_DATE": ["2019-12-05"],
                    "PAYMENT_ORDER_DATE_TIME": ["2023-05-15"],
                    "VALUE_BEFORE_WITH": [100000.0],
                    "VALUE_BEFORE_WITHOUT": [100000.0],
                    "WORK": [1.0],
                    "VICTIM_CAPACITY_ENGINE": [1.0],
                    "VICTIM_MAX_WEIGHT": [1.0],
                    "VICTIM_OBJECT_YEAR": [2015],
                    "LOSS_DATE_TIME": ["2019-12-06"],
                    "POLICYHOLDER_CONTRACT_END_DATE": ["2020-01-01"],
                }
            )
        )
        out = add_second_model_features(df)
        expected = float(
            deflate_to_base_year(
                pd.Series([100000.0]),
                pd.Series([2023]),
            ).iloc[0]
        )
        got = float(out["FE_VALUE_BEFORE_WITH_REAL_2022"].iloc[0])
        self.assertAlmostEqual(got, expected, places=2)
        by_event = float(
            deflate_to_base_year(
                pd.Series([100000.0]),
                pd.Series([2019]),
            ).iloc[0]
        )
        self.assertNotAlmostEqual(got, by_event, places=0)

# new
class TestVectorThreshold(unittest.TestCase):
    def test_resolve_from_vector(self):
        df = pd.DataFrame({"THRESHOLD": [0.42]})
        self.assertAlmostEqual(
            resolve_threshold(df, config.default_second_threshold), 0.42
        )

    def test_missing_threshold_uses_config_fallback(self):
        thr = resolve_threshold(pd.DataFrame({"X": [1]}), config.default_second_threshold)
        self.assertAlmostEqual(thr, float(config.default_second_threshold))


# new
class TestCalibrators(unittest.TestCase):
    def test_cf_rg_calibrators_load_and_apply(self):
        cf_path = require_file(ART.cf_calibrator, "cf_calibrator")
        rg_path = require_file(ART.rg_calibrator, "rg_calibrator")
        with open(cf_path, "rb") as f:
            cf_cal = pickle.load(f)
        with open(rg_path, "rb") as f:
            rg_cal = pickle.load(f)
        raw = [0.2, 0.5, 0.8]
        cal_cf = apply_cf_calibrator(cf_cal, raw)
        self.assertEqual(len(cal_cf), 3)
        cal_rg = apply_rg_calibrator(rg_cal, [1000.0, 5000.0])
        self.assertEqual(len(cal_rg), 2)
        self.assertTrue(all(v >= 0 for v in cal_rg))


if __name__ == "__main__":
    unittest.main()
