"""Специфика second_: CPI, FE_*, THRESHOLD из вектора, калибраторы."""
from __future__ import annotations

import unittest

import pandas as pd

from integration.features.cpi import (
    RU_CPI_LEVEL_VS_BASE,
    add_real_monetary_columns,
    resolve_cpi_year,
)
from integration.features.derived import add_second_model_features
from integration.scoring.second import (
    apply_cf_calibrator,
    apply_rg_calibrator,
    resolve_second_threshold,
)
from integration.tests.artifacts import ART, require_file


#new
class TestCpi(unittest.TestCase):
    def test_exact_year(self):
        self.assertEqual(resolve_cpi_year(2022), 2022)

    def test_future_year_uses_max(self):
        max_y = max(RU_CPI_LEVEL_VS_BASE)
        self.assertEqual(resolve_cpi_year(max_y + 5), max_y)

    def test_past_year_uses_min(self):
        min_y = min(RU_CPI_LEVEL_VS_BASE)
        self.assertEqual(resolve_cpi_year(min_y - 3), min_y)

    def test_deflate_bridge(self):
        df = pd.DataFrame(
            {
                "EVENT_DATE": ["2022-06-01", "2022-06-01"],
                "AMOUNT_REPAIR": [1000.0, 2000.0],
            }
        )
        out = add_real_monetary_columns(df)
        self.assertIn("VALUE_BEFORE_WITHOUT", out.columns)
        self.assertIn("FE_VALUE_BEFORE_WITHOUT_REAL_2022", out.columns)


#new
class TestDerivedFe(unittest.TestCase):
    def test_share_work_tier_from_works(self):
        df = pd.DataFrame(
            {
                "EVENT_DATE": ["2022-01-15"],
                "PAYMENT_ORDER_DATE_TIME": ["2022-02-01"],
                "VALUE_BEFORE_WITH": [80000.0],
                "VALUE_BEFORE_WITHOUT": [100000.0],
                "WORKS": [50000.0],
                "VICTIM_CAPACITY_ENGINE": [100.0],
                "VICTIM_MAX_WEIGHT": [1500.0],
                "VICTIM_OBJECT_YEAR": [2018],
                "LOSS_DATE_TIME": ["2022-01-20"],
                "POLICYHOLDER_CONTRACT_END_DATE": ["2022-08-01"],
            }
        )
        out = add_second_model_features(df)
        self.assertIn("FE_SHARE_WORK_TIER", out.columns)
        self.assertIn("FE_VALUE_BEFORE_RATIO", out.columns)
        self.assertIn("FE_VICTIM_POWER_PER_TON", out.columns)
        # 0.5 → mid (bins 0.3, 0.6)
        self.assertEqual(str(out["FE_SHARE_WORK_TIER"].iloc[0]), "mid")


#new
class TestVectorThreshold(unittest.TestCase):
    def test_resolve_from_vector(self):
        df = pd.DataFrame({"THRESHOLD": [0.42]})
        self.assertAlmostEqual(resolve_second_threshold(df), 0.42)

    def test_missing_threshold_uses_config_fallback(self):
        from integration import config

        thr = resolve_second_threshold(pd.DataFrame({"X": [1]}))
        self.assertAlmostEqual(thr, float(config.default_classification_threshold))


#new
class TestCalibrators(unittest.TestCase):
    def test_cf_rg_calibrators_load_and_apply(self):
        import pickle

        cf_path = require_file(ART.cf_calibrator, "cf_calibrator")
        rg_path = require_file(ART.rg_calibrator, "rg_calibrator")
        with open(cf_path, "rb") as f:
            cf_cal = pickle.load(f)
        with open(rg_path, "rb") as f:
            rg_cal = pickle.load(f)
        raw = [0.2, 0.5, 0.8]
        cal_cf = apply_cf_calibrator(cf_cal, raw, pd.DataFrame())
        self.assertEqual(len(cal_cf), 3)
        cal_rg = apply_rg_calibrator(rg_cal, [1000.0, 5000.0])
        self.assertEqual(len(cal_rg), 2)
        self.assertTrue(all(v >= 0 for v in cal_rg))


if __name__ == "__main__":
    unittest.main()
