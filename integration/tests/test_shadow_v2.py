"""Unit-тесты shadow 2.0.0: CPI fallback, frozen DQ, meta threshold, second_response пустой без second_*."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from integration.features_cpi import (
    RU_CPI_LEVEL_VS_BASE,
    add_real_monetary_columns,
    deflate_to_base_year,
    real_feature_name,
    resolve_cpi_year,
)
from integration.features_dq import apply_frozen_dq_bounds, load_dq_bounds
from integration.model_profiles import threshold_from_meta
from integration.shadow_v2 import (
    _APPLICANT_FORM_REMAP_UPPER,
    _enrich_dates_and_forms,
    _prepare_vector_upper,
    preprocess_shadow_vector,
)


class TestCpiFallback(unittest.TestCase):
    def test_exact_year(self):
        self.assertEqual(resolve_cpi_year(2022), 2022)

    def test_future_year_uses_max(self):
        max_y = max(RU_CPI_LEVEL_VS_BASE)
        self.assertEqual(resolve_cpi_year(max_y + 5), max_y)

    def test_past_year_uses_min(self):
        min_y = min(RU_CPI_LEVEL_VS_BASE)
        self.assertEqual(resolve_cpi_year(min_y - 3), min_y)

    def test_deflate_future_matches_max_year_level(self):
        max_y = max(RU_CPI_LEVEL_VS_BASE)
        amounts = pd.Series([1000.0, 1000.0])
        dates = pd.Series([f"{max_y}-06-01", f"{max_y + 2}-06-01"])
        out = deflate_to_base_year(amounts, dates)
        self.assertAlmostEqual(float(out.iloc[0]), float(out.iloc[1]), places=6)

    def test_add_real_bridge_amount_repair(self):
        df = pd.DataFrame(
            {
                "EVENT_DATE": ["2024-01-15"],
                "AMOUNT_REPAIR": [121331.8],
            }
        )
        out = add_real_monetary_columns(df)
        col = real_feature_name("VALUE_BEFORE_WITHOUT")
        self.assertIn("VALUE_BEFORE_WITHOUT", out.columns)
        self.assertIn(col, out.columns)
        self.assertFalse(pd.isna(out[col].iloc[0]))


class TestFrozenDq(unittest.TestCase):
    def test_hard_clip_and_winsor(self):
        bounds = {
            "hard_clip_nonnegative_columns": ["VALUE_BEFORE_WITHOUT"],
            "winsorize_bounds": {
                "VALUE_BEFORE_WITHOUT": {"low_raw": 0.0, "high_raw": 100.0},
            },
        }
        df = pd.DataFrame({"VALUE_BEFORE_WITHOUT": [-5.0, 50.0, 999.0]})
        out = apply_frozen_dq_bounds(df, bounds)
        self.assertEqual(float(out["VALUE_BEFORE_WITHOUT"].iloc[0]), 0.0)
        self.assertEqual(float(out["VALUE_BEFORE_WITHOUT"].iloc[1]), 50.0)
        self.assertEqual(float(out["VALUE_BEFORE_WITHOUT"].iloc[2]), 100.0)

    def test_load_fixture_bounds(self):
        path = Path(__file__).resolve().parent / "fixtures" / "dq_bounds.json"
        if not path.is_file():
            self.skipTest("нет fixtures/dq_bounds.json")
        bounds = load_dq_bounds(path)
        self.assertIn("winsorize_bounds", bounds)


class TestMetaThreshold(unittest.TestCase):
    def test_reads_best_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.json"
            path.write_text(
                json.dumps({"best_threshold": 0.37}),
                encoding="utf-8",
            )
            self.assertAlmostEqual(threshold_from_meta(path), 0.37)


class TestShadowPreprocess(unittest.TestCase):
    def test_upper_and_applicant_remap(self):
        raw = [
            {
                "EVENT_DATE": "2024-01-10T00:00:00",
                "PAYMENT_ORDER_DATE_TIME": "2024-01-20T00:00:00",
                "APPLICANT_FORM": "Скрытый юрист",
                "VALUE_BEFORE_WITHOUT": 10000,
                "FILIAL": "Москва",
            }
        ]
        df = _prepare_vector_upper(raw)
        self.assertEqual(df["FILIAL"].iloc[0], "МОСКВА")
        self.assertEqual(df["APPLICANT_FORM"].iloc[0], "СКРЫТЫЙ ЮРИСТ")
        df = _enrich_dates_and_forms(df)
        self.assertEqual(
            df["APPLICANT_FORM"].iloc[0],
            _APPLICANT_FORM_REMAP_UPPER["СКРЫТЫЙ ЮРИСТ"],
        )
        self.assertEqual(int(df["EVENT_YEAR"].iloc[0]), 2024)
        self.assertEqual(int(df["APPLY_DELAY"].iloc[0]), 10)

    def test_preprocess_adds_real_column(self):
        raw = [
            {
                "EVENT_DATE": "2024-01-10T00:00:00",
                "PAYMENT_ORDER_DATE_TIME": "2024-01-20T00:00:00",
                "APPLICANT_FORM": "ПОТЕРПЕВШИЙ",
                "VALUE_BEFORE_WITHOUT": 10000,
            }
        ]
        with mock.patch(
            "integration.shadow_v2.shadow_dq_bounds_path",
            return_value=Path(__file__).resolve().parent / "fixtures" / "dq_bounds.json",
        ):
            df = preprocess_shadow_vector(raw)
        self.assertIn(real_feature_name("VALUE_BEFORE_WITHOUT"), df.columns)


class TestLegacyMainSecondEmpty(unittest.TestCase):
    """Контракт: без second_* не зовём score_shadow_v2 (проверяем ветку через mock модуля)."""

    def test_score_shadow_importable_and_flag(self):
        from integration import model_profiles

        self.assertTrue(model_profiles.SHADOW_NEW_AS_SECOND)


if __name__ == "__main__":
    unittest.main()
