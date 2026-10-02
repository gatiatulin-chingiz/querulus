"""Контракт DatasetSchema: обязательные колонки и SEV⇒FREQ."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from querulus.dataset.schema import (  # noqa: E402
    DEFAULT_DATASET_SCHEMA,
    DatasetSchemaError,
)


class DatasetSchemaTests(unittest.TestCase):
    def test_required_columns_and_defaults(self):
        s = DEFAULT_DATASET_SCHEMA
        self.assertEqual(s.date_column, "PAYMENT_ORDER_DATE_TIME")
        self.assertEqual(s.frequency_target, "TARGET_FREQ")
        self.assertEqual(s.max_year_feature_value, 2026)
        self.assertIn("REGION", s.known_categorical)
        self.assertIn("TARGET_FREQ", s.other_cols)

    def test_validate_ok(self):
        df = pd.DataFrame(
            {
                "PAYMENT_ORDER_DATE_TIME": pd.date_range("2023-01-01", periods=20, freq="D"),
                "TARGET_FREQ": [0] * 18 + [1, 1],
                "TARGET_SEV": [0.0] * 18 + [100.0, 50.0],
            }
        )
        self.assertEqual(DEFAULT_DATASET_SCHEMA.validate(df), [])

    def test_validate_missing_raises(self):
        df = pd.DataFrame({"TARGET_FREQ": [0, 1]})
        with self.assertRaises(DatasetSchemaError):
            DEFAULT_DATASET_SCHEMA.validate(df, raise_on_error=True)

    def test_validate_sev_without_freq_warns(self):
        df = pd.DataFrame(
            {
                "PAYMENT_ORDER_DATE_TIME": pd.date_range("2023-01-01", periods=5, freq="D"),
                "TARGET_FREQ": [0, 0, 0, 0, 0],
                "TARGET_SEV": [0.0, 10.0, 0.0, 0.0, 0.0],
            }
        )
        issues = DEFAULT_DATASET_SCHEMA.validate(df, raise_on_error=False)
        self.assertTrue(any("TARGET_SEV>0" in x for x in issues))


if __name__ == "__main__":
    unittest.main()
