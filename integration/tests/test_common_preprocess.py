"""Общие preprocess-проверки для main_ и second_ (параметризация slot)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_INTEGRATION = Path(__file__).resolve().parents[1]
_QUERULUS = _INTEGRATION.parent
if str(_QUERULUS) not in sys.path:
    sys.path.insert(0, str(_QUERULUS))

from integration.service.mldataworker_shim import install_mldataworker_shim

install_mldataworker_shim()

import numpy as np
import pandas as pd

from integration.scoring.second import prepare_vector_upper, preprocess_second_vector
from integration.service.main import (
    enrich_common_vector_dataframe,
    prepare_common_vector_dataframe,
    preprocessing_and_validate_vector,
    validate_derived_dates_strict,
    validate_no_duplicate_columns,
    validate_preprocessing_invariants,
)
from integration.service.masking import masked_predictions


#new
def _base_vector(**overrides) -> list[dict]:
    row = {
        "EVENT_DATE": "2024-06-01T10:00:00",
        "PAYMENT_ORDER_DATE_TIME": "2024-06-10T00:00:00",
        "APPLICANT_FORM": "выгодоприобретатель",
        "FILIAL": "москва",
        "EMPTY_COL": "",
        "IS_LAWYER": 0,
        "VALUE_BEFORE_WITH": 100000,
        "VALUE_BEFORE_WITHOUT": 120000,
        "WORKS": 40000,
        "AMOUNT_REPAIR": 120000,
        "VICTIM_MAX_WEIGHT": 1500,
        "VICTIM_CAPACITY_ENGINE": 100,
        "VICTIM_OBJECT_YEAR": 2018,
        "LOSS_DATE_TIME": "2024-06-05T00:00:00",
        "POLICYHOLDER_CONTRACT_END_DATE": "2024-12-01T00:00:00",
        "EVENT_CREATED_BY_GIBDD_FLAG": 1,
        "RECIEVE_METHOD": "почта",
        "FL_PHOTO_VIDEO": 0,
        "PAYMENT_RECIPIENT_TYPE": "иное",
        "VICTIM_TS_REGION": "москва",
        "LOSS_UNIT_TYPE": "филиал",
        "VICTIM_VEHICLE_CATEGORY": "эконом",
        "VIC_TS_COUNTRY": "россия",
        "APPLICANT_AGE": 40,
        "VICTIM_AGE": 40,
        "NOT_NOTIFICATION": 0,
        "IS_TOTAL": 0,
        "REGION": "москва",
        "GUILTY_MAX_WEIGHT": 1500,
        "VICTIM_VEHICLE_COUNTRY": "россия",
        "VICTIM_VEHICLE_AGE": 5,
        "LOSS_UNIT_ZONE": "зона",
        "GUILTY_CAPACITY_ENGINE": 100,
    }
    row.update(overrides)
    return [row]


#new
class TestCommonPreprocess(unittest.TestCase):
    def test_empty_string_to_nan_main(self):
        df = prepare_common_vector_dataframe(_base_vector())
        self.assertTrue(pd.isna(df["EMPTY_COL"].iloc[0]))

    def test_empty_string_to_nan_second(self):
        df = prepare_vector_upper(_base_vector())
        self.assertTrue(pd.isna(df["EMPTY_COL"].iloc[0]))

    def test_upper_main(self):
        df = prepare_common_vector_dataframe(_base_vector())
        self.assertEqual(df["FILIAL"].iloc[0], "МОСКВА")

    def test_upper_second(self):
        df = prepare_vector_upper(_base_vector())
        self.assertEqual(df["FILIAL"].iloc[0], "МОСКВА")

    def test_no_duplicate_columns(self):
        df = pd.DataFrame([[1, 2]], columns=["A", "A"])
        with self.assertRaises(ValueError):
            validate_no_duplicate_columns(df)

    def test_dates_strict_main(self):
        df = preprocessing_and_validate_vector(_base_vector())
        validate_derived_dates_strict(df)
        self.assertIn("EVENT_YEAR", df.columns)
        self.assertIn("APPLY_DELAY", df.columns)

    def test_dates_strict_second(self):
        df = preprocess_second_vector(_base_vector())
        self.assertFalse(df["EVENT_DATE"].isna().any())
        self.assertIn("FE_VALUE_BEFORE_WITH_REAL_2022", df.columns)
        self.assertIn("FE_SHARE_WORK_TIER", df.columns)

    def test_bad_date_order_main(self):
        with self.assertRaises(ValueError):
            preprocessing_and_validate_vector(
                _base_vector(
                    EVENT_DATE="2024-07-01T00:00:00",
                    PAYMENT_ORDER_DATE_TIME="2024-06-01T00:00:00",
                )
            )

    def test_bad_date_order_second(self):
        with self.assertRaises(ValueError):
            preprocess_second_vector(
                _base_vector(
                    EVENT_DATE="2024-07-01T00:00:00",
                    PAYMENT_ORDER_DATE_TIME="2024-06-01T00:00:00",
                )
            )

    def test_is_lawyer_mask_both_slots(self):
        labels = [0.0, 0.0, 1.0]
        raw = [100.0, 200.0, 300.0]
        df = pd.DataFrame({"IS_LAWYER": [0, 1, 0]})
        masked = masked_predictions(labels, raw, df)
        self.assertEqual(masked, [0.0, 200.0, 300.0])


if __name__ == "__main__":
    unittest.main()
