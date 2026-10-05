"""Возраст ТС в месяцах из VICTIM_OBJECT_YEAR + EVENT_DATE."""
from __future__ import annotations

import unittest

import pandas as pd

from querulus.features.derived import (
    VICTIM_OBJECT_AGE_MONTHS_COL,
    ensure_victim_object_age_months,
    remap_feature_names,
    vehicle_age_months_from_year,
)
from querulus.features.integer_casts import is_year_feature


class VictimObjectAgeMonthsTests(unittest.TestCase):
    def test_age_months_january_manufacture(self) -> None:
        # 2020 год выпуска ≈ 2020-01-01; на 2024-07 → 4*12 + 6 = 54
        age = vehicle_age_months_from_year(
            pd.Series([2020, 2024, 2025]),
            pd.Series(["2024-07-15", "2024-01-01", "2024-06-01"]),
        )
        self.assertEqual(list(age), [54, 0, 0])

    def test_negative_clipped_to_zero(self) -> None:
        age = vehicle_age_months_from_year(
            pd.Series([2030]),
            pd.Series(["2024-01-01"]),
        )
        self.assertEqual(int(age.iloc[0]), 0)

    def test_ensure_uses_event_date(self) -> None:
        df = pd.DataFrame(
            {
                "VICTIM_OBJECT_YEAR": [2018],
                "EVENT_DATE": ["2022-03-10"],
                "PAYMENT_ORDER_DATE_TIME": ["2025-01-01"],
            }
        )
        out = ensure_victim_object_age_months(df)
        # (2022-2018)*12 + (3-1) = 48+2 = 50
        self.assertEqual(int(out[VICTIM_OBJECT_AGE_MONTHS_COL].iloc[0]), 50)

    def test_remap_feature_names(self) -> None:
        self.assertEqual(
            remap_feature_names(["REGION", "VICTIM_OBJECT_YEAR", "EVENT_YEAR"]),
            ["REGION", "VICTIM_OBJECT_AGE_MONTHS", "EVENT_YEAR"],
        )

    def test_age_months_is_not_year_feature(self) -> None:
        self.assertFalse(is_year_feature("VICTIM_OBJECT_AGE_MONTHS"))
        self.assertTrue(is_year_feature("VICTIM_OBJECT_YEAR"))
        self.assertTrue(is_year_feature("EVENT_YEAR"))


if __name__ == "__main__":
    unittest.main()
