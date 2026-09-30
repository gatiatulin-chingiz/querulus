"""Правила features[] для OutBoxML-конфигов + целочисленный DQ-winsorize.

Покрывают п.1–12 из постановки:
  * схлопывание категорий до топ-K + ``ПРОЧИЕ``; ``None``/``NaN`` → моду/``NAN``/``ПРОЧИЕ``;
  * ``clip`` у всех числовых не-бинарных, целые границы у int-подобных, год ≤ 2026;
  * ``EVENT_YEAR`` больше не превращается в ``2024.5009269176796``.

Запуск: python -m unittest discover -s tests -t . -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from querulus.features.data_quality import (  # noqa: E402
    MAX_YEAR_FEATURE_VALUE,
    apply_data_quality,
    clip_bounds_for_outboxml,
)
from querulus.training.build_outboxml_configs import (  # noqa: E402
    CAT_LEVEL_SHARE_MIN,
    CAT_MAX_CODES,
    CAT_MIN_CODES,
    CAT_NAN_SHARE_MIN,
    CAT_SMALL_CARD_MAX,
    KNOWN_CATEGORICAL,
    _categorical_feature_spec,
    _clip_for_feature,
    _is_categorical,
    _level_key,
    _numeric_feature_spec,
    build_features_block,
)


def _series(values) -> pd.Series:
    return pd.Series(values, dtype="object")


def _high_card_series(n_tail: int = 20, tail_size: int = 5, nan: int = 0) -> pd.Series:
    """400/200/120/80/60/40 + хвост по 5 + ``nan`` пропусков (всего 1000 + nan)."""
    values = (
        ["A"] * 400
        + ["B"] * 200
        + ["C"] * 120
        + ["D"] * 80
        + ["E"] * 60
        + ["F"] * 40
    )
    for index in range(n_tail):
        values += [f"T{index:03d}"] * tail_size
    values += [np.nan] * nan
    return _series(values)


class CategoricalSpecTests(unittest.TestCase):
    def test_level_key_treats_na_like_strings_as_missing(self):
        for value in (None, np.nan, pd.NA, "", "  ", "NaN", "n/a", "None"):
            self.assertIsNone(_level_key(value), msg=repr(value))
        self.assertEqual(_level_key("москва"), "МОСКВА")
        self.assertEqual(_level_key(2.0), "2")
        self.assertEqual(_level_key(True), "1")

    def test_top_k_collapse_and_other_bucket(self):
        spec = _categorical_feature_spec(_high_card_series(), "FILIAL", log=False)
        self.assertEqual(spec["replace"]["ПРОЧИЕ"], 0)
        self.assertEqual(spec["replace"]["A"], 1)
        self.assertEqual(spec["replace"]["F"], 6)
        # Хвост (доля 0.5% < 1%) — явно в ПРОЧИЕ, а не в default.
        self.assertEqual(spec["replace"]["T000"], 0)
        self.assertEqual(spec["default"], 0)
        self.assertNotIn("fillna", spec)  # NaN схлопывается в ПРОЧИЕ через default
        self.assertEqual(spec["encoding"], "to_int")

    def test_top_k_is_capped_and_tail_bounded_by_code_count(self):
        values = [f"L{index:03d}" for index in range(60) for _ in range(20)]  # 60 уровней по ~1.6%
        spec = _categorical_feature_spec(_series(values), "REGION", log=False)
        codes = {code for code in spec["replace"].values()}
        # ПРОЧИЕ (0) + не больше CAT_MAX_CODES своих кодов.
        self.assertLessEqual(len(codes - {0}), CAT_MAX_CODES)
        self.assertGreaterEqual(len(codes - {0}), CAT_MIN_CODES)

    def test_min_codes_even_below_share_threshold(self):
        # Доля ≥1% только у A и B → порог даёт 2 уровня, минимум CAT_MIN_CODES = 5.
        values = ["A"] * 900 + ["B"] * 90 + ["C"] * 4 + ["D"] * 3 + ["E"] * 2 + ["F"] * 1
        spec = _categorical_feature_spec(_series(values), "APPLICANT_FORM", log=False)
        own_codes = {
            code for key, code in spec["replace"].items() if key != "ПРОЧИЕ"
        }
        # F (доля 0.1%) — уже в ПРОЧИЕ (код 0), «своих» кодов ровно CAT_MIN_CODES.
        self.assertEqual(spec["replace"]["F"], 0)
        self.assertEqual(len(own_codes - {0}), CAT_MIN_CODES)
        self.assertGreater(CAT_LEVEL_SHARE_MIN, 0)

    def test_small_cardinality_sends_nan_to_mode(self):
        values = ["ПОЧТА"] * 40 + ["ОЧНО"] * 30 + ["ЭЛЕКТРОННО"] * 20 + [np.nan] * 5
        series = _series(values)
        self.assertLessEqual(series.nunique(dropna=True), CAT_SMALL_CARD_MAX)
        spec = _categorical_feature_spec(series, "RECIEVE_METHOD", log=False)
        self.assertEqual(spec["default"], spec["replace"]["ПОЧТА"])  # мода
        self.assertNotIn("fillna", spec)
        self.assertNotIn("NAN", spec["replace"])

    def test_many_nan_becomes_own_category(self):
        series = _high_card_series(nan=200)  # 200/1200 ≈ 16.7% ≥ 10%
        spec = _categorical_feature_spec(series, "VICTIM_TS_REGION", log=False)
        nan_code = spec["replace"]["NAN"]
        self.assertGreater(nan_code, 0)  # код 0 falsy → outboxml проигнорировал бы fillna
        self.assertEqual(spec["fillna"], nan_code)
        self.assertEqual(spec["default"], spec["replace"]["A"])  # default — мода, не NaN
        self.assertNotEqual(spec["default"], nan_code)

    def test_few_nan_high_cardinality_goes_to_other(self):
        series = _high_card_series(nan=50)  # 50/1050 ≈ 4.8% < 10%
        spec = _categorical_feature_spec(series, "REGION", log=False)
        self.assertNotIn("NAN", spec["replace"])
        self.assertNotIn("fillna", spec)
        self.assertEqual(spec["default"], 0)

    def test_binary_feature_keeps_historic_shape(self):
        spec = _categorical_feature_spec(_series([0] * 30 + [1] * 70 + [np.nan] * 10), "FLAG", log=False)
        self.assertEqual(spec["replace"], {"0": 0, "1": 1})
        self.assertEqual(spec["default"], 1)  # мода
        self.assertEqual(spec["fillna"], 1)

    def test_all_missing_feature(self):
        spec = _categorical_feature_spec(_series([np.nan] * 30), "PAYMENT_RECIPIENT_TYPE", log=False)
        self.assertEqual(spec["replace"], {"ПРОЧИЕ": 0, "NAN": 1})
        self.assertEqual(spec["fillna"], 1)

    def test_literal_other_level_is_merged_with_bucket(self):
        values = ["ПРОЧИЕ"] * 50 + ["МОСКВА"] * 50
        spec = _categorical_feature_spec(_series(values), "FILIAL", log=False)
        self.assertEqual(spec["replace"]["ПРОЧИЕ"], 0)
        self.assertEqual(spec["replace"]["МОСКВА"], 1)

    def test_thresholds_are_the_agreed_ones(self):
        self.assertEqual(CAT_NAN_SHARE_MIN, 0.10)
        self.assertEqual(CAT_LEVEL_SHARE_MIN, 0.01)
        self.assertEqual(CAT_MAX_CODES, 20)
        self.assertEqual(CAT_MIN_CODES, 5)
        self.assertEqual(CAT_SMALL_CARD_MAX, 10)

    def test_known_categorical_numeric_code_is_categorical(self):
        series = pd.Series(np.tile(np.arange(89), 10), name="REGION")
        self.assertIn("REGION", KNOWN_CATEGORICAL)
        self.assertTrue(
            _is_categorical("REGION", json_cats=set(), mvp_cats=set(), series=series)
        )


class NumericSpecTests(unittest.TestCase):
    def test_integer_like_features_get_to_int(self):
        age = pd.Series(np.linspace(18, 80, 200), name="APPLICANT_AGE")
        spec = _numeric_feature_spec(age, "APPLICANT_AGE")
        self.assertEqual(spec["encoding"], "to_int")
        self.assertEqual(spec["default"], "_MEDIAN_")

    def test_year_feature_default_is_max_and_clip_2026(self):
        year = pd.Series([2022] * 100 + [2023] * 60 + [2024] * 40, name="EVENT_YEAR")
        spec = _numeric_feature_spec(year, "EVENT_YEAR")
        self.assertEqual(spec["encoding"], "to_int")
        # «Последний (больший) год» — конкретным int: "_MAX_" ломает целые колонки
        # (np.int64 не проходит isinstance(default, (int, float)) в OutBoxML).
        self.assertEqual(spec["default"], 2024)
        self.assertIsInstance(spec["default"], int)
        clip = _clip_for_feature(year, "EVENT_YEAR", log=False)
        self.assertEqual(clip["max_value"], float(MAX_YEAR_FEATURE_VALUE))
        self.assertEqual(clip["min_value"], float(int(clip["min_value"])))

    def test_year_default_is_capped_by_current_year(self):
        year = pd.Series([2025] * 10 + [2027] * 5, name="VICTIM_OBJECT_YEAR")
        spec = _numeric_feature_spec(year, "VICTIM_OBJECT_YEAR")
        self.assertEqual(spec["default"], MAX_YEAR_FEATURE_VALUE)
        self.assertIsInstance(spec["default"], int)

    def test_year_default_for_all_missing_column(self):
        year = pd.Series([np.nan] * 30, name="GUILTY_OBJECT_YEAR")
        spec = _numeric_feature_spec(year, "GUILTY_OBJECT_YEAR")
        self.assertEqual(spec["default"], MAX_YEAR_FEATURE_VALUE)

    def test_year_dq_bounds_are_capped_to_2026(self):
        year = pd.Series([2022] * 100 + [2023] * 100, name="VICTIM_OBJECT_YEAR")
        clip = _clip_for_feature(
            year,
            "VICTIM_OBJECT_YEAR",
            dq_bounds={"VICTIM_OBJECT_YEAR": {"min_value": 2020.5009263071117, "max_value": 2024.5009269176796}},
            log=False,
        )
        self.assertEqual(clip, {"min_value": 2020.0, "max_value": 2026.0})

    def test_integer_bounds_are_floored_and_ceiled(self):
        age = pd.Series(np.linspace(18.4, 88.7, 300), name="VICTIM_AGE")
        clip = _clip_for_feature(
            age,
            "VICTIM_AGE",
            dq_bounds={"VICTIM_AGE": {"min_value": 18.8982483379393, "max_value": 88.70638870742215}},
            log=False,
        )
        self.assertEqual(clip, {"min_value": 18.0, "max_value": 89.0})

    def test_float_feature_keeps_fractional_bounds(self):
        weight = pd.Series(np.linspace(900, 6000, 300), name="VICTIM_MAX_WEIGHT")
        clip = _clip_for_feature(
            weight,
            "VICTIM_MAX_WEIGHT",
            dq_bounds={"VICTIM_MAX_WEIGHT": {"min_value": 879.3689875693901, "max_value": 6073.849159949769}},
            log=False,
        )
        self.assertAlmostEqual(clip["min_value"], 879.3689875693901)
        self.assertAlmostEqual(clip["max_value"], 6073.849159949769)

    def test_clip_fallback_by_quantiles_without_dq_bounds(self):
        values = pd.Series(np.r_[np.full(200, 5.0), np.linspace(6, 50, 200)], name="VICTIM_MAX_WEIGHT")
        clip = _clip_for_feature(values, "VICTIM_MAX_WEIGHT", dq_bounds={}, log=False)
        self.assertIsNotNone(clip)
        self.assertLessEqual(clip["min_value"], clip["max_value"])

    def test_degenerate_feature_gets_no_clip(self):
        clip = _clip_for_feature(pd.Series([7.0] * 100, name="X_CONST"), "X_CONST", log=False)
        self.assertIsNone(clip)

    def test_build_features_block_clips_every_numeric(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "FILIAL": ["МОСКВА", "СПБ"] * 100,
                "APPLICANT_AGE": rng.integers(18, 80, 200),
                "VICTIM_MAX_WEIGHT": rng.uniform(800, 3000, 200),
                "EVENT_YEAR": rng.integers(2022, 2026, 200),
            }
        )
        features, cats = build_features_block(df, list(df.columns), fit_index=df.index)
        self.assertEqual(cats, ["FILIAL"])
        for feature in features:
            if feature["name"] in cats:
                self.assertNotIn("clip", feature)
            else:
                self.assertIn("clip", feature)
        age = next(f for f in features if f["name"] == "APPLICANT_AGE")
        self.assertEqual(age["encoding"], "to_int")
        self.assertEqual(age["clip"]["min_value"], float(int(age["clip"]["min_value"])))


class DataQualityIntegerTests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        return pd.DataFrame(
            {
                "APPLICANT_AGE": rng.integers(18, 80, 400),
                "EVENT_YEAR": rng.choice([2022, 2023, 2024], 400),
                "VICTIM_MAX_WEIGHT": rng.uniform(900, 3000, 400),
            }
        )

    def test_integer_and_year_columns_use_integer_fence(self):
        df = self._frame()
        result, report = apply_data_quality(
            df,
            train_index=df.index,
            numeric_columns=list(df.columns),
        )
        integer_rows = {row["column"]: row for row in report.winsorize_iqr_integer}
        self.assertIn("APPLICANT_AGE", integer_rows)
        self.assertIn("EVENT_YEAR", integer_rows)
        self.assertNotIn("VICTIM_MAX_WEIGHT", integer_rows)
        self.assertEqual(
            {row["column"] for row in report.winsorize_log1p_iqr}, {"VICTIM_MAX_WEIGHT"}
        )
        for row in integer_rows.values():
            self.assertEqual(row["low_raw"], float(int(row["low_raw"])))
            self.assertEqual(row["high_raw"], float(int(row["high_raw"])))
            self.assertTrue(row["integer_rounded"])
        year_row = integer_rows["EVENT_YEAR"]
        self.assertTrue(year_row["year_capped"])
        self.assertEqual(year_row["high_raw"], float(MAX_YEAR_FEATURE_VALUE))
        # Никаких дробных годов в данных после DQ.
        self.assertTrue(
            np.all(np.mod(result["EVENT_YEAR"].to_numpy(dtype=float), 1.0) == 0.0)
        )

    def test_nonnegative_integer_fence_is_not_negative(self):
        df = self._frame()
        _, report = apply_data_quality(
            df, train_index=df.index, numeric_columns=list(df.columns)
        )
        age_row = next(r for r in report.winsorize_iqr_integer if r["column"] == "APPLICANT_AGE")
        # Возраст не может быть отрицательным: даже «широкий» IQR-забор → 0.
        self.assertEqual(age_row["low_raw"], 0.0)
        self.assertTrue(age_row["nonnegative"])

    def test_legitimately_negative_lag_keeps_negative_fence(self):
        rng = np.random.default_rng(3)
        df = pd.DataFrame(
            {"FE_DAYS_LOSS_TO_PAYMENT_ORDER": rng.integers(-30, 400, 400)}
        )
        result, report = apply_data_quality(
            df, train_index=df.index, numeric_columns=list(df.columns)
        )
        row = report.winsorize_iqr_integer[0]
        self.assertFalse(row["nonnegative"])
        self.assertLess(row["low_raw"], 0.0)
        self.assertLess(float(result["FE_DAYS_LOSS_TO_PAYMENT_ORDER"].min()), 0.0)

    def test_year_winsorize_does_not_create_fractional_years(self):
        df = pd.DataFrame(
            {
                "EVENT_YEAR": [2022] * 200 + [2023] * 150 + [2024] * 40 + [2027] * 10,
            }
        )
        result, report = apply_data_quality(
            df, train_index=df.index, numeric_columns=["EVENT_YEAR"]
        )
        self.assertLessEqual(float(result["EVENT_YEAR"].max()), float(MAX_YEAR_FEATURE_VALUE))
        self.assertTrue(np.all(np.mod(result["EVENT_YEAR"].to_numpy(dtype=float), 1.0) == 0.0))
        self.assertEqual(report.winsorize_log1p_iqr, [])

    def test_clip_bounds_merge_both_sections(self):
        df = self._frame()
        _, report = apply_data_quality(df, train_index=df.index, numeric_columns=list(df.columns))
        clips = clip_bounds_for_outboxml(report)
        self.assertIn("APPLICANT_AGE", clips)
        self.assertIn("EVENT_YEAR", clips)
        self.assertIn("VICTIM_MAX_WEIGHT", clips)
        self.assertEqual(clips["EVENT_YEAR"]["max_value"], float(MAX_YEAR_FEATURE_VALUE))


class DerivedFeatureScaleTests(unittest.TestCase):
    def test_victim_power_per_ton_is_scaled_by_1e6(self):
        from types import SimpleNamespace

        from querulus.features.derived import _add_victim_vehicle_features

        df = pd.DataFrame(
            {
                "VICTIM_CAPACITY_ENGINE": [150.0],
                "VICTIM_MAX_WEIGHT": [1500.0],
                "VICTIM_VEHICLE_AGE": [5],
            }
        )
        config = SimpleNamespace(
            thresholds=SimpleNamespace(vehicle_weight_heavy=1500.0),
            vehicle_age_bins=(3, 7, 15),
        )
        out = _add_victim_vehicle_features(df, config)
        value = float(out["FE_VICTIM_POWER_PER_TON"].iloc[0])
        self.assertAlmostEqual(value, 150.0 / 1500.0 * 1e6)
        self.assertGreater(value, 1000.0)  # нормальный диапазон, а не ~0.1

    def test_value_before_diff_is_without_minus_with(self):
        from types import SimpleNamespace

        from querulus.features.derived import _add_repair_features

        df = pd.DataFrame(
            {
                "PAYMENT_ORDER_DATE_TIME": pd.to_datetime(["2022-05-01", "2022-06-01"]),
                "VALUE_BEFORE_WITHOUT": [100_000.0, 50_000.0],
                "VALUE_BEFORE_WITH": [80_000.0, 20_000.0],
                "SHARE_WORK": [0.3, 0.8],
            }
        )
        config = SimpleNamespace(
            thresholds=SimpleNamespace(
                share_work_bins=(0.33, 0.66),
                amount_repair_bins=(50_000, 150_000),
                amount_repair_high=150_000,
            ),
            t0_column="PAYMENT_ORDER_DATE_TIME",
            inflation_base_year=2022,
        )
        out = _add_repair_features(df, config)
        # 2022 — базисный год инфляции: real == nominal.
        self.assertAlmostEqual(float(out["FE_VALUE_BEFORE_DIFF"].iloc[0]), 20_000.0)
        self.assertAlmostEqual(float(out["FE_VALUE_BEFORE_DIFF"].iloc[1]), 30_000.0)
        self.assertAlmostEqual(float(out["FE_VALUE_BEFORE_RATIO"].iloc[0]), 0.8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
