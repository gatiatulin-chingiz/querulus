"""Тесты flat-сервиса: пайплайны из volume, маска, CPI, dual-track smoke.

Запуск: ``python tests.py`` из ``integration/``.
Артефакты: ``./results`` или ``INTEGRATION_TEST_RESULTS_DIR``.
"""
from __future__ import annotations

import asyncio
import os
import sys
import unittest
from pathlib import Path

import pandas as pd

_INTEGRATION = Path(__file__).resolve().parent
if str(_INTEGRATION) not in sys.path:
    sys.path.insert(0, str(_INTEGRATION))

from main import (  # noqa: E402
    _load_module,
    cnfg,
    load_registry,
    score_track,
)


def _results_dir() -> Path:
    raw = os.environ.get("INTEGRATION_TEST_RESULTS_DIR", "").strip()
    return Path(raw) if raw else _INTEGRATION / "results"


def _pipelines_dir() -> Path:
    return _results_dir() / "pipelines"


def _load_pipeline(name: str):
    path = _pipelines_dir() / f"{name}.py"
    return _load_module(path, f"test_pipeline_{name}")


def _base_vector(**overrides) -> list[dict]:
    row = {
        "EVENT_DATE": "2024-06-01T10:00:00",
        "PAYMENT_ORDER_DATE_TIME": "2024-06-10T00:00:00",
        "APPLICANT_FORM": "выгодоприобретатель",
        "FILIAL": "москва",
        "EMPTY_COL": "",
        "IS_LAWYER": 0,
        "INCIDENT_NUMBER": "INC-1",
        "LOSS_NUMBER": "LOSS-1",
        "VALUE_BEFORE_WITH": 100000,
        "VALUE_BEFORE_WITHOUT": 120000,
        "WORK": 40000,
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
        "VICTIM_VEHICLE_BRAND": "toyota",
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


class TestCpi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.second = _load_pipeline("second_pipeline")

    def test_exact_year(self):
        self.assertEqual(self.second.resolve_cpi_year(2022), 2022)

    def test_future_year_uses_max(self):
        max_y = max(self.second.RU_CPI_LEVEL_VS_BASE)
        self.assertEqual(self.second.resolve_cpi_year(max_y + 5), max_y)

    def test_past_year_uses_min(self):
        min_y = min(self.second.RU_CPI_LEVEL_VS_BASE)
        self.assertEqual(self.second.resolve_cpi_year(min_y - 3), min_y)


class TestPreprocess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _load_pipeline("main_pipeline")
        cls.second = _load_pipeline("second_pipeline")
        cls.brands = cls.main.load_ts_country_by_brand(_results_dir())

    def test_empty_string_to_nan(self):
        df = self.main._prepare_vector_upper(_base_vector())
        self.assertTrue(pd.isna(df["EMPTY_COL"].iloc[0]))

    def test_main_no_fe_real(self):
        df = self.main.preprocess(_base_vector(), self.brands)
        self.assertNotIn("FE_VALUE_BEFORE_WITH_REAL_2022", df.columns)
        self.assertIn("VIC_TS_COUNTRY", df.columns)

    def test_second_has_fe(self):
        df = self.second.preprocess(_base_vector(), self.brands)
        self.assertIn("FE_VALUE_BEFORE_WITH_REAL_2022", df.columns)
        self.assertIn("FE_SHARE_WORK_TIER", df.columns)


class TestMaskAndCal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _load_pipeline("main_pipeline")

    def test_mask_zeros_when_not_lawyer(self):
        df = pd.DataFrame({"IS_LAWYER": [0, 1]})
        out = self.main.postprocess(df, [0.0, 0.0], [10.0, 20.0], None, 0.5)
        self.assertEqual(out["regression"], [0.0, 20.0])

    def test_mask_keeps_class_one(self):
        df = pd.DataFrame({"IS_LAWYER": [0]})
        out = self.main.postprocess(df, [1.0], [15.0], None, 0.5)
        self.assertEqual(out["regression"], [15.0])


class TestDualTrackSmoke(unittest.TestCase):
    """Smoke по registry: нужны все pickle из models.json в results/."""

    @classmethod
    def setUpClass(cls):
        try:
            cnfg.registry = load_registry(_results_dir())
        except FileNotFoundError as exc:
            raise unittest.SkipTest(f"нет pickle/models.json: {exc}")
        cls.registry = cnfg.registry

    def _first_stem(self, pipeline_name: str) -> str:
        return next(
            s for s, name in self.registry.pipelines.items() if name == pipeline_name
        )

    def test_models_json_loaded(self):
        main_stem = "querulus_ansamble_2026_04_18_v1"
        second_stem = "UU_querulus_2026_11_05_12_44_24"
        r = self.registry
        self.assertEqual(
            r.pipelines,
            {main_stem: "main_pipeline", second_stem: "second_pipeline"},
        )
        self.assertEqual(r.thresholds, {main_stem: 0.6, second_stem: 0.37})
        self.assertEqual(r.versions, {main_stem: "1.4.0", second_stem: "2.0.0"})
        self.assertEqual(set(r.models), {main_stem, second_stem})
        self.assertEqual(set(r.calibrators), {second_stem})
        self.assertIn(main_stem, r.ts_country_by_brand)
        self.assertIn(second_stem, r.ts_country_by_brand)

    def test_score_main(self):
        stem = self._first_stem("main_pipeline")
        resp = asyncio.run(score_track(self.registry, stem, _base_vector()))
        self.assertEqual(resp["usage_model"], stem)
        self.assertIn("input_raw", resp["df"])
        self.assertIn("output_preprocessed", resp["df"])
        self.assertIn("output_outboxml", resp["df"])
        self.assertNotIn("input", resp["df"])
        self.assertIn("CPMComment", resp)
        self.assertIn("classification_proba", resp["result"])

    def test_score_second(self):
        stem = self._first_stem("second_pipeline")
        resp = asyncio.run(score_track(self.registry, stem, _base_vector()))
        self.assertEqual(resp["usage_model"], stem)
        self.assertEqual(resp["version"], self.registry.versions[stem])
        self.assertIn("output_outboxml", resp["df"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
