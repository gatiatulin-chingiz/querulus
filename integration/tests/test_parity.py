"""Parity: preds сервиса vs ``df_for_service_{main,second}`` до 2 знаков.

Эталон в parquet: колонки ``preds_cf`` / ``preds_rg`` — ожидаемые
``classification_proba`` и ``regression_predictions`` ответа сервиса
(для second_ — уже после калибраторов и маски IS_LAWYER).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_INTEGRATION = Path(__file__).resolve().parents[1]
_QUERULUS = _INTEGRATION.parent
if str(_QUERULUS) not in sys.path:
    sys.path.insert(0, str(_QUERULUS))

from integration.service.mldataworker_shim import install_mldataworker_shim

install_mldataworker_shim()

import integration.config as config  # noqa: E402
from integration.scoring.second import score_second  # noqa: E402
from integration.service.main import main_predict  # noqa: E402
from integration.tests.artifacts import (  # noqa: E402
    ART,
    max_rows_limit,
    require_file,
)


#new
def _patch_models_dir() -> None:
    config.prod_models_path = str(ART.results_dir.resolve())
    config.base_path = _INTEGRATION
    config.shadow_models_subdir = ""
    config.second_cf_calibrator_filename = ART.cf_calibrator.name
    config.second_rg_calibrator_filename = ART.rg_calibrator.name
    if "config" in sys.modules and sys.modules["config"] is not config:
        top = sys.modules["config"]
        top.prod_models_path = config.prod_models_path
        top.base_path = config.base_path
        top.shadow_models_subdir = ""
        top.second_cf_calibrator_filename = ART.cf_calibrator.name
        top.second_rg_calibrator_filename = ART.rg_calibrator.name


#new
def _row_to_request(row: pd.Series) -> list[dict]:
    d = row.to_dict()
    for k in (ART.preds_cf_col, ART.preds_rg_col, ART.split_col):
        d.pop(k, None)
    clean = {}
    for k, v in d.items():
        if pd.isna(v):
            clean[k] = None
        elif hasattr(v, "isoformat"):
            clean[k] = v.isoformat()
        elif isinstance(v, (np.integer,)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean[k] = float(v)
        else:
            clean[k] = v
    return [clean]


#new
def _iter_rows(df: pd.DataFrame):
    """Все строки; если есть split — по train/test; лимит INTEGRATION_TEST_MAX_ROWS."""
    limit = max_rows_limit()
    if ART.split_col in df.columns:
        for split in ("train", "test"):
            part = df[df[ART.split_col] == split]
            if part.empty:
                continue
            if limit is not None:
                part = part.head(limit)
            for idx, row in part.iterrows():
                yield split, idx, row
    else:
        part = df if limit is None else df.head(limit)
        for idx, row in part.iterrows():
            yield "all", idx, row


#new
def _assert_preds(test: unittest.TestCase, cf: float, rg: float, row: pd.Series, tag: str):
    test.assertEqual(
        round(cf, 2),
        round(float(row[ART.preds_cf_col]), 2),
        msg=f"{tag} CF mismatch",
    )
    test.assertEqual(
        round(rg, 2),
        round(float(row[ART.preds_rg_col]), 2),
        msg=f"{tag} RG mismatch",
    )


#new
class TestParityMain(unittest.TestCase):
    """main_predict vs df_for_service_main."""

    @classmethod
    def setUpClass(cls):
        require_file(ART.main_pickle, "main_pickle")
        require_file(ART.df_for_service_main, "df_for_service_main")
        _patch_models_dir()

    def test_service_matches_df_for_service_main(self):
        df = pd.read_parquet(ART.df_for_service_main)
        self.assertIn(ART.preds_cf_col, df.columns)
        self.assertIn(ART.preds_rg_col, df.columns)
        n = 0
        for split, idx, row in _iter_rows(df):
            out = main_predict(ART.main_stem, _row_to_request(row))
            res = out["main_response"]["result"]
            _assert_preds(
                self,
                float(res["classification_proba"][0]),
                float(res["regression_predictions"][0]),
                row,
                f"main split={split} idx={idx}",
            )
            n += 1
        self.assertGreater(n, 0, "df_for_service_main пуст или отфильтрован")


#new
class TestParitySecond(unittest.TestCase):
    """score_second vs df_for_service_second."""

    @classmethod
    def setUpClass(cls):
        require_file(ART.second_pickle, "second_pickle")
        require_file(ART.df_for_service_second, "df_for_service_second")
        require_file(ART.cf_calibrator, "cf_calibrator")
        require_file(ART.rg_calibrator, "rg_calibrator")
        _patch_models_dir()

    def test_service_matches_df_for_service_second(self):
        df = pd.read_parquet(ART.df_for_service_second)
        self.assertIn(ART.preds_cf_col, df.columns)
        self.assertIn(ART.preds_rg_col, df.columns)
        n = 0
        for split, idx, row in _iter_rows(df):
            out = score_second(ART.second_stem, _row_to_request(row))
            res = out["result"]
            _assert_preds(
                self,
                float(res["classification_proba"][0]),
                float(res["regression_predictions"][0]),
                row,
                f"second split={split} idx={idx}",
            )
            n += 1
        self.assertGreater(n, 0, "df_for_service_second пуст или отфильтрован")


#new
class TestDualPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_file(ART.main_pickle, "main_pickle")
        require_file(ART.second_pickle, "second_pickle")
        require_file(ART.df_for_service_main, "df_for_service_main")
        require_file(ART.df_for_service_second, "df_for_service_second")
        _patch_models_dir()

    def test_both_slots_filled(self):
        main_df = pd.read_parquet(ART.df_for_service_main)
        second_df = pd.read_parquet(ART.df_for_service_second)
        main_row = main_df.iloc[0]
        second_row = second_df.iloc[0]
        out = main_predict(
            ART.main_stem,
            _row_to_request(main_row),
            ART.second_stem,
            _row_to_request(second_row),
        )
        self.assertIn("oisuu_responce", out)
        self.assertIn("main_response", out)
        self.assertTrue(out["second_response"].get("result"))


if __name__ == "__main__":
    unittest.main()
