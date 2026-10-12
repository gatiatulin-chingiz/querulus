"""Parity: CF predict_proba и RG predictions vs эталон (train+test) до 2 знаков."""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Shim до импорта service scoring
_INTEGRATION = Path(__file__).resolve().parents[1]
_QUERULUS = _INTEGRATION.parent
_SRC = _QUERULUS / "src"
_OUTBOXML = _QUERULUS.parent.parent
for p in (_SRC, _OUTBOXML, _QUERULUS, _INTEGRATION):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from integration.service.mldataworker_shim import install_mldataworker_shim

install_mldataworker_shim()

import integration.config as config  # noqa: E402
from integration.scoring.second import score_second  # noqa: E402
from integration.service.main import main_predict  # noqa: E402
from integration.tests.artifacts import ART, require_file  # noqa: E402


#new
def _patch_models_dir() -> None:
    config.prod_models_path = str(ART.results_dir.resolve())
    config.base_path = _INTEGRATION
    config.shadow_models_subdir = ""
    config.second_cf_calibrator_filename = ART.cf_calibrator.name
    config.second_rg_calibrator_filename = ART.rg_calibrator.name
    config.shadow_meta_filename = ART.metadata.name
    # Docker/flat layout may also import top-level ``config``
    if "config" in sys.modules and sys.modules["config"] is not config:
        top = sys.modules["config"]
        top.prod_models_path = config.prod_models_path
        top.base_path = config.base_path
        top.shadow_models_subdir = ""
        top.second_cf_calibrator_filename = ART.cf_calibrator.name
        top.second_rg_calibrator_filename = ART.rg_calibrator.name
        top.shadow_meta_filename = ART.metadata.name


#new
def _row_to_request(row: pd.Series) -> list[dict]:
    d = row.to_dict()
    for k in ("preds_cf", "preds_rg", "split"):
        d.pop(k, None)
    # JSON-friendly
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
class TestParityMain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_file(ART.main_pickle, "main_pickle")
        require_file(ART.df_main, "df_main")
        _patch_models_dir()

    def test_main_train_test_two_decimals(self):
        df = pd.read_parquet(ART.df_main)
        self.assertIn("split", df.columns)
        for split in ("train", "test"):
            part = df[df["split"] == split]
            self.assertGreater(len(part), 0, f"empty {split}")
            for _, row in part.iterrows():
                out = main_predict(ART.main_stem, _row_to_request(row))
                res = out["main_response"]["result"]
                cf = float(res["classification_proba"][0])
                rg = float(res["regression_predictions"][0])
                self.assertEqual(
                    round(cf, 2),
                    round(float(row["preds_cf"]), 2),
                    msg=f"main CF mismatch split={split}",
                )
                self.assertEqual(
                    round(rg, 2),
                    round(float(row["preds_rg"]), 2),
                    msg=f"main RG mismatch split={split}",
                )


#new
class TestParitySecond(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_file(ART.second_pickle, "second_pickle")
        require_file(ART.df_second, "df_second")
        require_file(ART.cf_calibrator, "cf_calibrator")
        require_file(ART.rg_calibrator, "rg_calibrator")
        require_file(ART.metadata, "metadata")
        _patch_models_dir()

    def test_second_train_test_two_decimals(self):
        df = pd.read_parquet(ART.df_second)
        self.assertIn("split", df.columns)
        for split in ("train", "test"):
            part = df[df["split"] == split]
            self.assertGreater(len(part), 0, f"empty {split}")
            for _, row in part.iterrows():
                out = score_second(ART.second_stem, _row_to_request(row))
                res = out["result"]
                cf = float(res["classification_proba"][0])
                rg = float(res["regression_predictions"][0])
                self.assertEqual(
                    round(cf, 2),
                    round(float(row["preds_cf"]), 2),
                    msg=f"second CF mismatch split={split}",
                )
                self.assertEqual(
                    round(rg, 2),
                    round(float(row["preds_rg"]), 2),
                    msg=f"second RG mismatch split={split}",
                )


#new
class TestDualPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_file(ART.main_pickle, "main_pickle")
        require_file(ART.second_pickle, "second_pickle")
        require_file(ART.df_main, "df_main")
        require_file(ART.df_second, "df_second")
        _patch_models_dir()

    def test_both_slots_filled(self):
        main_df = pd.read_parquet(ART.df_main)
        second_df = pd.read_parquet(ART.df_second)
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
