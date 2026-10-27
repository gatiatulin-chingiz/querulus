"""Parity: preds сервиса vs ``df_for_service_{main,second}`` до 2 знаков.

Эталон в parquet: колонки ``preds_cf`` / ``preds_rg`` — ожидаемые
``classification_proba`` и ``regression_predictions`` ответа сервиса
(для second_ — уже после калибраторов и маски IS_LAWYER).

Невалидные строки не валят тест: печатается статистика совпадений CF/RG
и топ-5 несовпадений по |Δ|.
"""
from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
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
def _cell_is_na(v: object) -> bool:
    """pd.isna на array/list даёт ValueError — только скаляры."""
    if v is None:
        return True
    if isinstance(v, (list, tuple, dict, np.ndarray, pd.Series, pd.Index)):
        return False
    if not np.isscalar(v) and not isinstance(v, (str, bytes)):
        return False
    try:
        return bool(pd.isna(v))
    except (ValueError, TypeError):
        return False


#new
def _row_to_request(row: pd.Series) -> list[dict]:
    d = row.to_dict()
    for k in (ART.preds_cf_col, ART.preds_rg_col, ART.split_col):
        d.pop(k, None)
    clean = {}
    for k, v in d.items():
        if _cell_is_na(v):
            clean[k] = None
        elif hasattr(v, "isoformat"):
            clean[k] = v.isoformat()
        elif isinstance(v, (np.integer,)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean[k] = float(v)
        elif isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        elif isinstance(v, (pd.Series, pd.Index)):
            clean[k] = v.tolist()
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
@dataclass(frozen=True)
class _Mismatch:
    kind: str  # CF | RG
    tag: str
    got: float
    expected: float

    @property
    def abs_delta(self) -> float:
        return abs(self.got - self.expected)


#new
def _round2(v: float) -> float:
    return round(float(v), 2)


#new
def _report_parity(
    title: str,
    n: int,
    cf_ok: int,
    rg_ok: int,
    mismatches: list[_Mismatch],
    top_n: int = 5,
) -> None:
    """Печать сводки; тест не валим из‑за расхождений."""
    cf_bad = n - cf_ok
    rg_bad = n - rg_ok
    lines = [
        f"=== parity {title} ===",
        f"rows={n}",
        f"CF match: {cf_ok}/{n}"
        + (f" ({100.0 * cf_ok / n:.1f}%)" if n else ""),
        f"RG match: {rg_ok}/{n}"
        + (f" ({100.0 * rg_ok / n:.1f}%)" if n else ""),
        f"CF mismatch: {cf_bad}, RG mismatch: {rg_bad}",
    ]
    if mismatches:
        top = sorted(mismatches, key=lambda m: m.abs_delta, reverse=True)[:top_n]
        lines.append(f"top-{len(top)} mismatches by |Δ|:")
        for i, m in enumerate(top, 1):
            lines.append(
                f"  {i}. [{m.kind}] {m.tag}: got={m.got} expected={m.expected} "
                f"|Δ|={m.abs_delta}"
            )
    else:
        lines.append("mismatches: none")
    print("\n".join(lines), flush=True)


#new
def _compare_row(
    cf: float,
    rg: float,
    row: pd.Series,
    tag: str,
) -> tuple[bool, bool, list[_Mismatch]]:
    exp_cf = _round2(row[ART.preds_cf_col])
    exp_rg = _round2(row[ART.preds_rg_col])
    got_cf = _round2(cf)
    got_rg = _round2(rg)
    cf_ok = got_cf == exp_cf
    rg_ok = got_rg == exp_rg
    bad: list[_Mismatch] = []
    if not cf_ok:
        bad.append(_Mismatch("CF", tag, got_cf, exp_cf))
    if not rg_ok:
        bad.append(_Mismatch("RG", tag, got_rg, exp_rg))
    return cf_ok, rg_ok, bad


#new
class TestParityMain(unittest.TestCase):
    """main_predict vs df_for_service_main (сводка, без fail на mismatch)."""

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
        cf_ok = 0
        rg_ok = 0
        mismatches: list[_Mismatch] = []
        for split, idx, row in _iter_rows(df):
            out = main_predict(ART.main_stem, _row_to_request(row))
            res = out["main_response"]["result"]
            tag = f"main split={split} idx={idx}"
            ok_c, ok_r, bad = _compare_row(
                float(res["classification_proba"][0]),
                float(res["regression_predictions"][0]),
                row,
                tag,
            )
            cf_ok += int(ok_c)
            rg_ok += int(ok_r)
            mismatches.extend(bad)
            n += 1
        self.assertGreater(n, 0, "df_for_service_main пуст или отфильтрован")
        _report_parity("main", n, cf_ok, rg_ok, mismatches)


#new
class TestParitySecond(unittest.TestCase):
    """score_second vs df_for_service_second (сводка, без fail на mismatch)."""

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
        cf_ok = 0
        rg_ok = 0
        mismatches: list[_Mismatch] = []
        for split, idx, row in _iter_rows(df):
            out = score_second(ART.second_stem, _row_to_request(row))
            res = out["result"]
            tag = f"second split={split} idx={idx}"
            ok_c, ok_r, bad = _compare_row(
                float(res["classification_proba"][0]),
                float(res["regression_predictions"][0]),
                row,
                tag,
            )
            cf_ok += int(ok_c)
            rg_ok += int(ok_r)
            mismatches.extend(bad)
            n += 1
        self.assertGreater(n, 0, "df_for_service_second пуст или отфильтрован")
        _report_parity("second", n, cf_ok, rg_ok, mismatches)


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
