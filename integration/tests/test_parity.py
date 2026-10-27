"""Parity: preds сервиса vs ``df_for_service_{main,second}`` до 2 знаков.

Эталон в parquet: колонки ``preds_cf`` / ``preds_rg`` — ожидаемые
``classification_proba`` и ``regression_predictions`` ответа сервиса
(для second_ — уже после калибраторов и маски IS_LAWYER).

Скоринг батчами (multi-row vector → сравнение массивов). Сводка match %
и топ-5 несовпадений; расхождения тест не валят.
"""
from __future__ import annotations

import heapq
import json
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

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
    batch_size,
    max_rows_limit,
    require_file,
)
from loguru import logger as _loguru_logger  # noqa: E402


#new
def _quiet_service_logs() -> None:
    """Parity гоняет десятки тысяч строк — INFO/DEBUG сервиса не нужен."""
    _loguru_logger.remove()
    _loguru_logger.add(sys.stderr, level="WARNING")


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
def _frame_to_request(df: pd.DataFrame) -> list[dict]:
    """Multi-row vector для сервиса (без эталонных preds / split)."""
    drop = [
        c
        for c in (ART.preds_cf_col, ART.preds_rg_col, ART.split_col)
        if c in df.columns
    ]
    part = df.drop(columns=drop, errors="ignore")
    return json.loads(part.to_json(orient="records", force_ascii=False, date_format="iso"))


#new
def _parity_frame(df: pd.DataFrame) -> pd.DataFrame:
    """train/test по split + лимит INTEGRATION_TEST_MAX_ROWS."""
    limit = max_rows_limit()
    if ART.split_col in df.columns:
        parts = []
        for split in ("train", "test"):
            part = df[df[ART.split_col] == split]
            if part.empty:
                continue
            if limit is not None:
                part = part.head(limit)
            parts.append(part)
        if not parts:
            return df.iloc[0:0]
        return pd.concat(parts, axis=0)
    return df if limit is None else df.head(limit)


#new
def _iter_batches(df: pd.DataFrame, size: int) -> Iterator[pd.DataFrame]:
    n = len(df)
    for start in range(0, n, size):
        yield df.iloc[start : start + size]


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
def _report_parity(
    title: str,
    n: int,
    full_ok: int,
    cf_ok: int,
    rg_ok: int,
    mismatches: list[_Mismatch],
    top_n: int = 5,
) -> None:
    """Печать сводки; тест не валим из‑за расхождений."""
    pct = (100.0 * full_ok / n) if n else 0.0
    lines = [
        f"=== parity {title} ===",
        f"{full_ok} из {n} строк полностью совпали, совпадение {pct:.2f}%",
        f"CF match: {cf_ok}/{n}"
        + (f" ({100.0 * cf_ok / n:.2f}%)" if n else ""),
        f"RG match: {rg_ok}/{n}"
        + (f" ({100.0 * rg_ok / n:.2f}%)" if n else ""),
    ]
    if mismatches:
        top = heapq.nlargest(top_n, mismatches, key=lambda m: m.abs_delta)
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
def _collect_batch_mismatches(
    *,
    lane: str,
    indices: list,
    got_cf: np.ndarray,
    got_rg: np.ndarray,
    exp_cf: np.ndarray,
    exp_rg: np.ndarray,
) -> tuple[int, int, int, list[_Mismatch]]:
    """Сверка массивов батча → счётчики + список несовпадений."""
    cf_eq = got_cf == exp_cf
    rg_eq = got_rg == exp_rg
    full_eq = cf_eq & rg_eq
    bad: list[_Mismatch] = []
    for i in np.flatnonzero(~cf_eq):
        bad.append(
            _Mismatch(
                "CF",
                f"{lane} idx={indices[i]}",
                float(got_cf[i]),
                float(exp_cf[i]),
            )
        )
    for i in np.flatnonzero(~rg_eq):
        bad.append(
            _Mismatch(
                "RG",
                f"{lane} idx={indices[i]}",
                float(got_rg[i]),
                float(exp_rg[i]),
            )
        )
    return int(full_eq.sum()), int(cf_eq.sum()), int(rg_eq.sum()), bad


#new
def _run_parity_batches(
    *,
    title: str,
    df: pd.DataFrame,
    score_batch: Callable[[pd.DataFrame], tuple[np.ndarray, np.ndarray]],
) -> tuple[int, int, int, int, list[_Mismatch]]:
    """Батч-predict + сравнение массивов; tqdm по батчам/строкам."""
    work = _parity_frame(df)
    n_total = len(work)
    bs = batch_size()
    n = 0
    full_ok = 0
    cf_ok = 0
    rg_ok = 0
    mismatches: list[_Mismatch] = []

    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            total=n_total,
            unit="row",
            desc=f"parity {title}",
            leave=True,
        )

    try:
        for chunk in _iter_batches(work, bs):
            got_cf, got_rg = score_batch(chunk)
            exp_cf = np.round(chunk[ART.preds_cf_col].to_numpy(dtype=float), 2)
            exp_rg = np.round(chunk[ART.preds_rg_col].to_numpy(dtype=float), 2)
            got_cf = np.round(np.asarray(got_cf, dtype=float).ravel(), 2)
            got_rg = np.round(np.asarray(got_rg, dtype=float).ravel(), 2)
            if len(got_cf) != len(chunk) or len(got_rg) != len(chunk):
                raise AssertionError(
                    f"{title}: длина ответа сервиса != батч "
                    f"(cf={len(got_cf)} rg={len(got_rg)} batch={len(chunk)})"
                )
            f_ok, c_ok, r_ok, bad = _collect_batch_mismatches(
                lane=title,
                indices=list(chunk.index),
                got_cf=got_cf,
                got_rg=got_rg,
                exp_cf=exp_cf,
                exp_rg=exp_rg,
            )
            full_ok += f_ok
            cf_ok += c_ok
            rg_ok += r_ok
            mismatches.extend(bad)
            n += len(chunk)
            if pbar is not None:
                pbar.update(len(chunk))
                pbar.set_postfix(
                    full=f"{100.0 * full_ok / n:.1f}%",
                    refresh=False,
                )
    finally:
        if pbar is not None:
            pbar.close()

    return n, full_ok, cf_ok, rg_ok, mismatches


#new
class TestParityMain(unittest.TestCase):
    """main_predict vs df_for_service_main — multi-row батчи + сводка."""

    @classmethod
    def setUpClass(cls):
        require_file(ART.main_pickle, "main_pickle")
        require_file(ART.df_for_service_main, "df_for_service_main")
        _patch_models_dir()
        _quiet_service_logs()

    def test_service_matches_df_for_service_main(self):
        df = pd.read_parquet(ART.df_for_service_main)
        self.assertIn(ART.preds_cf_col, df.columns)
        self.assertIn(ART.preds_rg_col, df.columns)

        def score_batch(chunk: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            out = main_predict(ART.main_stem, _frame_to_request(chunk))
            res = out["main_response"]["result"]
            return (
                np.asarray(res["classification_proba"], dtype=float),
                np.asarray(res["regression_predictions"], dtype=float),
            )

        n, full_ok, cf_ok, rg_ok, mismatches = _run_parity_batches(
            title="main",
            df=df,
            score_batch=score_batch,
        )
        self.assertGreater(n, 0, "df_for_service_main пуст или отфильтрован")
        _report_parity("main", n, full_ok, cf_ok, rg_ok, mismatches)


#new
class TestParitySecond(unittest.TestCase):
    """score_second vs df_for_service_second — multi-row батчи + сводка."""

    @classmethod
    def setUpClass(cls):
        require_file(ART.second_pickle, "second_pickle")
        require_file(ART.df_for_service_second, "df_for_service_second")
        require_file(ART.cf_calibrator, "cf_calibrator")
        require_file(ART.rg_calibrator, "rg_calibrator")
        _patch_models_dir()
        _quiet_service_logs()

    def test_service_matches_df_for_service_second(self):
        df = pd.read_parquet(ART.df_for_service_second)
        self.assertIn(ART.preds_cf_col, df.columns)
        self.assertIn(ART.preds_rg_col, df.columns)

        def score_batch(chunk: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            out = score_second(ART.second_stem, _frame_to_request(chunk))
            res = out["result"]
            return (
                np.asarray(res["classification_proba"], dtype=float),
                np.asarray(res["regression_predictions"], dtype=float),
            )

        n, full_ok, cf_ok, rg_ok, mismatches = _run_parity_batches(
            title="second",
            df=df,
            score_batch=score_batch,
        )
        self.assertGreater(n, 0, "df_for_service_second пуст или отфильтрован")
        _report_parity("second", n, full_ok, cf_ok, rg_ok, mismatches)


#new
class TestDualPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_file(ART.main_pickle, "main_pickle")
        require_file(ART.second_pickle, "second_pickle")
        require_file(ART.df_for_service_main, "df_for_service_main")
        require_file(ART.df_for_service_second, "df_for_service_second")
        _patch_models_dir()
        _quiet_service_logs()

    def test_both_slots_filled(self):
        main_df = pd.read_parquet(ART.df_for_service_main)
        second_df = pd.read_parquet(ART.df_for_service_second)
        # multi-row smoke: по 3 строки (или меньше, если parquet короткий)
        main_chunk = main_df.head(min(3, len(main_df)))
        second_chunk = second_df.head(min(3, len(second_df)))
        out = main_predict(
            ART.main_stem,
            _frame_to_request(main_chunk),
            ART.second_stem,
            _frame_to_request(second_chunk),
        )
        self.assertIn("oisuu_responce", out)
        self.assertIn("main_response", out)
        self.assertTrue(out["second_response"].get("result"))
        main_cf = out["main_response"]["result"]["classification_proba"]
        second_cf = out["second_response"]["result"]["classification_proba"]
        self.assertEqual(len(main_cf), len(main_chunk))
        self.assertEqual(len(second_cf), len(second_chunk))


if __name__ == "__main__":
    unittest.main()
