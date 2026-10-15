"""Единая точка путей к артефактам тестов main_ / second_.

Все тесты читают пути отсюда. Нет файла → fail (не skip).
Переопределение через env или правку дефолтов ниже.

Эталон для parity сервиса:
  ``df_for_service_main.parquet`` / ``df_for_service_second.parquet``
  с колонками ``preds_cf``, ``preds_rg`` (ожидаемый ответ сервиса до 2 знаков).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
_RESULTS = _INTEGRATION_ROOT / "results"


@dataclass(frozen=True)
#new
class TestArtifacts:
    """Пути к pickle / meta / калибраторам / эталонным df."""

    results_dir: Path
    main_pickle: Path
    second_pickle: Path
    metadata: Path
    cf_calibrator: Path
    rg_calibrator: Path
    df_for_service_main: Path
    df_for_service_second: Path
    main_stem: str
    second_stem: str
    preds_cf_col: str = "preds_cf"
    preds_rg_col: str = "preds_rg"
    split_col: str = "split"  # опционально: train | test


#new
def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    return Path(raw) if raw else default


#new
def load_artifacts() -> TestArtifacts:
    # По умолчанию — prod results/; синтетика: INTEGRATION_TEST_RESULTS_DIR=.../test_synth
    results = _env_path("INTEGRATION_TEST_RESULTS_DIR", _RESULTS)
    main_stem = os.environ.get("INTEGRATION_TEST_MAIN_STEM", "querulus_ansamble_legacy")
    second_stem = os.environ.get(
        "INTEGRATION_TEST_SECOND_STEM", "UU_querulus_2026_10_09_16_51_20"
    )
    return TestArtifacts(
        results_dir=results,
        main_pickle=_env_path(
            "INTEGRATION_TEST_MAIN_PICKLE", results / f"{main_stem}.pickle"
        ),
        second_pickle=_env_path(
            "INTEGRATION_TEST_SECOND_PICKLE", results / f"{second_stem}.pickle"
        ),
        metadata=_env_path("INTEGRATION_TEST_METADATA", results / "metadata.json"),
        cf_calibrator=_env_path(
            "INTEGRATION_TEST_CF_CAL", results / "cf_calibrator.pickle"
        ),
        rg_calibrator=_env_path(
            "INTEGRATION_TEST_RG_CAL", results / "rg_calibrator.pickle"
        ),
        df_for_service_main=_env_path(
            "INTEGRATION_TEST_DF_MAIN",
            results / "df_for_service_main.parquet",
        ),
        df_for_service_second=_env_path(
            "INTEGRATION_TEST_DF_SECOND",
            results / "df_for_service_second.parquet",
        ),
        main_stem=main_stem,
        second_stem=second_stem,
    )


#new
def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Нет артефакта {label}: {path}. "
            f"Положите parquet/pickle в results/ или задайте INTEGRATION_TEST_* . "
            f"Синтетика: python scripts/build_integration_test_artifacts.py"
        )
    return path


#new
def max_rows_limit() -> int | None:
    """Опциональный лимит строк для parity (INTEGRATION_TEST_MAX_ROWS)."""
    raw = os.environ.get("INTEGRATION_TEST_MAX_ROWS", "").strip()
    if not raw:
        return None
    return max(1, int(raw))


ART = load_artifacts()
