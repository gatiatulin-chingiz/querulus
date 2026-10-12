"""Единая точка путей к артефактам тестов main_ / second_.

Все тесты читают пути отсюда. Нет файла → fail (не skip).
Переопределение через env или правку дефолтов ниже.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
_RESULTS = _INTEGRATION_ROOT / "results"
_FIXTURES = Path(__file__).resolve().parent / "fixtures"


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
    df_main: Path
    df_second: Path
    main_stem: str
    second_stem: str
    preds_cf_col: str = "preds_cf"
    preds_rg_col: str = "preds_rg"
    split_col: str = "split"  # train | test


#new
def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    return Path(raw) if raw else default


#new
def load_artifacts() -> TestArtifacts:
    results = _env_path("INTEGRATION_TEST_RESULTS_DIR", _RESULTS / "test_synth")
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
        df_main=_env_path("INTEGRATION_TEST_DF_MAIN", results / "df_main.parquet"),
        df_second=_env_path("INTEGRATION_TEST_DF_SECOND", results / "df_second.parquet"),
        main_stem=main_stem,
        second_stem=second_stem,
    )


#new
def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Нет артефакта {label}: {path}. "
            f"Сгенерируйте: python scripts/build_integration_test_artifacts.py"
        )
    return path


ART = load_artifacts()
