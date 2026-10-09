"""Smoke: `SHARE_WORK` = `Работы` (calc) / `VALUE_BEFORE_WITHOUT` (общий victim-фрейм).

Проверяет `build_targets` на заглушках SQL-артефактов (без Hive/БД):
1) знаменатель берётся из колонки `VALUE_BEFORE_WITHOUT` общего фрейма, а не из
   `СуммаРемонтаБезУчётаИзноса` / `AMOUNT_REPAIR` из `df_calc`;
2) нулевой знаменатель → `NaN` (не делим);
3) если `VALUE_BEFORE_WITHOUT` в общем фрейме нет — фолбэк на `AMOUNT_REPAIR`
   с warning;
4) `Работы` в датасете → `WORK` (русская колонка не остаётся).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402

from querulus.dataset.load import targets as load_targets  # noqa: E402
from querulus.dataset.paths import DataPaths  # noqa: E402
from querulus.dataset.preprocess import targets as T  # noqa: E402

INCIDENTS = [1, 2, 3, 4]

# Ожидания: Работы / VALUE_BEFORE_WITHOUT и старое Работы / AMOUNT_REPAIR (из calc).
EXPECTED_FROM_VALUE_WITHOUT = {1: 0.25, 2: 0.05, 3: float("nan"), 4: 0.05}
EXPECTED_FROM_AMOUNT_REPAIR = {1: 0.5, 2: 0.1, 3: float("nan"), 4: 0.1}


def _paths(tmp: Path) -> DataPaths:
    """Пути без данных: артефакты отдаются заглушками, чекапов нет."""
    return DataPaths(
        data_root=tmp,
        raw_dir=tmp,
        processed_dir=tmp,
        victim_path=Path(),
        local_data_dir=tmp,
        artifact_overrides={},
    )


def _fake_calc() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "INCIDENT_NUMBER": INCIDENTS,
            "Работы": [100.0, 50.0, 30.0, 10.0],
            # Намеренно отличается от VALUE_BEFORE_WITHOUT в victim-фрейме.
            "СуммаРемонтаБезУчётаИзноса": [200.0, 500.0, 0.0, 100.0],
            "ПроцентИзноса": [10.0, 60.0, 20.0, 5.0],
        }
    )


def _victim(*, with_value_before_without: bool) -> pd.DataFrame:
    data = {
        "INCIDENT_NUMBER": INCIDENTS,
        "LOSS_NUMBER": [11, 12, 13, 14],
        "VICTIM_POLICYHOLDER_TYPE": ["Физ. Лицо"] * 4,
        "VICTIM_OBJECT_TYPE": ["Автотранспорт"] * 4,
        "APPLICANT_ID": [1, 2, 3, 4],
        "VICTIM_POLICYHOLDER_PERSON_ID": [1, 2, 3, 4],
    }
    if with_value_before_without:
        data["VALUE_BEFORE_WITHOUT"] = [400.0, 1000.0, 0.0, 200.0]
    return pd.DataFrame(data)


def _stub_fetchers() -> None:
    load_targets.fetch_calc_agg = lambda *a, **k: _fake_calc()
    load_targets.fetch_target_psr = lambda *a, **k: pd.DataFrame(
        {
            "Номер_инциндента": INCIDENTS,
            "Сумма_выплат_по_претензиям": [0.0, 100.0, 0.0, 0.0],
            "Сумма_взыскано_по_ФУ": [0.0, 0.0, 0.0, 0.0],
            "Суммы_взыскано_по_иску": [0.0, 0.0, 0.0, 0.0],
        }
    )
    load_targets.fetch_target_3_pretensions = lambda *a, **k: pd.DataFrame(
        {
            "INCIDENT_NUMBER": INCIDENTS,
            "SurchargeValue_cumsum_by_incident": [0.0, 0.0, 0.0, 0.0],
            "UTSSurchargeValue_cumsum_by_incident": [0.0, 0.0, 0.0, 0.0],
        }
    )
    load_targets.fetch_target_3_pretensions_all = lambda *a, **k: pd.DataFrame(
        {
            "INCIDENT_NUMBER": INCIDENTS,
            "SurchargeValue_cumsum_by_incident_all": [0.0, 0.0, 0.0, 0.0],
            "UTSSurchargeValue_cumsum_by_incident_all": [0.0, 0.0, 0.0, 0.0],
        }
    )
    load_targets.fetch_target_3_claims = lambda *a, **k: pd.DataFrame(
        {
            # 5 инстанций одного иска: pivot под TARGET_3_SEV ждёт колонки *_1..*_5.
            "INCIDENT_NUMBER": [2] * 5,
            "LOSS_NUMBER": [12] * 5,
            "INCOMING_CLAIM_NUMBER": [900] * 5,
            "INSTBYOISUU": [1, 2, 3, 4, 5],
            "CLAIMEDVALUEPERIOD": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"]
            ),
            "RECOVEREDVALUEWITHSD": [100.0, 101.0, 102.0, 103.0, 104.0],
            "RECOVEREDMAINDEBT": [90.0, 91.0, 92.0, 93.0, 94.0],
            "RECOVEREDWEAROUT": [10.0, 10.0, 10.0, 10.0, 10.0],
            "RECOVEREDLOSSCOMMODYVALUE": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )


def _run(*, with_value_before_without: bool) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        return T.build_targets(
            _paths(Path(tmp)),
            None,
            _victim(with_value_before_without=with_value_before_without),
            save_checkpoint=False,
            use_sql=False,
            maturity_enabled=False,
        )


def _same(actual, expected) -> bool:
    if pd.isna(actual) and pd.isna(expected):
        return True
    if pd.isna(actual) or pd.isna(expected):
        return False
    return abs(float(actual) - float(expected)) < 1e-9


def _check(name: str, got: dict, expected: dict, failures: list[str]) -> None:
    print(f"{name}: {got}")
    for incident, exp in expected.items():
        if not _same(got.get(incident), exp):
            failures.append(f"{name}: incident {incident}: got {got.get(incident)!r}, expected {exp!r}")


def main() -> int:
    _stub_fetchers()
    failures: list[str] = []

    out = _run(with_value_before_without=True)
    _check(
        "SHARE_WORK (VALUE_BEFORE_WITHOUT)",
        out.set_index("INCIDENT_NUMBER")["SHARE_WORK"].to_dict(),
        EXPECTED_FROM_VALUE_WITHOUT,
        failures,
    )
    if "Работы" in out.columns:
        failures.append("в датасете осталась русская колонка 'Работы'")
    if "WORK" not in out.columns:
        failures.append("в датасете нет колонки 'WORK'")
    else:
        work_got = out.set_index("INCIDENT_NUMBER")["WORK"].to_dict()
        work_exp = {1: 100.0, 2: 50.0, 3: 30.0, 4: 10.0}
        _check("WORK", work_got, work_exp, failures)

    fallback = _run(with_value_before_without=False)
    _check(
        "SHARE_WORK (fallback AMOUNT_REPAIR)",
        fallback.set_index("INCIDENT_NUMBER")["SHARE_WORK"].to_dict(),
        EXPECTED_FROM_AMOUNT_REPAIR,
        failures,
    )

    if failures:
        print("\nFAIL:")
        for item in failures:
            print(" -", item)
        return 1
    print("\nOK: SHARE_WORK = WORK / VALUE_BEFORE_WITHOUT (+ фолбэк AMOUNT_REPAIR); WORK сохранён")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
