"""Синтетические df_for_service + Excel для проверки vector_checker."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import FIXTURES_DIR, LOSS_NUMBER_COL, ensure_work_dirs


def build_synthetic_frames(
    n_rows: int = 5,
    *,
    with_mismatch: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Пара DataFrame: service и «выгрузка 1С» (идентичны или с расхождением)."""
    rows = []
    for i in range(n_rows):
        loss = 10_000_000 + i
        rows.append(
            {
                LOSS_NUMBER_COL: loss,
                "INCIDENT_NUMBER": loss,
                "FILIAL": ["Белгородский", "ВСК-Москва", "Краснодарский", "Казанский", "Тюменский"][
                    i % 5
                ],
                "EVENT_CREATED_BY_GIBDD_FLAG": [1, 0, 1, 0, 1][i % 5],
                "VICTIM_MAX_WEIGHT": 1500 + 10 * i,
                "RECIEVE_METHOD": ["SMS", "Email", "Лично", "Не оповещать", "Email и SMS"][i % 5],
                "APPLICANT_FORM": [
                    "Потерпевший",
                    "Представитель (не автоюрист)",
                    "Выгодоприобретатель",
                    "Представитель (по доверенности)",
                    "Потерпевший",
                ][i % 5],
                "APPLICANT_AGE": 30 + i,
                "VICTIM_VEHICLE_CATEGORY": ["Эконом 0-6", "Средний 7+", "Премиум 7+", "Иное", "Эконом 7+"][
                    i % 5
                ],
                "GUILTY_CAPACITY_ENGINE": round(1.2 + 0.2 * i, 1),
                "VICTIM_VEHICLE_AGE": 2 + i,
                "EVENT_YEAR": 2024 + (i % 2),
                "AMOUNT_REPAIR": 50_000 + 1000 * i,
                "LOSS_UNIT_ZONE": f"Зона тестовая {i}",
                "VICTIM_VEHICLE_COUNTRY": "Россия",
                "APPLY_DELAY": 3 + i,
                "EVENT_DATE": f"2025-0{(i % 9) + 1}-15T10:00:00",
                "PAYMENT_ORDER_DATE_TIME": f"2025-0{(i % 9) + 1}-20T00:00:00",
                "VICTIM_VEHICLE_BRAND": ["Toyota", "Mazda", "Skoda", "Lexus", "ЛиАЗ"][i % 5],
                "VALUE_BEFORE_WITH": 40_000 + i,
                "VALUE_BEFORE_WITHOUT": 45_000 + i,
                "REGION": "77",
                "preds_cf": 0.1 * i,
                "preds_rg": 1000.0 * i,
            }
        )

    service = pd.DataFrame(rows)
    # Excel из 1С обычно без preds_* и иногда без derived-полей.
    excel_cols = [
        c
        for c in service.columns
        if c not in {"preds_cf", "preds_rg", "VICTIM_VEHICLE_COUNTRY", "APPLY_DELAY"}
    ]
    excel = service.loc[:, excel_cols].copy()

    if with_mismatch and len(excel) > 0:
        excel = excel.copy()
        # Расхождение по значению на строке, которая останется в Excel.
        excel.loc[excel.index[0], "FILIAL"] = "ЛОМАЕМ_СОВПАДЕНИЕ"
        # Плюс потеря ключа: убрать другой убыток из выгрузки.
        if len(excel) > 1:
            drop_key = excel.loc[excel.index[1], LOSS_NUMBER_COL]
            excel = excel.loc[excel[LOSS_NUMBER_COL] != drop_key].reset_index(drop=True)

    return service, excel


def write_synthetic_fixtures(
    fixtures_dir: Path | None = None,
    *,
    n_rows: int = 5,
) -> dict[str, Path]:
    """Записать parquet/xlsx для demo (match + mismatch)."""
    ensure_work_dirs()
    root = fixtures_dir or FIXTURES_DIR
    root.mkdir(parents=True, exist_ok=True)

    service, excel_ok = build_synthetic_frames(n_rows, with_mismatch=False)
    _, excel_bad = build_synthetic_frames(n_rows, with_mismatch=True)

    paths = {
        "df": root / "df_for_service.parquet",
        "excel_ok": root / "export_1c_ok.xlsx",
        "excel_bad": root / "export_1c_mismatch.xlsx",
        "query_stub": root / "query_template_stub.txt",
    }
    service.to_parquet(paths["df"], index=False)
    excel_ok.to_excel(paths["excel_ok"], index=False, engine="openpyxl")
    excel_bad.to_excel(paths["excel_bad"], index=False, engine="openpyxl")

    # Минимальный шаблон с тремя маркерами &Убыток (как в Сутяжность.txt).
    stub = (
        "ВЫБРАТЬ\r\n"
        "\tУбыток.Ссылка КАК Убыток\r\n"
        "ПОМЕСТИТЬ втУбыток\r\n"
        "ИЗ\r\n"
        "\tДокумент.Убыток КАК Убыток\r\n"
        "ГДЕ\r\n"
        "\tУбыток.Ссылка = &Убыток\r\n"
        ";\r\n"
        "\r\n"
        "ВЫБРАТЬ\r\n"
        "\tЗаявлениеНаВыплату.Убыток.Номер КАК LOSS_NUMBER\r\n"
        "ИЗ\r\n"
        "\tДокумент.ЗаявлениеНаВыплату КАК ЗаявлениеНаВыплату\r\n"
        "\t\tЛЕВОЕ СОЕДИНЕНИЕ РегистрСведений.СуммыКалькуляцийAudanet.СрезПервых(, Убыток = &Убыток) "
        "КАК СуммыКалькуляцийAudanetСрезПервых\r\n"
        "\t\tПО ЗаявлениеНаВыплату.Убыток = СуммыКалькуляцийAudanetСрезПервых.Убыток\r\n"
        "ГДЕ\r\n"
        "\tЗаявлениеНаВыплату.Убыток = &Убыток\r\n"
    )
    paths["query_stub"].write_text(stub, encoding="cp1251")
    return paths
