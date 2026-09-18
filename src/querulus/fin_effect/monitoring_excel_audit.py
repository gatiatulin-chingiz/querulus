"""Excel-аудит финэффекта: вводные + формулы + bootstrap как значения."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from querulus.fin_effect.excel_monitoring import MonitoringEffectResult
from querulus.fin_effect.monitoring_report import FORMULA_VERSION

# sheet → [(column, description, formula_or_source), ...]
_SHEET_COLUMN_DOCS: dict[str, list[tuple[str, str, str]]] = {
    "inputs": [
        ("param", "Имя вводного параметра", "ключ строки"),
        (
            "value",
            "Значение параметра (можно менять)",
            "ячейки $B$ — ссылки calc/filial/effect",
        ),
        ("unit / note", "Единицы и краткий смысл", "справочно"),
        ("how used", "Как параметр входит в формулы", "текстовая формула"),
    ],
    "rows": [
        (
            "loss",
            "Номер(а) убытка внутри инцидента",
            "после схлопывания — список LossID через «; »",
        ),
        (
            "incident",
            "Номер инцидента — единица ITT",
            "ключ схлопывания; одна строка = один инцидент",
        ),
        (
            "group",
            "Группа ITT: control или model",
            "Result=−100 → control; Result∈{0,1} → model",
        ),
        ("filial", "Филиал (мода по убыткам инцидента)", "страта рандомизации"),
        (
            "paid_SummaPlatezha",
            "Оплата на дату расчёта, ₽",
            "Σ СуммаПлатежа по убыткам инцидента (= Yfact)",
        ),
        (
            "od",
            "Основной долг заявленный, ₽",
            "СуммаОсновногоДолгаЗаявлено (сумма при схлопе)",
        ),
        (
            "psr_pretension",
            "Наблюдаемый ПСР претензий на строке, ₽",
            "обычно 0 на первичных",
        ),
        ("psr_fu", "Наблюдаемый ПСР ФУ на строке, ₽", "обычно 0 на первичных"),
        (
            "psr_court",
            "Наблюдаемый ПСР суда на строке, ₽",
            "обычно 0 на первичных",
        ),
        (
            "observed_psr",
            "Сумма наблюдаемого ПСР",
            "psr_pretension + psr_fu + psr_court",
        ),
        (
            "agreement_01",
            "Признак соглашения",
            "1 если форма возмещения = соглашение, иначе 0",
        ),
        (
            "age_days",
            "Возраст инцидента на t_calc, дни",
            "t_calc − дата заявления",
        ),
        ("q", "Доля остатка ПСР", "IF(agreement=1, residual_share, 1)"),
        (
            "ultimate_base",
            "База ultimate ПСР, ₽",
            "IF(OD>0, OD×k_U, m_U)",
        ),
        (
            "expected_open_psr",
            "Ожидаемый открытый ПСР, ₽",
            "p_U × (ultimate_base + e_U)",
        ),
        (
            "remaining",
            "Остаток ПСР к дисконтированию, ₽",
            "MAX(q×expected_open_psr − observed_psr, 0)",
        ),
        (
            "midpoint_days",
            "Середина оставшегося горизонта, дни",
            "MAX(horizon_days − age_days, 0) / 2",
        ),
        ("Yfact", "Фактический исход, ₽", "Yfact = paid_SummaPlatezha"),
        (
            "Y365",
            "Исход с NPV хвоста на горизонте 365, ₽",
            "Yfact + remaining / (1+r)^(midpoint_days/365)",
        ),
        ("python_Yfact", "Эталон Yfact из Python", "сверка с колонкой Yfact"),
        ("python_Y365", "Эталон Y365 из Python", "сверка с колонкой Y365"),
    ],
    "filial": [
        ("filial", "Филиал", "уникальные значения из rows.filial"),
        ("n", "Число инцидентов филиала", "COUNTIF(rows.filial, filial)"),
        (
            "mean_Yfact_control",
            "Средний Yfact control в филиале",
            "AVERAGEIFS(Yfact | filial, group=control)",
        ),
        (
            "mean_Yfact_model",
            "Средний Yfact model в филиале",
            "AVERAGEIFS(Yfact | filial, group=model)",
        ),
        (
            "effect_Yfact",
            "Локальный ITT Yfact филиала",
            "mean_Yfact_control − mean_Yfact_model",
        ),
        (
            "mean_Y365_control",
            "Средний Y365 control в филиале",
            "AVERAGEIFS(Y365 | filial, group=control)",
        ),
        (
            "mean_Y365_model",
            "Средний Y365 model в филиале",
            "AVERAGEIFS(Y365 | filial, group=model)",
        ),
        (
            "effect_Y365",
            "Локальный ITT Y365 филиала",
            "mean_Y365_control − mean_Y365_model",
        ),
        (
            "weight",
            "Вес филиала в стратифицированном ITT",
            "n / Σ n  (по всем филиалам листа)",
        ),
    ],
    "effect": [
        (
            "metric",
            "Имя итоговой метрики",
            "effect_Yfact / effect_Y365 / annual_network_Y365",
        ),
        (
            "excel_formula_value",
            "Значение, посчитанное формулами Excel",
            "ссылки на filial и inputs",
        ),
        ("python_value", "Эталон из Python", "сверка с excel_formula_value"),
        ("note", "Как считается метрика", "текстовая формула"),
    ],
    "bootstrap": [
        ("horizon", "Горизонт ITT: fact или 365", "из effect_summary"),
        (
            "effect_per_case",
            "Точечный стратифицированный ITT, ₽/инцидент",
            "из Python",
        ),
        ("ci_low", "Нижняя граница 95% CI (2.5% квантиль)", "bootstrap ITT"),
        ("ci_high", "Верхняя граница 95% CI (97.5% квантиль)", "bootstrap ITT"),
        (
            "n_bootstrap",
            "Число успешных bootstrap-повторений",
            "из Python",
        ),
        (
            "bootstrap_samples.horizon",
            "Горизонт семпла",
            "блок bootstrap_samples ниже CI-таблицы",
        ),
        (
            "bootstrap_samples.effect_per_case",
            "ITT одной bootstrap-итерации",
            "сырые семплы; в Excel не пересчитываются",
        ),
    ],
}


def _require_openpyxl():
    try:
        from openpyxl import Workbook
        from openpyxl.comments import Comment
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Для Excel-аудита нужен openpyxl. Установите: pip install openpyxl"
        ) from exc
    return Workbook, Alignment, Font, PatternFill, get_column_letter, Comment


def _autosize(ws, max_width: int = 42) -> None:
    get_column_letter = _require_openpyxl()[4]
    for idx, column in enumerate(ws.columns, start=1):
        width = 10
        for cell in column:
            value = "" if cell.value is None else str(cell.value)
            width = max(width, min(len(value) + 2, max_width))
        ws.column_dimensions[get_column_letter(idx)].width = width


def _annotate_headers(
    ws,
    sheet_key: str,
    *,
    header_row: int = 1,
    start_col: int = 1,
) -> None:
    """Комментарии к заголовкам таблицы: описание + формула."""
    Comment = _require_openpyxl()[5]
    docs = {
        name: (desc, formula)
        for name, desc, formula in _SHEET_COLUMN_DOCS.get(sheet_key, [])
    }
    col = start_col
    while True:
        cell = ws.cell(header_row, col)
        name = cell.value
        if name is None or name == "":
            break
        key = str(name)
        if key in docs:
            desc, formula = docs[key]
            text = f"{desc}"
            if formula:
                text += f"\nФормула/источник: {formula}"
            cell.comment = Comment(text, "fin_effect_audit", height=80, width=280)
        col += 1


def _write_columns_sheet(ws) -> None:
    """Словарь колонок всех листов audit-книги."""
    Font = _require_openpyxl()[2]
    PatternFill = _require_openpyxl()[3]
    header = Font(bold=True, color="075F66")
    fill = PatternFill("solid", fgColor="E7F3F4")
    ws["A1"] = "Описание колонок всех листов fin_effect_audit.xlsx"
    ws["A1"].font = header
    ws.merge_cells("A1:D1")
    ws["A2"] = (
        "Единица анализа — инцидент после дедупа по убытку и схлопывания. "
        "Наведите курсор на заголовок колонки на листе — там тот же текст в комментарии."
    )
    headers = ("sheet", "column", "description", "formula / source")
    for col, name in enumerate(headers, start=1):
        cell = ws.cell(4, col, name)
        cell.font = header
        cell.fill = fill

    row = 5
    for sheet_name, cols in _SHEET_COLUMN_DOCS.items():
        for column, desc, formula in cols:
            ws.cell(row, 1, sheet_name)
            ws.cell(row, 2, column)
            ws.cell(row, 3, desc)
            ws.cell(row, 4, formula)
            row += 1
    ws.cell(row + 1, 1, "readme")
    ws.cell(row + 1, 2, "—")
    ws.cell(row + 1, 3, "Инструкция; табличных колонок нет")
    ws.cell(row + 1, 4, "см. лист readme")


def _write_inputs(ws, result: MonitoringEffectResult) -> dict[str, str]:
    """Лист вводных. Возвращает map имя → адрес ячейки значения (например B3)."""
    Font = _require_openpyxl()[2]
    PatternFill = _require_openpyxl()[3]
    header = Font(bold=True, color="075F66")
    fill = PatternFill("solid", fgColor="E7F3F4")
    ws["A1"] = (
        "Вводные параметры (можно менять — пересчитаются формулы на calc/filial/effect)"
    )
    ws["A1"].font = header
    ws.merge_cells("A1:D1")

    pilot = result.priors.pilot
    annual = result.annual_summary.set_index("horizon")
    n_year = (
        float(annual.loc["365", "N_pilot_eligible_year"])
        if "365" in annual.index
        else 0.0
    )
    multiplier = (
        float(annual.loc["365", "network_multiplier"])
        if "365" in annual.index
        else 1.0
    )

    rows = [
        ("discount_rate", result.discount_rate, "Годовая ставка дисконта r"),
        ("residual_share", result.residual_share, "Остаток ПСР при соглашении q"),
        ("horizon_days", 365, "Горизонт Y365, дни"),
        ("p_U", pilot.p_ultimate, "Доля убытков с ПСР (ретро pilot)"),
        ("k_U", pilot.k_ultimate, "Коэффициент ПСР к OD"),
        ("m_U", pilot.mean_positive_psr, "Средний положительный ПСР (fallback), ₽"),
        ("e_U", pilot.expected_fee, "Ожидаемые взносы ФУ/суда, ₽"),
        ("t_calc", result.t_calc.date().isoformat(), "Дата расчёта (справочно)"),
        ("N_pilot_eligible_year", n_year, "Годовой поток eligible пилота"),
        ("network_multiplier", multiplier, "Множитель сети (шаринг на все филиалы)"),
        (
            "formula_version",
            FORMULA_VERSION,
            "Версия методики (справочно)",
        ),
    ]
    ws["A3"] = "param"
    ws["B3"] = "value"
    ws["C3"] = "unit / note"
    ws["D3"] = "how used"
    for col in ("A3", "B3", "C3", "D3"):
        ws[col].font = header
        ws[col].fill = fill

    how = {
        "discount_rate": "Y365 = Yfact + remaining / (1+r)^(midpoint/365)",
        "residual_share": "q = residual_share при соглашении, иначе 1",
        "horizon_days": "remaining_days = max(horizon − age, 0)",
        "p_U": "expected_open = p_U × (ultimate_base + e_U)",
        "k_U": "ultimate_base = OD × k_U если OD > 0",
        "m_U": "ultimate_base = m_U если OD отсутствует/≤0",
        "e_U": "оргвзносы в expected_open",
        "N_pilot_eligible_year": "annual = effect × N × multiplier",
        "network_multiplier": "annual network scale",
    }
    cells: dict[str, str] = {}
    for offset, (name, value, note) in enumerate(rows, start=4):
        ws.cell(offset, 1, name)
        cell = ws.cell(offset, 2, value)
        ws.cell(offset, 3, note)
        ws.cell(offset, 4, how.get(name, ""))
        cells[name] = f"$B${offset}"
        if isinstance(value, float):
            cell.number_format = "0.0000" if abs(value) < 10 else "#,##0.00"
    ws["A16"] = (
        "Меняйте только столбец value (B). Листы calc / filial / effect "
        "ссылаются на эти ячейки."
    )
    ws["A17"] = (
        "Описание колонок таблицы — лист columns и комментарии к заголовкам (строка 3)."
    )
    _annotate_headers(ws, "inputs", header_row=3)
    return cells


def _write_rows(ws, result: MonitoringEffectResult, inputs: dict[str, str]) -> int:
    """Строки выборки: значения + формулы. Возвращает число data-строк."""
    Font = _require_openpyxl()[2]
    PatternFill = _require_openpyxl()[3]
    frame = result.frame.reset_index(drop=True)
    headers = [
        "loss",
        "incident",
        "group",
        "filial",
        "paid_SummaPlatezha",
        "od",
        "psr_pretension",
        "psr_fu",
        "psr_court",
        "observed_psr",
        "agreement_01",
        "age_days",
        "q",
        "ultimate_base",
        "expected_open_psr",
        "remaining",
        "midpoint_days",
        "Yfact",
        "Y365",
        "python_Yfact",
        "python_Y365",
    ]
    header_font = Font(bold=True, color="075F66")
    fill = PatternFill("solid", fgColor="E7F3F4")
    for col, name in enumerate(headers, start=1):
        cell = ws.cell(1, col, name)
        cell.font = header_font
        cell.fill = fill
    _annotate_headers(ws, "rows", header_row=1)

    r = inputs["discount_rate"]
    q_share = inputs["residual_share"]
    horizon = inputs["horizon_days"]
    p_u = inputs["p_U"]
    k_u = inputs["k_U"]
    m_u = inputs["m_U"]
    e_u = inputs["e_U"]

    n = len(frame)
    for i in range(n):
        excel_row = i + 2
        get = frame.iloc[i]
        ws.cell(excel_row, 1, get["_loss"])
        ws.cell(excel_row, 2, get["_incident"])
        ws.cell(excel_row, 3, get["_group"])
        ws.cell(excel_row, 4, get["_filial"])
        ws.cell(excel_row, 5, float(get["_paid_to_date"]))
        od = get["_od"]
        ws.cell(excel_row, 6, float(od) if pd.notna(od) else None)
        ws.cell(excel_row, 7, float(get["_psr_pretension"]))
        ws.cell(excel_row, 8, float(get["_psr_fu"]))
        ws.cell(excel_row, 9, float(get["_psr_court"]))
        ws.cell(excel_row, 10, f"=G{excel_row}+H{excel_row}+I{excel_row}")
        ws.cell(excel_row, 11, int(bool(get["_agreement"])))
        ws.cell(excel_row, 12, int(get["_age_days"]))
        ws.cell(excel_row, 13, f"=IF(K{excel_row}=1,inputs!{q_share},1)")
        ws.cell(
            excel_row,
            14,
            f'=IF(AND(F{excel_row}<>"",F{excel_row}>0),'
            f"F{excel_row}*inputs!{k_u},inputs!{m_u})",
        )
        ws.cell(
            excel_row,
            15,
            f"=inputs!{p_u}*(N{excel_row}+inputs!{e_u})",
        )
        ws.cell(
            excel_row,
            16,
            f"=MAX(M{excel_row}*O{excel_row}-J{excel_row},0)",
        )
        ws.cell(
            excel_row,
            17,
            f"=MAX(inputs!{horizon}-L{excel_row},0)/2",
        )
        ws.cell(excel_row, 18, f"=E{excel_row}")
        ws.cell(
            excel_row,
            19,
            f"=R{excel_row}+P{excel_row}/POWER(1+inputs!{r},Q{excel_row}/365)",
        )
        ws.cell(excel_row, 20, float(get["Yfact"]))
        ws.cell(excel_row, 21, float(get["Y365"]))
        for col in (5, 6, 7, 8, 9, 16, 18, 19, 20, 21):
            ws.cell(excel_row, col).number_format = "#,##0.00"

    ws.cell(n + 3, 1, "Примечание")
    ws.cell(
        n + 3,
        2,
        "Одна строка = один инцидент после дедупа/схлопывания. "
        "Yfact = СуммаПлатежа (оплата), не observed_PSR. "
        "Колонки psr_* вычитаются из хвоста remaining, а не прибавляются к Yfact. "
        "python_Y* — эталон из Python для сверки. "
        "Полное описание колонок — лист columns; краткое — комментарий к заголовку.",
    )
    return n


def _write_filial(ws, n_rows: int, filials: list[str]) -> int:
    Font = _require_openpyxl()[2]
    PatternFill = _require_openpyxl()[3]
    headers = [
        "filial",
        "n",
        "mean_Yfact_control",
        "mean_Yfact_model",
        "effect_Yfact",
        "mean_Y365_control",
        "mean_Y365_model",
        "effect_Y365",
        "weight",
    ]
    header_font = Font(bold=True, color="075F66")
    fill = PatternFill("solid", fgColor="E7F3F4")
    for col, name in enumerate(headers, start=1):
        cell = ws.cell(1, col, name)
        cell.font = header_font
        cell.fill = fill
    _annotate_headers(ws, "filial", header_row=1)

    last = n_rows + 1  # header row 1, data 2..last
    for i, filial in enumerate(filials, start=2):
        ws.cell(i, 1, filial)
        ws.cell(i, 2, f"=COUNTIF(rows!$D$2:$D${last},A{i})")
        ws.cell(
            i,
            3,
            f"=IFERROR(AVERAGEIFS(rows!$R$2:$R${last},rows!$D$2:$D${last},A{i},"
            f'rows!$C$2:$C${last},"control"),"")',
        )
        ws.cell(
            i,
            4,
            f"=IFERROR(AVERAGEIFS(rows!$R$2:$R${last},rows!$D$2:$D${last},A{i},"
            f'rows!$C$2:$C${last},"model"),"")',
        )
        ws.cell(i, 5, f'=IF(OR(C{i}="",D{i}=""),"",C{i}-D{i})')
        ws.cell(
            i,
            6,
            f"=IFERROR(AVERAGEIFS(rows!$S$2:$S${last},rows!$D$2:$D${last},A{i},"
            f'rows!$C$2:$C${last},"control"),"")',
        )
        ws.cell(
            i,
            7,
            f"=IFERROR(AVERAGEIFS(rows!$S$2:$S${last},rows!$D$2:$D${last},A{i},"
            f'rows!$C$2:$C${last},"model"),"")',
        )
        ws.cell(i, 8, f'=IF(OR(F{i}="",G{i}=""),"",F{i}-G{i})')
        end_f = 1 + len(filials)
        ws.cell(i, 9, f"=IF(SUM($B$2:$B${end_f})=0,0,B{i}/SUM($B$2:$B${end_f}))")
        for col in range(3, 9):
            ws.cell(i, col).number_format = "#,##0.00"
        ws.cell(i, 9).number_format = "0.0000"

    note_row = 3 + len(filials)
    ws.cell(note_row, 1, "Описание колонок")
    ws.cell(
        note_row,
        2,
        "См. лист columns и комментарии к заголовкам. "
        "Плюс effect = control дороже model = экономия модели.",
    )
    return len(filials)


def _write_effect(ws, n_filials: int, result: MonitoringEffectResult) -> None:
    Font = _require_openpyxl()[2]
    PatternFill = _require_openpyxl()[3]
    header = Font(bold=True, color="075F66")
    fill = PatternFill("solid", fgColor="E7F3F4")
    ws["A1"] = "Стратифицированный ITT (формулы Excel)"
    ws["A1"].font = header
    end = 1 + n_filials
    ws["A3"] = "metric"
    ws["B3"] = "excel_formula_value"
    ws["C3"] = "python_value"
    ws["D3"] = "note"
    for col in ("A3", "B3", "C3", "D3"):
        ws[col].font = header
        ws[col].fill = fill
    _annotate_headers(ws, "effect", header_row=3)

    effects = result.effect_summary.set_index("horizon")
    annual = result.annual_summary.set_index("horizon")

    ws["A4"] = "effect_Yfact"
    ws["B4"] = (
        f"=IFERROR(SUMPRODUCT(filial!$B$2:$B${end},filial!$E$2:$E${end})"
        f'/SUM(filial!$B$2:$B${end}),"")'
    )
    ws["C4"] = (
        float(effects.loc["fact", "effect_per_case"])
        if "fact" in effects.index
        else None
    )
    ws["D4"] = "Σ w_f × (mean_control − mean_model), w_f ∝ N_f"

    ws["A5"] = "effect_Y365"
    ws["B5"] = (
        f"=IFERROR(SUMPRODUCT(filial!$B$2:$B${end},filial!$H$2:$H${end})"
        f'/SUM(filial!$B$2:$B${end}),"")'
    )
    ws["C5"] = (
        float(effects.loc["365", "effect_per_case"])
        if "365" in effects.index
        else None
    )
    ws["D5"] = "То же для Y365"

    ws["A6"] = "annual_network_Y365"
    ws["B6"] = '=IFERROR(B5*inputs!$B$12*inputs!$B$13,"")'
    ws["C6"] = (
        float(annual.loc["365", "annual_network_full"])
        if "365" in annual.index
        else None
    )
    ws["D6"] = "effect × N_year × network_multiplier"

    ws["A8"] = "Плюс = экономия (control дороже model)."
    ws["A9"] = (
        "Марийский и Архангельский в эту книгу не входят: filial_scope=pilot."
    )
    ws["A10"] = (
        "Описание колонок таблицы — лист columns и комментарии к заголовкам (строка 3)."
    )
    for cell in ("B4", "B5", "B6", "C4", "C5", "C6"):
        ws[cell].number_format = "#,##0.00"


def _write_bootstrap(ws, result: MonitoringEffectResult) -> None:
    Font = _require_openpyxl()[2]
    PatternFill = _require_openpyxl()[3]
    header = Font(bold=True, color="075F66")
    fill = PatternFill("solid", fgColor="FFF4DA")
    ws["A1"] = "Bootstrap CI — значения из Python (не формулы Excel)"
    ws["A1"].font = header
    ws["A1"].fill = fill
    ws.merge_cells("A1:E1")
    ws["A2"] = (
        f"ITT iterations={result.bootstrap_iterations}; "
        f"compliance iterations={result.bootstrap_compliance_iterations}. "
        "Пересчёт CI в Excel не поддерживается. Описание колонок — лист columns."
    )

    effects = result.effect_summary
    ws["A4"] = "horizon"
    ws["B4"] = "effect_per_case"
    ws["C4"] = "ci_low"
    ws["D4"] = "ci_high"
    ws["E4"] = "n_bootstrap"
    for col in range(1, 6):
        ws.cell(4, col).font = header
        ws.cell(4, col).fill = fill
    _annotate_headers(ws, "bootstrap", header_row=4)
    for i, row in enumerate(effects.to_dict("records"), start=5):
        ws.cell(i, 1, row.get("horizon"))
        ws.cell(i, 2, row.get("effect_per_case"))
        ws.cell(i, 3, row.get("ci_low"))
        ws.cell(i, 4, row.get("ci_high"))
        ws.cell(i, 5, row.get("n_bootstrap"))
        for col in (2, 3, 4):
            ws.cell(i, col).number_format = "#,##0.00"

    samples = result.bootstrap_samples
    start = 5 + len(effects) + 2
    ws.cell(start, 1, "bootstrap_samples (ITT)")
    ws.cell(start, 1).font = header
    if samples is not None and not samples.empty:
        Comment = _require_openpyxl()[5]
        cols = list(samples.columns)
        sample_docs = {
            "horizon": "Горизонт семпла (fact / 365)",
            "effect_per_case": "ITT одной bootstrap-итерации, ₽/инцидент",
        }
        for j, name in enumerate(cols, start=1):
            cell = ws.cell(start + 1, j, name)
            cell.font = header
            cell.fill = fill
            if name in sample_docs:
                cell.comment = Comment(
                    sample_docs[name],
                    "fin_effect_audit",
                    height=60,
                    width=240,
                )
        for i, rec in enumerate(samples.to_dict("records"), start=start + 2):
            for j, name in enumerate(cols, start=1):
                ws.cell(i, j, rec.get(name))


def _write_readme(ws) -> None:
    Font = _require_openpyxl()[2]
    ws["A1"] = "Как пользоваться audit-книгой"
    ws["A1"].font = Font(bold=True, color="075F66")
    lines = [
        "1. Лист columns — словарь колонок всех листов (описание + формула/источник).",
        "2. Лист inputs — крутите discount_rate, residual_share, p_U/k_U/m_U/e_U, N_year, multiplier.",
        "3. Лист rows — одна строка = инцидент; исходные поля и расчёт Yfact/Y365 формулами Excel.",
        "4. Лист filial — средние control/model и веса филиалов (AVERAGEIFS / COUNTIF).",
        "5. Лист effect — стратифицированный ITT = Σ N_f×effect_f / Σ N_f; сверяйте с python_value.",
        "6. Лист bootstrap — только снимок Python (CI и семплы); в Excel не пересчитывается.",
        "7. Наведите курсор на заголовок колонки — во всплывающем комментарии то же описание.",
        "8. Yfact = СуммаПлатежа; претензия/ФУ/суд на строке — observed_PSR для хвоста remaining.",
        "9. Данные из frame estimate_monitoring_effect (после дедупа убытка и схлопывания на инцидент).",
    ]
    for i, line in enumerate(lines, start=3):
        ws.cell(i, 1, line)


def export_monitoring_audit_xlsx(
    result: MonitoringEffectResult,
    path: str | Path,
) -> Path:
    """Записать .xlsx с вводными, формулами и bootstrap-значениями."""
    Workbook = _require_openpyxl()[0]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    ws_readme = wb.active
    ws_readme.title = "readme"
    _write_readme(ws_readme)

    ws_columns = wb.create_sheet("columns")
    _write_columns_sheet(ws_columns)

    ws_inputs = wb.create_sheet("inputs")
    cells = _write_inputs(ws_inputs, result)

    ws_rows = wb.create_sheet("rows")
    n_rows = _write_rows(ws_rows, result, cells)

    filials = sorted(result.frame["_filial"].astype(str).unique().tolist())
    ws_filial = wb.create_sheet("filial")
    n_filials = _write_filial(ws_filial, n_rows, filials)

    ws_effect = wb.create_sheet("effect")
    _write_effect(ws_effect, n_filials, result)

    ws_boot = wb.create_sheet("bootstrap")
    _write_bootstrap(ws_boot, result)

    for ws in (ws_readme, ws_columns, ws_inputs, ws_filial, ws_effect, ws_boot):
        _autosize(ws)
    from openpyxl.utils import get_column_letter

    for idx in range(1, 22):
        ws_rows.column_dimensions[get_column_letter(idx)].width = 14
    ws_columns.column_dimensions["C"].width = 56
    ws_columns.column_dimensions["D"].width = 48

    wb.save(path)
    return path


__all__ = ["export_monitoring_audit_xlsx"]
