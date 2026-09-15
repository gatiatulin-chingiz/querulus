"""Excel-аудит финэффекта: вводные + формулы + bootstrap как значения."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from querulus.fin_effect.excel_monitoring import MonitoringEffectResult
from querulus.fin_effect.monitoring_report import FORMULA_VERSION


def _require_openpyxl():
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Для Excel-аудита нужен openpyxl. Установите: pip install openpyxl"
        ) from exc
    return Workbook, Alignment, Font, PatternFill, get_column_letter


def _autosize(ws, max_width: int = 42) -> None:
    _, _, _, _, get_column_letter = _require_openpyxl()
    for idx, column in enumerate(ws.columns, start=1):
        width = 10
        for cell in column:
            value = "" if cell.value is None else str(cell.value)
            width = max(width, min(len(value) + 2, max_width))
        ws.column_dimensions[get_column_letter(idx)].width = width


def _write_inputs(ws, result: MonitoringEffectResult) -> dict[str, str]:
    """Лист вводных. Возвращает map имя → адрес ячейки значения (например B3)."""
    Font = _require_openpyxl()[2]
    PatternFill = _require_openpyxl()[3]
    header = Font(bold=True, color="075F66")
    fill = PatternFill("solid", fgColor="E7F3F4")
    ws["A1"] = "Вводные параметры (можно менять — пересчитаются формулы на calc/filial/effect)"
    ws["A1"].font = header
    ws.merge_cells("A1:D1")

    pilot = result.priors.pilot
    annual = result.annual_summary.set_index("horizon")
    n_year = float(annual.loc["365", "N_pilot_eligible_year"]) if "365" in annual.index else 0.0
    multiplier = (
        float(annual.loc["365", "network_multiplier"]) if "365" in annual.index else 1.0
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
        ("network_multiplier", multiplier, "Множитель сети (full rollout)"),
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
        "Меняйте только столбец value (B). Листы calc / filial / effect ссылаются на эти ячейки."
    )
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
        "Yfact в методике = только СуммаПлатежа первичного убытка. "
        "Колонки psr_* на первичных строках обычно 0; они вычитаются из хвоста, "
        "а не прибавляются к Yfact. python_Y* — эталон из Python для сверки.",
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

    last = n_rows + 1  # header row 1, data 2..last
    for i, filial in enumerate(filials, start=2):
        ws.cell(i, 1, filial)
        ws.cell(i, 2, f'=COUNTIF(rows!$D$2:$D${last},A{i})')
        ws.cell(
            i,
            3,
            f'=IFERROR(AVERAGEIFS(rows!$R$2:$R${last},rows!$D$2:$D${last},A{i},'
            f'rows!$C$2:$C${last},"control"),"")',
        )
        ws.cell(
            i,
            4,
            f'=IFERROR(AVERAGEIFS(rows!$R$2:$R${last},rows!$D$2:$D${last},A{i},'
            f'rows!$C$2:$C${last},"model"),"")',
        )
        ws.cell(i, 5, f'=IF(OR(C{i}="",D{i}=""),"",C{i}-D{i})')
        ws.cell(
            i,
            6,
            f'=IFERROR(AVERAGEIFS(rows!$S$2:$S${last},rows!$D$2:$D${last},A{i},'
            f'rows!$C$2:$C${last},"control"),"")',
        )
        ws.cell(
            i,
            7,
            f'=IFERROR(AVERAGEIFS(rows!$S$2:$S${last},rows!$D$2:$D${last},A{i},'
            f'rows!$C$2:$C${last},"model"),"")',
        )
        ws.cell(i, 8, f'=IF(OR(F{i}="",G{i}=""),"",F{i}-G{i})')
        end_f = 1 + len(filials)
        ws.cell(i, 9, f"=IF(SUM($B$2:$B${end_f})=0,0,B{i}/SUM($B$2:$B${end_f}))")
        for col in range(3, 9):
            ws.cell(i, col).number_format = "#,##0.00"
        ws.cell(i, 9).number_format = "0.0000"
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

    effects = result.effect_summary.set_index("horizon")
    annual = result.annual_summary.set_index("horizon")

    ws["A4"] = "effect_Yfact"
    ws["B4"] = f'=IFERROR(SUMPRODUCT(filial!$B$2:$B${end},filial!$E$2:$E${end})/SUM(filial!$B$2:$B${end}),"")'
    ws["C4"] = float(effects.loc["fact", "effect_per_case"]) if "fact" in effects.index else None
    ws["D4"] = "Σ w_f × (mean_control − mean_model), w_f ∝ N_f"

    ws["A5"] = "effect_Y365"
    ws["B5"] = f'=IFERROR(SUMPRODUCT(filial!$B$2:$B${end},filial!$H$2:$H${end})/SUM(filial!$B$2:$B${end}),"")'
    ws["C5"] = float(effects.loc["365", "effect_per_case"]) if "365" in effects.index else None
    ws["D5"] = "То же для Y365"

    ws["A6"] = "annual_network_Y365"
    ws["B6"] = "=IFERROR(B5*inputs!$B$12*inputs!$B$13,\"\")"
    ws["C6"] = (
        float(annual.loc["365", "annual_network_full"]) if "365" in annual.index else None
    )
    ws["D6"] = "effect × N_year × network_multiplier"

    ws["A8"] = "Плюс = экономия (control дороже model)."
    ws["A9"] = (
        "Марийский и Архангельский в эту книгу не входят: filial_scope=pilot."
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
        "Пересчёт CI в Excel не поддерживается."
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
        cols = list(samples.columns)
        for j, name in enumerate(cols, start=1):
            ws.cell(start + 1, j, name)
        for i, rec in enumerate(samples.to_dict("records"), start=start + 2):
            for j, name in enumerate(cols, start=1):
                ws.cell(i, j, rec.get(name))


def _write_readme(ws) -> None:
    Font = _require_openpyxl()[2]
    ws["A1"] = "Как пользоваться audit-книгой"
    ws["A1"].font = Font(bold=True, color="075F66")
    lines = [
        "1. Лист inputs — крутите discount_rate, residual_share, p_U/k_U/m_U/e_U, N_year, multiplier.",
        "2. Лист rows — исходные поля (значения) и расчёт Yfact/Y365 формулами Excel.",
        "3. Лист filial — средние control/model и веса филиалов (AVERAGEIFS / COUNTIF).",
        "4. Лист effect — стратифицированный ITT = Σ N_f×effect_f / Σ N_f; сверяйте с python_value.",
        "5. Лист bootstrap — только снимок Python (CI и семплы); в Excel не пересчитывается.",
        "6. Yfact = СуммаПлатежа первичного убытка; претензия/ФУ/суд на строке — observed_PSR для хвоста.",
        "7. Данные из прод-витрины MSSQL (или того frame, что подан в estimate_monitoring_effect).",
    ]
    for i, line in enumerate(lines, start=3):
        ws.cell(i, 1, line)


def export_monitoring_audit_xlsx(
    result: MonitoringEffectResult,
    path: str | Path,
) -> Path:
    """Записать .xlsx с вводными, формулами и bootstrap-значениями."""
    Workbook, _, _, _, _ = _require_openpyxl()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    ws_readme = wb.active
    ws_readme.title = "readme"
    _write_readme(ws_readme)

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

    for ws in (ws_readme, ws_inputs, ws_filial, ws_effect, ws_boot):
        _autosize(ws)
    from openpyxl.utils import get_column_letter

    for idx in range(1, 22):
        ws_rows.column_dimensions[get_column_letter(idx)].width = 14

    wb.save(path)
    return path


__all__ = ["export_monitoring_audit_xlsx"]
