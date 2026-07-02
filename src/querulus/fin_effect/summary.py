"""Сводная таблица финансового эффекта по квадрантам TARGET × pred_freq."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from querulus.fin_effect.config import FinEffectConfig


def _neg_column_sum(group: pd.DataFrame, column: str) -> float:
    """Сумма колонки с инверсией знака (расходы отрицательные)."""
    if column not in group.columns:
        return 0.0
    return float(-pd.to_numeric(group[column], errors="coerce").fillna(0).sum())


def create_summary_table(
    effect_df: pd.DataFrame,
    config: FinEffectConfig | None = None,
) -> pd.DataFrame:
    """Сводная таблица по комбинациям TARGET и pred_freq (Litigant)."""
    config = config or FinEffectConfig()
    masks = {
        "1_1": (effect_df[config.frequency_target_column] == 1)
        & (effect_df["pred_freq"] == 1),
        "1_0": (effect_df[config.frequency_target_column] == 1)
        & (effect_df["pred_freq"] == 0),
        "0_1": (effect_df[config.frequency_target_column] == 0)
        & (effect_df["pred_freq"] == 1),
        "0_0": (effect_df[config.frequency_target_column] == 0)
        & (effect_df["pred_freq"] == 0),
    }

    rows: list[dict[str, float | int]] = []
    for mask_name, mask in masks.items():
        group = effect_df.loc[mask]
        target_2, pred_freq = map(int, mask_name.split("_"))

        count = int(group.shape[0])
        payout_main = _neg_column_sum(group, config.base_payment_column)
        sum_od_uts = _neg_column_sum(group, config.severity_target_column)
        regression = _neg_column_sum(group, "pred_sev")
        sum_claims = _neg_column_sum(group, config.pretension_payments_column)
        sum_fu = _neg_column_sum(group, config.fu_recovery_column)
        sum_lawsuit = _neg_column_sum(group, config.court_recovery_column)
        contributions = _neg_column_sum(group, config.premiums_column)

        fin_effect_model = float(group["fin_effect_model"].sum())
        fin_effect_fact = float(sum_claims + sum_fu + sum_lawsuit + contributions)

        if target_2 == 1 and pred_freq == 1:
            total = fin_effect_model
        elif target_2 == 1 and pred_freq == 0:
            total = fin_effect_fact
        elif target_2 == 0 and pred_freq == 1:
            total = fin_effect_model - fin_effect_fact
        else:
            total = 0.0

        rows.append(
            {
                "Количество инцидентов с иными взысканиями": count,
                "Факт": target_2,
                "Классификация": pred_freq,
                "Выплата по основному убытку": payout_main,
                "Сумма ОД+УТС+Износ": sum_od_uts,
                "Регрессия": regression,
                "Сумма выплат по претензиям": sum_claims,
                "Сумма взыскано по ФУ": sum_fu,
                "Суммы взыскано по иску": sum_lawsuit,
                "Взносы": contributions,
                "ФИН. ЭФФЕКТ МОДЕЛЬ": fin_effect_model,
                "ФИН. ЭФФЕКТ ФАКТ": fin_effect_fact,
                "ИТОГО": total,
            }
        )

    return pd.DataFrame(rows)


def color_excel_table(writer, sheet_name: str, summary_df: pd.DataFrame) -> None:
    """Раскрасить лист Excel как в Litigant."""
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    worksheet = writer.book[sheet_name]
    colors = {
        "gray": "4F4F4F",
        "purple": "8B479C",
        "yellow": "FFC000",
        "blue": "00B0F0",
        "pink": "FF9999",
        "green": "00B050",
        "red": "FF0000",
        "white": "FFFFFF",
    }
    column_colors = {
        "Количество инцидентов с иными взысканиями": colors["gray"],
        "Факт": colors["gray"],
        "Классификация": colors["purple"],
        "Выплата по основному убытку": colors["yellow"],
        "Сумма ОД+УТС+Износ": colors["blue"],
        "Регрессия": colors["purple"],
        "Сумма выплат по претензиям": colors["pink"],
        "Сумма взыскано по ФУ": colors["pink"],
        "Суммы взыскано по иску": colors["pink"],
        "Взносы": colors["pink"],
        "ФИН. ЭФФЕКТ МОДЕЛЬ": colors["green"],
        "ФИН. ЭФФЕКТ ФАКТ": colors["red"],
        "ИТОГО": colors["gray"],
    }
    header_font = Font(bold=True, color=colors["white"])
    header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell_alignment = Alignment(horizontal="right", vertical="center")
    thin_border = Border(
        left=Side(style="thin"),
        right=Side(style="thin"),
        top=Side(style="thin"),
        bottom=Side(style="thin"),
    )

    for col_num, cell in enumerate(worksheet[1], 1):
        col_name = summary_df.columns[col_num - 1] if col_num - 1 < len(summary_df.columns) else None
        if col_name and col_name in column_colors:
            fill_color = column_colors[col_name]
            cell.fill = PatternFill(start_color=fill_color, end_color=fill_color, fill_type="solid")
            cell.font = header_font
            cell.alignment = header_alignment
            cell.border = thin_border

    for row_num in range(2, worksheet.max_row + 1):
        for col_num, cell in enumerate(worksheet[row_num], 1):
            col_name = summary_df.columns[col_num - 1] if col_num - 1 < len(summary_df.columns) else None
            if col_name and col_name in column_colors:
                fill_color = column_colors[col_name]
                cell.fill = PatternFill(start_color=fill_color, end_color=fill_color, fill_type="solid")
                cell.alignment = cell_alignment
                cell.border = thin_border
                if isinstance(cell.value, (int, float)):
                    cell.number_format = "#,##0"

    column_widths = {
        "Количество инцидентов с иными взысканиями": 25,
        "Факт": 8,
        "Классификация": 15,
        "Выплата по основному убытку": 20,
        "Сумма ОД+УТС+Износ": 20,
        "Регрессия": 15,
        "Сумма выплат по претензиям": 22,
        "Сумма взыскано по ФУ": 18,
        "Суммы взыскано по иску": 20,
        "Взносы": 12,
        "ФИН. ЭФФЕКТ МОДЕЛЬ": 18,
        "ФИН. ЭФФЕКТ ФАКТ": 18,
        "ИТОГО": 15,
    }
    for col_num, col_name in enumerate(summary_df.columns, 1):
        width = column_widths.get(col_name, 15)
        worksheet.column_dimensions[get_column_letter(col_num)].width = width


def export_summary_excel(
    summary_df: pd.DataFrame,
    path: str | Path,
    *,
    sheet_name: str = "Summary",
) -> Path:
    """Сохранить сводную таблицу в Excel с форматированием."""
    output_path = Path(path)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name=sheet_name, index=False)
        color_excel_table(writer, sheet_name, summary_df)
    return output_path
