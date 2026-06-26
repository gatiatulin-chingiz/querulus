from __future__ import annotations

from datetime import datetime
import numpy as np
import pandas as pd


def _excel_autofit_and_style(
    writer: pd.ExcelWriter,
    sheet_to_df: dict[str, pd.DataFrame],
    *,
    zoom: int = 90,
    max_width: int = 60,
    sample_rows: int = 1000,
) -> None:
    workbook = writer.book
    base_fmt = workbook.add_format({"text_wrap": True, "align": "center", "valign": "vcenter"})

    for sheet_name, df_sheet in sheet_to_df.items():
        if sheet_name not in writer.sheets:
            continue

        ws = writer.sheets[sheet_name]
        ws.set_zoom(zoom)

        if df_sheet is None or df_sheet.empty:
            ws.set_column(0, 0, 20, base_fmt)
            continue

        for col_idx, col in enumerate(df_sheet.columns):
            header_w = len(str(col)) if col is not None else 0

            series = df_sheet.iloc[:sample_rows, col_idx]
            values = series.astype(str).replace("nan", "").fillna("")
            cell_w = int(values.map(len).max()) if len(values) else 0

            width = min(max_width, max(header_w, cell_w) + 2)
            ws.set_column(col_idx, col_idx, width, base_fmt)


def _to_datetime_safe(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False)


def _to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _normalize_pred(x) -> str:
    if pd.isna(x):
        return np.nan
    return str(x).strip()


def _group_predictions_ru(pred: pd.Series) -> pd.Series:
    pred = pred.map(_normalize_pred)
    num = pd.to_numeric(pred, errors="coerce")
    return pred.where(~(num > 0), other="Предсказаний > 0")


def _is_error_row(df: pd.DataFrame) -> pd.Series:
    """Строка — ошибка, если задано поле error или пропущен comment."""
    err = pd.Series(False, index=df.index)

    if "error" in df.columns:
        err = err | df["error"].notna()

    if "comment" in df.columns:
        err = err | df["comment"].isna()

    return err


def _confidence_bucket(p: pd.Series) -> pd.Series:
    bins = [0.0, 0.5, 0.7, 0.85, 0.95, 1.0000001]
    labels = ["<=0.50", "0.50-0.70", "0.70-0.85", "0.85-0.95", ">=0.95"]
    return pd.cut(p, bins=bins, labels=labels, include_lowest=True)


def build_model_prod_report_ru(
    df: pd.DataFrame,
    output_path: str | None = None,
) -> str:
    df = df.copy()

    if "FILIAL" not in df.columns:
        df["FILIAL"] = "UNKNOWN"

    for c in ["TIME_STAMP_x", "TIME_STAMP_y", "EVENT_DATE", "PAYMENT_ORDER_DATE_TIME"]:
        if c in df.columns:
            df[c] = _to_datetime_safe(df[c])

    if "classification_proba" in df.columns:
        df["classification_proba"] = _to_numeric_safe(df["classification_proba"])

    if "threshold" in df.columns:
        df["threshold"] = _to_numeric_safe(df["threshold"])

    if "AMOUNT_REPAIR" in df.columns:
        df["AMOUNT_REPAIR"] = _to_numeric_safe(df["AMOUNT_REPAIR"])

    if "predictions" in df.columns:
        df["predictions"] = df["predictions"].map(_normalize_pred)
        pred_num = pd.to_numeric(df["predictions"], errors="coerce")
        df["_in_pilot"] = pred_num.ne(-100) | pred_num.isna()
        df["_pred_one_predictions"] = pred_num.gt(0)
    else:
        df["_in_pilot"] = True
        df["_pred_one_predictions"] = False

    if "regression_predictions" in df.columns:
        df["regression_predictions"] = _to_numeric_safe(df["regression_predictions"])
        reg_num = pd.to_numeric(df["regression_predictions"], errors="coerce")
        df["_pred_one_regression"] = reg_num.gt(0)
        pred_for_stats = df["regression_predictions"]
    else:
        df["_pred_one_regression"] = False
        pred_for_stats = df["predictions"] if "predictions" in df.columns else pd.Series(np.nan, index=df.index)

    df["_pred_group"] = _group_predictions_ru(pred_for_stats)

    ts_col = None
    for cand in ["TIME_STAMP_x", "TIME_STAMP_y", "PAYMENT_ORDER_DATE_TIME", "EVENT_DATE"]:
        if cand in df.columns and df[cand].notna().any():
            ts_col = cand
            break

    if ts_col is not None:
        df["_date"] = df[ts_col].dt.date
    else:
        df["_date"] = pd.NaT

    df["_is_error"] = _is_error_row(df)
    df["_has_pred"] = df["predictions"].notna() if "predictions" in df.columns else False

    # ---------- Summary (русские метрики) ----------
    summary_rows = []

    def add_metric(ru_name: str, value):
        summary_rows.append({"Показатель": ru_name, "Значение": value})

    add_metric("Всего записей", len(df))

    if "LOSS_NUMBER" in df.columns:
        add_metric("Уникальных убытков", df["LOSS_NUMBER"].nunique(dropna=True))

    add_metric("Филиалов", df["FILIAL"].nunique(dropna=True))

    if ts_col is not None:
        add_metric("Использованная колонка даты/времени", ts_col)
        add_metric("Начало периода", df[ts_col].min())
        add_metric("Конец периода", df[ts_col].max())
    else:
        add_metric("Использованная колонка даты/времени", None)
        add_metric("Начало периода", pd.NaT)
        add_metric("Конец периода", pd.NaT)

    add_metric("Записей с ошибками", int(df["_is_error"].sum()))
    add_metric("Доля ошибок", float(df["_is_error"].mean()) if len(df) else np.nan)

    if "predictions" in df.columns:
        add_metric("Записей без предсказания", int((~df["_has_pred"]).sum()))
        add_metric(
            "Доля записей без предсказания",
            float((~df["_has_pred"]).mean()) if len(df) else np.nan,
        )

        pred_counts = df["_pred_group"].value_counts(dropna=True)
        for cls, cnt in pred_counts.items():
            if cls == "Предсказаний > 0":
                add_metric("Предсказаний > 0", int(cnt))
            else:
                add_metric(f"Предсказаний класса '{cls}'", int(cnt))

    if "classification_proba" in df.columns:
        p = df["classification_proba"].dropna()
        add_metric("Средняя вероятность", float(p.mean()) if len(p) else np.nan)
        add_metric("Медианная вероятность", float(p.median()) if len(p) else np.nan)
        add_metric("5-й процентиль вероятности", float(p.quantile(0.05)) if len(p) else np.nan)
        add_metric("95-й процентиль вероятности", float(p.quantile(0.95)) if len(p) else np.nan)

        if len(p):
            buckets = _confidence_bucket(df["classification_proba"])
            bucket_counts = buckets.value_counts(dropna=True).sort_index()
            for b, cnt in bucket_counts.items():
                add_metric(f"Корзина уверенности {b}", int(cnt))

    summary_df = pd.DataFrame(summary_rows)

    # ---------- By_FILIAL ----------
    agg_dict = {
        "LOSS_NUMBER": ["count"] if "LOSS_NUMBER" in df.columns else [],
        "_is_error": ["sum", "mean"],
        "_has_pred": ["mean"],
    }
    agg_dict = {k: v for k, v in agg_dict.items() if v}

    if "classification_proba" in df.columns:
        agg_dict["classification_proba"] = ["count", "mean", "median", "min", "max"]

    if ts_col is not None:
        agg_dict[ts_col] = ["min", "max"]

    by_filial = df.groupby("FILIAL", dropna=False).agg(agg_dict)
    by_filial.columns = [
        "_".join([c for c in col if c]) for col in by_filial.columns.to_flat_index()
    ]
    by_filial = by_filial.reset_index()

    rename_by_filial = {
        "FILIAL": "Филиал",
        "LOSS_NUMBER_count": "Записей",
        "_is_error_sum": "Ошибок",
        "_is_error_mean": "Доля ошибок",
        "_has_pred_mean": "Доля с предсказанием",
        "classification_proba_count": "Вероятностей (кол-во)",
        "classification_proba_mean": "Вероятность (средняя)",
        "classification_proba_median": "Вероятность (медиана)",
        "classification_proba_min": "Вероятность (мин)",
        "classification_proba_max": "Вероятность (макс)",
    }
    if ts_col is not None:
        rename_by_filial[f"{ts_col}_min"] = "Начало периода"
        rename_by_filial[f"{ts_col}_max"] = "Конец периода"

    by_filial = by_filial.rename(columns=rename_by_filial)

    # добавляем распределение predictions по филиалу (и переименуем колонки на русский формат)
    if "predictions" in df.columns:
        pred_dist = (
            df.groupby("FILIAL")["_pred_group"]
            .value_counts(dropna=True)
            .rename("cnt")
            .reset_index()
            .pivot_table(
                index="FILIAL",
                columns="_pred_group",
                values="cnt",
                fill_value=0,
                aggfunc="sum",
            )
            .reset_index()
            .rename(columns={"FILIAL": "Филиал"})
        )

        # переименуем колонку каждого класса в "Предсказаний: <класс>"
        pred_rename = {}
        for c in pred_dist.columns:
            if c == "Филиал":
                continue
            if c == "Предсказаний > 0":
                pred_rename[c] = "Предсказаний > 0"
            else:
                pred_rename[c] = f"Предсказаний: {c}"
        pred_dist = pred_dist.rename(columns=pred_rename)

        by_filial = by_filial.merge(pred_dist, on="Филиал", how="left")

    # ---------- By_day ----------
    if df["_date"].notna().any():
        base_count_col = "LOSS_NUMBER" if "LOSS_NUMBER" in df.columns else "FILIAL"
        by_day = (
            df.groupby("_date", dropna=False)
            .agg(
                rows=(base_count_col, "count"),
                error_rate=("_is_error", "mean"),
                error_rows=("_is_error", "sum"),
                predicted_ones_predictions_rate=("_pred_one_predictions", "mean"),
                predicted_ones_regression_rate=("_pred_one_regression", "mean"),
                proba_mean=("classification_proba", "mean")
                if "classification_proba" in df.columns
                else ("_is_error", "mean"),
                proba_median=("classification_proba", "median")
                if "classification_proba" in df.columns
                else ("_is_error", "mean"),
            )
            .reset_index()
            .rename(
                columns={
                    "_date": "Дата",
                    "rows": "Записей",
                    "error_rate": "Доля ошибок",
                    "error_rows": "Ошибок",
                    "predicted_ones_predictions_rate": "Доля единичек (predictions > 0)",
                    "predicted_ones_regression_rate": "Доля единичек (regression_predictions > 0)",
                    "proba_mean": "Вероятность (средняя)",
                    "proba_median": "Вероятность (медиана)",
                }
            )
        )
    else:
        by_day = pd.DataFrame(
            columns=[
                "Дата",
                "Записей",
                "Доля ошибок",
                "Ошибок",
                "Доля единичек (predictions > 0)",
                "Доля единичек (regression_predictions > 0)",
                "Вероятность (средняя)",
                "Вероятность (медиана)",
            ]
        )

    # ---------- By_day_and_filial ----------
    if df["_date"].notna().any():
        base_count_col = "LOSS_NUMBER" if "LOSS_NUMBER" in df.columns else "FILIAL"
        by_day_filial = (
            df.groupby(["_date", "FILIAL"], dropna=False)
            .agg(
                rows=(base_count_col, "count"),
                error_rate=("_is_error", "mean"),
                error_rows=("_is_error", "sum"),
                has_prediction_rate=("_has_pred", "mean"),
                proba_mean=("classification_proba", "mean")
                if "classification_proba" in df.columns
                else ("_is_error", "mean"),
                proba_median=("classification_proba", "median")
                if "classification_proba" in df.columns
                else ("_is_error", "mean"),
            )
            .reset_index()
            .rename(
                columns={
                    "_date": "Дата",
                    "FILIAL": "Филиал",
                    "rows": "Записей",
                    "error_rate": "Доля ошибок",
                    "error_rows": "Ошибок",
                    "has_prediction_rate": "Доля с предсказанием",
                    "proba_mean": "Вероятность (средняя)",
                    "proba_median": "Вероятность (медиана)",
                }
            )
        )
        all_dates = pd.Index(sorted(df["_date"].dropna().unique()), name="Дата")
        all_filials = pd.Index(df["FILIAL"].unique(), name="Филиал")
        full_idx = pd.MultiIndex.from_product([all_dates, all_filials], names=["Дата", "Филиал"])

        by_day_filial = (
            by_day_filial.set_index(["Дата", "Филиал"])
            .reindex(full_idx)
            .reset_index()
            .sort_values(["Дата", "Филиал"], ascending=[False, True])
        )
        if "Записей" in by_day_filial.columns:
            by_day_filial["Записей"] = by_day_filial["Записей"].fillna(0).astype(int)
        if "Ошибок" in by_day_filial.columns:
            by_day_filial["Ошибок"] = by_day_filial["Ошибок"].fillna(0).astype(int)
    else:
        by_day_filial = pd.DataFrame(
            columns=[
                "Дата",
                "Филиал",
                "Записей",
                "Доля ошибок",
                "Ошибок",
                "Доля с предсказанием",
                "Вероятность (средняя)",
                "Вероятность (медиана)",
            ]
        )

    key_cols = [
        c
        for c in [
            "PROCESS_ID",
            "LOSS_NUMBER",
            "FILIAL",
            "predictions",
            "classification_proba",
            "threshold",
            "error",
            "comment",
            ts_col,
        ]
        if c and c in df.columns
    ]

    # ---------- Missingness by FILIAL ----------
    missing_focus_cols = [
        "RECIEVE_METHOD",
        "APPLICANT_FORM",
        "APPLICANT_AGE",
        "FILIAL",
        "EVENT_CREATED_BY_GIBDD_FLAG",
        "VICTIM_MAX_WEIGHT",
        "GUILTY_CAPACITY_ENGINE",
        "VICTIM_VEHICLE_AGE",
        "EVENT_YEAR",
        "LOSS_UNIT_ZONE",
        "EVENT_DATE",
        "PAYMENT_ORDER_DATE_TIME",
        "LOSS_NUMBER",
        "AMOUNT_REPAIR",
        "VICTIM_VEHICLE_BRAND",
        "VICTIM_VEHICLE_CATEGORY",
        "INCIDENT_NUMBER",
    ]
    missing_by_filial_cols = [c for c in missing_focus_cols if c in df.columns and c != "FILIAL"]
    if missing_by_filial_cols:
        miss = (
            df.groupby(["FILIAL", "_date"], dropna=False)[missing_by_filial_cols]
            .apply(lambda g: g.isna().sum())
            .reset_index()
            .melt(id_vars=["FILIAL", "_date"], var_name="Колонка", value_name="Пусто")
        )
        filled = (
            df.groupby(["FILIAL", "_date"], dropna=False)[missing_by_filial_cols]
            .apply(lambda g: g.notna().sum())
            .reset_index()
            .melt(id_vars=["FILIAL", "_date"], var_name="Колонка", value_name="Заполнено")
        )
        total = (
            df.groupby(["FILIAL", "_date"], dropna=False)
            .size()
            .rename("Всего")
            .reset_index()
            .rename(columns={"FILIAL": "Филиал", "_date": "Дата"})
        )
        missing_by_filial = (
            miss.merge(filled, on=["FILIAL", "_date", "Колонка"], how="left")
            .rename(columns={"FILIAL": "Филиал", "_date": "Дата"})
            .merge(total, on=["Филиал", "Дата"], how="left")
        )
        missing_by_filial["Доля пустых"] = (
            missing_by_filial["Пусто"] / missing_by_filial["Всего"].replace(0, np.nan)
        )
        missing_by_filial = missing_by_filial.sort_values(
            ["Дата", "Пусто", "Доля пустых", "Филиал", "Колонка"],
            ascending=[False, False, False, True, True],
        )
    else:
        missing_by_filial = pd.DataFrame(
            columns=["Филиал", "Дата", "Колонка", "Пусто", "Заполнено", "Всего", "Доля пустых"]
        )

    # ---------- Samples sheets ----------
    sample_cols = [
        c
        for c in [
            "PROCESS_ID",
            "LOSS_NUMBER",
            "FILIAL",
            ts_col,
            "predictions",
            "classification_proba",
            "threshold",
            "comment",
            "error",
        ]
        if c and c in df.columns
    ]

    sample_rename = {
        "PROCESS_ID": "ID процесса",
        "LOSS_NUMBER": "Номер убытка",
        "FILIAL": "Филиал",
        "predictions": "Предсказание (класс)",
        "classification_proba": "Вероятность (уверенность)",
        "threshold": "Порог",
        "comment": "Комментарий",
        "error": "Ошибка",
    }
    if ts_col is not None:
        sample_rename[ts_col] = "Дата/время"

    err_sample_cols = [
        c
        for c in [
            "PROCESS_ID",
            "LOSS_NUMBER",
            "FILIAL",
            ts_col,
            "error",
        ]
        if c and c in df.columns
    ]
    err_samples = df.loc[df["_is_error"], err_sample_cols].copy()
    if ts_col is not None and ts_col in err_samples.columns:
        err_samples = err_samples.sort_values(ts_col, ascending=False)
    err_samples = err_samples.head(200).rename(columns=sample_rename)

    if "AMOUNT_REPAIR" in df.columns:
        amount_cols = [c for c in sample_cols if c != "classification_proba"] + ["AMOUNT_REPAIR"]
        amount_rename = dict(sample_rename)
        amount_rename["predictions"] = "Предсказанная сумма"
        amount_rename["AMOUNT_REPAIR"] = "Сумма ремонта"

        if "regression_predictions" in df.columns:
            amount_cols = [c for c in amount_cols if c != "predictions"] + ["regression_predictions"]
            amount_rename["regression_predictions"] = "Предсказанная сумма"

        high_amount = (
            df.loc[df["AMOUNT_REPAIR"].notna() & (df["AMOUNT_REPAIR"] >= 0), amount_cols]
            .copy()
            .sort_values(
                ["regression_predictions", "AMOUNT_REPAIR"] if "regression_predictions" in df.columns else ["predictions", "AMOUNT_REPAIR"],
                ascending=[False, False],
            )
            .head(200)
            .rename(columns=amount_rename)
        )
    else:
        high_amount = pd.DataFrame()

    # ---------- Threshold impact ----------
    if "threshold" in df.columns and "classification_proba" in df.columns:
        tmp = df.loc[
            df["threshold"].notna() & df["classification_proba"].notna(),
            ["FILIAL", "threshold", "classification_proba"],
        ].copy()
        tmp["_above_threshold"] = tmp["classification_proba"] >= tmp["threshold"]

        threshold_impact = (
            tmp.groupby("FILIAL")
            .agg(
                rows=("threshold", "count"),
                above_threshold_rate=("_above_threshold", "mean"),
                proba_mean=("classification_proba", "mean"),
                threshold_mean=("threshold", "mean"),
            )
            .reset_index()
            .rename(
                columns={
                    "FILIAL": "Филиал",
                    "rows": "Записей",
                    "above_threshold_rate": "Доля выше порога",
                    "proba_mean": "Вероятность (средняя)",
                    "threshold_mean": "Порог (средний)",
                }
            )
        )
    else:
        threshold_impact = pd.DataFrame(
            columns=[
                "Филиал",
                "Записей",
                "Доля выше порога",
                "Вероятность (средняя)",
                "Порог (средний)",
            ]
        )

    # ---------- Save Excel ----------
    if output_path is None:
        output_path = f"Краткий отчет ML в проде на {datetime.now().date()}.xlsx"

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Сводка", index=False)
        by_filial.to_excel(writer, sheet_name="По филиалам", index=False)
        by_day.to_excel(writer, sheet_name="Динамика по дням", index=False)
        by_day_filial.to_excel(writer, sheet_name="Динамика по дням по филиалам", index=False)
        missing_by_filial.to_excel(writer, sheet_name="Пропуски по филиалам", index=False)
        err_samples.to_excel(writer, sheet_name="Ошибки", index=False)
        if not high_amount.empty:
            high_amount.to_excel(writer, sheet_name="Крупные суммы", index=False)
        threshold_impact.to_excel(writer, sheet_name="Влияние порога", index=False)

        # Принудительно центрируем значения "Начало периода"/"Конец периода" (pandas пишет дату своим форматом)
        ws_summary = writer.sheets.get("Сводка")
        if ws_summary is not None:
            wb = writer.book
            dt_fmt = wb.add_format(
                {"align": "center", "valign": "vcenter", "text_wrap": True, "num_format": "yyyy-mm-dd hh:mm:ss"}
            )
            for metric_name in ["Начало периода", "Конец периода"]:
                idxs = summary_df.index[summary_df["Показатель"] == metric_name].tolist()
                if not idxs:
                    continue
                row = int(idxs[0]) + 1  # +1 из-за заголовка
                value = summary_df.loc[idxs[0], "Значение"]
                if pd.isna(value):
                    ws_summary.write_blank(row, 1, None, dt_fmt)
                else:
                    if hasattr(value, "to_pydatetime"):
                        value = value.to_pydatetime()
                    ws_summary.write_datetime(row, 1, value, dt_fmt)

        # Убираем лишние колонки по требованиям
        for col in ["Доля с предсказанием", "Вероятностей (кол-во)"]:
            if col in by_filial.columns:
                by_filial.drop(columns=[col], inplace=True)

        if "Доля с предсказанием" in by_day.columns:
            by_day.drop(columns=["Доля с предсказанием"], inplace=True)

        _excel_autofit_and_style(
            writer,
            {
                "Сводка": summary_df,
                "По филиалам": by_filial,
                "Динамика по дням": by_day,
                "Динамика по дням по филиалам": by_day_filial,
                "Пропуски по филиалам": missing_by_filial,
                "Ошибки": err_samples,
                "Крупные суммы": high_amount if not high_amount.empty else pd.DataFrame(),
                "Влияние порога": threshold_impact,
            },
            zoom=90,
        )

    return output_path
    


# Использование:
path = build_model_prod_report_ru(df)
print("Saved:", path)