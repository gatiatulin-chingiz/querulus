"""Этап 1: derived FE_* колонки (блоки A–J)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from querulus.features.config import FeatureConfig, FeatureThresholds


def _series(df: pd.DataFrame, column: str) -> pd.Series:
    """Колонка или NA-series, если колонки нет."""
    if column in df.columns:
        return df[column]
    return pd.Series(pd.NA, index=df.index)


def _to_datetime(series: pd.Series) -> pd.Series:
    """Привести к datetime64."""
    return pd.to_datetime(series, errors="coerce")


def _days_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    """Разница в днях (later − earlier)."""
    return (_to_datetime(later) - _to_datetime(earlier)).dt.days


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Деление с защитой от нуля."""
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return num / den.where(den != 0)


def _as_flag(series: pd.Series) -> pd.Series:
    """Привести к int 0/1 для известных флагов."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(int)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return (numeric.fillna(0) != 0).astype(int)
    text = series.fillna("").astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "да", "y", "yes"}).astype(int)


def _vehicle_age_bin(age: pd.Series, bins: tuple[float, ...]) -> pd.Series:
    """Бакеты возраста ТС: 0–3 / 3–7 / 7–15 / 15+."""
    values = pd.to_numeric(age, errors="coerce")
    b0, b1, b2 = bins
    labels = ["0-3", "3-7", "7-15", "15+"]
    return pd.cut(
        values,
        bins=[-np.inf, b0, b1, b2, np.inf],
        labels=labels,
        right=False,
    ).astype(str).replace("nan", np.nan)


def _participants_bin(count: pd.Series) -> pd.Series:
    """2 / 3 / 4+ участников."""
    values = pd.to_numeric(count, errors="coerce")
    result = pd.Series(pd.NA, index=count.index, dtype="object")
    result = result.mask(values == 2, "2")
    result = result.mask(values == 3, "3")
    result = result.mask(values >= 4, "4+")
    return result


def _count_bin(count: pd.Series) -> pd.Series:
    """0 / 1 / 2 / 3+ для истории убытков."""
    values = pd.to_numeric(count, errors="coerce").fillna(0)
    result = pd.Series("0", index=count.index, dtype="object")
    result = result.mask(values == 1, "1")
    result = result.mask(values == 2, "2")
    result = result.mask(values >= 3, "3+")
    return result


def _amount_bins(amount: pd.Series, edges: tuple[float, ...]) -> pd.Series:
    """Трёхуровневые бакеты суммы."""
    values = pd.to_numeric(amount, errors="coerce")
    low, high = edges
    labels = [f"<{int(low)}", f"{int(low)}-{int(high)}", f">{int(high)}"]
    return pd.cut(
        values,
        bins=[-np.inf, low, high, np.inf],
        labels=labels,
        right=False,
    ).astype(str).replace("nan", np.nan)


def _tier_from_bins(value: pd.Series, edges: tuple[float, ...], labels: tuple[str, ...]) -> pd.Series:
    """Универсальные tier-бакеты."""
    values = pd.to_numeric(value, errors="coerce")
    low, high = edges
    return pd.cut(
        values,
        bins=[-np.inf, low, high, np.inf],
        labels=list(labels),
        right=False,
    ).astype(str).replace("nan", np.nan)


def _season(month: pd.Series) -> pd.Series:
    """Сезон по номеру месяца."""
    values = pd.to_numeric(month, errors="coerce")
    result = pd.Series(pd.NA, index=month.index, dtype="object")
    winter = values.isin([12, 1, 2])
    spring = values.isin([3, 4, 5])
    summer = values.isin([6, 7, 8])
    autumn = values.isin([9, 10, 11])
    result = result.mask(winter, "winter")
    result = result.mask(spring, "spring")
    result = result.mask(summer, "summer")
    result = result.mask(autumn, "autumn")
    return result


def _hour_bucket(hour: pd.Series) -> pd.Series:
    """night / morning / day / evening."""
    values = pd.to_numeric(hour, errors="coerce")
    result = pd.Series(pd.NA, index=hour.index, dtype="object")
    result = result.mask((values >= 0) & (values < 6), "night")
    result = result.mask((values >= 6) & (values < 12), "morning")
    result = result.mask((values >= 12) & (values < 18), "day")
    result = result.mask((values >= 18) & (values <= 23), "evening")
    return result


def _kbm_bin(kbm: pd.Series, thresholds: FeatureThresholds) -> pd.Series:
    """Бакеты КБМ."""
    values = pd.to_numeric(kbm, errors="coerce")
    result = pd.Series(pd.NA, index=kbm.index, dtype="object")
    result = result.mask(values <= thresholds.kbm_low, "le_1")
    result = result.mask(
        (values > thresholds.kbm_low) & (values <= thresholds.kbm_mid),
        "1_1.17",
    )
    result = result.mask(values > thresholds.kbm_mid, "gt_1.17")
    return result


def _doors_bin(doors: pd.Series) -> pd.Series:
    """Бакеты числа дверей."""
    values = pd.to_numeric(doors, errors="coerce")
    result = pd.Series(pd.NA, index=doors.index, dtype="object")
    for door_count in (2, 3, 4):
        result = result.mask(values == door_count, str(door_count))
    result = result.mask(values >= 5, "5+")
    return result


def _seats_bin(seats: pd.Series) -> pd.Series:
    """Бакеты числа мест."""
    values = pd.to_numeric(seats, errors="coerce")
    result = pd.Series(pd.NA, index=seats.index, dtype="object")
    result = result.mask(values <= 4, "le_4")
    result = result.mask((values > 4) & (values <= 7), "5-7")
    result = result.mask(values > 7, "8+")
    return result


def _equals(left: pd.Series, right: pd.Series) -> pd.Series:
    """1 если значения равны и оба не NA."""
    both = left.notna() & right.notna()
    return (left == right).where(both).astype("Int64")


def _pick_column(df: pd.DataFrame, *names: str) -> pd.Series:
    """Первый существующий столбец из списка."""
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(pd.NA, index=df.index)


def _add_timeline_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Блок A: timeline."""
    t0 = _series(df, config.t0_column)
    th = config.thresholds

    df["FE_DAYS_LOSS_TO_T0"] = _days_between(t0, _series(df, "LOSS_DATE_TIME"))
    df["FE_DAYS_EVENT_TO_LOSS"] = _days_between(
        _series(df, "LOSS_DATE_TIME"),
        _series(df, "EVENT_DATE"),
    )
    df["FE_DAYS_TO_PH_CONTRACT_END"] = _days_between(
        _series(df, "POLICYHOLDER_CONTRACT_END_DATE"),
        _series(df, "EVENT_DATE"),
    )
    df["FE_DAYS_TO_VICTIM_CONTRACT_END"] = _days_between(
        _series(df, "VICTIM_CONTRACT_END_DATE"),
        _series(df, "EVENT_DATE"),
    )

    event_day = pd.to_numeric(_series(df, "EVENT_DAY"), errors="coerce")
    event_date = _to_datetime(_series(df, "EVENT_DATE"))
    weekend_by_day = event_day.isin([5, 6])
    weekend_by_date = event_date.dt.dayofweek >= 5
    df["FE_IS_WEEKEND_EVENT"] = (weekend_by_day | weekend_by_date).astype("Int64")

    df["FE_SEASON_EVENT"] = _season(_series(df, "EVENT_MONTH"))
    df["FE_HOUR_BUCKET_EVENT"] = _hour_bucket(_series(df, "EVENT_HOUR"))

    apply_delay = pd.to_numeric(_series(df, "APPLY_DELAY"), errors="coerce")
    df["FE_HIGH_APPLY_DELAY"] = (apply_delay > th.apply_delay_high).astype("Int64")
    return df


def _add_accident_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Блок B: ДТП."""
    th = config.thresholds
    df["FE_PARTICIPANTS_BIN"] = _participants_bin(_series(df, "PARTICIPANTS_COUNT"))

    not_notify = _as_flag(_series(df, "NOT_NOTIFICATION"))
    apply_delay = pd.to_numeric(_series(df, "APPLY_DELAY"), errors="coerce")
    df["FE_DELAY_AND_NO_NOTIFY"] = (
        (not_notify == 1) & (apply_delay > th.apply_delay_notify)
    ).astype("Int64")
    return df


def _add_victim_vehicle_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Блок C: ТС потерпевшего."""
    th = config.thresholds
    age_bins = config.vehicle_age_bins

    df["FE_VICTIM_AGE_BIN"] = _vehicle_age_bin(_series(df, "VICTIM_VEHICLE_AGE"), age_bins)
    df["FE_VICTIM_POWER_PER_TON"] = _safe_div(
        _series(df, "VICTIM_CAPACITY_ENGINE"),
        _series(df, "VICTIM_MAX_WEIGHT"),
    )
    weight = pd.to_numeric(_series(df, "VICTIM_MAX_WEIGHT"), errors="coerce")
    df["FE_VICTIM_HEAVY"] = (weight > th.vehicle_weight_heavy).astype("Int64")
    df["FE_VICTIM_DOORS_BIN"] = _doors_bin(_series(df, "VICTIM_NUM_DOORS"))
    df["FE_VICTIM_SEATS_BIN"] = _seats_bin(_series(df, "VICTIM_NUM_PLACE"))

    japan = _as_flag(_series(df, "VICTIM_VEHICLE_IS_JAPAN"))
    made_rf = _as_flag(_series(df, "VICTIM_VEHICLE_MADE_IN_RF"))
    df["FE_VICTIM_JAPAN_RF"] = ((japan == 1) & (made_rf == 1)).astype("Int64")

    df["FE_VICTIM_ENGINE_BUCKET"] = _series(df, "VICTIM_TYPE_ENGINE").astype("string")
    df["FE_VICTIM_BODY_BUCKET"] = _series(df, "VICTIM_TYPE_BODY").astype("string")
    return df


def _add_guilty_vehicle_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Блок D: ТС виновника."""
    th = config.thresholds
    age_bins = config.vehicle_age_bins

    df["FE_GUILTY_AGE_BIN"] = _vehicle_age_bin(_series(df, "GUILTY_VEHICLE_AGE"), age_bins)
    df["FE_GUILTY_POWER_PER_TON"] = _safe_div(
        _series(df, "GUILTY_CAPACITY_ENGINE"),
        _series(df, "GUILTY_MAX_WEIGHT"),
    )
    weight = pd.to_numeric(_series(df, "GUILTY_MAX_WEIGHT"), errors="coerce")
    df["FE_GUILTY_HEAVY"] = (weight > th.vehicle_weight_heavy).astype("Int64")
    df["FE_GUILTY_ENGINE_BUCKET"] = _series(df, "GUILTY_TYPE_ENGINE").astype("string")
    return df


def _add_victim_guilty_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """Блок E: victim vs guilty."""
    victim_power = pd.to_numeric(_series(df, "VICTIM_CAPACITY_ENGINE"), errors="coerce")
    guilty_power = pd.to_numeric(_series(df, "GUILTY_CAPACITY_ENGINE"), errors="coerce")
    victim_weight = pd.to_numeric(_series(df, "VICTIM_MAX_WEIGHT"), errors="coerce")
    guilty_weight = pd.to_numeric(_series(df, "GUILTY_MAX_WEIGHT"), errors="coerce")

    df["FE_DIFF_VEHICLE_POWER"] = victim_power - guilty_power
    df["FE_RATIO_VEHICLE_POWER"] = _safe_div(victim_power, guilty_power)
    df["FE_DIFF_VEHICLE_WEIGHT"] = victim_weight - guilty_weight

    df["FE_SAME_VEHICLE_CATEGORY"] = _equals(
        _series(df, "VICTIM_VEHICLE_CATEGORY"),
        _series(df, "GUILTY_VEHICLE_CATEGORY"),
    )
    df["FE_SAME_VEHICLE_COUNTRY"] = _equals(
        _pick_column(df, "VICTIM_VEHICLE_COUNTRY", "VIC_TS_COUNTRY"),
        _pick_column(df, "GUILTY_VEHICLE_COUNTRY", "GUIL_TS_COUNTRY"),
    )
    df["FE_SAME_VEHICLE_BRAND"] = _equals(
        _series(df, "VICTIM_VEHICLE_BRAND"),
        _series(df, "GUILTY_VEHICLE_BRAND"),
    )
    df["FE_SAME_VEHICLE_BODY"] = _equals(
        _series(df, "VICTIM_TYPE_BODY"),
        _series(df, "GUILTY_TYPE_BODY"),
    )
    df["FE_SAME_VEHICLE_DRIVE"] = _equals(
        _series(df, "VICTIM_TYPE_PRIVOD"),
        _series(df, "GUILTY_TYPE_PRIVOD"),
    )

    japan_v = _as_flag(_series(df, "VICTIM_VEHICLE_IS_JAPAN"))
    japan_g = _as_flag(_series(df, "GUILTY_VEHICLE_IS_JAPAN"))
    df["FE_JAPAN_MISMATCH"] = (japan_v != japan_g).astype("Int64")

    ev_v = _as_flag(_series(df, "VIC_IS_EV_REG"))
    ev_g = _as_flag(_series(df, "GUIL_IS_EV_REG"))
    df["FE_EV_MISMATCH"] = (ev_v != ev_g).astype("Int64")

    df["FE_SAME_TS_REGION"] = _equals(
        _series(df, "VICTIM_TS_REGION"),
        _series(df, "GUILTY_TS_REGION"),
    )
    df["FE_SAME_POLICY_ISSUER_GROUP"] = _equals(
        _series(df, "VICTIM_POLICY_ISSUER_GROUP"),
        _series(df, "GUILTY_POLICY_ISSUER_GROUP"),
    )
    return df


def _add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Блок F: гео."""
    region = _series(df, "REGION")
    region_event = _series(df, "REGION_EVENT")
    region_corrected = _series(df, "REGION_CORRECTED")

    df["FE_SAME_REGION_EVENT"] = _equals(region, region_event)
    corrected_filled = region_corrected.notna() & (region_corrected.astype(str).str.strip() != "")
    df["FE_REGION_CORRECTED"] = (
        corrected_filled & (region_corrected != region)
    ).astype("Int64")

    accepted = _series(df, "ACCEPTED_UNIT")
    loss_unit = _series(df, "LOSS_UNIT")
    both_filled = accepted.notna() & loss_unit.notna()
    df["FE_SAME_ACCEPTED_LOSS_UNIT"] = _equals(accepted, loss_unit).where(both_filled)
    return df


def _add_policy_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Блок G: полис."""
    th = config.thresholds

    df["FE_KBM_BIN"] = _kbm_bin(_series(df, "RSAPolicyKBM"), th)

    taxi = _as_flag(_series(df, "USED_AS_TAXI"))
    carsharing = _as_flag(_series(df, "USED_AS_CARSH"))
    df["FE_COMMERCIAL_USE"] = ((taxi == 1) | (carsharing == 1)).astype("Int64")

    franchise = pd.to_numeric(_series(df, "FRANCHISE_VALUE"), errors="coerce")
    df["FE_HAS_FRANCHISE"] = (franchise > 0).astype("Int64")

    premium_sum = pd.to_numeric(_series(df, "PREMIUM_SUM_ALL"), errors="coerce")
    premium_count = pd.to_numeric(_series(df, "PREMIUM_COUNT_ALL"), errors="coerce")
    df["FE_PREMIUM_PER_POLICY"] = _safe_div(premium_sum, premium_count)

    df["FE_INSURANCE_AMOUNT_BIN"] = _amount_bins(
        _series(df, "INSURANCE_AMOUNT"),
        th.insurance_amount_bins,
    )
    return df


def _add_process_features(df: pd.DataFrame) -> pd.DataFrame:
    """Блок H: refund / minimization."""
    refund_detailed = _series(df, "REFUND_FORM_DETAILED").astype("string")
    refund_order = _series(df, "REFUND_FORM_BY_PAYMENT_ORDER").astype("string")
    refund = _series(df, "REFUND_FORM").astype("string")

    both_refund = refund_detailed.notna() & refund_order.notna()
    df["FE_REFUND_FORM_MATCH"] = (refund_detailed == refund_order).where(both_refund).astype("Int64")

    both_mismatch = refund.notna() & refund_detailed.notna()
    df["FE_REFUND_FORM_MISMATCH"] = (refund != refund_detailed).where(both_mismatch).astype("Int64")

    df["FE_REFUND_IS_CASH"] = refund_detailed.str.contains(
        "Денежн", case=False, na=False
    ).astype("Int64")
    df["FE_REFUND_IS_REPAIR"] = refund_detailed.str.contains(
        "Ремонт", case=False, na=False
    ).astype("Int64")

    rec = pd.to_numeric(_series(df, "MINIMIZATION_REC"), errors="coerce")
    fact = pd.to_numeric(_series(df, "MINIMIZATION_FACT"), errors="coerce")
    df["FE_MINIMIZATION_GAP"] = rec - fact

    minim_kind = _series(df, "MINIMIZATION_KIND")
    df["FE_HAS_MINIMIZATION"] = minim_kind.notna().astype("Int64")
    return df


def _add_repair_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Блок I: калькуляция / ремонт."""
    th = config.thresholds

    df["FE_WEAROUT_TIER"] = _tier_from_bins(
        _series(df, "SHARE_WEAROUT"),
        th.share_wearout_bins,
        ("0-20", "20-50", "50+"),
    )
    df["FE_SHARE_WORK_TIER"] = _tier_from_bins(
        _series(df, "SHARE_WORK"),
        th.share_work_bins,
        ("low", "mid", "high"),
    )

    amount_repair = pd.to_numeric(_series(df, "AMOUNT_REPAIR"), errors="coerce")
    share_wearout = pd.to_numeric(_series(df, "SHARE_WEAROUT"), errors="coerce")
    df["FE_EXPECTED_WEAROUT_RUB"] = amount_repair * share_wearout / 100

    df["FE_AMOUNT_REPAIR_BIN"] = _amount_bins(amount_repair, th.amount_repair_bins)
    df["FE_HIGH_REPAIR"] = (amount_repair > th.amount_repair_high).astype("Int64")

    repair_value = pd.to_numeric(_series(df, "REPAIR_VALUE"), errors="coerce")
    both_positive = (amount_repair > 0) & (repair_value > 0)
    df["FE_REPAIR_TO_VALUE_RATIO"] = _safe_div(repair_value, amount_repair).where(both_positive)
    return df


def _add_loss_history_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Блок J: история убытков (past only)."""
    th = config.thresholds

    victim_count = _series(df, "VICTIM_LOSS_COUNT")
    guilty_count = _series(df, "GUILTY_LOSS_COUNT")

    df["FE_VICTIM_LOSS_COUNT_BIN"] = _count_bin(victim_count)
    df["FE_VICTIM_REPEAT"] = (
        pd.to_numeric(victim_count, errors="coerce").fillna(0) > 0
    ).astype("Int64")
    df["FE_VICTIM_LOSS_SUM_BIN"] = _amount_bins(
        _series(df, "VICTIM_LOSS_SUM"),
        th.victim_loss_sum_bins,
    )

    df["FE_GUILTY_LOSS_COUNT_BIN"] = _count_bin(guilty_count)
    df["FE_GUILTY_REPEAT"] = (
        pd.to_numeric(guilty_count, errors="coerce").fillna(0) > 0
    ).astype("Int64")
    return df


def add_derived_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Добавить все FE_* колонки по каталогу v4."""
    out = df.copy()
    out = _add_timeline_features(out, config)
    out = _add_accident_features(out, config)
    out = _add_victim_vehicle_features(out, config)
    out = _add_guilty_vehicle_features(out, config)
    out = _add_victim_guilty_diff_features(out)
    out = _add_geo_features(out)
    out = _add_policy_features(out, config)
    out = _add_process_features(out)
    out = _add_repair_features(out, config)
    out = _add_loss_history_features(out, config)
    return out