"""Exploratory анализ боевой Excel-выгрузки (фаза 1 мониторинга финэффекта).

Профиль колонок, денежные сверки и кросстабы флагов. Отчёт — только агрегаты
(без сырых номеров инцидентов) для копипаста из закрытого контура.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_TOLERANCE = 1.0

# Алиасы заголовков со скринов (имена плывут: «к доплате» / «к выплате», «в инциденте»).
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "incident": ("НомерИнцидента", "INCIDENT_NUMBER", "Номер инцидента"),
    "result_check": ("РезультатПроверки", "Результат проверки"),
    "agreement": (
        "Заключено соглашение",
        "Был ли учет соглашения",
        "ВызовМодельСогласен",
        "Учет соглашения",
    ),
    "model_payout": (
        "Выплата по модели в Инциденте",
        "Выплата по модели в инциденте",
        "Выплата по модели",
    ),
    "recommended": (
        "Сумма рекомендованная к доплате по модулю",
        "Сумма рекомендованная к выплате по модели",
        "Сумма рекомендованная к доплате по модели",
    ),
    "payment": ("СуммаПлатежа", "Сумма платежа"),
    "to_pay": ("СуммаКВыплате", "Сумма к выплате"),
    "wear_cost": ("СтоимостьСУчетомИзноса", "Стоимость с учетом износа"),
    "other_costs": ("Иные затраты", "ИныеЗатраты"),
    "od_claimed": ("СуммаОсновногоДолгаЗаявлено", "Сумма основного долга заявлено"),
    "od_paid": (
        "СуммаОсновногоДолгаВыплачено",
        "Сумма основного долга выплачено",
        "ОсновныеВыплатыСуммаОплаченоДолиВыплата",
    ),
    "od_to_pay": (
        "СуммаОсновногоДолгаКВыплате",
        "ОсновныеВыплатыСуммаОплаченоДолиВыплата",
    ),
    "sum_ou": ("СуммаОУ", "Сумма ОУ"),
    "sum_od": ("СуммаОД", "Сумма ОД"),
    "claimed_od": (
        "ОсновныеВыплатыЗаявлено_Основной долг",
        "ОсновныеВыплатыЗаявление_Основной долг",
    ),
    "claimed_uts": (
        "ОсновныеВыплатыЗаявлено_Утрата товарной стоимости",
        "ОсновныеВыплатыЗаявление_Утрата товарной стоимости",
    ),
    "claimed_evac": (
        "ОсновныеВыплатыЗаявлено_Затраты на эвакуацию ТС",
        "ОсновныеВыплатыЗаявление_Затраты на эвакуацию ТС",
    ),
    "claimed_storage": (
        "ОсновныеВыплатыЗаявлено_Затраты на хранение",
        "ОсновныеВыплатыЗаявление_Затраты на хранение",
    ),
    "refund_form": ("ФормаВозмещения", "Форма возмещения"),
}

_ID_HINTS = (
    "номер",
    "incident",
    "id",
    "инцидент",
    "loss_number",
    "убытокномер",
)

_CANDIDATE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "paid": ("рекоменд", "платеж", "к выплате", "к доплате", "основного долга"),
    "flags": (
        "результатпроверки",
        "соглаш",
        "выплата по модели",
        "модель",
    ),
    "refund": ("формавозмещ", "форма возмещ"),
}


@dataclass(frozen=True)
class ReconcileSpec:
    """Гипотеза сверки: сумма left_cols ≈ сумма right_cols (допуск в ₽)."""

    name: str
    left_cols: tuple[str, ...]
    right_cols: tuple[str, ...]
    subset: str | None = None  # None | "I"


@dataclass
class ExploreResult:
    """Результаты exploratory-прогона."""

    df: pd.DataFrame
    profile: pd.DataFrame
    reconciliations: pd.DataFrame
    crosstabs: dict[str, pd.DataFrame] = field(default_factory=dict)
    candidates: dict[str, list[str]] = field(default_factory=dict)
    report: str = ""


def load_excel(
    path: str | Path,
    *,
    sheet_name: str | int = 0,
) -> pd.DataFrame:
    """Загрузить Excel через openpyxl."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Excel не найден: {path}")
    return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")


def resolve_column(
    df: pd.DataFrame,
    key_or_name: str,
    *,
    aliases: dict[str, tuple[str, ...]] | None = None,
) -> str | None:
    """Найти колонку по ключу алиасов или точному имени (без учёта регистра)."""
    aliases = aliases or COLUMN_ALIASES
    columns = list(df.columns)
    lower_map = {str(c).strip().lower(): c for c in columns}

    candidates: Sequence[str]
    if key_or_name in aliases:
        candidates = aliases[key_or_name]
    else:
        candidates = (key_or_name,)

    for name in candidates:
        hit = lower_map.get(str(name).strip().lower())
        if hit is not None:
            return hit
    return None


def resolve_columns(
    df: pd.DataFrame,
    keys: Iterable[str],
) -> list[str]:
    """Резолв нескольких ключей; пропуски отбрасываются."""
    found: list[str] = []
    for key in keys:
        col = resolve_column(df, key)
        if col is None and key in df.columns:
            col = key
        if col is not None and col not in found:
            found.append(col)
    return found


def _to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _is_id_like(name: str, nunique: int, nrows: int) -> bool:
    low = name.lower().replace(" ", "")
    if any(h in low for h in _ID_HINTS) and nunique > max(50, int(0.5 * nrows)):
        return True
    return False


def profile_columns(
    df: pd.DataFrame,
    *,
    top_n: int = 10,
    skip_id_values: bool = True,
) -> pd.DataFrame:
    """Профиль колонок: тип, % null, nunique; числа — min/max/mean/median; категории — top-N."""
    nrows = len(df)
    rows: list[dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        nunique = int(s.nunique(dropna=True))
        null_pct = float(s.isna().mean() * 100.0) if nrows else 0.0
        numeric = _to_numeric(s)
        numeric_ok = numeric.notna().mean() >= 0.7 and numeric.notna().any()
        dtype = str(s.dtype)
        row: dict[str, Any] = {
            "column": str(col),
            "dtype": dtype,
            "null_pct": round(null_pct, 2),
            "nunique": nunique,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "top_values": "",
        }
        if numeric_ok:
            vals = numeric.dropna()
            row["min"] = float(vals.min()) if len(vals) else np.nan
            row["max"] = float(vals.max()) if len(vals) else np.nan
            row["mean"] = float(vals.mean()) if len(vals) else np.nan
            row["median"] = float(vals.median()) if len(vals) else np.nan
            # Флаги / мало категорий — top counts тоже полезны
            if nunique <= max(top_n * 2, 20):
                vc = numeric.fillna("__NA__").value_counts(normalize=True).head(top_n)
                row["top_values"] = "; ".join(
                    f"{k}={v * 100:.1f}%" for k, v in vc.items()
                )
        else:
            if skip_id_values and _is_id_like(str(col), nunique, nrows):
                row["top_values"] = "(id-like, values omitted)"
            else:
                vc = s.fillna("__NA__").astype(str).value_counts(normalize=True).head(top_n)
                row["top_values"] = "; ".join(
                    f"{k}={v * 100:.1f}%" for k, v in vc.items()
                )
        rows.append(row)
    return pd.DataFrame(rows)


def intervention_mask(df: pd.DataFrame) -> pd.Series:
    """Черновик I: РезультатПроверки=1 ∧ соглашение=1 ∧ выплата по модели=1."""
    parts: list[pd.Series] = []
    for key in ("result_check", "agreement", "model_payout"):
        col = resolve_column(df, key)
        if col is None:
            return pd.Series(False, index=df.index)
        parts.append(_to_numeric(df[col]).fillna(0).eq(1))
    mask = parts[0]
    for p in parts[1:]:
        mask = mask & p
    return mask


def default_reconcile_specs(df: pd.DataFrame) -> list[ReconcileSpec]:
    """Гипотезы сверки со скринов; отсутствуют колонки — спека пропускается."""

    def cols(*keys: str) -> tuple[str, ...] | None:
        resolved = []
        for key in keys:
            col = resolve_column(df, key)
            if col is None:
                return None
            resolved.append(col)
        return tuple(resolved)

    specs: list[ReconcileSpec] = []
    claimed = cols("claimed_od", "claimed_uts", "claimed_evac", "claimed_storage")
    # Если нет отдельной «заявлено всего» — сверяем сумму компонент с od_claimed
    if claimed is not None:
        right = resolve_column(df, "od_claimed")
        if right is not None:
            specs.append(
                ReconcileSpec(
                    "заявлено_компоненты ?= СуммаОсновногоДолгаЗаявлено",
                    claimed,
                    (right,),
                )
            )
        # Иногда «заявлено» — это сумма компонент без отдельного тотала:
        # сверяем компоненты между собой через to_pay / payment не делаем.

    pair = cols("wear_cost", "other_costs")
    to_pay = resolve_column(df, "to_pay")
    if pair is not None and to_pay is not None:
        specs.append(
            ReconcileSpec(
                "износ+иные ?= СуммаКВыплате",
                pair,
                (to_pay,),
            )
        )

    payment = resolve_column(df, "payment")
    if payment is not None and to_pay is not None:
        specs.append(
            ReconcileSpec("СуммаПлатежа ?= СуммаКВыплате", (payment,), (to_pay,))
        )

    od_c = resolve_column(df, "od_claimed")
    od_p = resolve_column(df, "od_paid")
    if od_c is not None and od_p is not None:
        specs.append(
            ReconcileSpec(
                "ОД заявлено ?= ОД выплачено",
                (od_c,),
                (od_p,),
            )
        )

    recommended = resolve_column(df, "recommended")
    if recommended is not None and payment is not None:
        specs.append(
            ReconcileSpec(
                "на I: рекомендованная ?= СуммаПлатежа",
                (recommended,),
                (payment,),
                subset="I",
            )
        )
    od_to_pay = resolve_column(df, "od_to_pay")
    if recommended is not None and od_to_pay is not None:
        specs.append(
            ReconcileSpec(
                "на I: рекомендованная ?= ОД к выплате",
                (recommended,),
                (od_to_pay,),
                subset="I",
            )
        )
    if recommended is not None and od_p is not None:
        specs.append(
            ReconcileSpec(
                "на I: рекомендованная ?= ОД выплачено",
                (recommended,),
                (od_p,),
                subset="I",
            )
        )
    return specs


def _sum_cols(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    total = pd.Series(0.0, index=df.index, dtype=float)
    for col in cols:
        total = total + _to_numeric(df[col]).fillna(0.0)
    return total


def reconcile_amounts(
    df: pd.DataFrame,
    specs: Sequence[ReconcileSpec] | None = None,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> pd.DataFrame:
    """Таблица сверок: % совпадений / медиана |Δ| / n сравнений."""
    specs = list(specs) if specs is not None else default_reconcile_specs(df)
    i_mask = intervention_mask(df)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        missing = [c for c in (*spec.left_cols, *spec.right_cols) if c not in df.columns]
        if missing:
            continue
        work = df
        n_all = len(df)
        if spec.subset == "I":
            work = df.loc[i_mask]
            if work.empty:
                rows.append(
                    {
                        "hypothesis": spec.name,
                        "subset": "I",
                        "n": 0,
                        "n_comparable": 0,
                        "match_pct": np.nan,
                        "median_abs_delta": np.nan,
                        "mean_abs_delta": np.nan,
                    }
                )
                continue
        left = _sum_cols(work, spec.left_cols)
        right = _sum_cols(work, spec.right_cols)
        both = left.notna() & right.notna()
        n_comp = int(both.sum())
        if n_comp == 0:
            match_pct = np.nan
            med = np.nan
            mean_abs = np.nan
        else:
            delta = (left[both] - right[both]).abs()
            match_pct = float((delta <= tolerance).mean() * 100.0)
            med = float(delta.median())
            mean_abs = float(delta.mean())
        rows.append(
            {
                "hypothesis": spec.name,
                "subset": spec.subset or "all",
                "n": int(len(work)),
                "n_comparable": n_comp,
                "match_pct": round(match_pct, 2) if pd.notna(match_pct) else np.nan,
                "median_abs_delta": round(med, 2) if pd.notna(med) else np.nan,
                "mean_abs_delta": round(mean_abs, 2) if pd.notna(mean_abs) else np.nan,
                "n_total_frame": n_all,
            }
        )
    return pd.DataFrame(rows)


def _crosstab_counts(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    if len(present) < 2:
        return pd.DataFrame()
    work = df[list(present)].copy()
    for c in present:
        work[c] = _to_numeric(work[c]).fillna(-999).astype(int)
    ct = work.groupby(list(present), dropna=False).size().reset_index(name="n")
    ct["pct"] = (ct["n"] / len(df) * 100.0).round(2) if len(df) else 0.0
    return ct.sort_values("n", ascending=False).reset_index(drop=True)


def flag_crosstabs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Кросстабы флагов модели / соглашения / выплаты и форм возмещения."""
    out: dict[str, pd.DataFrame] = {}
    result = resolve_column(df, "result_check")
    agreement = resolve_column(df, "agreement")
    model = resolve_column(df, "model_payout")
    form = resolve_column(df, "refund_form")

    triple = [c for c in (result, agreement, model) if c is not None]
    if len(triple) >= 2:
        out["flags"] = _crosstab_counts(df, triple)

    if form is not None and model is not None:
        work = df[[form, model]].copy()
        work[model] = _to_numeric(work[model]).fillna(-999).astype(int)
        work[form] = work[form].fillna("__NA__").astype(str)
        ct = work.groupby([form, model], dropna=False).size().reset_index(name="n")
        ct["pct"] = (ct["n"] / len(df) * 100.0).round(2) if len(df) else 0.0
        out["refund_form_x_model"] = ct.sort_values("n", ascending=False).reset_index(
            drop=True
        )
    elif form is not None:
        vc = df[form].fillna("__NA__").astype(str).value_counts(normalize=False)
        out["refund_form"] = (
            vc.rename_axis(form).reset_index(name="n").assign(
                pct=lambda x: (x["n"] / len(df) * 100.0).round(2)
            )
        )
    return out


def suggest_candidates(df: pd.DataFrame) -> dict[str, list[str]]:
    """Эвристики: колонки-кандидаты на paid / флаги / форму возмещения."""
    result: dict[str, list[str]] = {k: [] for k in _CANDIDATE_KEYWORDS}
    # Явные алиасы — в начало
    for key, bucket in (
        ("recommended", "paid"),
        ("payment", "paid"),
        ("to_pay", "paid"),
        ("od_to_pay", "paid"),
        ("result_check", "flags"),
        ("agreement", "flags"),
        ("model_payout", "flags"),
        ("refund_form", "refund"),
    ):
        col = resolve_column(df, key)
        if col is not None and col not in result[bucket]:
            result[bucket].append(col)

    for col in df.columns:
        low = str(col).lower()
        for bucket, keywords in _CANDIDATE_KEYWORDS.items():
            if any(k in low for k in keywords):
                if col not in result[bucket]:
                    result[bucket].append(str(col))
    return result


def _df_to_md(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df is None or df.empty:
        return "_пусто_"
    view = df.head(max_rows)
    try:
        return view.to_markdown(index=False)
    except Exception:
        return view.to_string(index=False)


def format_explore_report(
    df: pd.DataFrame,
    *,
    profile: pd.DataFrame | None = None,
    reconciliations: pd.DataFrame | None = None,
    crosstabs: dict[str, pd.DataFrame] | None = None,
    candidates: dict[str, list[str]] | None = None,
    title: str = "Excel explore report",
) -> str:
    """Markdown-отчёт только с агрегатами (для копипаста в чат)."""
    profile = profile if profile is not None else profile_columns(df)
    reconciliations = (
        reconciliations
        if reconciliations is not None
        else reconcile_amounts(df)
    )
    crosstabs = crosstabs if crosstabs is not None else flag_crosstabs(df)
    candidates = candidates if candidates is not None else suggest_candidates(df)

    lines: list[str] = [
        f"# {title}",
        "",
        f"- rows: **{len(df)}**",
        f"- columns: **{df.shape[1]}**",
        "",
        "## Columns",
        "",
        ", ".join(f"`{c}`" for c in df.columns),
        "",
        "## Profile",
        "",
        _df_to_md(profile),
        "",
        "## Money reconciliations (tolerance ~1 RUB)",
        "",
        _df_to_md(reconciliations),
        "",
        "## Flag crosstabs",
        "",
    ]
    if not crosstabs:
        lines.append("_нет подходящих флагов_")
    else:
        for name, table in crosstabs.items():
            lines.extend([f"### {name}", "", _df_to_md(table), ""])

    lines.extend(["## Candidates (heuristics)", ""])
    for bucket, cols in candidates.items():
        shown = ", ".join(f"`{c}`" for c in cols) if cols else "_нет_"
        lines.append(f"- **{bucket}**: {shown}")
    lines.append("")
    lines.append(
        "_В отчёте нет сырых ID инцидентов — только агрегаты для закрытого контура._"
    )
    return "\n".join(lines)


def run_explore(
    df: pd.DataFrame,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> ExploreResult:
    """Полный exploratory-прогон + текстовый отчёт."""
    profile = profile_columns(df)
    reconciliations = reconcile_amounts(df, tolerance=tolerance)
    crosstabs = flag_crosstabs(df)
    candidates = suggest_candidates(df)
    report = format_explore_report(
        df,
        profile=profile,
        reconciliations=reconciliations,
        crosstabs=crosstabs,
        candidates=candidates,
    )
    return ExploreResult(
        df=df,
        profile=profile,
        reconciliations=reconciliations,
        crosstabs=crosstabs,
        candidates=candidates,
        report=report,
    )


def save_explore_report(report: str, path: str | Path) -> Path:
    """Сохранить отчёт локально (только на контуре)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return path
