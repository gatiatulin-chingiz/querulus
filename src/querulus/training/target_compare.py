"""Сверка legacy/new таргетов между датасетами или внутри одного df."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_BINARY_TARGETS: frozenset[str] = frozenset({"TARGET", "TARGET_2", "TARGET_FREQ"})

# Слагаемые таргетов в df_final_3 (если колонка есть — попадёт в top_pair_mismatches)
TARGET_COMPONENTS: dict[str, tuple[str, ...]] = {
    "TARGET_2": (
        "Сумма_выплат_по_претензиям",
        "Сумма_взыскано_по_ФУ",
        "Суммы_взыскано_по_иску",
    ),
    "TARGET_FREQ": ("TARGET_FREQ_CLAIMS_AMOUNT", "TARGET_FREQ_PRET_AMOUNT"),
    "TARGET_3_SEV": (
        "SurchargeValue_cumsum_by_incident",
        "UTSSurchargeValue_cumsum_by_incident",
    ),
    "TARGET_SEV": (
        "TARGET_SEV_CLAIMS_AMOUNT",
        "SurchargeValue_cumsum_by_incident_all",
        "UTSSurchargeValue_cumsum_by_incident_all",
    ),
}

PAIRS_LEGACY: list[tuple[str, str]] = [
    ("TARGET_2", "TARGET_2"),
    ("TARGET_3_SEV", "TARGET_3_SEV"),
]
PAIRS_OLD_VS_NEW: list[tuple[str, str]] = [
    ("TARGET_2", "TARGET_FREQ"),
    ("TARGET_3_SEV", "TARGET_SEV"),
]
REGRESSION_PAIR: tuple[str, str] = ("TARGET_3_SEV", "TARGET_SEV")
CLASSIFICATION_PAIR: tuple[str, str] = ("TARGET_2", "TARGET_FREQ")


@dataclass(frozen=True)
class TargetComparisonResult:
    """Результат сверки таргетов."""

    merged: pd.DataFrame
    report: pd.DataFrame
    top_regression: pd.DataFrame
    top_classification: pd.DataFrame


def compare_target_pairs(
    df_reference: pd.DataFrame,
    df_candidate: pd.DataFrame,
    pairs: list[tuple[str, str]],
    *,
    key: str = "INCIDENT_NUMBER",
    reference_name: str = "reference",
    candidate_name: str = "candidate",
    binary_targets: frozenset[str] = DEFAULT_BINARY_TARGETS,
    float_rtol: float = 0.0,
    float_atol: float = 0.0,
    float_pct_threshold: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Сверка пар (col_reference, col_candidate) по ключу инцидента."""
    if not pairs:
        raise ValueError("pairs не может быть пустым")

    for name, frame in ((reference_name, df_reference), (candidate_name, df_candidate)):
        if key not in frame.columns:
            raise KeyError(f"{name}: нет колонки {key!r}")

    ref = df_reference.copy()
    if "TARGET_2" not in ref.columns and "TARGET" in ref.columns:
        ref["TARGET_2"] = pd.to_numeric(ref["TARGET"], errors="coerce").fillna(0).astype(int)

    cols_ref = {key} | {p[0] for p in pairs}
    cols_cnd = {key} | {p[1] for p in pairs}
    miss_ref = [c for c in cols_ref if c != key and c not in ref.columns]
    miss_cnd = [c for c in cols_cnd if c != key and c not in df_candidate.columns]
    if miss_ref:
        raise KeyError(f"{reference_name}: нет колонок {miss_ref}")
    if miss_cnd:
        raise KeyError(f"{candidate_name}: нет колонок {miss_cnd}")

    ref_k = (
        ref[list(cols_ref)]
        .drop_duplicates(key)
        .rename(columns={c: f"{c}__ref" for c in cols_ref if c != key})
    )
    cnd_k = (
        df_candidate[list(cols_cnd)]
        .drop_duplicates(key)
        .rename(columns={c: f"{c}__cnd" for c in cols_cnd if c != key})
    )
    merged = ref_k.merge(cnd_k, on=key, how="outer", indicator=True)

    print(f"=== {reference_name} vs {candidate_name} (key={key}) ===")
    print(f"  всего ключей: {len(merged):,}")
    print(f"  только {reference_name}: {int((merged['_merge'] == 'left_only').sum()):,}")
    print(f"  только {candidate_name}: {int((merged['_merge'] == 'right_only').sum()):,}")
    print(f"  в обоих: {int((merged['_merge'] == 'both').sum()):,}\n")

    both = merged[merged["_merge"] == "both"].copy()
    report_rows: list[dict] = []
    pair_masks: dict[str, pd.Series] = {}

    for col_ref, col_cnd in pairs:
        pair_label = f"{col_ref} vs {col_cnd}"
        a_col, b_col = f"{col_ref}__ref", f"{col_cnd}__cnd"
        a = pd.to_numeric(both[a_col], errors="coerce")
        b = pd.to_numeric(both[b_col], errors="coerce")
        is_binary = col_ref in binary_targets or col_cnd in binary_targets

        if is_binary:
            a = a.fillna(0).astype(int)
            b = b.fillna(0).astype(int)
            exact = a == b
            pct_ok = exact
        else:
            both_nan = a.isna() & b.isna()
            numeric = np.isclose(a, b, rtol=float_rtol, atol=float_atol, equal_nan=False)
            exact = both_nan | (numeric & a.notna() & b.notna())
            denom = b.abs().clip(lower=1.0)
            pct_ok = both_nan | ((a.sub(b).abs() / denom) <= float_pct_threshold)

        pair_masks[pair_label] = exact
        n = len(both)
        report_rows.append(
            {
                "pair": pair_label,
                "col_reference": col_ref,
                "col_candidate": col_cnd,
                "kind": "binary" if is_binary else "float",
                "common_n": n,
                "exact_match_n": int(exact.sum()),
                "exact_match_pct": round(100 * float(exact.mean()), 4),
                "mismatch_n": int((~exact).sum()),
                "mismatch_pct": round(100 * (1 - float(exact.mean())), 4),
                "pct_match_at_threshold": None
                if is_binary
                else round(100 * float(pct_ok.mean()), 4),
            }
        )
        print(f"--- {pair_label} ({'binary' if is_binary else 'float'}) ---")
        print(f"  точное совпадение: {int(exact.sum()):,} / {n:,} ({100 * exact.mean():.2f}%)")
        print(f"  расхождения:       {int((~exact).sum()):,} ({100 * (1 - exact.mean()):.2f}%)")
        if not is_binary:
            print(f"  ≤{float_pct_threshold:.0%} от candidate: {100 * pct_ok.mean():.2f}%")
        print()

    if pair_masks:
        all_ok = pd.concat(pair_masks, axis=1).all(axis=1)
        print("--- ВСЕ указанные пары ---")
        print(
            f"  полное совпадение: {int(all_ok.sum()):,} / {len(both):,} "
            f"({100 * all_ok.mean():.2f}%)\n"
        )

    return merged, pd.DataFrame(report_rows)


def attach_target_components(
    out: pd.DataFrame,
    *,
    key: str,
    target: str,
    side: str,
    df: pd.DataFrame,
    component_map: dict[str, tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    """Добавить слагаемые таргета с суффиксом __ref / __cnd."""
    component_map = component_map or TARGET_COMPONENTS
    cols = [c for c in component_map.get(target, ()) if c in df.columns]
    if not cols:
        return out
    comp = df[[key, *cols]].drop_duplicates(key)
    comp = comp.rename(columns={c: f"{c}__{side}" for c in cols})
    return out.merge(comp, on=key, how="left")


def top_pair_mismatches(
    merged: pd.DataFrame,
    col_reference: str,
    col_candidate: str,
    *,
    df_reference: pd.DataFrame,
    df_candidate: pd.DataFrame,
    key: str = "INCIDENT_NUMBER",
    binary_targets: frozenset[str] = DEFAULT_BINARY_TARGETS,
    component_map: dict[str, tuple[str, ...]] | None = None,
    n: int = 50,
) -> pd.DataFrame:
    """Топ расхождений по паре таргетов + слагаемые обеих сторон."""
    component_map = component_map or TARGET_COMPONENTS
    both = merged[merged["_merge"] == "both"].copy()
    a_col, b_col = f"{col_reference}__ref", f"{col_candidate}__cnd"
    a = pd.to_numeric(both[a_col], errors="coerce")
    b = pd.to_numeric(both[b_col], errors="coerce")
    is_binary = col_reference in binary_targets or col_candidate in binary_targets

    if is_binary:
        a = a.fillna(0).astype(int)
        b = b.fillna(0).astype(int)
        diff = (a - b).abs()
    else:
        diff = (a - b).abs()

    out = pd.DataFrame(
        {
            key: both[key],
            f"{col_reference}__ref": a,
            f"{col_candidate}__cnd": b,
            "abs_diff": diff,
            "pct_diff_vs_candidate": np.where(
                b.abs() > 0,
                diff / b.abs(),
                np.where(diff > 0, 1.0, 0.0),
            ),
        }
    )
    out = out.loc[diff > 0].sort_values(["abs_diff", "pct_diff_vs_candidate"], ascending=False).head(n)
    out = attach_target_components(
        out, key=key, target=col_reference, side="ref", df=df_reference, component_map=component_map
    )
    out = attach_target_components(
        out, key=key, target=col_candidate, side="cnd", df=df_candidate, component_map=component_map
    )
    return out


def compare_old_vs_new_targets(
    df: pd.DataFrame,
    *,
    pairs: list[tuple[str, str]] | None = None,
    top_n: int = 50,
    key: str = "INCIDENT_NUMBER",
) -> TargetComparisonResult:
    """Сверка legacy vs new таргетов на одном датасете."""
    pairs = pairs or PAIRS_OLD_VS_NEW
    merged, report = compare_target_pairs(
        df, df, pairs, key=key, reference_name="old", candidate_name="new"
    )
    return TargetComparisonResult(
        merged=merged,
        report=report,
        top_regression=top_pair_mismatches(
            merged,
            *REGRESSION_PAIR,
            df_reference=df,
            df_candidate=df,
            key=key,
            n=top_n,
        ),
        top_classification=top_pair_mismatches(
            merged,
            *CLASSIFICATION_PAIR,
            df_reference=df,
            df_candidate=df,
            key=key,
            n=top_n,
        ),
    )


def compare_litigant_vs_querulus(
    df_litigant: pd.DataFrame,
    df_querulus: pd.DataFrame,
    *,
    pairs: list[tuple[str, str]] | None = None,
    top_n: int = 50,
    key: str = "INCIDENT_NUMBER",
) -> TargetComparisonResult:
    """Сверка Litigant parquet vs Querulus."""
    pairs = pairs or PAIRS_LEGACY
    merged, report = compare_target_pairs(
        df_litigant,
        df_querulus,
        pairs,
        key=key,
        reference_name="litigant",
        candidate_name="querulus",
    )
    return TargetComparisonResult(
        merged=merged,
        report=report,
        top_regression=top_pair_mismatches(
            merged,
            *REGRESSION_PAIR,
            df_reference=df_litigant,
            df_candidate=df_querulus,
            key=key,
            n=top_n,
        ),
        top_classification=top_pair_mismatches(
            merged,
            *CLASSIFICATION_PAIR,
            df_reference=df_litigant,
            df_candidate=df_querulus,
            key=key,
            n=top_n,
        ),
    )


def default_litigant_dataset_path(project_root: Path) -> Path:
    """Путь к legacy parquet Litigant."""
    return project_root / "data" / "processed" / "df_final_3_litigant.parquet"
