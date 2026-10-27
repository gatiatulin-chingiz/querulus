"""Одноразово: метки + маска RG для ``df_for_service_main.parquet``.

Читает parquet, считает ``preds_label = (preds_cf >= τ)``, маскирует ``preds_rg``
как сервис (класс 1 или IS_LAWYER), пишет ``THRESHOLD`` / ``preds_label``, сохраняет.

Запуск из корня querulus::

  python scripts/enrich_df_for_service_main_labels.py
  python scripts/enrich_df_for_service_main_labels.py --threshold 0.6 \\
      --path integration/results/df_for_service_main.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


#new
def enrich(path: Path, threshold: float, out: Path | None = None) -> Path:
    df = pd.read_parquet(path)
    if "preds_cf" not in df.columns or "preds_rg" not in df.columns:
        raise SystemExit(f"{path}: нужны колонки preds_cf и preds_rg")

    preds_cf = pd.to_numeric(df["preds_cf"], errors="coerce")
    labels = (preds_cf >= float(threshold)).astype(float)
    if "IS_LAWYER" in df.columns:
        is_lawyer = (
            pd.to_numeric(df["IS_LAWYER"], errors="coerce").fillna(0).eq(1)
        )
    else:
        is_lawyer = pd.Series(False, index=df.index)
    keep = labels.eq(1.0) | is_lawyer
    preds_rg = pd.to_numeric(df["preds_rg"], errors="coerce")
    df["preds_rg"] = preds_rg.where(keep, 0.0)
    df["preds_label"] = labels
    df["THRESHOLD"] = float(threshold)

    target = out if out is not None else path
    df.to_parquet(target, index=True)
    n_pos = int(labels.eq(1.0).sum())
    n_keep = int(keep.sum())
    print(f"OK -> {target}")
    print(f"  rows={len(df):,} threshold={threshold:.4f}")
    print(f"  preds_label=1: {n_pos:,}  keep RG (label|lawyer): {n_keep:,}")
    return target


#new
def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=root / "integration" / "results" / "df_for_service_main.parquet",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="τ main_ (как DEFAULT_MAIN_THRESHOLD)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="куда писать (по умолчанию overwrite --path)",
    )
    args = parser.parse_args()
    if not args.path.is_file():
        raise SystemExit(f"нет файла: {args.path}")
    enrich(args.path, args.threshold, out=args.out)


if __name__ == "__main__":
    main()
