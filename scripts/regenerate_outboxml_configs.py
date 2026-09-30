"""Перегенерация ``config_parity.json`` / ``config_prod.json`` из собранного датасета.

Локальный smoke: берёт parquet, который указан в текущих конфигах (по умолчанию
синтетический ``df_final_3_synthetic.parquet``), и перезаписывает конфиги через
``querulus.training.build_outboxml_configs.write_outboxml_configs``.

В прод-контуре те же конфиги пишет collect (ячейка export); здесь — чтобы
проверить правила features[] без полной сборки датасета.

Запуск:
    python scripts/regenerate_outboxml_configs.py [--parquet PATH] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from querulus.training.build_outboxml_configs import (  # noqa: E402
    default_model_version,
    write_outboxml_configs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        default=str(PROJECT_ROOT / "data" / "processed" / "df_final_3_synthetic.parquet"),
        help="датасет, на котором считаются частоты/клипы",
    )
    parser.add_argument("--dq-report", default=None, help="data_quality_report.json")
    parser.add_argument("--dry-run", action="store_true", help="только печать результата")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )

    parquet_path = Path(args.parquet)
    df = pd.read_parquet(parquet_path)
    date_column = "PAYMENT_ORDER_DATE_TIME"
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    print(f"dataset: {parquet_path} shape={df.shape} version={default_model_version()}")

    if args.dry_run:
        from querulus.training.build_outboxml_configs import (
            compute_period_windows,
            load_hpo_best_params,
            load_selected_task,
            build_features_block,
            catboost_params_from_hpo,
        )
        from querulus.features.data_quality import clip_bounds_for_outboxml

        report = args.dq_report or str(
            PROJECT_ROOT / "data" / "processed" / "data_quality_report.json"
        )
        clips = clip_bounds_for_outboxml(report_path=report)
        periods = compute_period_windows(df, date_column=date_column)
        fit_index = periods["splits"].train.union(periods["splits"].val)
        for task in ("frequency", "severity"):
            feats, cats = load_selected_task(task)
            block, cat_out = build_features_block(
                df, feats, categorical_names=cats, fit_index=fit_index, clip_bounds=clips
            )
            print(f"\n=== {task} ({len(feats)} фич) ===")
            print(json.dumps(block, ensure_ascii=False, indent=2))
            print("cat_features:", cat_out)
        _ = load_hpo_best_params, catboost_params_from_hpo
        return 0

    result = write_outboxml_configs(
        df,
        parquet_path=str(parquet_path.as_posix()),
        dq_report_path=args.dq_report,
        date_column=date_column,
    )
    print(f"parity: {result['parity_path']}")
    print(f"prod:   {result['prod_path']}")
    print(
        f"features: cf={result['n_frequency_features']} rg={result['n_severity_features']} "
        f"clip_bounds={result['n_clip_bounds']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
