"""CLI: preprocess → AHunter → Querulus check.

Примеры:
  python -m pipeline.run --limit 1 --skip-ahunter
  python -m pipeline.run --limit 15
  python -m pipeline.run --all
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# запуск и как модуль, и как скрипт
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "pipeline"

from . import config
from .ahunter_client import run_ahunter
from .preprocess import run_preprocess
from .querulus_check import run_querulus_check


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Пайплайн справочника автоюристов → AHunter → /api/predict"
    )
    p.add_argument(
        "--input",
        type=Path,
        default=config.DEFAULT_INPUT,
        help="Разложенная таблица (xlsx)",
    )
    p.add_argument(
        "--artifacts",
        type=Path,
        default=config.ARTIFACTS,
        help="Каталог промежуточных файлов",
    )
    p.add_argument(
        "--df-for-service",
        type=Path,
        default=config.DEFAULT_DF_FOR_SERVICE,
        help="parquet базовых убытков",
    )
    p.add_argument(
        "--predict-url",
        default=config.PREDICT_URL,
        help="URL шлюза /api/predict",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Сколько контактов слать в Querulus (1, 15, …). Без флага и без --all = все",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Явно отправить все строки справочника",
    )
    p.add_argument("--skip-preprocess", action="store_true")
    p.add_argument("--skip-ahunter", action="store_true")
    p.add_argument("--skip-querulus", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    art = args.artifacts
    art.mkdir(parents=True, exist_ok=True)

    path_01 = art / "01_input.xlsx"
    path_02 = art / "02_preprocessed.xlsx"
    path_03 = art / "03_cleaned.xlsx"
    path_04 = art / "04_querulus_check.xlsx"
    path_summary = art / "04_querulus_summary.json"

    if not args.input.is_file():
        print(f"Нет входного файла: {args.input}")
        return 1

    shutil.copy2(args.input, path_01)
    print(f"01_input ← {args.input.name}")

    if args.skip_preprocess:
        shutil.copy2(path_01, path_02)
        print("preprocess: skipped")
    else:
        run_preprocess(path_01, path_02)

    if args.skip_ahunter:
        shutil.copy2(path_02, path_03)
        print("ahunter: skipped")
    else:
        run_ahunter(path_02, path_03)

    if args.skip_querulus:
        print("querulus: skipped")
        return 0

    if not args.df_for_service.is_file():
        print(f"Нет df_for_service: {args.df_for_service}")
        return 1

    limit = None if args.all else args.limit
    # если ни --limit ни --all — шлём все (limit=None)
    run_querulus_check(
        path_03,
        args.df_for_service,
        path_04,
        path_summary,
        limit=limit,
        predict_url=args.predict_url,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
