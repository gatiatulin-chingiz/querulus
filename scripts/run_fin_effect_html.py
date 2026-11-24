"""Собрать fin_effect_report.html + fin_effect_conclusion.html.

Запуск из корня querulus::

    python scripts/run_fin_effect_html.py
    python scripts/run_fin_effect_html.py --source mssql --bootstrap 200
    python scripts/run_fin_effect_html.py --source excel --excel path/to.xlsx
    python scripts/run_fin_effect_html.py --source synthetic --bootstrap 0

Нужны: витрина (MSSQL / Excel / synthetic) и ретро-parquet
(``data/processed/querulus_train_dataset.parquet`` или ``--retro`` /
``FIN_EFFECT_RETRO_PATH``). Для mssql — ``OISUU_DB_*`` в env.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402

from querulus.fin_effect.excel_explore import (  # noqa: E402
    VITRINA_TABLE_DEFAULT,
    load_monitoring_frame,
)
from querulus.fin_effect.excel_monitoring import (  # noqa: E402
    RETRO_AS_OF_DEFAULT,
    estimate_monitoring_effect,
)
from querulus.fin_effect.monitoring_report import (  # noqa: E402
    CONCLUSION_FILENAME,
    FORMULA_VERSION,
    REPORT_FILENAME,
    write_all_monitoring_htmls,
    write_error_html,
)

DEFAULT_RETRO = PROJECT_ROOT / "data" / "processed" / "querulus_train_dataset.parquet"
DEFAULT_OUT = PROJECT_ROOT / "monitoring" / "data"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ITT fin-effect → report.html + conclusion.html",
    )
    parser.add_argument(
        "--source",
        choices=("mssql", "excel", "synthetic"),
        default="mssql",
        help="источник витрины (default: mssql)",
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=None,
        help="путь к Excel витрины (для --source excel)",
    )
    parser.add_argument(
        "--table",
        default=VITRINA_TABLE_DEFAULT,
        help=f"таблица MSSQL (default: {VITRINA_TABLE_DEFAULT})",
    )
    parser.add_argument(
        "--retro",
        type=Path,
        default=None,
        help="ретро parquet (default: data/processed/querulus_train_dataset.parquet "
        "или FIN_EFFECT_RETRO_PATH)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"каталог HTML (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--retro-as-of",
        default=RETRO_AS_OF_DEFAULT,
        help=f"as_of для ретро-priors (default: {RETRO_AS_OF_DEFAULT})",
    )
    parser.add_argument(
        "--t-calc",
        default=None,
        help="дата расчёта YYYY-MM-DD (default: сегодня)",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="bootstrap ITT iterations (default: 200; полный ~1000)",
    )
    parser.add_argument(
        "--bootstrap-compliance",
        type=int,
        default=100,
        help="bootstrap 100%% compliance iterations (default: 100)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="параллелизм bootstrap (default: -1 = все ядра)",
    )
    parser.add_argument(
        "--no-flat",
        action="store_true",
        help="не копировать flat fin_effect_report/conclusion в --out",
    )
    return parser.parse_args()


def _resolve_retro(path: Path | None) -> Path:
    if path is not None:
        return path
    env = os.environ.get("FIN_EFFECT_RETRO_PATH", "").strip()
    if env:
        return Path(env)
    return DEFAULT_RETRO


def main() -> int:
    args = _parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    retro_path = _resolve_retro(args.retro)

    print(f"formula {FORMULA_VERSION}")
    print(f"source  {args.source}")
    print(f"retro   {retro_path}")
    print(f"out     {out_dir}")

    try:
        if args.source == "excel" and args.excel is None:
            raise ValueError("для --source excel нужен --excel PATH")
        if args.source != "synthetic" and not retro_path.exists():
            raise FileNotFoundError(
                f"ретро parquet не найден: {retro_path}. "
                "Укажи --retro или FIN_EFFECT_RETRO_PATH."
            )

        monitoring_df = load_monitoring_frame(
            source=args.source,
            excel_path=args.excel,
            table=args.table,
        )
        retro_df = pd.read_parquet(retro_path)
        print(f"monitoring rows={len(monitoring_df):,}  retro rows={len(retro_df):,}")

        result = estimate_monitoring_effect(
            monitoring_df,
            retro_df,
            t_calc=args.t_calc,
            retro_as_of=args.retro_as_of,
            bootstrap_iterations=args.bootstrap,
            bootstrap_compliance_iterations=args.bootstrap_compliance,
            bootstrap_n_jobs=args.n_jobs,
        )
        for warning in result.warnings:
            print(f"WARN: {warning}")

        report, conclusion, _, names = write_all_monitoring_htmls(
            result,
            out_dir,
            also_flat_dir=None if args.no_flat else out_dir,
        )
        print(f"OK report     {report}")
        print(f"OK conclusion {conclusion}")
        if not args.no_flat:
            print(f"OK flat       {out_dir / REPORT_FILENAME}")
            print(f"OK flat       {out_dir / CONCLUSION_FILENAME}")
        print(f"stamp names   {names}")
        return 0
    except Exception as exc:
        err_path = out_dir / "fin_effect_error.html"
        write_error_html(exc, err_path, source_label=str(args.table))
        print(f"FAIL: {exc}")
        print(f"error html → {err_path}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
