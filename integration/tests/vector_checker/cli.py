"""CLI: prepare / compare / demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .compare_features import (
    compare_feature_frames,
    load_excel_export,
    write_compare_artifacts,
)
from .paths import (
    DEFAULT_QUERY_OUT,
    DEFAULT_QUERY_PATH,
    DEFAULT_SAMPLE_META,
    EXCEL_DIR,
    LOSS_NUMBER_COL,
    REPORTS_DIR,
    WORK_DIR,
    ensure_work_dirs,
    resolve_df_for_service,
)
from .query_inject import (
    load_sample_meta,
    read_query_text,
    sample_loss_numbers,
    unique_loss_numbers,
    write_injected_query,
    write_sample_meta,
)
from .synthetic import write_synthetic_fixtures


def _filter_service_by_losses(service: pd.DataFrame, losses: list[str]) -> pd.DataFrame:
    keys = {str(x).strip() for x in losses}
    mask = service[LOSS_NUMBER_COL].map(lambda x: str(x).strip() if pd.notna(x) else "").isin(keys)
    return service.loc[mask].copy()


def _cmd_prepare(args: argparse.Namespace) -> int:
    ensure_work_dirs()
    df_path = resolve_df_for_service(args.df)
    query_path = Path(args.query)
    out_path = Path(args.out)
    sample_meta_path = Path(args.sample_meta)

    df = pd.read_parquet(df_path)
    if LOSS_NUMBER_COL not in df.columns:
        raise SystemExit(f"В {df_path} нет колонки {LOSS_NUMBER_COL}")

    all_losses = unique_loss_numbers(df[LOSS_NUMBER_COL].tolist())
    n = args.n if args.n is not None else args.limit
    seed = int(args.seed)
    losses = sample_loss_numbers(all_losses, n, random_seed=seed)

    write_injected_query(
        query_path,
        losses,
        out_path,
        source_df=df_path,
        as_strings=not args.as_numbers,
        n_total=len(all_losses),
        random_seed=seed if n is not None else None,
    )
    write_sample_meta(
        sample_meta_path,
        losses,
        source_df=df_path,
        n_total=len(all_losses),
        n=n,
        random_seed=seed,
    )
    print(f"df_for_service : {df_path}")
    print(f"убытков всего   : {len(all_losses)}")
    print(f"в выборке      : {len(losses)}" + (f" (n={n}, seed={seed})" if n is not None else ""))
    print(f"шаблон         : {query_path}")
    print(f"запрос для 1С  : {out_path}")
    print(f"sample meta    : {sample_meta_path}")
    print(f"положите Excel-выгрузку в: {EXCEL_DIR}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    ensure_work_dirs()
    df_path = resolve_df_for_service(args.df)
    excel_path = Path(args.excel)
    if not excel_path.is_file():
        raise SystemExit(f"Excel не найден: {excel_path}")

    service = pd.read_parquet(df_path)
    excel = load_excel_export(excel_path)

    sample_meta_path = Path(args.sample_meta) if args.sample_meta else DEFAULT_SAMPLE_META
    if args.use_sample_meta and sample_meta_path.is_file():
        meta = load_sample_meta(sample_meta_path)
        losses = [str(x) for x in meta.get("loss_numbers") or []]
        before = len(service)
        service = _filter_service_by_losses(service, losses)
        print(
            f"sample meta    : {sample_meta_path} "
            f"(service {before} → {len(service)} rows, seed={meta.get('random_seed')})"
        )

    report, mismatches = compare_feature_frames(
        service,
        excel,
        all_overlap=args.all_overlap,
        rtol=args.rtol,
    )
    if args.strict_features and report.features_missing_in_excel:
        report.ok = False

    stem = args.report_stem or "vector_compare"
    json_path, csv_path = write_compare_artifacts(
        report, mismatches, Path(args.reports_dir), stem=stem
    )

    print(f"df_for_service : {df_path}")
    print(f"excel          : {excel_path}")
    print(f"keys match     : {report.n_matched_keys}")
    print(f"only service   : {report.n_only_service}")
    print(f"only excel     : {report.n_only_excel}")
    print(f"mismatch cells : {report.n_mismatch_cells}")
    print(f"features       : {len(report.features_compared)}")
    if report.features_missing_in_excel:
        print(f"нет в Excel    : {report.features_missing_in_excel}")
    print(f"report json    : {json_path}")
    print(f"mismatches csv : {csv_path}")
    print(f"RESULT         : {'OK' if report.ok else 'FAIL'}")
    return 0 if report.ok else 1


def _cmd_demo(args: argparse.Namespace) -> int:
    """Синтетика: prepare + compare (ok) + compare (mismatch)."""
    ensure_work_dirs()
    paths = write_synthetic_fixtures(n_rows=args.n_rows)
    print("synthetic fixtures:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    query_out = WORK_DIR / "Сутяжность_for_1c_demo.txt"
    df = pd.read_parquet(paths["df"])
    all_losses = unique_loss_numbers(df[LOSS_NUMBER_COL].tolist())
    n = args.n if args.n is not None else min(3, len(all_losses))
    seed = int(args.seed)
    losses = sample_loss_numbers(all_losses, n, random_seed=seed)
    write_injected_query(
        paths["query_stub"],
        losses,
        query_out,
        source_df=paths["df"],
        as_strings=True,
        n_total=len(all_losses),
        random_seed=seed,
    )
    write_sample_meta(
        DEFAULT_SAMPLE_META,
        losses,
        source_df=paths["df"],
        n_total=len(all_losses),
        n=n,
        random_seed=seed,
    )
    injected, _ = read_query_text(query_out)
    if "&Убыток" in injected:
        print("FAIL: в demo-запросе остался &Убыток")
        return 1
    if "Убыток.Номер В (" not in injected:
        print("FAIL: не найдена подстановка Убыток.Номер В (...)")
        return 1
    print(f"prepare demo   : {query_out} ({len(losses)}/{len(all_losses)} losses, seed={seed})")

    service_full = pd.read_parquet(paths["df"])
    service = _filter_service_by_losses(service_full, losses)
    excel_ok = load_excel_export(paths["excel_ok"])
    excel_ok = _filter_service_by_losses(excel_ok, losses)
    report_ok, mism_ok = compare_feature_frames(service, excel_ok)
    write_compare_artifacts(report_ok, mism_ok, REPORTS_DIR, stem="demo_ok")
    print(
        f"compare ok     : RESULT={'OK' if report_ok.ok else 'FAIL'} "
        f"(mismatches={report_ok.n_mismatch_cells}, "
        f"missing_excel={report_ok.features_missing_in_excel})"
    )

    excel_bad = load_excel_export(paths["excel_bad"])
    # mismatch-фикстура ломает конкретные ключи полного набора — сверяем на полном service
    report_bad, mism_bad = compare_feature_frames(service_full, excel_bad)
    write_compare_artifacts(report_bad, mism_bad, REPORTS_DIR, stem="demo_mismatch")
    print(
        f"compare bad    : RESULT={'OK' if report_bad.ok else 'FAIL'} "
        f"(mismatches={report_bad.n_mismatch_cells}, "
        f"only_service={report_bad.n_only_service})"
    )

    # Воспроизводимость sample
    losses2 = sample_loss_numbers(all_losses, n, random_seed=seed)
    if losses != losses2:
        print("FAIL: sample_loss_numbers не детерминирован при том же seed")
        return 1

    ok_cells = report_ok.n_mismatch_cells == 0 and report_ok.n_only_service == 0
    bad_detected = (not report_bad.ok) and (
        report_bad.n_mismatch_cells > 0 or report_bad.n_only_service > 0
    )
    if ok_cells and bad_detected:
        print("RESULT         : OK (synthetic pipeline works)")
        return 0

    print("RESULT         : FAIL (synthetic expectations not met)")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m integration.tests.vector_checker",
        description="Сверка вектора 1С (Сутяжность) с df_for_service",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="Подставить LOSS_NUMBER в запрос для 1С")
    p_prep.add_argument("--df", default=None, help="Путь к df_for_service.parquet")
    p_prep.add_argument("--query", default=str(DEFAULT_QUERY_PATH), help="Шаблон Сутяжность.txt")
    p_prep.add_argument("--out", default=str(DEFAULT_QUERY_OUT), help="Выходной запрос для 1С")
    p_prep.add_argument(
        "-n",
        "--n",
        type=int,
        default=None,
        help="Случайная выборка N убытков (по умолчанию — все)",
    )
    p_prep.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random_seed для выборки (default: 42)",
    )
    p_prep.add_argument(
        "--limit",
        type=int,
        default=None,
        help="устарело: алиас для --n (случайная выборка с --seed)",
    )
    p_prep.add_argument(
        "--sample-meta",
        default=str(DEFAULT_SAMPLE_META),
        help="Куда писать sampled_losses.json",
    )
    p_prep.add_argument(
        "--as-numbers",
        action="store_true",
        help="Литералы без кавычек (по умолчанию строки \"...\")",
    )
    p_prep.set_defaults(func=_cmd_prepare)

    p_cmp = sub.add_parser("compare", help="Сверить Excel 1С с df_for_service")
    p_cmp.add_argument("--excel", required=True, help="Путь к xlsx/csv выгрузке из 1С")
    p_cmp.add_argument("--df", default=None, help="Путь к df_for_service.parquet")
    p_cmp.add_argument("--reports-dir", default=str(REPORTS_DIR))
    p_cmp.add_argument("--report-stem", default="vector_compare")
    p_cmp.add_argument(
        "--sample-meta",
        default=str(DEFAULT_SAMPLE_META),
        help="sampled_losses.json из prepare",
    )
    p_cmp.add_argument(
        "--use-sample-meta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Сузить service до выборки из prepare (default: да)",
    )
    p_cmp.add_argument(
        "--all-overlap",
        action="store_true",
        help="Сверять все общие колонки, а не только SERVICE_VECTOR_FEATURES",
    )
    p_cmp.add_argument(
        "--strict-features",
        action="store_true",
        help="FAIL, если ожидаемые фичи отсутствуют в Excel",
    )
    p_cmp.add_argument("--rtol", type=float, default=0.0, help="Допуск для float")
    p_cmp.set_defaults(func=_cmd_compare)

    p_demo = sub.add_parser("demo", help="Синтетический прогон prepare+compare")
    p_demo.add_argument("--n-rows", type=int, default=5)
    p_demo.add_argument("-n", "--n", type=int, default=None, help="Размер выборки убытков")
    p_demo.add_argument("--seed", type=int, default=42)
    p_demo.set_defaults(func=_cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
