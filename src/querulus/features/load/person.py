"""Person FE loaders — делегируют в dataset.load (без дубли SQL)."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from querulus.dataset.load.claims import fetch_claims_incoming, fetch_claims_persons
from querulus.dataset.load.io import LazyOisuuConnection, read_artifact
from querulus.dataset.load.pretensions import fetch_pretensions_base, fetch_pretensions_penalty
from querulus.dataset.load.sql import render_claims_predicate
from querulus.dataset.paths import DataPaths
from querulus.dataset.preprocess.pretension import dedupe_pretension_rows

TARGET_CLAIMS_ARTIFACT = "target_3_claims.parquet"

_PRETENSION_HISTORY_COLUMNS = (
    "INCIDENTNUMBER",
    "INCIDENT_NUMBER",
    "PRETENSIONNUMBER",
    "PRETENSION_NUMBER",
    "PRETENSIONGETDATE",
    "PRETENSION_GET_DATE",
    "APPLICANTPERSONID",
    "APPLICANT_PERSON_ID",
    "RECIPIENTPERSONID",
    "RECIPIENT_PERSON_ID",
    "PRETENSIONVALUE",
    "PRETENSION_VALUE",
    "SURCHARGEVALUE",
    "SURCHARGE_VALUE",
    "UTSSURCHARGEVALUE",
    "UTS_SURCHARGE_VALUE",
    "PRETENSIONTYPES",
    "PRETENSION_TYPES",
    "PRETENSIONGETMETHOD",
    "PRETENSION_GET_METHOD",
    "ANSWERTYPE",
    "ANSWER_TYPE",
    "PRETENSION_VALUE_PENALTY",
    "SURCHARGE_VALUE_PENALTY",
)


def _subset_columns(df: pd.DataFrame, preferred: Iterable[str]) -> pd.DataFrame:
    keep = [col for col in preferred if col in df.columns]
    if not keep:
        return df
    extra = [
        col
        for col in df.columns
        if col.startswith("CLAIMED") or col.startswith("RECOVERED")
    ]
    for col in ("INCIDENT_NUMBER", "INCOMING_CLAIM_NUMBER", "INCOMING_CLAIM_GET_DATE", "CLAIMITEM", "CLAIMORIGIN"):
        if col in df.columns and col not in keep:
            keep.append(col)
    for col in extra:
        if col not in keep:
            keep.append(col)
    return df[keep]


def _require_conn(conn: LazyOisuuConnection | None, use_sql: bool) -> LazyOisuuConnection:
    if use_sql and conn is None:
        raise ValueError("conn обязателен при use_sql=True")
    return conn or LazyOisuuConnection()


def _pretensions_parquet_columns(paths: DataPaths) -> list[str] | None:
    path = paths.resolve_artifact(paths.raw_dir, "df_pretensions.parquet")
    if path is None or not path.exists():
        return None
    try:
        import pyarrow.parquet as pq

        names = {str(n) for n in pq.read_schema(path).names}
        upper_map = {n.upper(): n for n in names}
        wanted = []
        for pref in _PRETENSION_HISTORY_COLUMNS:
            if pref in names:
                wanted.append(pref)
            elif pref.upper() in upper_map:
                wanted.append(upper_map[pref.upper()])
        wanted.extend(n for n in names if str(n).upper().startswith("DECLARED_"))
        return list(dict.fromkeys(wanted)) or None
    except Exception:
        return None


def load_pretensions_base(
    paths: DataPaths,
    conn: LazyOisuuConnection | None,
    *,
    use_sql: bool,
    save_checkpoint: bool,
) -> pd.DataFrame:
    """Загрузить претензии с INCIDENT_NUMBER (без enrich/cumsum)."""
    _conn = _require_conn(conn, use_sql)
    columns = None if use_sql else _pretensions_parquet_columns(paths)
    df = fetch_pretensions_base(
        paths,
        _conn,
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
        columns=columns,
    )
    df.columns = df.columns.str.upper()
    df = _subset_columns(df, _PRETENSION_HISTORY_COLUMNS)
    return dedupe_pretension_rows(df)


def load_target_claims_for_features(
    paths: DataPaths,
    conn: LazyOisuuConnection | None,
    *,
    use_sql: bool,
    save_checkpoint: bool,
) -> pd.DataFrame:
    """Иски icnl из кэша build_targets."""
    _conn = _require_conn(conn, use_sql)
    try:
        df = read_artifact(paths, paths.raw_dir, TARGET_CLAIMS_ARTIFACT)
    except FileNotFoundError:
        return fetch_claims_incoming(
            paths,
            _conn,
            use_sql=use_sql,
            save_checkpoint=save_checkpoint,
            claims_where=render_claims_predicate(icnl_alias="icnl", loss_alias="l"),
        )
    return _subset_columns(df, ())


def load_pretensions_penalty_surcharge(
    paths: DataPaths,
    conn: LazyOisuuConnection | None,
    *,
    use_sql: bool,
    save_checkpoint: bool,
) -> pd.DataFrame:
    """Загрузить доплаты/неустойку по претензиям."""
    _conn = _require_conn(conn, use_sql)
    df = fetch_pretensions_penalty(
        paths, _conn, use_sql=use_sql, save_checkpoint=save_checkpoint
    )
    df.columns = df.columns.str.upper()
    return df


def load_claims_persons(
    paths: DataPaths,
    conn: LazyOisuuConnection | None,
    *,
    use_sql: bool,
    save_checkpoint: bool,
) -> pd.DataFrame:
    """Загрузить истцов (person_id -> INCOMING_CLAIM_NUMBER)."""
    _conn = _require_conn(conn, use_sql)
    return fetch_claims_persons(
        paths, _conn, use_sql=use_sql, save_checkpoint=save_checkpoint
    )


def load_claims_incoming(
    paths: DataPaths,
    conn: LazyOisuuConnection | None,
    *,
    use_sql: bool,
    save_checkpoint: bool,
    claims_where_sql: str,
) -> pd.DataFrame:
    """Загрузить суды (incoming claim)."""
    _conn = _require_conn(conn, use_sql)
    df = fetch_claims_incoming(
        paths,
        _conn,
        use_sql=use_sql,
        save_checkpoint=save_checkpoint,
        claims_where=claims_where_sql,
    )
    df.columns = df.columns.str.upper()
    return df


def normalize_hex_person_id(value: Any) -> str | None:
    """Привести person_id из SQL к строке hex upper."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex().upper()
    raw = str(value).strip()
    return raw.upper() if raw else None


def normalize_person_id_series(series: pd.Series | None) -> pd.Series:
    """Нормализовать person_id."""
    if series is None:
        return pd.Series(dtype="object")
    return series.map(normalize_hex_person_id)
