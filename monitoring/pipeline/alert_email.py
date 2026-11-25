"""Weekly email alert from Querulus_DataDrift (MSSQL agrafana).

Reads already-written metrics (no recalculation). Дайджест: последний
``date`` в Motor, строки с PSI выше порога (без усреднения).

Секреты SMTP — из Airflow Connection (EMAIL_LOGIN / EMAIL_PASSWORD).
Получатели — дефолты в ``pipeline.config`` / ``settings`` (env перекрывает).

SMTP через stdlib + pretty_html_table (без ``outboxml.core.email`` —
на контуре outboxml ставится ``--no-deps`` и тянет polars/plots).

  ALERT_DRY_RUN=1 — build mail + save HTML, skip SMTP
  ALERT_RESULTS_DIR — каталог для email.html
"""
from __future__ import annotations

import logging
import os
import smtplib
import ssl
import sys
import tempfile
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd
from pretty_html_table import build_table
from sqlalchemy import text

_PIPELINE_ROOT = Path(__file__).resolve().parent
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))
import _bootstrap_path  # noqa: F401 — monorepo + standalone GitLab clone

from pipeline.settings import load_settings
from pipeline.write_sql import get_engine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("alert_email")

# SMTP (контур задаёт через DAG env; локально — pipeline/.env)
EMAIL_SMTP_SERVER = "internalsmtp.vsk.ru"
EMAIL_PORT = 2525
EMAIL_SENDER = "artificial.intelligence@vsk.ru"

_TABLE_COLS = [
    "model_name",
    "date",
    "track",
    "col",
    "PSI",
    "KL",
    "NaN_train",
    "NaN_test",
    "mode_train",
    "mode_test",
    "mean_train",
    "mean_test",
]


@dataclass
class _EmailEnvConfig:
    """SMTP + получатели (как ожидал outboxml Mail)."""

    email_smtp_server: str
    email_port: int
    email_sender: str
    email_login: str
    email_pass: str
    email_receivers: list[str]
    results_path: str


class _HtmlMail:
    """Минимальный HTML-mail без outboxml (smtplib + pretty_html_table)."""

    _BODY_START = (
        "<html><head><meta charset=\"utf-8\"><style>"
        "p,font{font-family:sans-serif,'Times New Roman',Times,serif;}"
        "</style></head><body>"
    )

    def __init__(self, config: _EmailEnvConfig) -> None:
        self.config = config
        self.msg = MIMEMultipart()
        self.subject = ""
        self.body = self._BODY_START
        self.signature = (
            "<p><strong><br />С уважением,<br />Искусственный Интеллект</strong></p>"
        )

    def add_email_subject(self, subject: str) -> None:
        self.subject = subject

    def add_text(self, text: str, n_line_breaks: int = 0) -> None:
        self.body += (
            "<font>"
            + text.replace("\n", "<br>")
            + "</font>"
            + "<br>" * n_line_breaks
        )

    def add_pandas_table(self, table: pd.DataFrame, *, params: dict | None = None) -> None:
        self.body += build_table(table, color="blue_dark", **(params or {}))

    def save_html(self) -> Path:
        html = self.body + self.signature + "</body></html>"
        out = Path(self.config.results_path) / "email.html"
        out.write_text(html, encoding="utf-8")
        return out

    def send(self) -> None:
        self.body = self.body + self.signature
        self.msg["From"] = self.config.email_sender
        self.msg["To"] = "; ".join(self.config.email_receivers)
        self.msg["Subject"] = self.subject
        self.msg.attach(MIMEText(self.body, "html"))
        with smtplib.SMTP(
            host=self.config.email_smtp_server, port=self.config.email_port
        ) as server:
            server.starttls(context=ssl._create_unverified_context())
            server.login(self.config.email_login, self.config.email_pass)
            server.sendmail(
                self.config.email_sender,
                self.config.email_receivers,
                self.msg.as_string(),
            )


class QuerulusDriftAlertMail:
    """Weekly DataDrift digest for Querulus."""

    def __init__(self, config: _EmailEnvConfig) -> None:
        self.config = config
        self.mail = _HtmlMail(config)

    def send_report(
        self,
        drift_df: pd.DataFrame,
        *,
        psi_threshold: float,
        dry_run: bool = False,
    ) -> None:
        self.mail.add_email_subject("Querulus Monitoring (weekly)")
        self.mail.add_text(
            "Еженедельный отчёт по мониторингу модели сутяжничества",
            n_line_breaks=2,
        )
        self.mail.add_text(
            "Произведён расчёт датадрифта (полный train, фичи cf∪rg) "
            "по отношению к train-датасету.",
            n_line_breaks=1,
        )
        run_date = ""
        if drift_df is not None and not drift_df.empty and "date" in drift_df.columns:
            run_date = str(drift_df["date"].iloc[0])
        self.mail.add_text(
            "Дрифт за последний прогон"
            + (f" ({run_date})" if run_date else "")
            + f", PSI > {psi_threshold}:",
            n_line_breaks=1,
        )
        table = self._prepare_drift_df(drift_df, psi_threshold=psi_threshold)
        if table.empty:
            self.mail.add_text(
                "В последнем прогоне признаков с PSI выше порога нет.",
                n_line_breaks=1,
            )
        else:
            self.mail.add_pandas_table(
                table,
                params=dict(
                    text_align="right", font_family="sans-serif", width="180px"
                ),
            )
        self.mail.add_text(
            "PSI (Population Stability Index): изменение распределения данных.",
            n_line_breaks=1,
        )
        self.mail.add_text(
            "KL (Kullback–Leibler Divergence): различие между распределениями.",
            n_line_breaks=1,
        )
        if dry_run:
            path = self.mail.save_html()
            logger.info("ALERT_DRY_RUN=1 — SMTP skipped, html in %s", path)
            return
        self.mail.send()

    def _prepare_drift_df(
        self, df: pd.DataFrame, *, psi_threshold: float
    ) -> pd.DataFrame:
        """Фильтр PSI > порог по последнему прогону (без усреднения)."""
        if df is None or df.empty or "PSI" not in df.columns:
            return pd.DataFrame()
        work = df.copy()
        work["PSI"] = pd.to_numeric(work["PSI"], errors="coerce")
        alarm = work.loc[work["PSI"] > psi_threshold].copy()
        if alarm.empty:
            return pd.DataFrame()
        cols = [c for c in _TABLE_COLS if c in alarm.columns]
        out = alarm[cols].sort_values(by="PSI", ascending=False)
        return out.rename(
            columns={
                "NaN_train": "NaN_dataset",
                "NaN_test": "NaN_prod",
                "mode_train": "mode_dataset",
                "mode_test": "mode_prod",
                "mean_train": "mean_dataset",
                "mean_test": "mean_prod",
            }
        )


def _load_email_config() -> _EmailEnvConfig:
    settings = load_settings()
    receivers = list(settings.email_receivers)
    if not receivers:
        raise ValueError(
            "email_receivers empty (DEFAULT_EMAIL_RECEIVERS / EMAIL_RECEIVERS)"
        )

    login = os.environ.get("EMAIL_LOGIN", "").strip()
    password = os.environ.get("EMAIL_PASSWORD", "").strip()
    dry = os.environ.get("ALERT_DRY_RUN", "").strip().lower() in {"1", "true", "yes"}
    if not dry and (not login or not password):
        raise ValueError("Set EMAIL_LOGIN and EMAIL_PASSWORD (or ALERT_DRY_RUN=1)")

    results = Path(
        os.environ.get(
            "ALERT_RESULTS_DIR",
            str(Path(tempfile.gettempdir()) / "querulus_alert"),
        )
    )
    results.mkdir(parents=True, exist_ok=True)

    return _EmailEnvConfig(
        email_smtp_server=os.environ.get("EMAIL_SMTP_SERVER", EMAIL_SMTP_SERVER).strip()
        or EMAIL_SMTP_SERVER,
        email_port=int(
            os.environ.get("EMAIL_PORT", str(EMAIL_PORT)) or str(EMAIL_PORT)
        ),
        email_sender=os.environ.get("EMAIL_SENDER", EMAIL_SENDER).strip()
        or EMAIL_SENDER,
        email_login=login,
        email_pass=password,
        email_receivers=receivers,
        results_path=str(results),
    )


def _fetch_latest_drift() -> pd.DataFrame:
    """Только строки с максимальным date в Querulus_DataDrift."""
    settings = load_settings()
    engine = get_engine(settings.grafana_sqlalchemy_url)
    sql = text(
        f"""
        SELECT *
        FROM [{settings.schema}].[{settings.table_drift}]
        WHERE [date] = (
            SELECT MAX([date]) FROM [{settings.schema}].[{settings.table_drift}]
        )
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    logger.info(
        "drift latest || rows=%s schema=%s table=%s date=%s",
        len(df),
        settings.schema,
        settings.table_drift,
        df["date"].iloc[0] if not df.empty and "date" in df.columns else "-",
    )
    return df


def main() -> int:
    dry = os.environ.get("ALERT_DRY_RUN", "").strip().lower() in {"1", "true", "yes"}
    try:
        settings = load_settings()
        drift = _fetch_latest_drift()
        mail = QuerulusDriftAlertMail(config=_load_email_config())
        mail.send_report(
            drift,
            psi_threshold=settings.psi_threshold,
            dry_run=dry,
        )
    except Exception:
        # Полный traceback в stdout pod — UI часто не отдаёт логи (http://:8793)
        logger.exception("alert_email FAILED")
        raise
    logger.info("alert_email done || dry_run=%s", dry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
