"""Weekly email alert from querulus_DataDrift (MSSQL agrafana).

Reads already-written metrics (no recalculation). Same shape as archive
datadrift_* UUEMailMonitoring: PSI over threshold table + Grafana link.

Env (no secrets in code):
  EMAIL_SMTP_SERVER, EMAIL_PORT, EMAIL_SENDER, EMAIL_LOGIN, EMAIL_PASSWORD
  EMAIL_RECEIVERS — comma-separated
  GRAFANA_DASHBOARD_URL — link in body
  PSI_THRESHOLD — default 0.3
  ALERT_LOOKBACK_DAYS — default 7
  ALERT_RESULTS_DIR — where to save email.html (EMail.save_mail_as_html)
  ALERT_DRY_RUN=1 — build mail + save HTML, skip SMTP
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from outboxml.core.email import EMailMonitoring
from sqlalchemy import text

_PIPELINE_ROOT = Path(__file__).resolve().parent
_MONITORING_ROOT = _PIPELINE_ROOT.parent
if str(_MONITORING_ROOT) not in sys.path:
    sys.path.insert(0, str(_MONITORING_ROOT))

from pipeline.config import load_dotenv, load_settings
from pipeline.write_sql import get_engine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("alert_email")

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
    """Attrs expected by outboxml.dsml.mailing.Mail / EMail."""

    email_smtp_server: str
    email_port: int
    email_sender: str
    email_login: str
    email_pass: str
    email_receivers: list[str]
    results_path: str


class QuerulusDriftAlertMail(EMailMonitoring):
    """Weekly DataDrift digest (cf+rg) for Querulus."""

    def send_report(
        self,
        drift_df: pd.DataFrame,
        *,
        psi_threshold: float,
        grafana_url: str,
        dry_run: bool = False,
    ) -> None:
        self.base_mail(
            header_name="Querulus Monitoring (weekly)",
            text="Еженедельный отчёт по мониторингу модели сутяжничества",
        )
        self.mail.add_text(
            "Произведён расчёт датадрифта (cf/rg) по отношению к train-датасету.",
            n_line_breaks=1,
        )
        self.mail.add_text(
            f"Обнаружен дрифт в фичах выше PSI={psi_threshold}:",
            n_line_breaks=1,
        )
        table = self._prepare_drift_df(drift_df, psi_threshold=psi_threshold)
        if table.empty:
            self.mail.add_text(
                "За выбранный период признаков с PSI выше порога нет.",
                n_line_breaks=1,
            )
        else:
            self.mail.add_pandas_table(
                table,
                params=dict(text_align="right", font_family="sans-serif", width="180px"),
            )
        if grafana_url:
            self.mail.add_text(
                f"Полные результаты в Grafana: {grafana_url}",
                n_line_breaks=2,
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
            self.save_mail_as_html()
            logger.info("ALERT_DRY_RUN=1 — SMTP skipped, html in %s", self.config.results_path)
            return
        self.send()

    def _prepare_drift_df(
        self, df: pd.DataFrame, *, psi_threshold: float
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        alarm = df.loc[df["PSI"] > psi_threshold].copy()
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
    load_dotenv()
    receivers_raw = os.environ.get("EMAIL_RECEIVERS", "").strip()
    receivers = [r.strip() for r in receivers_raw.split(",") if r.strip()]
    if not receivers:
        raise ValueError("Set EMAIL_RECEIVERS (comma-separated)")

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
        email_smtp_server=os.environ.get("EMAIL_SMTP_SERVER", "internalsmtp.vsk.ru").strip()
        or "internalsmtp.vsk.ru",
        email_port=int(os.environ.get("EMAIL_PORT", "2525") or "2525"),
        email_sender=os.environ.get(
            "EMAIL_SENDER", "artificial.intelligence@vsk.ru"
        ).strip()
        or "artificial.intelligence@vsk.ru",
        email_login=login,
        email_pass=password,
        email_receivers=receivers,
        results_path=str(results),
    )


def _fetch_drift(lookback_days: int) -> pd.DataFrame:
    settings = load_settings()
    engine = get_engine(settings.grafana_sqlalchemy_url)
    # date column is written by datadrift; look back N days including today.
    sql = text(
        f"""
        SELECT *
        FROM [{settings.schema}].[{settings.table_drift}]
        WHERE [date] >= DATEADD(day, - :lookback, CAST(GETDATE() AS date))
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"lookback": lookback_days})
    logger.info(
        "drift rows=%s schema=%s table=%s lookback_days=%s",
        len(df),
        settings.schema,
        settings.table_drift,
        lookback_days,
    )
    return df


def main() -> int:
    lookback = int(os.environ.get("ALERT_LOOKBACK_DAYS", "7") or "7")
    grafana_url = os.environ.get("GRAFANA_DASHBOARD_URL", "").strip()
    dry = os.environ.get("ALERT_DRY_RUN", "").strip().lower() in {"1", "true", "yes"}
    settings = load_settings()

    drift = _fetch_drift(lookback)
    mail = QuerulusDriftAlertMail(config=_load_email_config())
    mail.send_report(
        drift,
        psi_threshold=settings.psi_threshold,
        grafana_url=grafana_url,
        dry_run=dry,
    )
    over = int((drift["PSI"] > settings.psi_threshold).sum()) if not drift.empty and "PSI" in drift else 0
    logger.info("alert_email done || psi_over_threshold=%s || dry_run=%s", over, dry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
