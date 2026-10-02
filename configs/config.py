import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


base_path = Path(__file__).resolve().parent
results_path = PROJECT_ROOT / "data" / "processed"

# OutBoxML DataSetsManager (пакет ≥ с work_type_*)
work_type_fit = os.getenv("work_type_fit", "CPU")
work_type_hptune = os.getenv("work_type_hptune", "CPU")

email_smtp_server = f"{os.getenv('EMAIL_SERVER')}"
email_port = f"{os.getenv('EMAIL_PORT')}"

email_sender = f"{os.getenv('EMAIL_SENDER')}"
email_login = f"{os.getenv('EMAIL_LOGIN')}"
email_pass = f"{os.getenv('EMAIL_PASSWORD')}"
_email_receivers_raw = os.getenv("EMAIL_RECEIVERS", "gatyatulin@vsk.ru")
email_receivers = [
    addr.strip()
    for addr in _email_receivers_raw.replace(";", ",").split(",")
    if addr.strip()
]

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.vsk.ru/")
mlflow_experiment = os.getenv("MLFLOW_EXPERIMENT", "Querulus")
