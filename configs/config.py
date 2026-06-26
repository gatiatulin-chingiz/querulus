import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


base_path = Path(__file__).resolve().parent
results_path = PROJECT_ROOT / "data" / "processed"

email_smtp_server = f"{os.getenv('EMAIL_SERVER')}"
email_port = f"{os.getenv('EMAIL_PORT')}"

email_sender = f"{os.getenv('EMAIL_SENDER')}"
email_login = f"{os.getenv('EMAIL_LOGIN')}"
email_pass = f"{os.getenv('EMAIL_PASSWORD')}"
email_receivers = ['gatyatulin@vsk.ru']

mlflow_tracking_uri = "https://mlflow.vsk.ru/"