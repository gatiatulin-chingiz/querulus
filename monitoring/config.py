from pathlib import Path

from querulus import PROJECT_ROOT
from querulus.env import load_project_env

env_reader = load_project_env()
project_root = PROJECT_ROOT
base_path = Path(__file__).resolve().parent

results_path = project_root / "data" / "processed"
prod_path = project_root / "data" / "processed"
prod_models_folder = base_path / "prod_models"
prod_models_path = prod_path / prod_models_folder

email_smtp_server = "internalsmtp.vsk.ru"
email_port = 2525
email_sender = "artificial.intelligence@vsk.ru"
email_login = env_reader.str("EMAIL_LOGIN", "")
email_pass = env_reader.str("EMAIL_PASSWORD", "")
email_receivers = ["Gatyatulin@vsk.ru"]

grafana_sqlalchemy_url = env_reader.str("GRAFANA_SQLALCHEMY_URL", "")
