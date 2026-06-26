from pathlib import Path
from environs import Env

env_reader = Env()
_env_file = base_path / ".env"
if _env_file.exists():
    env_reader.read_env(_env_file)
else:
    env_reader.read_env(base_path / "env_template")

base_path = Path(__file__).resolve().parent
project_root = base_path.parent
results_path = project_root / "data" / "processed"
prod_path = project_root / "data" / "processed"
prod_models_folder = base_path / "prod_models"
prod_models_path = prod_path / prod_models_folder

# database_host = env_reader.str("database_host","o-rtdm-tst-p1")
# database_port = env_reader.str("database_port", "1521")
# database_service_name = env_reader.str("database_service_name", "rtdmtst")
# database_user_name = env_reader.str("user_name", "")
# database_user_pass = env_reader.str("user_pass", "")

email_smtp_server = "internalsmtp.vsk.ru"
email_port = 2525
email_sender = "artificial.intelligence@vsk.ru"
email_login = env_reader.str("EMAIL_LOGIN", "")
email_pass = env_reader.str("EMAIL_PASSWORD", "")
email_receivers = ["Gatyatulin@vsk.ru"] # , "AVBondarev@vsk.ru", "Matsera@VSK.RU", "VASuvorov@vsk.ru"

# SQLAlchemy URL for Grafana export (avoid hardcoding secrets in code).
# Example for SQL Server (pyodbc):
# mssql+pyodbc://USER:PASSWORD@HOST:1433/DB?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes
grafana_sqlalchemy_url = env_reader.str("GRAFANA_SQLALCHEMY_URL", "")