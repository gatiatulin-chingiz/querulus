-- Metrics schema (Airflow uses public in the same database)
CREATE SCHEMA IF NOT EXISTS agrafana;
GRANT ALL ON SCHEMA agrafana TO querulus;
ALTER DEFAULT PRIVILEGES IN SCHEMA agrafana GRANT ALL ON TABLES TO querulus;
