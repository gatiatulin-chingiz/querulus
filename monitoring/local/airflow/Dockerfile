FROM apache/airflow:2.8.4-python3.11

USER root

# ODBC Driver 18 for SQL Server (mssql+pyodbc → Motor)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates apt-transport-https \
    && curl -fsSL https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb \
        -o /tmp/packages-microsoft-prod.deb \
    && dpkg -i /tmp/packages-microsoft-prod.deb \
    && rm -f /tmp/packages-microsoft-prod.deb \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 unixodbc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/querulus/monitoring /opt/airflow/db

# requirements-airflow.txt omits SQLAlchemy (Airflow ships 1.4.x)
COPY pipeline/requirements-airflow.txt /opt/querulus/monitoring/pipeline-requirements.txt

USER airflow
RUN pip install --no-cache-dir -r /opt/querulus/monitoring/pipeline-requirements.txt
