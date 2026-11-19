#!/usr/bin/env bash
# Usage:
#   ./run.sh setup      first time only: install everything and create the database
#   ./run.sh pipeline   run the ETL pipeline with Airflow
#   ./run.sh train      train the models (MLflow) and run monitoring
#   ./run.sh start      start database + API + dashboard + MLflow UI
#   ./run.sh airflow    open the Airflow web UI on http://localhost:8080
#   ./run.sh stop       stop everything
set -e
cd "$(dirname "$0")"
export AIRFLOW_HOME="$PWD/airflow_home" AIRFLOW__CORE__DAGS_FOLDER="$PWD/dags" \
       AIRFLOW__CORE__LOAD_EXAMPLES=False MLFLOW_DISABLE_AGENT_HINT=1

start_db() {
    .pgenv/bin/pg_ctl -D .pgdata status >/dev/null 2>&1 || \
        .pgenv/bin/pg_ctl -D .pgdata -o "-p 5434 -k /tmp" -l data/postgres.log -w start >/dev/null
    echo "database running on port 5434"
}

case "$1" in
setup)
    uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -r requirements.txt
    uv venv --python 3.12 .venv-airflow && VIRTUAL_ENV=.venv-airflow uv pip install \
        "apache-airflow==2.10.5" pandas requests python-dotenv psycopg2-binary \
        --constraint https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.12.txt
    conda create -y -p ./.pgenv -c conda-forge postgresql=16
    mkdir -p data && .pgenv/bin/initdb -D .pgdata -U postgres --auth=trust
    start_db
    PASSWORD=$(openssl rand -hex 8)
    .pgenv/bin/psql -h localhost -p 5434 -U postgres -c "CREATE ROLE accidents LOGIN PASSWORD '$PASSWORD'"
    .pgenv/bin/psql -h localhost -p 5434 -U postgres -c "CREATE DATABASE accidents OWNER accidents"
    echo "DATABASE_URL=postgresql+psycopg2://accidents:$PASSWORD@localhost:5434/accidents" > .env
    .venv-airflow/bin/airflow db migrate
    echo "Setup done. Next: ./run.sh pipeline" ;;
pipeline)
    start_db
    .venv-airflow/bin/airflow dags test accident_etl_pipeline "$(date -u +%Y-%m-%dT%H:%M:%S)" \
        | tee data/airflow_run.log | grep -E "\[[0-9:]{8}\]|Marking task as SUCCESS|FAILED" ;;
train)
    start_db
    .venv/bin/python train.py
    .venv/bin/python monitor.py ;;
start)
    start_db
    nohup setsid .venv/bin/uvicorn api:app --port 8000 > data/api.log 2>&1 < /dev/null &
    nohup setsid .venv/bin/streamlit run dashboard.py --server.headless true > data/dashboard.log 2>&1 < /dev/null &
    nohup setsid .venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 > data/mlflow.log 2>&1 < /dev/null &
    sleep 8
    echo "Dashboard: http://localhost:8501"
    echo "API docs:  http://localhost:8000/docs"
    echo "MLflow:    http://localhost:5000" ;;
airflow)
    start_db
    export PATH="$PWD/.venv-airflow/bin:$PATH"
    echo "login: admin / password in airflow_home/standalone_admin_password.txt"
    airflow standalone ;;
stop)
    pkill -f "uvicorn api:app" || true
    pkill -f "streamlit run dashboard.py" || true
    pkill -f "mlflow" || true
    pkill -f "airflow" || true
    .pgenv/bin/pg_ctl -D .pgdata stop >/dev/null 2>&1 || true
    echo "stopped" ;;
*)
    sed -n '2,8p' "$0" ;;
esac
