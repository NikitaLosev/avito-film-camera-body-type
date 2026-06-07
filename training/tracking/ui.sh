#!/usr/bin/env bash
# Поднять MLflow UI на нашей sqlite-базе
# Открыть в браузере: http://127.0.0.1:5000
set -euo pipefail
cd "$(dirname "$0")"
exec mlflow ui \
  --backend-store-uri "sqlite:///$(pwd)/store/mlflow.db" \
  --host 127.0.0.1 --port 5000
