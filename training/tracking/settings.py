"""Настройки трекинга экспериментов: пути к локальному mlflow и имя эксперимента"""

from pathlib import Path

TRACKING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TRACKING_ROOT.parents[1]

# локальный стор mlflow: sqlite + артефакты, в git не идёт
STORE_DIR = TRACKING_ROOT / 'store'
TRACKING_DB = STORE_DIR / 'mlflow.db'
ARTIFACTS_DIR = STORE_DIR / 'mlartifacts'

# sqlite-бэкенд, потому что Model Registry для версий весов нужен БД-бэкенд
TRACKING_URI = f'sqlite:///{TRACKING_DB}'
EXPERIMENT = 'film_camera_body_type'

# сюда выгружаем таблицу всех ранов для отчёта
REPORTS_DIR = TRACKING_ROOT.parent / 'reports'
RUNS_CSV = REPORTS_DIR / 'experiments_summary.csv'
