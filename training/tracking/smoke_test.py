"""Дымовой тест трекинга: пишет тестовый ран и читает его назад

Если печатает OK - mlflow настроен и к настройке возвращаться не нужно
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow

from tracking.client import init_mlflow, start_run
from tracking.settings import EXPERIMENT, TRACKING_DB


def main():
    init_mlflow()
    with start_run('smoke_test', params={'demo_param': 1}) as run:
        run_id = run.info.run_id
        mlflow.log_metric('demo_metric', 0.42)
        mlflow.log_text('hello mlflow', 'smoke.txt')

    runs = mlflow.search_runs(experiment_names=[EXPERIMENT])
    assert len(runs) >= 1, 'ран не записался в базу'

    print(f'db: {TRACKING_DB}')
    print(f'run_id: {run_id}')
    print(f'ранов в эксперименте: {len(runs)}')
    print('OK')


if __name__ == '__main__':
    main()
