"""Обёртки над mlflow: единый старт рана, теги воспроизводимости, реестр моделей

Папка называется tracking а не mlflow, чтобы import mlflow тянул настоящий пакет.
Логирование идёт прямо в sqlite, отдельный сервер поднимать не нужно
"""

import contextlib
import platform
import subprocess

import mlflow
from mlflow.tracking import MlflowClient

from tracking.settings import (
    ARTIFACTS_DIR,
    EXPERIMENT,
    PROJECT_ROOT,
    REPORTS_DIR,
    RUNS_CSV,
    STORE_DIR,
    TRACKING_URI,
)


def init_mlflow():
    """Настроить трекинг и выбрать эксперимент, можно звать в начале любого скрипта"""
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACTS_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT)


def _git_commit() -> str:
    try:
        out = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True)
        return out.strip()
    except Exception:
        return 'unknown'


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(['git', 'status', '--porcelain'], cwd=PROJECT_ROOT, text=True)
        return bool(out.strip())
    except Exception:
        return False


def log_repro_tags(extra: dict = None) -> None:
    """Залогировать теги воспроизводимости: код и окружение

    extra нужен для data_sha / split_sha когда они появятся
    """
    tags = {
        'git_commit': _git_commit(),
        'git_dirty': str(_git_dirty()),
        'python_version': platform.python_version(),
    }
    if extra:
        tags.update(extra)
    mlflow.set_tags(tags)


@contextlib.contextmanager
def start_run(name: str, params: dict = None, tags: dict = None):
    """Открыть ран, автоматически залогировать repro-теги и переданные параметры"""
    init_mlflow()
    with mlflow.start_run(run_name=name) as run:
        log_repro_tags()
        if tags:
            mlflow.set_tags(tags)
        if params:
            mlflow.log_params(params)
        yield run


def register_model(run_id: str, artifact_path: str, name=EXPERIMENT, alias=None):
    """Зарегистрировать артефакт рана как новую версию модели в реестре

    alias (например champion) метит версию которая идёт в сервис
    """
    init_mlflow()
    version = mlflow.register_model(f'runs:/{run_id}/{artifact_path}', name)
    if alias:
        MlflowClient().set_registered_model_alias(name, alias, version.version)
    return version


def export_runs_csv(dst=RUNS_CSV):
    """Выгрузить все раны эксперимента в csv, чтобы отчёт читался без sqlite-базы"""
    init_mlflow()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT])
    runs.to_csv(dst, index=False)
    return dst
