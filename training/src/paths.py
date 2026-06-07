"""Пути артефактов этапа обучения - единый источник, как data-pipeline/lib/paths.py

Тяжёлые артефакты (эмбеддинги, предсказания, бандлы) лежат в artifacts/ и гитигнорятся,
ноутбуки-извлечения и код - в git. Эмбеддинги извлекаются на Kaggle и кладутся в
artifacts/embeddings/ руками
"""

from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = TRAINING_ROOT / 'notebooks'

ARTIFACTS_DIR = TRAINING_ROOT / 'artifacts'
EMBEDDINGS_DIR = ARTIFACTS_DIR / 'embeddings'    # кэш эмбеддингов с Kaggle (вход)
PREDICTIONS_DIR = ARTIFACTS_DIR / 'predictions'  # таблицы-контракты по экспериментам (выход)
MODELS_DIR = ARTIFACTS_DIR / 'models'            # инференс-бандлы (выход)
