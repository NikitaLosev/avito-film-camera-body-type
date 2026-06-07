"""Контракт таблицы предсказаний между моделью и evaluator

Любая модель пишет parquet с этими колонками, evaluator читает только его.
Друг про друга они ничего не знают - связь идёт через формат, а не через импорты
"""

import numpy as np
import pandas as pd

from validation.settings import ALL_LABELS

PRED_REQUIRED = ('item_id', 'pred_label', 'confidence')


def prob_col(label: str) -> str:
    """Имя колонки вероятности класса, например p_SLR"""
    return f'p_{label}'


def _check_probabilities(preds: pd.DataFrame) -> None:
    """Если есть колонки p_<class> - проверить что вероятности согласованы

    Это закрывает класс ошибок где модель пишет кривые вероятности,
    а порог и калибровка молча считаются по мусору
    """
    prob_cols = [prob_col(label) for label in ALL_LABELS]
    if not all(col in preds.columns for col in prob_cols):
        return
    probs = preds[prob_cols].to_numpy(float)
    if not np.isfinite(probs).all():
        raise ValueError('вероятности содержат NaN или inf')
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError('вероятности по классам не суммируются в 1')
    argmax = np.array(ALL_LABELS)[probs.argmax(axis=1)]
    if not (preds['pred_label'].astype(str).to_numpy() == argmax).all():
        raise ValueError('pred_label не совпадает с argmax вероятностей')
    if not np.allclose(preds['confidence'].astype(float), probs.max(axis=1), atol=1e-6):
        raise ValueError('confidence не совпадает с максимальной вероятностью')


def validate_predictions(preds: pd.DataFrame) -> None:
    """Проверить формат таблицы предсказаний, упасть с понятной ошибкой если что-то не так"""
    missing = [col for col in PRED_REQUIRED if col not in preds.columns]
    if missing:
        raise ValueError(f'нет обязательных колонок: {missing}')
    bad = set(preds['pred_label'].astype(str)) - set(ALL_LABELS)
    if bad:
        raise ValueError(f'метки вне схемы: {bad}')
    if not preds['confidence'].astype(float).between(0.0, 1.0).all():
        raise ValueError('confidence вне диапазона 0..1')
    if preds['item_id'].duplicated().any():
        raise ValueError('item_id в предсказаниях не уникален')
    _check_probabilities(preds)
