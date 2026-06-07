"""Реестр голов классификатора за общим интерфейсом fit/predict_proba

logreg, catboost, lightgbm, random_forest. Пайплайн не знает какая голова - выбор по
имени из конфига, все sklearn-совместимы. accepts_sparse страхует: catboost не кормим
50k разреженным tfidf. Дефолты лёгкие (не тюненные) + balanced под дисбаланс классов
(как у logreg) + фикс сид, чтобы сравнение голов было честным
"""

import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import LOGREG_PARAMS

CATBOOST_PARAMS = dict(loss_function='MultiClass', auto_class_weights='Balanced', random_seed=42, verbose=0)
LGBM_PARAMS = dict(n_estimators=400, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
RF_PARAMS = dict(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)

# голова -> принимает ли разреженную матрицу (catboost на 50k sparse tfidf не годится)
ACCEPTS_SPARSE = {'logreg': True, 'lightgbm': True, 'random_forest': True, 'catboost': False}

# дефолты по голове, для логирования параметров рана
HEAD_DEFAULTS = {
    'logreg': LOGREG_PARAMS,
    'random_forest': RF_PARAMS,
    'lightgbm': LGBM_PARAMS,
    'catboost': CATBOOST_PARAMS,
}


def make_head(name: str, params: dict = None):
    """Создать классификатор по имени, params мержатся поверх дефолтов головы"""
    params = params or {}
    if name == 'logreg':
        return LogisticRegression(**{**LOGREG_PARAMS, **params})
    if name == 'random_forest':
        return RandomForestClassifier(**{**RF_PARAMS, **params})
    if name == 'lightgbm':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**{**LGBM_PARAMS, **params})
    if name == 'catboost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**{**CATBOOST_PARAMS, **params})
    raise ValueError(f'неизвестная голова: {name}')
