"""Каскад: стейдж-1 (камера/не) -> стейдж-2 (тип среди 5). Композиция в 6-классовое распределение

Стейдж-2 учится только на камерах train. Обе модели независимы -> утечек нет, OOF не нужен.
Выход - тот же контракт (item_id, pred_label, confidence, p_<class>), дальше tau/evaluate штатно

Композиция:  p_other = 1 - P(camera);  p_<type> = P(camera) * P(type | camera)  (сумма = 1)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import issparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import contract_frame
from features import build_features
from models import ACCEPTS_SPARSE, make_head
from validation.settings import ABSTAIN, MAIN_SET


def _check_sparse(feats: dict, head: str, stage: str) -> None:
    if issparse(feats['X_train']) and not ACCEPTS_SPARSE[head]:
        raise ValueError(f'{stage}: голова {head} не принимает разреженные фичи (есть tfidf-блок)')


class CascadeModel:
    """Двухстадийный каскад: фит обеих голов, композиция вероятностей в 6 классов"""

    def __init__(self, stage1_cfg: dict, stage2_cfg: dict):
        self.s1 = stage1_cfg
        self.s2 = stage2_cfg

    def fit(self, data):
        """Стейдж-1 - бинарка camera/other на всём train; стейдж-2 - 5-класс на камерах train"""
        self.feats1 = build_features(data, self.s1['blocks'])
        self.feats2 = build_features(data, self.s2['blocks'])
        _check_sparse(self.feats1, self.s1['head'], 'стейдж-1')
        _check_sparse(self.feats2, self.s2['head'], 'стейдж-2')

        # стейдж-1: бинарка camera/other на всём train
        y1 = self.feats1['y_train']
        self.head1 = make_head(self.s1['head'], self.s1.get('head_params'))
        self.head1.fit(self.feats1['X_train'], np.where(y1 == ABSTAIN, 'other', 'camera'))

        # стейдж-2: 5-класс только на камерах train (своя y, чтобы не зависеть от выравнивания с feats1)
        y2 = self.feats2['y_train']
        cam = y2 != ABSTAIN
        self.head2 = make_head(self.s2['head'], self.s2.get('head_params'))
        self.head2.fit(self.feats2['X_train'][cam], y2[cam])
        return self

    def _p_camera(self, split: str) -> np.ndarray:
        proba = self.head1.predict_proba(self.feats1[f'X_{split}'])
        return proba[:, list(self.head1.classes_).index('camera')]

    def frame(self, split: str) -> pd.DataFrame:
        """Контракт-таблица: 6-класс из композиции P(camera) x P(type | camera)"""
        p_cam = self._p_camera(split)
        type_cls = list(self.head2.classes_)
        p_type = self.head2.predict_proba(self.feats2[f'X_{split}'])
        n = len(p_cam)

        by_class = {ABSTAIN: 1 - p_cam}
        for t in MAIN_SET:
            by_class[t] = p_cam * p_type[:, type_cls.index(t)] if t in type_cls else np.zeros(n)
        return contract_frame(self.feats1[f'id_{split}'], by_class)

    def stage1_frame(self, split: str) -> pd.DataFrame:
        """Стейдж-1 отдельно: item_id + p_camera, для анализа границы камера/не"""
        return pd.DataFrame({'item_id': self.feats1[f'id_{split}'], 'p_camera': self._p_camera(split)})
