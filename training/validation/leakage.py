"""Группы утечки через connected components и проверка непересечения сплитов

Идея: два объявления попадают в одну группу если связаны хотя бы по одному ключу -
один продавец, одинаковое или почти одинаковое фото, одинаковый текст. Группа целиком
уходит в один сплит, поэтому модель не подсмотрит ответ через утёкшего соседа
"""

import re

import numpy as np
import pandas as pd

from validation.settings import LEAKAGE_KEYS, MIN_DESC_CHARS

TOKEN_RE = re.compile(r'[а-яёa-z0-9]+', flags=re.IGNORECASE)


def normalize_for_dup(text) -> str:
    """Оставить только буквы и цифры в нижнем регистре через один пробел"""
    if pd.isna(text):
        return ''
    return ' '.join(TOKEN_RE.findall(str(text).lower()))


def build_text_key(title, description) -> str | None:
    """Ключ дедупа по тексту, короткое описание даёт None

    Иначе generic-заголовки вроде 'пленочный фотоаппарат' от тысяч разных продавцов
    с пустым описанием слиплись бы в одну гигантскую группу
    """
    desc = normalize_for_dup(description)
    if len(desc) < MIN_DESC_CHARS:
        return None
    return normalize_for_dup(title) + '\x00' + desc


def _find(parent, node):
    root = node
    while parent[root] != root:
        root = parent[root]
    # сжатие пути: подвешиваем пройденные узлы прямо к корню
    while parent[node] != root:
        parent[node], node = root, parent[node]
    return root


def _union(parent, left, right):
    root_left, root_right = _find(parent, int(left)), _find(parent, int(right))
    if root_left != root_right:
        parent[root_right] = root_left


def build_leakage_groups(data: pd.DataFrame, keys=LEAKAGE_KEYS) -> pd.Series:
    """Построить leakage_group_id: union-find по всем ключам сразу"""
    n = len(data)
    parent = np.arange(n)
    work = data.reset_index(drop=True)
    # для каждого ключа сливаем в одну компоненту все строки с общим значением
    for key in keys:
        for members in work.groupby(key, sort=False).indices.values():
            for other in members[1:]:
                _union(parent, members[0], other)
    roots = np.fromiter((_find(parent, i) for i in range(n)), dtype=np.int64, count=n)
    _, group_ids = np.unique(roots, return_inverse=True)
    return pd.Series(group_ids, index=data.index, name='leakage_group_id')


def assert_no_overlap(data: pd.DataFrame, keys=LEAKAGE_KEYS) -> dict:
    """Проверить что train/val/test не пересекаются ни по одному ключу

    Падает если есть пересечение, иначе возвращает словарь с нулями для манифеста
    """
    parts = {name: data[data['split'] == name] for name in ('train', 'val', 'test')}
    overlaps = {}
    for key in keys:
        values = {name: set(parts[name][key].dropna()) for name in parts}
        overlaps[f'train_val_{key}'] = len(values['train'] & values['val'])
        overlaps[f'train_test_{key}'] = len(values['train'] & values['test'])
        overlaps[f'val_test_{key}'] = len(values['val'] & values['test'])
    leaked = {name: count for name, count in overlaps.items() if count > 0}
    assert not leaked, f'утечка между сплитами: {leaked}'
    return overlaps
