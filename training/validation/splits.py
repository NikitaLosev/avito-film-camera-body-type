"""Раскладка групп утечки по train/val/test

Три фазы, детерминированно при фиксированном seed:
- группы с audit/gold строками уходят в test, чтобы не учиться на eval-данных
- мелкие группы с TLR добираем в val, иначе редкий класс (1.3%) туда не попадёт
- остальное раскидываем жадно по убыванию размера, минимизируя перекос классов
"""

import random

import numpy as np
import pandas as pd

from validation.settings import ALL_LABELS, MIN_TLR_VAL, SEED, SPLIT_RATIOS


def attach_group_features(data: pd.DataFrame, group_col='leakage_group_id') -> pd.DataFrame:
    """По каждой группе: размер, счётчики классов, флаги audit/gold/tlr"""
    labels = list(ALL_LABELS)
    counts = data.pivot_table(index=group_col, columns='final_label', values='item_id',
                              aggfunc='count', fill_value=0).reindex(columns=labels, fill_value=0)
    feats = pd.DataFrame({'n_rows': data.groupby(group_col).size()}).join(counts)
    flags = data.groupby(group_col).agg(has_audit=('is_audit', 'any'), has_gold=('is_gold', 'any'))
    feats = feats.join(flags)
    feats['has_tlr'] = feats['TLR'] > 0
    return feats


def assign_splits(feats: pd.DataFrame, ratios=SPLIT_RATIOS, seed=SEED, min_tlr_val=MIN_TLR_VAL) -> dict:
    """Сопоставить каждой группе сплит, вернуть group_id -> split"""
    labels = list(ALL_LABELS)
    splits = list(ratios)
    counts = feats[labels].to_numpy(float)
    nrows = feats['n_rows'].to_numpy(int)
    total = int(nrows.sum())
    targets = {name: ratios[name] * total for name in splits}
    global_share = counts.sum(axis=0) / total
    tlr = labels.index('TLR')

    assignment = {}
    rows = {name: 0 for name in splits}
    cls = {name: np.zeros(len(labels)) for name in splits}

    def place(i, split):
        assignment[i] = split
        rows[split] += int(nrows[i])
        cls[split] += counts[i]

    def pick(i):
        # сплит где группа меньше всего портит баланс классов, тай-брейк по меньшей заполненности
        options = []
        for split in splits:
            if rows[split] + int(nrows[i]) > 1.02 * targets[split]:
                continue
            share = (cls[split] + counts[i]) / (rows[split] + int(nrows[i]))
            cost = float(np.max(np.abs(share - global_share)))
            options.append((cost, rows[split] / targets[split], split))
        if not options:
            return min(splits, key=lambda name: rows[name] / targets[name])
        return min(options)[2]

    # фаза 1: eval-якоря (audit и gold) принудительно в test
    for i in np.where(feats['has_audit'].to_numpy() | feats['has_gold'].to_numpy())[0]:
        place(i, 'test')
    # фаза 2: добираем TLR в val мелкими группами пока не наберём минимум
    tlr_free = [i for i in np.where(feats['has_tlr'].to_numpy())[0] if i not in assignment]
    for i in sorted(tlr_free, key=lambda i: int(nrows[i])):
        if cls['val'][tlr] >= min_tlr_val:
            break
        place(i, 'val')
    # фаза 3: остальное жадно, большие группы первыми чтобы влезли без перекоса
    rest = [i for i in range(len(feats)) if i not in assignment]
    random.Random(seed).shuffle(rest)
    rest.sort(key=lambda i: -int(nrows[i]))
    for i in rest:
        place(i, pick(i))

    return {feats.index[i]: split for i, split in assignment.items()}


def build_split_table(data: pd.DataFrame, assignment: dict, group_col='leakage_group_id') -> pd.DataFrame:
    """Собрать итоговую таблицу item_id -> leakage_group_id -> split"""
    table = data[['item_id', group_col]].copy()
    table['split'] = table[group_col].map(assignment)
    return table
