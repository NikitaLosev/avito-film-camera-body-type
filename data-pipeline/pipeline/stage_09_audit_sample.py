"""Достаёт 210 случайных строк из labels_final для ручной проверки

Стратифицировано по final_label (около 35 на класс), чтобы редкие классы
тоже попали в выборку. Открываем потом в Jupyter через tools/label_audit.py,
руками проставляем audit_label, а stage_10 считает по этому precision
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.io import atomic_write_parquet
from lib.paths import AUDIT_SAMPLE, ITEMS, LABELS_FINAL

PER_CLASS = 35
SEED = 42


def main():
    labels = pd.read_parquet(LABELS_FINAL)
    items = pd.read_parquet(ITEMS)[['item_id', 'title', 'description']]
    df = labels.merge(items, on='item_id')

    parts = []
    for cls in sorted(df['final_label'].unique()):
        sub = df[df['final_label'] == cls]
        n = min(PER_CLASS, len(sub))
        parts.append(sub.sample(n, random_state=SEED))

    out = pd.concat(parts).reset_index(drop=True)
    out['audit_label'] = None
    out['audit_notes'] = None

    cols = ['item_id', 'image_id', 'title', 'description',
            'final_label', 'label_source', 'confidence',
            'audit_label', 'audit_notes']
    out = out[cols]

    atomic_write_parquet(out, AUDIT_SAMPLE)
    print(f'аудит-сэмпл {len(out)} строк, по классам:')
    print(out['final_label'].value_counts().to_string())
    print(f'\nсохранил {AUDIT_SAMPLE}')
    print('\nоткрывай в Jupyter, заполни колонку audit_label, потом запусти stage_10_audit_eval')


if __name__ == '__main__':
    main()
