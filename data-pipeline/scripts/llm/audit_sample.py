"""Достаёт 200 случайных строк из labels_final для ручной проверки

Стратифицированно по final_label (~30 на класс)
Сохраняет audit_sample.parquet с title, description, image_id, final_label, label_source, confidence
Открывай в Jupyter, размечай вручную, считай precision на 144k
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from progress import atomic_write_parquet

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ITEMS = PROJECT_ROOT / 'data' / 'labeling' / 'items.parquet'
LABELS = PROJECT_ROOT / 'data' / 'training' / 'labels_final.parquet'
DST = PROJECT_ROOT / 'data' / 'labeling' / 'audit_sample.parquet'

PER_CLASS = 35
SEED = 42


def main():
    labels = pd.read_parquet(LABELS)
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

    atomic_write_parquet(out, DST)
    print(f'аудит-сэмпл {len(out)} строк, по классам:')
    print(out['final_label'].value_counts().to_string())
    print(f'\nсохранил {DST}')
    print('\nоткрывай в Jupyter, заполни колонку audit_label, потом сравни с final_label')


if __name__ == '__main__':
    main()
