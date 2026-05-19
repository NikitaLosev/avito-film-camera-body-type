"""Отбирает ~500 объявлений для ручной разметки

Стратифицирует выборку грубыми regex по title (slr/tlr/rangefinder/instant/compact)
чтобы в gold попали все классы а не одни мыльницы и Instax
Сохраняет gold.parquet с пустыми колонками разметки по схеме из data/taxonomy.yaml
"""

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / 'data' / 'labeling' / 'items.parquet'
DST = PROJECT_ROOT / 'data' / 'labeling' / 'gold.parquet'

SEED = 42
MAX_PER_USER = 2

# bucket -> (regex для стратификации, сколько строк брать)
BUCKETS = {
    'slr':         (r'зенит|canon ae|nikon f[ma]|pentax k|minolta|olympus om', 80),
    'tlr':         (r'любитель|rolleiflex|yashica mat',                        80),
    'rangefinder': (r'фэд|зоркий|leica|kiev|киев|смена',                       80),
    'instant':     (r'polaroid|instax|fujifilm mini',                          80),
    'compact':     (r'mju|prima|skina|zoom',                                   80),
    'unknown':     (None,                                                     100),
}

KEEP_COLS = ['item_id', 'user_id', 'title', 'description', 'image_id']
LABEL_COLS = ['object_status', 'body_type', 'final_label',
              'model_name', 'evidence', 'label_source', 'confidence']


def detect_bucket(title):
    text = title.lower()
    for name, (pattern, _) in BUCKETS.items():
        if pattern and re.search(pattern, text):
            return name
    return 'unknown'


def main():
    df = pd.read_parquet(SRC)
    df['bucket'] = df['title'].apply(detect_bucket)

    samples = [
        df[df['bucket'] == name].sample(min(quota, (df['bucket'] == name).sum()), random_state=SEED)
        for name, (_, quota) in BUCKETS.items()
    ]

    gold = (pd.concat(samples)
            .drop_duplicates('item_id')
            .groupby('user_id').head(MAX_PER_USER)
            .reset_index(drop=True))

    for col in LABEL_COLS:
        gold[col] = None

    DST.parent.mkdir(parents=True, exist_ok=True)
    gold[KEEP_COLS + LABEL_COLS].to_parquet(DST)

    print(f'Записал {len(gold)} строк в {DST}')
    print(gold['bucket'].value_counts())


if __name__ == '__main__':
    main()
