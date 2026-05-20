"""Отбирает ~500 объявлений для ручной разметки в gold

Случайной выборки мало - почти всё уйдёт в мыльницы и Instax, потому что они
доминируют в категории. Поэтому стратифицируем по грубому regex на title:
ловим SLR, TLR, дальномерки, инстант, компакт + добираем 'unknown' случайно

Ограничение MAX_PER_USER чтобы не словить кучу объявлений от одного продавца
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.paths import GOLD, ITEMS

SEED = 42
MAX_PER_USER = 2

# bucket -> (regex для стратификации, квота)
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
    df = pd.read_parquet(ITEMS)
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

    GOLD.parent.mkdir(parents=True, exist_ok=True)
    gold[KEEP_COLS + LABEL_COLS].to_parquet(GOLD)

    print(f'записал {len(gold)} строк в {GOLD}')
    print(gold['bucket'].value_counts())


if __name__ == '__main__':
    main()
