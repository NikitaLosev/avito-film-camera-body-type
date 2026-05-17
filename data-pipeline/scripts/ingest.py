'''Читает сырой CSV с объявлениями и сохраняет в parquet для быстрой работы дальше'''

import hashlib
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / 'data' / 'raw' / 'items_project_aaa.csv'
DST = PROJECT_ROOT / 'data' / 'labeling' / 'items.parquet'


def main():
    df = pd.read_csv(SRC, dtype={'item_id': str, 'user_id': str})
    DST.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DST)

    sha = hashlib.sha256(SRC.read_bytes()).hexdigest()
    print(f'Строк: {len(df):,}')
    print(f'Колонки: {list(df.columns)}')
    print(f'CSV sha256: {sha}')
    print(f'Сохранил: {DST}')


if __name__ == '__main__':
    main()
