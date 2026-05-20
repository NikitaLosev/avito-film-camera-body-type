"""Читает сырой csv и пересохраняет в parquet, дальше работаем только с ним

Зачем parquet: типы фиксированные (item_id и user_id как строки, остальное pandas угадает),
читается в разы быстрее csv, занимает меньше места, колонки доступны по одной
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.paths import ITEMS, RAW_CSV


def main():
    df = pd.read_csv(RAW_CSV, dtype={'item_id': str, 'user_id': str})
    ITEMS.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ITEMS)

    sha = hashlib.sha256(RAW_CSV.read_bytes()).hexdigest()
    print(f'строк: {len(df):,}')
    print(f'колонки: {list(df.columns)}')
    print(f'csv sha256: {sha}')
    print(f'сохранил: {ITEMS}')


if __name__ == '__main__':
    main()
