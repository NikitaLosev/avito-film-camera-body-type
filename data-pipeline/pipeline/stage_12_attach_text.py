"""Соединяет labels_final с title, description и user_id из items.parquet

На выходе $-$ data/training/data_eda.parquet, в котором все колонки labels_final
плюс текстовые поля объявления и id продавца. Это рабочий артефакт для EDA.
Финальный тренировочный датасет соберём отдельно после EDA, когда решим что
из колонок реально оставлять и какие строки исключать

Текстовые поля и user_id сами по себе живут в items.parquet (артефакт сырых
данных). Тут мы их не копируем как новые сущности, а аккуратно мержим к
финальным меткам по item_id $-$ чтобы дальше работать с одним файлом
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.io import atomic_write_parquet
from lib.paths import EDA_DATASET, ITEMS, LABELS_FINAL


def main():
    labels = pd.read_parquet(LABELS_FINAL)
    items = pd.read_parquet(ITEMS, columns=['item_id', 'user_id', 'title', 'description'])

    df = labels.merge(items, on='item_id', how='left')

    assert len(df) == len(labels), f'left join изменил число строк: {len(df)} != {len(labels)}'
    assert df['item_id'].is_unique, 'после join появились дубликаты item_id'

    print(f'строк:    {len(df):,}')
    print(f'колонок:  {len(df.columns)} (было {len(labels.columns)}, добавили user_id, title, description)')
    print()
    print('пропуски в добавленных колонках:')
    print(f'  user_id:     {df["user_id"].isna().sum():,}')
    print(f'  title:       {df["title"].isna().sum():,}')
    print(f'  description: {df["description"].isna().sum():,}')

    EDA_DATASET.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(df, EDA_DATASET)
    print(f'\nсохранил {EDA_DATASET}')


if __name__ == '__main__':
    main()
