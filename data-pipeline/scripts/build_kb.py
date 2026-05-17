'''Собирает справочник "модель камеры -> класс" из ручной разметки gold.parquet

Берёт строки с object_status='valid_single_film_camera' где model_name заполнено,
нормализует написания через text_norm.normalize и пишет в data/labeling/kb.yaml
'''

from pathlib import Path

import pandas as pd
import yaml

from text_norm import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / 'data' / 'labeling' / 'gold.parquet'
DST = PROJECT_ROOT / 'data' / 'labeling' / 'kb.yaml'


def main():
    gold = pd.read_parquet(SRC)
    valid = gold[
        (gold['object_status'] == 'valid_single_film_camera')
        & gold['model_name'].notna()
    ].assign(model=lambda d: d['model_name'].map(normalize))

    kb = valid.groupby('model')['final_label'].first().to_dict()

    DST.write_text(yaml.safe_dump(kb, allow_unicode=True, sort_keys=False))
    print(f'моделей: {len(kb)}, сохранил {DST}')


if __name__ == '__main__':
    main()
