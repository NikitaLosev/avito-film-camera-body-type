"""Собирает справочник 'модель -> класс' из размеченного вручную gold.parquet

Берём только строки с object_status = valid_single_film_camera где заполнено
model_name, нормализуем написание через text_norm и сохраняем как простой словарь
в kb.yaml. На следующем этапе по этому словарю размечаются 55k объявлений
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.paths import GOLD, KB_YAML
from lib.text_norm import normalize


def main():
    gold = pd.read_parquet(GOLD)
    valid = gold[
        (gold['object_status'] == 'valid_single_film_camera')
        & gold['model_name'].notna()
    ].assign(model=lambda d: d['model_name'].map(normalize))

    kb = valid.groupby('model')['final_label'].first().to_dict()

    KB_YAML.write_text(yaml.safe_dump(kb, allow_unicode=True, sort_keys=False))
    print(f'моделей: {len(kb)}, сохранил {KB_YAML}')


if __name__ == '__main__':
    main()
