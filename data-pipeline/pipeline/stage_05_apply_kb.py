"""Прогоняет справочник по всем объявлениям и ставит метку где модель явно указана

Перед матчингом отсекаем заведомо плохие случаи через два фильтра:

1. Негативный фильтр на title - стоп-слова в самом начале (главный товар не камера),
   признаки цифровой камеры где угодно, множественное число фотоаппаратов (лот)
   и формулировки 'в футляре' / 'в чехле' (на фото скорее всего камеры не видно)

2. Multi-match в title - если по title находится больше одного ключа справочника,
   это лот из нескольких камер, тоже пропускаем

В чистых строках ищем ключ kb в нормализованном title + description, longest-first
(чтобы 'зенит 11' нашёлся раньше 'зенит'), с word boundaries ('зенит 1' не должен
поймать 'зенит 11')
"""

import re
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.paths import ITEMS, KB_LABELS, KB_YAML
from lib.text_norm import normalize

# стоп-слова в начале title + явные маркеры цифровой камеры + лот-паттерны
NEGATIVE = re.compile(
    r'^\s*(?:картридж|плёнк|пленк|фотобумаг|проявител|фиксаж|'
    r'объектив|вспышк|чехол|ремешок|крышк|штатив|'
    r'коробк|инструкц|паспорт|гаранти|'
    r'запчаст|детал|адаптер|'
    r'лот|комплект|набор|коллекци)'
    r'|\bцифров|\bdigital|\bdslr'
    r'|\bфотоаппарат(?:ы|ов|ам|ами|ах)\b'
    r'|\bв футляре\b|\bв чехле\b',
    re.IGNORECASE,
)


def main():
    items = pd.read_parquet(ITEMS)
    kb = yaml.safe_load(KB_YAML.read_text())

    patterns = sorted(kb.keys(), key=len, reverse=True)
    kb_pattern = r'\b(' + '|'.join(re.escape(p) for p in patterns) + r')\b'
    kb_compiled = re.compile(kb_pattern)

    titles = items['title'].fillna('')
    descs = items['description'].fillna('')

    # pandas 3.0 str.contains с string dtype ломается на \b + кириллица
    # поэтому матчинг делаем через map(search) на чистом re
    dirty = titles.map(lambda t: bool(NEGATIVE.search(t)))
    title_normed = titles.map(normalize)
    multi = title_normed.map(lambda t: len(kb_compiled.findall(t)) > 1)
    skip = dirty | multi

    text = (titles + ' ' + descs).where(~skip, '').map(normalize)
    kb_model = text.str.extract(kb_pattern, expand=False)
    kb_label = kb_model.map(kb)

    pd.DataFrame({
        'item_id': items['item_id'],
        'kb_model': kb_model,
        'kb_label': kb_label,
        'label_source': kb_label.where(kb_label.isna(), 'kb'),
        'confidence': kb_label.where(kb_label.isna(), 1.0),
    }).to_parquet(KB_LABELS)

    n = len(items)
    matched = kb_label.notna().sum()
    print(f'всего: {n:,}')
    print(f'отсечено негативным фильтром: {dirty.sum():,} ({dirty.sum() / n * 100:.1f}%)')
    print(f'отсечено multi-match (лот в title): {multi.sum():,} ({multi.sum() / n * 100:.1f}%)')
    print(f'размечено kb: {matched:,} ({matched / n * 100:.1f}%)')
    print()
    print('распределение kb-меток:')
    print(kb_label.value_counts(dropna=False).to_string())
    print()
    print('топ-15 моделей по числу матчей:')
    print(kb_model.value_counts().head(15).to_string())


if __name__ == '__main__':
    main()
