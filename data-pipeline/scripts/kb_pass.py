'''Прогоняет kb по всем объявлениям и ставит метку где модель явно указана

Сначала на title применяется негативный фильтр (слова которые означают что продают
НЕ камеру или лот): картридж, плёнка, объектив, чехол, коробка, лот, цифров, мн.ч.
"фотоаппараты", "в футляре/чехле" и т.п. Если фильтр сработал, kb не применяется

Дополнительно: если в title найдено 2+ ключа kb (multi-match), это тоже лот → skip

В чистых строках ищется ключ kb (longest-first, с word-boundary \\b) в
нормализованном title + description через text_norm.normalize
'''

import re
from pathlib import Path

import pandas as pd
import yaml

from text_norm import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ITEMS = PROJECT_ROOT / 'data' / 'labeling' / 'items.parquet'
KB = PROJECT_ROOT / 'data' / 'labeling' / 'kb.yaml'
DST = PROJECT_ROOT / 'data' / 'labeling' / 'kb_labels.parquet'

# негативный фильтр на title:
# - стоп-слова в НАЧАЛЕ title (главный товар не камера)
# - явные маркеры цифровой камеры где угодно
# - мн.ч. "фотоаппараты/фотоаппаратов" где угодно (признак лота)
# - "в футляре"/"в чехле" где угодно (на фото скорее всего не видно камеры)
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
    kb = yaml.safe_load(KB.read_text())

    # longest-first чтобы "зенит 11" нашлось раньше "зенит"
    # \b...\b чтобы "зенит 1" не поймал "зенит 11"
    patterns = sorted(kb.keys(), key=len, reverse=True)
    kb_pattern = r'\b(' + '|'.join(re.escape(p) for p in patterns) + r')\b'

    titles = items['title'].fillna('')
    descs = items['description'].fillna('')

    # pandas 3.0 str.contains с string dtype ломается на \b + кириллица,
    # поэтому через map(search) на чистом re
    dirty = titles.map(lambda t: bool(NEGATIVE.search(t)))
    # multi-match в title = лот (например "Смена 6, Смена 8, Polaroid")
    kb_compiled = re.compile(kb_pattern)
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
    }).to_parquet(DST)

    n, matched, n_dirty, n_multi = len(items), kb_label.notna().sum(), dirty.sum(), multi.sum()
    print(f'всего: {n:,}')
    print(f'отсечено негативным фильтром: {n_dirty:,} ({n_dirty/n*100:.1f}%)')
    print(f'отсечено multi-match (лот в title): {n_multi:,} ({n_multi/n*100:.1f}%)')
    print(f'размечено kb: {matched:,} ({matched/n*100:.1f}%)')
    print()
    print('распределение kb-меток:')
    print(kb_label.value_counts(dropna=False).to_string())
    print()
    print('топ-15 моделей по числу матчей:')
    print(kb_model.value_counts().head(15).to_string())


if __name__ == '__main__':
    main()
