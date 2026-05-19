"""Интерактивная разметка gold.parquet

Запускай ячейки (# %%) в VS Code interactive python или в Jupyter
Цикл: смотришь карточку через show() -> ставишь метку через lbl() -> сам перейдёт к следующей
"""

# %% setup
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
assert (ROOT / 'data').exists(), f'запускай из корня проекта, текущая папка: {ROOT}'

GOLD = ROOT / 'data' / 'labeling' / 'gold.parquet'
IMG_DIR = ROOT / 'data' / 'raw' / 'image'

# короткие алиасы для object_status (полный список - в data/taxonomy.yaml)
STATUS = {
    'v': 'valid_single_film_camera',
    'lot': 'multi_camera_lot',
    'acc': 'accessory_or_part',
    'film': 'film_or_consumable',
    'dig': 'digital_camera',
    'box': 'box_manual_packaging',
    'not': 'not_camera',
    'info': 'insufficient_info',
    'noimg': 'image_unavailable',
    'conf': 'conflicting_evidence',
}

# короткие алиасы для body_type
BODY = {
    'slr': 'SLR',
    'tlr': 'TLR',
    'rf': 'rangefinder_viewfinder',
    'comp': 'compact_point_and_shoot',
    'inst': 'instant',
}


gold = pd.read_parquet(GOLD)
done = gold['object_status'].notna().sum()
print(f'размечено {done} / {len(gold)}')


# %% helpers
def show(i):
    row = gold.loc[i]
    print(f'[{i}] {row.title}')
    print(row.description[:600])
    path = IMG_DIR / str(row.image_id % 1000) / f'{row.image_id}.jpg'
    display(Image(filename=str(path), width=400))


def next_idx():
    todo = gold[gold['object_status'].isna()]
    return None if todo.empty else todo.index[0]


def lbl(i, status, body=None, model=None, ev=None):
    s = STATUS[status]
    b = BODY[body] if body else None
    final = b if s == 'valid_single_film_camera' and b else 'other_unknown'
    gold.loc[i, ['object_status', 'body_type', 'final_label',
                 'model_name', 'evidence', 'label_source', 'confidence']] = [
        s, b, final, model, ev, 'human', 1.0,
    ]
    gold.to_parquet(GOLD)
    left = gold['object_status'].isna().sum()
    print(f'[{i}] -> {final}    осталось {left}')
    new_i = next_idx()
    if new_i is not None:
        show(new_i)
    return new_i


# %% открыть первую неразмеченную
i = next_idx()
show(i)


# %% разметить и перейти к следующей (эту ячейку гоняешь по кругу)
# примеры:
#   i = lbl(i, 'v', 'slr', model='Зенит TTL', ev='в заголовке')
#   i = lbl(i, 'v', 'inst', model='Polaroid 636', ev='в заголовке')
#   i = lbl(i, 'lot', ev='в объявлении 3 камеры')
#   i = lbl(i, 'acc', ev='только объектив, камеры нет')
#   i = lbl(i, 'box', ev='на фото пустая коробка')
_ = lbl(420, 'v', 'inst', model='Palaroid 636', ev='Polaroid 636 Closeup на фото')
# %%
