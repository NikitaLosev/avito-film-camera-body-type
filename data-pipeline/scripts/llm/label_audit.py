"""Интерактивная ручная разметка audit_sample.parquet

Запускай в VS Code interactive python или Jupyter (cell-by-cell через # %%)
Цикл: смотришь карточку через show() -> agree/mark -> само переходит к следующей

agree(i)              # модель угадала, просто копирует final_label в audit_label
mark(i, 'slr')        # модель ошиблась, правильно SLR
mark(i, 'other')      # правильно other_unknown
"""

# %% setup
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
assert (ROOT / 'data').exists(), f'запускай из корня проекта, текущая папка: {ROOT}'

AUDIT = ROOT / 'data' / 'labeling' / 'audit_sample.parquet'
IMG_DIR = ROOT / 'data' / 'raw' / 'image'

# короткие алиасы для аудит-метки
LBL = {
    'slr': 'SLR',
    'tlr': 'TLR',
    'rf': 'rangefinder_viewfinder',
    'comp': 'compact_point_and_shoot',
    'inst': 'instant',
    'other': 'other_unknown',
}

audit = pd.read_parquet(AUDIT)
done = audit['audit_label'].notna().sum()
print(f'размечено {done} / {len(audit)}')


# %% helpers
def show(i):
    r = audit.loc[i]
    desc = r.description[:600] if pd.notna(r.description) else '(пусто)'
    print(f'[{i}] {r.title}')
    print(desc)
    print(f'\n→ модель пометила: {r.final_label}  (source={r.label_source}, conf={r.confidence:.2f})')
    path = IMG_DIR / str(r.image_id % 1000) / f'{r.image_id}.jpg'
    if path.exists():
        display(Image(filename=str(path), width=400))
    else:
        print('[фото не найдено]')


def next_idx():
    todo = audit[audit['audit_label'].isna()]
    return None if todo.empty else todo.index[0]


def mark(i, code, notes=None):
    label = LBL.get(code, code)
    audit.loc[i, 'audit_label'] = label
    if notes:
        audit.loc[i, 'audit_notes'] = notes
    audit.to_parquet(AUDIT)
    left = audit['audit_label'].isna().sum()
    flag = '✓' if label == audit.loc[i, 'final_label'] else '✗'
    print(f'[{i}] {flag} {label}    осталось {left}')
    new_i = next_idx()
    if new_i is not None:
        show(new_i)
    return new_i


def agree(i, notes=None):
    return mark(i, audit.loc[i, 'final_label'], notes)


# %% открыть первую неразмеченную
i = next_idx()
show(i)


# %% размечать и идти дальше (гоняешь эту ячейку по кругу)
# i = agree(i)                                      # если модель угадала
# i = mark(i, 'slr')                                # если модель ошиблась → ставишь правильно
# i = mark(i, 'other', notes='лот из 3 камер')      # с комментарием
i = agree(i)
# %%
# %%
