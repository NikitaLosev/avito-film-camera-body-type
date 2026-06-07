"""Срезы метрик по группам где ждём проблем

Главный срез - по популярности kb_model: если на популярных моделях качество резко
выше чем на редких, модель выучила название модели, а не форму корпуса (утечка)
"""

import pandas as pd

# слишком общие заголовки по которым класс не восстановить (из EDA)
GENERIC_TITLES = {'фотоаппарат', 'пленочный фотоаппарат', 'камера', 'фотик'}


def _seller_bucket(count: int) -> str:
    if count >= 50:
        return 'large'
    if count >= 5:
        return 'medium'
    return 'small'


def add_slice_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить колонки срезов: продавец, generic-заголовок, популярность kb, источник, TLR"""
    out = df.copy()
    out['slice_seller'] = out['user_id'].map(out['user_id'].value_counts()).map(_seller_bucket)
    # сравниваем нормализовав ё, чтобы 'плёночный' тоже попал в generic
    title = out['title'].fillna('').str.lower().str.replace('ё', 'е', regex=False).str.strip()
    out['slice_generic_title'] = title.isin(GENERIC_TITLES)
    kb_counts = out['kb_model'].map(out['kb_model'].value_counts())
    out['slice_kb_model'] = pd.cut(kb_counts.fillna(0), bins=[-1, 0, 20, 1e9],
                                   labels=['no_kb', 'rare_kb', 'popular_kb'])
    out['slice_source'] = out['label_source']
    out['slice_tlr'] = out['final_label'].eq('TLR')
    return out


def slice_metrics(df: pd.DataFrame, slice_col: str, metric_fn) -> pd.DataFrame:
    """Посчитать метрику по каждому значению среза"""
    rows = []
    for value, sub in df.groupby(slice_col, observed=True):
        row = {'slice': slice_col, 'value': value, 'n': len(sub)}
        row.update(metric_fn(sub))
        rows.append(row)
    return pd.DataFrame(rows)
