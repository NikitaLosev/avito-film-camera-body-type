"""Реестр блоков фич и сборка матрицы: tfidf по тексту + эмбеддинги текста/фото

Блок - источник (текст или parquet эмбеддингов), ключ джойна, набор колонок и скейлинг.
Конфиг выбирает какие блоки склеить и как нормировать. Все трансформеры фитятся СТРОГО
на train - так нормировка не знает про val/test и сплит остаётся без утечек

Эквивалентности, на которых держится воспроизводимость прежних скриптов:
- StandardScaler нормирует каждую колонку независимо -> один скейлер на склейку блоков
  даёт то же что по скейлеру на блок (embfusion воспроизводится per-block standard)
- хранение CSR vs dense не меняет X·w -> LogReg даёт те же коэффициенты
- переименование emb-колонок меняет только метки, не числа и не порядок
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import TFIDF_PARAMS
from paths import EMBEDDINGS_DIR
from validation.settings import EDA_DATASET, SPLIT_PARQUET

# источник эмбеддингов фото: имя -> (файл, имя модели, размерность)
# pe_sim - prompt-similarity PE-Core (косинусы к промптам), тоже image-keyed плотный блок
IMAGE_SOURCES = {
    'dinov2': ('image_emb_dinov2.parquet', 'facebook/dinov2-small', 384),
    'dinov3': ('image_emb_dinov3.parquet', 'facebook/dinov3-vits16-pretrain-lvd1689m', 384),
    'pe': ('image_emb_pe.parquet', 'timm/PE-Core-L-14-336', 1024),
    'pe_sim': ('text_sim_pe.parquet', 'timm/PE-Core-L-14-336', 14),
}
TEXT_EMB_FILE = 'text_emb_qwen3.parquet'
TEXT_EMB_MODEL = 'Qwen/Qwen3-Embedding-0.6B'
TEXT_EMB_DIM = 1024

# внутренние префиксы колонок: текст один (te_), фото per-source (ie_<source>_) чтобы источники не сталкивались
TEXT_PREFIX, IMAGE_PREFIX = 'te_', 'ie_'
SPLITS = ('train', 'val', 'test')

# бинарные лексические флаги из error-analysis (эксп.9): признаки что объект НЕ
# автозаполняемая одиночная плёнка (лот/запчасти/аксессуар/киноаппарат/цифра)
FLAGS = {name: re.compile(pat) for name, pat in {
    'lot': r'ассортимент|комплект|\bлот\b|набор|несколько|\+',
    'parts': r'запчаст|в ремонт|неисправ|не работает|разбит',
    'accessory': r'видоискатель|чехол|коробк|футляр|вспышк|ремень',
    'movie': r'\bкино',
    'digital': r'цифров',
}.items()}


def text_flags_matrix(texts) -> np.ndarray:
    """Матрица бинарных флагов (n, len(FLAGS)) по lower-тексту - стейтлесс, без фита"""
    low = [str(t).lower() for t in texts]
    cols = [[1 if rx.search(t) else 0 for t in low] for rx in FLAGS.values()]
    return np.array(cols, dtype='int8').T


def _emb_meta(spec: dict) -> tuple:
    """Метаданные emb-блока: (внутренний префикс колонок, размерность, ключ джойна)

    Ключ РАЗНЫЙ: текст джойнится по item_id (строка), фото по image_id (int64). Префикс фото
    per-source (ie_<source>_), чтобы dinov3 и pe не сталкивались колонками в одном конфиге
    """
    if spec['block'] == 'text_emb':
        return TEXT_PREFIX, TEXT_EMB_DIM, 'item_id'
    if spec['block'] == 'image_emb':
        source = spec['source']
        return f'{IMAGE_PREFIX}{source}_', IMAGE_SOURCES[source][2], 'image_id'
    raise ValueError(f'у блока {spec["block"]} нет колонок эмбеддингов')


def _block_cols(spec: dict) -> list:
    """Имена колонок эмбеддинг-блока в загруженном фрейме (source-aware)"""
    prefix, dim, _ = _emb_meta(spec)
    return [f'{prefix}{i}' for i in range(dim)]


def _spec_key(spec: dict) -> str:
    """Ключ блока для словаря скейлеров: текст один, фото per-source"""
    return spec['block'] if spec['block'] == 'text_emb' else f"image_emb:{spec['source']}"


def _load_emb(file: str, prefix: str, key: str) -> pd.DataFrame:
    """Прочитать parquet эмбеддингов: ключ оставить, фичи переименовать ПО ПОЗИЦИИ в {prefix}{i}

    Позиционно (а не по индексу emb_i), чтобы работало и для emb_* (raw), и для sim_* (prompt-sim).
    Для emb_* страхуемся: порядок колонок == порядок индексов 0..n-1 (защита якоря от переколонки)
    """
    emb = pd.read_parquet(EMBEDDINGS_DIR / file)
    feats = [c for c in emb.columns if c != key]
    embcols = [c for c in feats if c.startswith('emb_')]
    if embcols:
        assert feats == embcols and [c.split('_', 1)[1] for c in embcols] == [str(i) for i in range(len(embcols))], \
            f'{file}: колонки emb_* не в порядке 0..n-1'
    return emb.rename(columns={c: f'{prefix}{i}' for i, c in enumerate(feats)})


def _emb_file(spec: dict) -> str:
    """Файл parquet эмбеддингов для спека"""
    return TEXT_EMB_FILE if spec['block'] == 'text_emb' else IMAGE_SOURCES[spec['source']][0]


def load_blocks(specs: list) -> pd.DataFrame:
    """Собрать единый фрейм: метка + сплит + сырьё всех нужных блоков, всё джойнится 1:1

    Все emb-источники (text + любое число image) грузятся каждый на СВОЙ ключ со своим префиксом
    """
    blocks = [s['block'] for s in specs]
    cols = ['item_id', 'final_label']
    if 'image_emb' in blocks:
        cols.insert(1, 'image_id')
    if 'tfidf' in blocks or 'text_flags' in blocks:
        cols += ['title', 'description']
    data = pd.read_parquet(EDA_DATASET, columns=cols)
    n = len(data)

    seen = set()
    for spec in specs:
        if spec['block'] not in ('text_emb', 'image_emb'):
            continue
        tag = (spec['block'], spec.get('source', ''))
        if tag in seen:
            continue
        seen.add(tag)
        prefix, _, key = _emb_meta(spec)
        data = data.merge(_load_emb(_emb_file(spec), prefix, key), on=key, how='left', validate='one_to_one')
        assert data[f'{prefix}0'].notna().all(), f'{tag}: эмбеддинги покрывают не все строки'
    split = pd.read_parquet(SPLIT_PARQUET)[['item_id', 'split']]
    data = data.merge(split, on='item_id', how='left', validate='one_to_one')

    assert len(data) == n, 'merge изменил число строк'
    assert data['split'].notna().all(), 'split покрывает не все строки'
    if 'tfidf' in blocks or 'text_flags' in blocks:
        data['text'] = (data['title'].fillna('') + ' ' + data['description'].fillna('')).str.strip()
    return data


def build_features(data: pd.DataFrame, specs: list) -> dict:
    """Фит трансформеров на train, трансформ всех сплитов, склейка блоков в порядке specs"""
    parts = {name: data[data['split'] == name] for name in SPLITS}
    train = parts['train']
    has_sparse = any(s['block'] == 'tfidf' for s in specs)

    # фитим (только на train) и запоминаем как трансформировать каждый блок
    plan, scalers, vectorizer = [], {}, None
    for spec in specs:
        block, scaling = spec['block'], spec['scaling']
        if block == 'tfidf':
            vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
            vectorizer.fit(train['text'])
            plan.append(('tfidf', vectorizer, None))
        elif block == 'text_flags':
            plan.append(('flags', None, None))
        elif scaling == 'standard':
            cols = _block_cols(spec)
            scaler = StandardScaler().fit(train[cols].to_numpy())
            scalers[_spec_key(spec)] = scaler
            plan.append(('standard', scaler, cols))
        elif scaling == 'l2':
            plan.append(('l2', None, _block_cols(spec)))
        else:
            raise ValueError(f'недопустимый scaling {scaling!r} для блока {block!r}')

    out = {'vectorizer': vectorizer, 'scalers': scalers}
    for name, part in parts.items():
        mats = []
        for kind, transformer, cols in plan:
            if kind == 'tfidf':
                mats.append(transformer.transform(part['text']))            # уже L2 по строке
            elif kind == 'flags':
                m = text_flags_matrix(part['text'].to_numpy())
                mats.append(csr_matrix(m) if has_sparse else m)
            elif kind == 'standard':
                m = transformer.transform(part[cols].to_numpy())
                mats.append(csr_matrix(m) if has_sparse else m)
            else:
                m = normalize(part[cols].to_numpy(), norm='l2')             # per-row, stateless
                mats.append(csr_matrix(m) if has_sparse else m)
        if has_sparse:
            X = hstack(mats).tocsr()
        else:
            X = mats[0] if len(mats) == 1 else np.hstack(mats)
        out[f'X_{name}'] = X
        out[f'id_{name}'] = part['item_id'].to_numpy()
    out['y_train'] = train['final_label'].to_numpy()
    return out
