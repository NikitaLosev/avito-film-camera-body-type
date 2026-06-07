"""Список экспериментов как данные: имя -> конфиг. Единственный файл для нового опыта

Конфиг: name (= имя рана в mlflow), blocks (порядок = порядок hstack), head, head_params.
Имена ранов совпадают с историческими, чтобы experiments_summary.csv остался согласован.
dinov2-конфиги НЕ выкидываем - это задокументированный якорь сравнения
"""

TFIDF = {'block': 'tfidf', 'scaling': 'none'}
QWEN_STD = {'block': 'text_emb', 'scaling': 'standard'}
QWEN_L2 = {'block': 'text_emb', 'scaling': 'l2'}
TEXT_FLAGS = {'block': 'text_flags', 'scaling': 'none'}


def _img(source: str, scaling: str = 'standard') -> dict:
    return {'block': 'image_emb', 'scaling': scaling, 'source': source}


def _cfg(name: str, blocks: list, head: str = 'logreg', head_params: dict = None) -> dict:
    return {'name': name, 'blocks': blocks, 'head': head, 'head_params': head_params or {}}


def _cascade(name: str, s1_head: str, s1_blocks: list, s2_head: str, s2_blocks: list) -> dict:
    """Каскад: стейдж-1 (камера/не) -> стейдж-2 (тип среди 5)"""
    return {'name': name, 'kind': 'cascade',
            'stage1': {'head': s1_head, 'blocks': s1_blocks, 'head_params': {}},
            'stage2': {'head': s2_head, 'blocks': s2_blocks, 'head_params': {}}}


GRID_HEADS = ['logreg', 'lightgbm', 'catboost']


def _grid_blocks(head: str, stage1: bool) -> list:
    """Фичи стейджа под голову: logreg с tfidf, бустинги плотные; стейдж-1 + флаги"""
    flags = [TEXT_FLAGS] if stage1 else []
    if head == 'logreg':
        return flags + [TFIDF, QWEN_L2, _img('dinov3', 'l2')]
    return flags + [QWEN_STD, _img('dinov3')]


# блоки чемпиона (один источник, переиспользуем в его конфиге и в logreg-гриде)
CHAMP_BLOCKS = [TFIDF, QWEN_L2, _img('dinov3', 'l2'), _img('pe', 'l2')]

# шаг 4: грид logreg-головы на блоках чемпиона. class_weight=None может расширить фронтир -
# balanced тянет редкий класс, но портит калибровку, а метрика = порог на вероятностях
LR_GRID_C = [0.1, 0.3, 1.0, 3.0]
LR_GRID_CW = ['balanced', None]


def _lr_grid() -> list:
    """Грид logreg (C x class_weight) на блоках чемпиона; ЧЕМПИОН (шаг 4) = lrgrid__c1.0__none
    (None > balanced на фронтире + лучше калибровка: audit AER_hi 0.078->0.062; ячейка (1.0,balanced)=прежний)"""
    return [_cfg(f'lrgrid__c{c}__{cw or "none"}', CHAMP_BLOCKS, 'logreg', {'C': c, 'class_weight': cw})
            for c in LR_GRID_C for cw in LR_GRID_CW]


EXPERIMENTS = {cfg['name']: cfg for cfg in [
    # --- исторические (логрег) ---
    _cfg('tfidf+dinov2s+logreg', [TFIDF, _img('dinov2')]),
    _cfg('tfidf+dinov3s+logreg', [TFIDF, _img('dinov3')]),                       # якорь raw f1 0.917
    _cfg('dinov2s+logreg', [_img('dinov2')]),
    _cfg('dinov3s+logreg', [_img('dinov3')]),
    _cfg('tfidf+logreg', [TFIDF]),
    _cfg('qwen3emb+logreg', [QWEN_STD]),
    _cfg('qwen3emb+dinov3+logreg', [QWEN_STD, _img('dinov3')]),                  # embfusion
    _cfg('tfidf+qwen3+logreg', [TFIDF, QWEN_L2]),                                # hybrid A
    _cfg('tfidf+qwen3+dinov3+logreg', [TFIDF, QWEN_L2, _img('dinov3', 'l2')]),   # прежний чемпион (raw f1 0.928)

    # --- сравнение голов на ОДНИХ плотных emb-фичах (qwen3+dinov3, standard) ---
    # logreg тут = qwen3emb+dinov3+logreg (0.911) выше, остальные головы рядом для bake-off
    _cfg('qwen3emb+dinov3+catboost', [QWEN_STD, _img('dinov3')], 'catboost'),
    _cfg('qwen3emb+dinov3+rf', [QWEN_STD, _img('dinov3')], 'random_forest'),
    _cfg('qwen3emb+dinov3+lgbm', [QWEN_STD, _img('dinov3')], 'lightgbm'),

    # --- catboost на solo-эмбеддингах (vs соответствующие solo-logreg) ---
    _cfg('dinov3s+catboost', [_img('dinov3')], 'catboost'),
    _cfg('qwen3emb+catboost', [QWEN_STD], 'catboost'),

    # --- бонус: lightgbm держит sparse -> на полном гибриде против logreg 0.928 ---
    _cfg('tfidf+qwen3+dinov3+lgbm', [TFIDF, QWEN_L2, _img('dinov3', 'l2')], 'lightgbm'),

    # --- бонус: text_flags в ПЛОСКОЙ модели (помогают ли лексические флаги чемпиону) ---
    _cfg('tfidf+qwen3+dinov3+flags+logreg', [TFIDF, QWEN_L2, _img('dinov3', 'l2'), TEXT_FLAGS]),

    # --- каскад (рычаг из эксп.9): стейдж-1 камера/не -> стейдж-2 тип ---
    _cascade('cascade__lr__lr',
             'logreg', [TFIDF, QWEN_L2, _img('dinov3', 'l2')],
             'logreg', [TFIDF, QWEN_L2, _img('dinov3', 'l2')]),                  # sanity
    _cascade('cascade__lr+flags__lr',
             'logreg', [TFIDF, TEXT_FLAGS, QWEN_L2, _img('dinov3', 'l2')],
             'logreg', [TFIDF, QWEN_L2, _img('dinov3', 'l2')]),                  # флаги в стейдж-1
    _cascade('cascade__lr+flags__lgbm',
             'logreg', [TFIDF, TEXT_FLAGS, QWEN_L2, _img('dinov3', 'l2')],
             'lightgbm', [QWEN_STD, _img('dinov3')]),                            # стейдж-2 lgbm
    _cascade('cascade__lgbm+flags__lgbm',
             'lightgbm', [TEXT_FLAGS, QWEN_STD, _img('dinov3')],                 # lgbm без sparse: флаги+эмб
             'lightgbm', [QWEN_STD, _img('dinov3')]),
    _cascade('cascade__textflags__lgbm',
             'logreg', [TFIDF, TEXT_FLAGS],                                      # дёшево: стейдж-1 только лексика
             'lightgbm', [QWEN_STD, _img('dinov3')]),

    # --- стейдж-1 head bake-off: кто детектит «не камера» лучше. Фичи стейджа-1 ПЛОТНЫЕ
    #     (флаги+эмб, без tfidf - чтобы catboost мог); стейдж-2 фиксирован = logreg чемпион ---
    _cascade('cascade__s1logreg', 'logreg', [TEXT_FLAGS, QWEN_STD, _img('dinov3')],
             'logreg', [TFIDF, QWEN_L2, _img('dinov3', 'l2')]),
    _cascade('cascade__s1lgbm', 'lightgbm', [TEXT_FLAGS, QWEN_STD, _img('dinov3')],
             'logreg', [TFIDF, QWEN_L2, _img('dinov3', 'l2')]),
    _cascade('cascade__s1catboost', 'catboost', [TEXT_FLAGS, QWEN_STD, _img('dinov3')],
             'logreg', [TFIDF, QWEN_L2, _img('dinov3', 'l2')]),

    # --- эксп.4: ПОЛНЫЙ грид стейдж-1 x стейдж-2 головы (9 клеток), каждая голова в своих фичах
    #     (logreg+tfidf, бустинги+плотные), стейдж-1 +флаги. Покрывает все комбинации ---
    *[_cascade(f'cgrid__{a}__{b}', a, _grid_blocks(a, True), b, _grid_blocks(b, False))
      for a in GRID_HEADS for b in GRID_HEADS],

    # --- шаг 2: VLM Perception Encoder как НОВЫЙ блок рядом с dinov3 (НЕ вместо) ---
    #     pe = сырой PE-emb (1024d, l2 как dinov3 + контроль standard, PE ненормирован);
    #     pe_sim = prompt-similarity (14 косинусов, standard) - целевой сигнал под should_abstain
    # чемпион шага 2 (C=1.0 balanced); грид -> None лучше, чемпион = lrgrid__c1.0__none
    _cfg('tfidf+qwen3+dinov3+pe+logreg', CHAMP_BLOCKS),
    _cfg('tfidf+qwen3+dinov3+peSTD+logreg', [TFIDF, QWEN_L2, _img('dinov3', 'l2'), _img('pe', 'standard')]),
    _cfg('tfidf+qwen3+dinov3+pesim+logreg', [TFIDF, QWEN_L2, _img('dinov3', 'l2'), _img('pe_sim', 'standard')]),
    _cfg('tfidf+qwen3+dinov3+pe+pesim+logreg',
         [TFIDF, QWEN_L2, _img('dinov3', 'l2'), _img('pe', 'l2'), _img('pe_sim', 'standard')]),

    # --- шаг 4: грид logreg-головы (C x class_weight) на блоках чемпиона ---
    *_lr_grid(),
]}

# какие эксперименты гоняем сейчас: правишь этот список и запускаешь `python train.py`
# (без аргументов train.py берёт ровно его; --all гоняет весь каталог выше)
# шаг 4: грид logreg-головы C x class_weight на блоках чемпиона (8 ячеек, logreg, минуты)
TO_RUN = [f'lrgrid__c{c}__{cw or "none"}' for c in LR_GRID_C for cw in LR_GRID_CW]
