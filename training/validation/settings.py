"""Константы блока валидации: классы, доли сплита, ключи утечки, пути к артефактам"""

import sys
from pathlib import Path

VALIDATION_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = VALIDATION_ROOT.parents[1]

# пути к данным берём из общего lib пайплайна, чтобы не дублировать их тут
sys.path.insert(0, str(PROJECT_ROOT / 'data-pipeline'))
from lib.paths import AUDIT_SAMPLE, EDA_DATASET, GOLD

# кэш хэшей фото из EDA: sha256 + dhash на все 144k, пересканировать фото не нужно
PHOTO_META = PROJECT_ROOT / 'eda' / '_cache' / 'photo_meta.parquet'

# замороженный сплит и его манифест
SPLIT_DIR = VALIDATION_ROOT / 'splits'
SPLIT_PARQUET = SPLIT_DIR / 'split_v1.parquet'
SPLIT_MANIFEST = SPLIT_DIR / 'split_v1_manifest.yaml'

SEED = 42
SPLIT_RATIOS = {'train': 0.70, 'val': 0.15, 'test': 0.15}

# 5 основных классов идут в автозаполнение, other_unknown это отказ
MAIN_SET = ('SLR', 'TLR', 'rangefinder_viewfinder', 'compact_point_and_shoot', 'instant')
ALL_LABELS = MAIN_SET + ('other_unknown',)
ABSTAIN = 'other_unknown'

# бизнес-ограничение из solution-design: ошибка автозаполнения не выше 5%
AER_CAP = 0.05
CAR_TARGET = 0.70

# минимум TLR-строк который гарантируем в val, иначе редкий класс туда не попадёт
MIN_TLR_VAL = 30

LEAKAGE_KEYS = ('user_id', 'sha256', 'dhash', 'text_hash')

# текст становится ключом утечки только если описание содержательное, иначе generic
# заголовки от тысяч продавцов слиплись бы в одну мега-группу через цепочку user_id <-> text
MIN_DESC_CHARS = 40
