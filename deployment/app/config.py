"""Константы и пути прод-сервиса film_camera_body_type"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / 'training' / 'final_model'
_BUNDLE_NAME = 'tfidf_qwen3_dinov3_pe_logreg_reg1_0_unweighted_final_pipeline.joblib'
BUNDLE_PATH = Path(os.environ.get('BUNDLE_PATH', str(MODEL_DIR / _BUNDLE_NAME)))

ATTRIBUTE = 'film_camera_body_type'

# 5 основных классов идут в автозаполнение, other_unknown это отказ
MAIN_SET = ('SLR', 'TLR', 'rangefinder_viewfinder', 'compact_point_and_shoot', 'instant')
ABSTAIN = 'other_unknown'
ALL_LABELS = MAIN_SET + (ABSTAIN,)

# размерности emb-блоков, порядок hstack: tfidf -> qwen3 -> dinov3 -> pe
QWEN_DIM = 1024
DINO_DIM = 384
PE_DIM = 1024
TOTAL_DIM = 52432

# решения и причины ответа
DECISION_AUTO = 'auto_fill'
DECISION_ABSTAIN = 'abstain'
REASON_ARGMAX = 'model_argmax'
REASON_ABSTAIN = 'model_abstained'
REASON_NO_PHOTO = 'photo_required'
REASON_URL_FAIL = 'input_unavailable'

# энкодеры (паритет с training/notebooks/extract_*.ipynb)
QWEN_MODEL = 'Qwen/Qwen3-Embedding-0.6B'
DINO_MODEL = 'facebook/dinov3-vits16-pretrain-lvd1689m'
PE_MODEL = 'hf-hub:timm/PE-Core-L-14-336'
MAX_SEQ_LEN = 512

# человекочитаемые названия классов и причин для веб-страницы
LABELS_RU = {
    'SLR': 'Зеркальная (SLR)',
    'TLR': 'Двухобъективная (TLR)',
    'rangefinder_viewfinder': 'Дальномерная',
    'compact_point_and_shoot': 'Компактная «мыльница»',
    'instant': 'Моментальная (Polaroid/Instax)',
    'other_unknown': 'Не определено',
}
REASONS_RU = {
    'model_argmax': 'определено моделью',
    'model_abstained': 'модель не уверена',
    'photo_required': 'нужно фото',
    'input_unavailable': 'не удалось открыть объявление',
}
