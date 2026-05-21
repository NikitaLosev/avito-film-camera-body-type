"""Все пути проекта в одном месте, чтобы не дублировать PROJECT_ROOT по скриптам"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# сырые данные с бакета
RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
RAW_CSV = RAW_DIR / 'items_project_aaa.csv'
IMG_DIR = RAW_DIR / 'image'

# артефакты разметки
LABELING_DIR = PROJECT_ROOT / 'data' / 'labeling'
ITEMS = LABELING_DIR / 'items.parquet'
GOLD = LABELING_DIR / 'gold.parquet'
GOLD_DEV = LABELING_DIR / 'gold_dev.parquet'
GOLD_HOLDOUT = LABELING_DIR / 'gold_holdout.parquet'
KB_YAML = LABELING_DIR / 'kb.yaml'
KB_LABELS = LABELING_DIR / 'kb_labels.parquet'
LLM_LABELS = LABELING_DIR / 'llm_labels.parquet'
LLM_KB_CHECK = LABELING_DIR / 'llm_kb_check.parquet'
AUDIT_SAMPLE = LABELING_DIR / 'audit_sample.parquet'

# финальный артефакт для обучения
TRAINING_DIR = PROJECT_ROOT / 'data' / 'training'
LABELS_FINAL = TRAINING_DIR / 'labels_final.parquet'

# схема, правила, манифест
TAXONOMY = PROJECT_ROOT / 'data' / 'taxonomy.yaml'
DECISION_RULES = PROJECT_ROOT / 'data' / 'decision_rules.md'
MANIFEST = PROJECT_ROOT / 'data' / 'manifest.yaml'

# промпты
PROMPTS_DIR = PROJECT_ROOT / 'data-pipeline' / 'prompts'
PROMPT_ACTIVE = PROMPTS_DIR / 'v3_with_vision.md'

# env-файл
ENV_FILE = PROJECT_ROOT / '.env'


def image_path(image_id):
    """Путь к фото по его id: шардим по последним 3 цифрам чтобы не было миллиона файлов в одной папке"""
    return IMG_DIR / str(image_id % 1000) / f'{image_id}.jpg'
