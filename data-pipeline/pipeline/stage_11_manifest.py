"""Собирает data/manifest.yaml - sha256 всех артефактов, версии промпта и модели

Для воспроизводимости фиксируем что получилось на выходе: каждый parquet и
yaml записываем с его sha256, числом строк, родительскими артефактами.
Плюс версию промпта, id модели и id кеша Gemini который использовался
"""

import hashlib
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.io import read_state
from lib.paths import (AUDIT_SAMPLE, EDA_DATASET, GOLD, GOLD_DEV, GOLD_HOLDOUT,
                       ITEMS, KB_LABELS, KB_YAML, LABELING_DIR, LABELS_FINAL,
                       MANIFEST, PROJECT_ROOT, PROMPT_ACTIVE, RAW_CSV,
                       TAXONOMY)


def sha256_of(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_of_parquet(path: Path) -> int | None:
    if not path.exists():
        return None
    return len(pd.read_parquet(path))


def artifact(path: Path, parent=None, extra=None) -> dict:
    rel = path.relative_to(PROJECT_ROOT)
    info = {'path': str(rel), 'sha256': sha256_of(path), 'exists': path.exists()}
    if path.suffix == '.parquet':
        info['rows'] = rows_of_parquet(path)
    if parent:
        info['parent'] = parent if isinstance(parent, list) else [parent]
    if extra:
        info.update(extra)
    return info


def main():
    state = read_state(LABELING_DIR / 'batch_state.json')

    m = {
        'project': 'film-camera-body-type',
        'generated_at': datetime.utcnow().isoformat(),
        'model': 'gemini-3.1-flash-lite',
        'prompt_version': PROMPT_ACTIVE.stem,
        'prompt_sha256': sha256_of(PROMPT_ACTIVE),
        'cache_id': state.get('_cache', {}).get('name'),
        'batch_jobs': {k: v.get('batch_name') for k, v in state.items() if k != '_cache'},
        'artifacts': {
            'raw_csv': artifact(RAW_CSV),
            'items_parquet': artifact(ITEMS, parent='raw_csv'),
            'gold_parquet': artifact(GOLD, parent='items_parquet'),
            'taxonomy_yaml': artifact(TAXONOMY),
            'kb_yaml': artifact(KB_YAML, parent='gold_parquet'),
            'kb_labels': artifact(KB_LABELS, parent=['items_parquet', 'kb_yaml']),
            'gold_dev': artifact(GOLD_DEV, parent='gold_parquet'),
            'gold_holdout': artifact(GOLD_HOLDOUT, parent='gold_parquet'),
            'llm_labels': artifact(LABELING_DIR / 'llm_labels.parquet',
                                   parent=['items_parquet', 'kb_labels']),
            'llm_kb_check': artifact(LABELING_DIR / 'llm_kb_check.parquet',
                                     parent=['items_parquet', 'kb_labels']),
            'labels_final': artifact(LABELS_FINAL,
                                     parent=['gold_parquet', 'kb_labels', 'llm_labels']),
            'audit_sample': artifact(AUDIT_SAMPLE, parent='labels_final'),
            'eda_dataset': artifact(EDA_DATASET, parent=['labels_final', 'items_parquet']),
        },
    }

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        yaml.safe_dump(m, allow_unicode=True, sort_keys=False, default_flow_style=False)
    )
    print(f'manifest: {MANIFEST}')
    print('\nартефакты:')
    for name, info in m['artifacts'].items():
        if info['exists']:
            extra = f' ({info.get("rows", "?")} rows)' if 'rows' in info else ''
            print(f'  [+] {name}{extra}: {info["sha256"][:16]}...')
        else:
            print(f'  [-] {name}: нет')


if __name__ == '__main__':
    main()
