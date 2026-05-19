"""Генерит data/manifest.yaml - data lineage с sha256 всех артефактов

По правилам Avito: для каждого parquet/yaml фиксируем sha256, rows, parent,
дату, прометить версию промпта и модель
"""

import hashlib
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent / 'llm'))
from progress import read_state

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DST = PROJECT_ROOT / 'data' / 'manifest.yaml'


def sha256_of(path: Path) -> str:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_of_parquet(path: Path) -> int:
    if not path.exists():
        return None
    return len(pd.read_parquet(path))


def artifact(path: Path, parent=None, extra=None):
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
    state = read_state(PROJECT_ROOT / 'data' / 'labeling' / 'batch_state.json')

    m = {
        'project': 'film-camera-body-type',
        'generated_at': datetime.utcnow().isoformat(),
        'model': 'gemini-3.1-flash-lite',
        'prompt_version': 'v3_with_vision',
        'prompt_sha256': sha256_of(PROJECT_ROOT / 'data-pipeline' / 'scripts' / 'llm' / 'prompts' / 'v3_with_vision.md'),
        'cache_id': state.get('_cache', {}).get('name'),
        'batch_jobs': {k: v.get('batch_name') for k, v in state.items() if k != '_cache'},
        'artifacts': {
            'raw_csv': artifact(PROJECT_ROOT / 'data' / 'raw' / 'items_project_aaa.csv'),
            'items_parquet': artifact(PROJECT_ROOT / 'data' / 'labeling' / 'items.parquet', parent='raw_csv'),
            'gold_parquet': artifact(PROJECT_ROOT / 'data' / 'labeling' / 'gold.parquet', parent='items_parquet'),
            'taxonomy_yaml': artifact(PROJECT_ROOT / 'data' / 'taxonomy.yaml'),
            'kb_yaml': artifact(PROJECT_ROOT / 'data' / 'labeling' / 'kb.yaml', parent='gold_parquet'),
            'kb_labels': artifact(PROJECT_ROOT / 'data' / 'labeling' / 'kb_labels.parquet', parent=['items_parquet', 'kb_yaml']),
            'gold_dev': artifact(PROJECT_ROOT / 'data' / 'labeling' / 'gold_dev.parquet', parent='gold_parquet'),
            'gold_holdout': artifact(PROJECT_ROOT / 'data' / 'labeling' / 'gold_holdout.parquet', parent='gold_parquet'),
            'image_uris': artifact(PROJECT_ROOT / 'data' / 'labeling' / 'image_uris.parquet'),
            'llm_labels': artifact(PROJECT_ROOT / 'data' / 'labeling' / 'llm_labels.parquet', parent=['items_parquet', 'kb_labels']),
            'labels_final': artifact(PROJECT_ROOT / 'data' / 'training' / 'labels_final.parquet', parent=['gold_parquet', 'kb_labels', 'llm_labels']),
            'audit_sample': artifact(PROJECT_ROOT / 'data' / 'labeling' / 'audit_sample.parquet', parent='labels_final'),
        },
    }

    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(yaml.safe_dump(m, allow_unicode=True, sort_keys=False, default_flow_style=False))
    print(f'manifest: {DST}')
    print('\nартефакты:')
    for name, info in m['artifacts'].items():
        if info['exists']:
            extra = f' ({info.get("rows", "?")} rows)' if 'rows' in info else ''
            print(f'  ✓ {name}{extra}: {info["sha256"][:16]}...')
        else:
            print(f'  ✗ {name}: НЕТ')


if __name__ == '__main__':
    main()
