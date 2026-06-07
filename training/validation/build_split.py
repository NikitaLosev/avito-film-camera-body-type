"""Построить и заморозить leakage-free сплит раз и навсегда

Читает data_eda + кэш хэшей фото + audit + gold, строит группы утечки,
раскладывает по train/val/test, проверяет непересечение и пишет два артефакта:
split_v1.parquet (item_id -> split) и split_v1_manifest.yaml (для отчёта)
"""

import hashlib
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation.leakage import assert_no_overlap, build_leakage_groups, build_text_key
from validation.settings import (
    ALL_LABELS,
    AUDIT_SAMPLE,
    EDA_DATASET,
    GOLD,
    LEAKAGE_KEYS,
    PHOTO_META,
    SEED,
    SPLIT_MANIFEST,
    SPLIT_PARQUET,
    SPLIT_RATIOS,
)
from validation.splits import assign_splits, attach_group_features, build_split_table

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'data-pipeline'))
from lib.io import atomic_write_parquet


def data_sha256(data):
    """Стабильный хэш данных (item_id + final_label) для версии сплита"""
    payload = data[['item_id', 'final_label']].sort_values('item_id')
    digest = pd.util.hash_pandas_object(payload, index=False).values.tobytes()
    return hashlib.sha256(digest).hexdigest()


def write_yaml(data, dst):
    """Атомарная запись yaml через .tmp + os.replace как в lib/io"""
    tmp = dst.with_suffix(dst.suffix + '.tmp')
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    os.replace(tmp, dst)


def load_inputs():
    """Собрать таблицу с ключами утечки и флагами audit/gold"""
    cols = ['item_id', 'user_id', 'image_id', 'title', 'description', 'final_label']
    data = pd.read_parquet(EDA_DATASET, columns=cols)
    photo = pd.read_parquet(PHOTO_META, columns=['image_id', 'sha256', 'dhash'])
    data = data.merge(photo, on='image_id', how='left')
    data['text_hash'] = [build_text_key(t, d) for t, d in zip(data['title'], data['description'])]
    audit_ids = set(pd.read_parquet(AUDIT_SAMPLE, columns=['item_id'])['item_id'])
    gold_ids = set(pd.read_parquet(GOLD, columns=['item_id'])['item_id'])
    data['is_audit'] = data['item_id'].isin(audit_ids)
    data['is_gold'] = data['item_id'].isin(gold_ids)
    return data


def split_stats(merged, name):
    """Статистика одного сплита для манифеста"""
    sub = merged[merged['split'] == name]
    dist = sub['final_label'].value_counts().reindex(ALL_LABELS, fill_value=0)
    return {
        'n_rows': int(len(sub)),
        'row_share': round(len(sub) / len(merged), 4),
        'n_leakage_groups': int(sub['leakage_group_id'].nunique()),
        'class_counts': {label: int(dist[label]) for label in ALL_LABELS},
        'tlr_rows': int(dist['TLR']),
    }


def build_manifest(data, split_df, overlaps):
    """Манифест сплита: версия данных, распределение, проверки утечек"""
    merged = split_df.merge(data[['item_id', 'final_label', 'is_audit', 'is_gold']], on='item_id')
    test = merged[merged['split'] == 'test']
    return {
        'split_id': 'split_v1',
        'seed': SEED,
        'ratios': SPLIT_RATIOS,
        'leakage_keys': list(LEAKAGE_KEYS),
        'data_sha256': data_sha256(data),
        'n_rows_total': int(len(merged)),
        'n_leakage_groups': int(merged['leakage_group_id'].nunique()),
        'per_split': {name: split_stats(merged, name) for name in SPLIT_RATIOS},
        'leakage_overlaps': overlaps,
        'audit_in_test': int(test['is_audit'].sum()),
        'gold_in_test': int(test['is_gold'].sum()),
    }


def print_report(manifest):
    """Напечатать итог раскладки в консоль"""
    for name, info in manifest['per_split'].items():
        head = f'{info["n_rows"]:>7,} строк {info["row_share"]:.1%}'
        print(f'  {name:5s} {head}  TLR={info["tlr_rows"]}')
    print(f'пересечений между сплитами: {sum(manifest["leakage_overlaps"].values())}')
    print(f'audit в test: {manifest["audit_in_test"]} / 210')
    print(f'gold в test: {manifest["gold_in_test"]} / 471')


def main():
    data = load_inputs()
    data['leakage_group_id'] = build_leakage_groups(data, LEAKAGE_KEYS)
    feats = attach_group_features(data)
    split_df = build_split_table(data, assign_splits(feats))

    # к строкам сплита подмешиваем сами ключи и проверяем что 0 пересечений
    keyed = split_df.merge(data[['item_id', *LEAKAGE_KEYS]], on='item_id')
    overlaps = assert_no_overlap(keyed, LEAKAGE_KEYS)

    manifest = build_manifest(data, split_df, overlaps)
    atomic_write_parquet(split_df[['item_id', 'leakage_group_id', 'split']], SPLIT_PARQUET)
    write_yaml(manifest, SPLIT_MANIFEST)
    print_report(manifest)


if __name__ == '__main__':
    main()
