"""Считает метрики аудита по заполненному audit_sample.parquet

Запускать после того как заполнил колонку audit_label вручную
Сравнивает audit_label с АКТУАЛЬНОЙ final_label из labels_final.parquet
(audit_sample.parquet содержит snapshot final_label на момент создания, но
после decision.py метки могут поменяться - подтягиваем свежие)

Выдаёт:
- macro precision - простое среднее per-class precision (на стратифицированном sample)
- weighted precision - precision взвешенный по доле классов в labels_final
  (это и есть честная цифра precision на 144k)
- precision_by_label_source - отдельно kb / llm / human / abstain / kb_overridden
- confusion matrix
- разбор всех несовпадений
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT = PROJECT_ROOT / 'data' / 'labeling' / 'audit_sample.parquet'
FINAL = PROJECT_ROOT / 'data' / 'training' / 'labels_final.parquet'


def main():
    if not AUDIT.exists():
        sys.exit(f'нет {AUDIT} - сначала прогони audit_sample.py')
    if not FINAL.exists():
        sys.exit(f'нет {FINAL} - сначала прогони decision.py')

    audit_raw = pd.read_parquet(AUDIT)
    final = pd.read_parquet(FINAL)

    # берём ТОЛЬКО audit_label/notes/title из audit, всё остальное из текущего labels_final
    audit_keep = audit_raw[['item_id', 'title', 'audit_label', 'audit_notes']]
    final_cols = final[['item_id', 'final_label', 'label_source', 'confidence']]
    audit = audit_keep.merge(final_cols, on='item_id', how='inner')

    labeled = audit[audit['audit_label'].notna()].copy()
    if len(labeled) == 0:
        sys.exit('колонка audit_label пустая - заполни её в Jupyter и попробуй снова')

    labeled['correct'] = labeled['audit_label'] == labeled['final_label']
    print(f'audit-sample размечено: {len(labeled)} / {len(audit)}')
    print(f'raw accuracy: {labeled["correct"].mean():.1%}\n')

    # per-class precision на audit
    classes = sorted(labeled['final_label'].unique())
    per_class = {}
    for cls in classes:
        sub = labeled[labeled['final_label'] == cls]
        if len(sub):
            per_class[cls] = (sub['correct'].mean(), len(sub))

    print('per-class precision (на audit sample):')
    for cls, (p, n) in per_class.items():
        print(f'  {cls:30s}  {p:.1%}  ({n} в audit)')

    # macro = простое среднее по классам
    macro = sum(p for p, _ in per_class.values()) / len(per_class)
    print(f'\nMACRO precision: {macro:.1%}')

    # weighted = взвешенное по доле классов в labels_final
    final_dist = final['final_label'].value_counts(normalize=True).to_dict()
    weighted = sum(per_class[cls][0] * final_dist.get(cls, 0) for cls in per_class)
    coverage = sum(final_dist.get(cls, 0) for cls in per_class)
    print(f'WEIGHTED precision (на 144k): {weighted:.1%} (coverage по классам: {coverage:.1%})')

    # precision by label_source
    print('\nprecision by label_source:')
    for src in sorted(labeled['label_source'].unique()):
        sub = labeled[labeled['label_source'] == src]
        if len(sub):
            print(f'  {src:10s}  {sub["correct"].mean():.1%}  ({len(sub)} в audit, share в labels_final: {(final["label_source"] == src).mean():.1%})')

    # confusion matrix
    print('\nconfusion matrix (audit_label rows, final_label cols):')
    print(pd.crosstab(labeled['audit_label'], labeled['final_label'], margins=True))

    # ошибки
    wrong = labeled[~labeled['correct']]
    if len(wrong):
        print(f'\nошибок ({len(wrong)}):')
        for _, r in wrong.iterrows():
            print(f'  [{r["label_source"]}] {r["final_label"]} -> {r["audit_label"]}  conf={r["confidence"]:.2f}')
            print(f'    {r["title"][:70]}')


if __name__ == '__main__':
    main()
