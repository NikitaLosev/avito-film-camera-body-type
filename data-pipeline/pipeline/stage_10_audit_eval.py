"""Метрики precision по заполненному audit_sample.parquet

Запускать после того как заполнил audit_label в Jupyter руками. Сравниваем
с актуальной final_label из labels_final.parquet (а не со snapshot из
audit_sample), потому что после decision.py итоговая метка могла поменяться -
подтягиваем свежую

Выдаёт:
- macro precision - простое среднее точности по классам на стратифицированной выборке
- weighted precision - точность взвешенная по доле классов в labels_final,
  это и есть честная цифра precision на полных 144k для отчёта
- точность по источнику (kb / llm / human / abstain / kb_overridden)
- confusion matrix
- разбор всех ошибок
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.paths import AUDIT_SAMPLE, LABELS_FINAL


def main():
    if not AUDIT_SAMPLE.exists():
        sys.exit(f'нет {AUDIT_SAMPLE} - сначала запусти stage_09_audit_sample')
    if not LABELS_FINAL.exists():
        sys.exit(f'нет {LABELS_FINAL} - сначала запусти stage_08_merge_decision')

    audit_raw = pd.read_parquet(AUDIT_SAMPLE)
    final = pd.read_parquet(LABELS_FINAL)

    # из audit берём только то что заполнил руками: title (для разбора ошибок)
    # и audit_label/notes. Всё остальное - из текущего labels_final
    audit_keep = audit_raw[['item_id', 'title', 'audit_label', 'audit_notes']]
    final_cols = final[['item_id', 'final_label', 'label_source', 'confidence']]
    audit = audit_keep.merge(final_cols, on='item_id', how='inner')

    labeled = audit[audit['audit_label'].notna()].copy()
    if len(labeled) == 0:
        sys.exit('колонка audit_label пустая - заполни в Jupyter и запусти ещё раз')

    labeled['correct'] = labeled['audit_label'] == labeled['final_label']
    print(f'audit-sample размечено: {len(labeled)} / {len(audit)}')
    print(f'raw accuracy: {labeled["correct"].mean():.1%}\n')

    classes = sorted(labeled['final_label'].unique())
    per_class = {}
    for cls in classes:
        sub = labeled[labeled['final_label'] == cls]
        if len(sub):
            per_class[cls] = (sub['correct'].mean(), len(sub))

    print('per-class precision (на audit-сэмпле):')
    for cls, (p, n) in per_class.items():
        print(f'  {cls:30s}  {p:.1%}  ({n} в аудите)')

    macro = sum(p for p, _ in per_class.values()) / len(per_class)
    print(f'\nMACRO precision: {macro:.1%}')

    final_dist = final['final_label'].value_counts(normalize=True).to_dict()
    weighted = sum(per_class[cls][0] * final_dist.get(cls, 0) for cls in per_class)
    coverage = sum(final_dist.get(cls, 0) for cls in per_class)
    print(f'WEIGHTED precision (на 144k): {weighted:.1%} (coverage: {coverage:.1%})')

    print('\nprecision по источнику:')
    for src in sorted(labeled['label_source'].unique()):
        sub = labeled[labeled['label_source'] == src]
        if len(sub):
            share = (final['label_source'] == src).mean()
            print(f'  {src:15s}  {sub["correct"].mean():.1%}  '
                  f'({len(sub)} в аудите, доля в labels_final: {share:.1%})')

    print('\nconfusion matrix (audit_label - строки, final_label - колонки):')
    print(pd.crosstab(labeled['audit_label'], labeled['final_label'], margins=True))

    wrong = labeled[~labeled['correct']]
    if len(wrong):
        print(f'\nошибок ({len(wrong)}):')
        for _, r in wrong.iterrows():
            print(f'  [{r["label_source"]}] {r["final_label"]} -> {r["audit_label"]}  '
                  f'conf={r["confidence"]:.2f}')
            print(f'    {r["title"][:70]}')


if __name__ == '__main__':
    main()
