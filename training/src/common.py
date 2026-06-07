"""Общие хелперы пайплайна обучения: хэш файла, контракт предсказаний, метрики рана, вывод

Чистые функции без I/O и mlflow - их зовут features и pipeline. Поведение портировано
один-в-один из прежних baseline-скриптов, чтобы числа экспериментов не поехали
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from validation.settings import ALL_LABELS

# крепкие дефолты, зафиксированы при настоящем бейзлайне
TFIDF_PARAMS = dict(analyzer='char_wb', ngram_range=(3, 5), min_df=5, max_features=50000, sublinear_tf=True)
# lbfgs: правильный солвер для L2-multinomial, сходится за сотни итераций и детерминирован
LOGREG_PARAMS = dict(solver='lbfgs', C=1.0, class_weight='balanced', max_iter=1000, random_state=42)


def file_sha256(path) -> str:
    """sha256 файла по содержимому, для версии данных в логе"""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def contract_frame(item_ids, by_class: dict) -> pd.DataFrame:
    """Контракт из словаря {label: вектор вероятностей}, в каноничном порядке ALL_LABELS

    argmax берём по каноничному порядку -> pred_label совпадает с argmax контракта даже при
    ничьих (RF даёт ничьи долей деревьев, порядок classes_ у sklearn алфавитный, не как ALL_LABELS)
    """
    canonical = np.column_stack([by_class[label] for label in ALL_LABELS])
    labels = np.array(ALL_LABELS)
    frame = pd.DataFrame({
        'item_id': item_ids,
        'pred_label': labels[canonical.argmax(axis=1)],
        'confidence': canonical.max(axis=1),
    })
    for i, label in enumerate(ALL_LABELS):
        frame[f'p_{label}'] = canonical[:, i]
    return frame


def predictions_frame(clf, x, item_ids) -> pd.DataFrame:
    """Контракт по предсказаниям головы: раскладываем proba головы по каноничному порядку"""
    proba = clf.predict_proba(x)
    return contract_frame(item_ids, {str(cls): proba[:, i] for i, cls in enumerate(clf.classes_)})


def run_metrics(report_val, report_test, report_internal) -> dict:
    """Числа для лога: val (proxy), test на audit (human), test internal (обобщение)"""
    return {
        'val_CAR': report_val['business']['CorrectAutofillRate'],
        'val_AER': report_val['business']['AutoErrorRate'],
        'val_macro_f1': report_val['classification']['macro_f1'],
        'test_audit_CAR': report_test['business']['CorrectAutofillRate'],
        'test_audit_AER': report_test['business']['AutoErrorRate'],
        'test_audit_AER_hi': report_test['business']['AutoErrorRate_hi'],
        'test_internal_macro_f1': report_internal['classification']['macro_f1'],
        'test_internal_TLR_f1': report_internal['classification']['TLR_f1'],
    }


def raw_metrics(report_raw_val) -> dict:
    """Raw (без отказа, tau:0) метрики val - отдельно от post-tau, чтобы видеть качество до границы

    Аддитивно к run_metrics: показывает, что text-only и пр. плохи именно из-за отказа, а не модели
    """
    return {
        'raw_val_macro_f1': report_raw_val['classification']['macro_f1'],
        'raw_val_acc': report_raw_val['classification']['accuracy'],
        'raw_val_CAR': report_raw_val['business']['CorrectAutofillRate'],
        'raw_val_AER': report_raw_val['business']['AutoErrorRate'],
    }


def print_summary(name: str, metrics: dict, tau: float) -> None:
    """Короткая выжимка рана в консоль"""
    print(f'\n[{name}]  tau = {tau:.3f}')
    print(f"val  (proxy):    CAR={metrics['val_CAR']:.3f}  AER={metrics['val_AER']:.3f}  "
          f"macroF1={metrics['val_macro_f1']:.3f}")
    print(f"test (audit):    CAR={metrics['test_audit_CAR']:.3f}  AER={metrics['test_audit_AER']:.3f}  "
          f"AER_hi={metrics['test_audit_AER_hi']:.3f}")
    print(f"test (internal): macroF1={metrics['test_internal_macro_f1']:.3f}  "
          f"TLR_f1={metrics['test_internal_TLR_f1']:.3f}")
