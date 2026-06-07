"""Оркестратор валидации - единственный модуль который читает файлы

Берёт predictions-таблицу (контракт), нужный сплит и источник истины, считает
полный отчёт. Импортирует contract/metrics/slices, но НЕ модели и НЕ tracking -
блок ничего не знает про то как предсказания получены

Истина бывает двух родов:
- final_label - метки учителя Gemini (~22k на test), меряет обобщение, это proxy
- audit_label - ручная разметка (210), меряет настоящую бизнес-метрику
AER против final_label это 'насколько похоже на Gemini', а не настоящая ошибка
"""

import pandas as pd

from validation.contract import validate_predictions
from validation.metrics import (
    brier_score,
    business_metrics,
    calibration_metrics,
    classification_metrics,
)
from validation.settings import ABSTAIN, AUDIT_SAMPLE, EDA_DATASET, SPLIT_PARQUET
from validation.slices import add_slice_columns, slice_metrics

# колонки из data_eda которые нужны для срезов и обобщённой истины
CONTEXT_COLS = ['item_id', 'user_id', 'title', 'kb_model', 'label_source', 'final_label']
SLICE_COLS = ['slice_source', 'slice_seller', 'slice_generic_title', 'slice_kb_model', 'slice_tlr']


def load_eval_frame(predictions_path, split_name: str, truth_col: str) -> pd.DataFrame:
    """Собрать таблицу для оценки: предсказания + нужный сплит + контекст + истина

    Требуем чтобы предсказания покрывали ВЕСЬ сплит. Иначе метрика посчитается по
    случайному подмножеству (модель могла предсказать только лёгкие строки) и взлетит
    """
    preds = pd.read_parquet(predictions_path)
    validate_predictions(preds)
    split = pd.read_parquet(SPLIT_PARQUET)
    split = split[split['split'] == split_name][['item_id', 'leakage_group_id']]

    missing = set(split['item_id']) - set(preds['item_id'])
    if missing:
        raise ValueError(f'предсказания не покрывают split {split_name}: нет {len(missing)} строк')

    context = pd.read_parquet(EDA_DATASET, columns=CONTEXT_COLS)
    frame = split.merge(preds, on='item_id', validate='one_to_one')
    frame = frame.merge(context, on='item_id', validate='one_to_one')
    # final_label уже пришёл из контекста, audit_label подмешиваем отдельно
    if truth_col == 'audit_label':
        audit = pd.read_parquet(AUDIT_SAMPLE, columns=['item_id', 'audit_label'])
        frame = frame.merge(audit, on='item_id', validate='one_to_one')
    elif truth_col != 'final_label':
        raise ValueError(f'неизвестный источник истины: {truth_col}')
    return frame


def _slices_report(frame: pd.DataFrame, truth_col: str) -> dict:
    frame = add_slice_columns(frame)

    def metric(sub):
        return business_metrics(sub[truth_col], sub['pred_label'])

    return {col: slice_metrics(frame, col, metric).to_dict('records') for col in SLICE_COLS}


def evaluate(predictions_path, split_name: str, truth_col: str, tau=None) -> dict:
    """Полный отчёт по предсказаниям на сплите: бизнес, классификация, калибровка, срезы"""
    frame = load_eval_frame(predictions_path, split_name, truth_col)
    # порог отказа: всё ниже tau переводим в other_unknown
    if tau is not None:
        keep = frame['confidence'].fillna(0) >= tau
        frame['pred_label'] = frame['pred_label'].where(keep, ABSTAIN)
    return {
        'n': len(frame),
        'split': split_name,
        'truth': truth_col,
        'business_kind': 'real' if truth_col == 'audit_label' else 'proxy',
        'business': business_metrics(frame[truth_col], frame['pred_label']),
        'classification': classification_metrics(frame[truth_col], frame['pred_label']),
        'calibration': calibration_metrics(frame, truth_col, 'pred_label', 'confidence'),
        'brier': brier_score(frame, truth_col),
        'slices': _slices_report(frame, truth_col),
    }
