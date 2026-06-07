"""Чистые функции метрик: бизнес (CAR/AER), классификация, калибровка

На вход массивы или таблицы, на выход числа и словари. Ничего не знают про модель
и про сплит, их зовёт evaluator. Бизнес-метрики и калибровка портированы из EDA
"""

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from validation.contract import prob_col
from validation.settings import ABSTAIN, AER_CAP, ALL_LABELS, MAIN_SET


def wilson_ci(k: int, n: int) -> tuple:
    """95% доверительный интервал Wilson для доли k из n"""
    if n == 0:
        return np.nan, np.nan
    z = 1.96
    p = k / n
    d = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / d
    half = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / d
    return max(0, center - half), min(1, center + half)


def business_metrics(y_true, y_pred, positive_labels=MAIN_SET) -> dict:
    """CorrectAutofillRate и AutoErrorRate с интервалами Wilson

    N - все строки, A - автозаполнено (вернули один из основных классов), C - из них
    правильных, W - ошибочных. Держим AER = W/A <= 5%, максимизируем CAR = C/N
    """
    y_true = pd.Series(y_true).astype(str).reset_index(drop=True)
    y_pred = pd.Series(y_pred).astype(str).reset_index(drop=True)
    auto = y_pred.isin(set(positive_labels))
    correct = auto & y_pred.eq(y_true)
    n, a, c = len(y_true), int(auto.sum()), int(correct.sum())
    w = a - c
    car_lo, car_hi = wilson_ci(c, n)
    aer_lo, aer_hi = wilson_ci(w, a) if a else (np.nan, np.nan)
    return {
        'N': n, 'A_auto': a, 'C_correct': c, 'W_wrong': w,
        'coverage': a / n if n else np.nan,
        'CorrectAutofillRate': c / n if n else np.nan,
        'CorrectAutofillRate_lo': car_lo, 'CorrectAutofillRate_hi': car_hi,
        'AutoErrorRate': w / a if a else np.nan,
        'AutoErrorRate_lo': aer_lo, 'AutoErrorRate_hi': aer_hi,
    }


def threshold_sweep(df, true_col, pred_col, conf_col, positive_labels=MAIN_SET, n=101) -> pd.DataFrame:
    """Бизнес-метрики на сетке порогов уверенности

    Ниже порога предсказание заменяется на other_unknown - это и есть отказ
    """
    rows = []
    for tau in np.linspace(0, 1, n):
        pred = df[pred_col].where(df[conf_col].fillna(0) >= tau, ABSTAIN)
        row = business_metrics(df[true_col], pred, positive_labels)
        row['threshold'] = float(tau)
        rows.append(row)
    return pd.DataFrame(rows)


def pick_operating_point(curve: pd.DataFrame, aer_cap=AER_CAP, use_ci=True) -> dict:
    """Порог с максимальным CAR при AER не выше порога

    use_ci берёт верхнюю границу Wilson - честный выбор на маленькой выборке,
    страхует от того что точечная оценка ошибки занижена
    """
    aer_col = 'AutoErrorRate_hi' if use_ci else 'AutoErrorRate'
    feasible = curve[curve[aer_col].fillna(1.0) <= aer_cap]
    if feasible.empty:
        return {'tau': 1.0, 'feasible': False}
    best = feasible.loc[feasible['CorrectAutofillRate'].idxmax()]
    return {
        'tau': float(best['threshold']),
        'feasible': True,
        'CorrectAutofillRate': float(best['CorrectAutofillRate']),
        'AutoErrorRate': float(best['AutoErrorRate']),
        'AutoErrorRate_hi': float(best['AutoErrorRate_hi']),
        'coverage': float(best['coverage']),
    }


def _per_class_scores(y_true, y_pred, labels) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return {label: {'precision': float(precision[i]), 'recall': float(recall[i]),
                    'f1': float(f1[i]), 'support': int(support[i])}
            for i, label in enumerate(labels)}


def classification_metrics(y_true, y_pred, labels=ALL_LABELS) -> dict:
    """Accuracy, macro-F1, метрики по классам, отдельно TLR и матрица ошибок"""
    labels = list(labels)
    y_true = pd.Series(y_true).astype(str)
    y_pred = pd.Series(y_pred).astype(str)
    per_class = _per_class_scores(y_true, y_pred, labels)
    macro = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(macro),
        'TLR_recall': per_class['TLR']['recall'],
        'TLR_f1': per_class['TLR']['f1'],
        'per_class': per_class,
        'confusion_matrix': matrix.tolist(),
        'labels': labels,
    }


def calibration_metrics(df, true_col, pred_col, conf_col, positive_labels=MAIN_SET, n_bins=10) -> dict:
    """Ошибка калибровки ECE по бинам уверенности на автозаполненных строках"""
    auto = df[df[pred_col].isin(set(positive_labels))].copy()
    if auto.empty:
        return {'ece': np.nan, 'n_auto': 0}
    auto['conf'] = auto[conf_col].fillna(0).clip(0, 1)
    auto['bin'] = pd.cut(auto['conf'], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    auto['correct'] = auto[pred_col].astype(str).eq(auto[true_col].astype(str))
    grouped = auto.groupby('bin', observed=True).agg(
        n=('conf', 'size'), mean_conf=('conf', 'mean'), acc=('correct', 'mean'))
    ece = ((grouped['acc'] - grouped['mean_conf']).abs() * grouped['n']).sum() / grouped['n'].sum()
    return {'ece': float(ece), 'n_auto': int(len(auto))}


def _onehot(truth, labels) -> np.ndarray:
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(truth), len(labels)))
    for row, label in enumerate(truth.astype(str)):
        if label in index:
            matrix[row, index[label]] = 1.0
    return matrix


def brier_score(df, true_col, labels=ALL_LABELS):
    """Multiclass Brier score, None если в таблице нет колонок вероятностей p_<class>"""
    labels = list(labels)
    cols = [prob_col(label) for label in labels]
    if not all(col in df.columns for col in cols):
        return None
    probs = df[cols].to_numpy(float)
    onehot = _onehot(df[true_col], labels)
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
