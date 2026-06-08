"""Контракт инференса: бандл -> вектор 52432 -> proba -> canonical argmax -> tau -> decision

Энкодеры (Qwen3/DINOv3/PE-Core) считаются отдельно (encoders.py, шаг 2) и приходят сюда
готовыми векторами. Здесь только сборка фич и решение - зеркало training/src/features.py
"""

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import normalize

from app.config import (
    ABSTAIN,
    ALL_LABELS,
    BUNDLE_PATH,
    DECISION_ABSTAIN,
    DECISION_AUTO,
    REASON_ABSTAIN,
    REASON_ARGMAX,
)
from app.schemas import PredictResponse


def build_vector(bundle, text, qwen_vec, dino_vec, pe_vec):
    """Разреженный вектор 52432: tfidf | L2(qwen) | L2(dino) | L2(pe)"""
    blocks = [bundle['vectorizer'].transform([text])]
    for vec in (qwen_vec, dino_vec, pe_vec):
        normed = normalize(np.asarray(vec, dtype='float64').reshape(1, -1))
        blocks.append(csr_matrix(normed))
    return hstack(blocks).tocsr()


def predict_proba_row(bundle, x):
    """proba головы -> (pred_label, confidence, probabilities) в каноничном порядке ALL_LABELS"""
    clf = bundle['classifier']
    proba = clf.predict_proba(x)[0]
    by_class = {str(cls): float(proba[i]) for i, cls in enumerate(clf.classes_)}
    probabilities = {label: by_class[label] for label in ALL_LABELS}
    probs = np.array([probabilities[label] for label in ALL_LABELS])
    best = int(probs.argmax())
    return ALL_LABELS[best], float(probs[best]), probabilities


def decide(pred_label, confidence, tau):
    """Решение и причина: ниже tau или other_unknown -> abstain, иначе auto_fill"""
    if confidence < tau or pred_label == ABSTAIN:
        return ABSTAIN, DECISION_ABSTAIN, REASON_ABSTAIN
    return pred_label, DECISION_AUTO, REASON_ARGMAX


def abstain(reason):
    """Готовый ответ-отказ: нет фото, недоступна ссылка и т.п."""
    return PredictResponse(value=ABSTAIN, decision=DECISION_ABSTAIN, confidence=0.0, reason=reason)


def build_text(title, description):
    """Текст для tfidf и Qwen3: (title + ' ' + description) без краёв"""
    return (str(title) + ' ' + str(description)).strip()


class Predictor:
    """Грузит бандл один раз и считает ответ; энкодеры инжектятся (torch сюда не тянем)"""

    def __init__(self, bundle_path=BUNDLE_PATH, encoders=None):
        self.bundle = joblib.load(bundle_path)
        self.tau = float(self.bundle['tau'])
        self.encoders = encoders

    def predict(self, title, description, image):
        """Заголовок/описание/фото -> PredictResponse (энкодеры считают эмбеддинги)"""
        text = build_text(title, description)
        qwen_vec, dino_vec, pe_vec = self.encoders.embed(text, image)
        return self.predict_from_embeddings(text, qwen_vec, dino_vec, pe_vec)

    def predict_from_embeddings(self, text, qwen_vec, dino_vec, pe_vec):
        """Текст + три готовых эмбеддинга -> PredictResponse"""
        x = build_vector(self.bundle, text, qwen_vec, dino_vec, pe_vec)
        pred_label, confidence, probabilities = predict_proba_row(self.bundle, x)
        value, decision, reason = decide(pred_label, confidence, self.tau)
        return PredictResponse(
            value=value,
            decision=decision,
            confidence=confidence,
            reason=reason,
            probabilities=probabilities,
        )
