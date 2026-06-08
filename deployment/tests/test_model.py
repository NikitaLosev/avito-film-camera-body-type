"""Тесты контракта инференса без энкодеров: размерность 52432, decision-маппинг, отказ"""

import numpy as np
from app.config import ABSTAIN, ALL_LABELS, DINO_DIM, PE_DIM, QWEN_DIM, REASON_NO_PHOTO, TOTAL_DIM
from app.model import abstain, build_vector, decide


def _fake_embeddings():
    """Случайные эмбеддинги нужных размерностей вместо энкодеров"""
    rng = np.random.default_rng(0)
    return rng.normal(size=QWEN_DIM), rng.normal(size=DINO_DIM), rng.normal(size=PE_DIM)


def test_bundle_contract(predictor):
    clf = predictor.bundle['classifier']
    assert clf.n_features_in_ == TOTAL_DIM
    assert predictor.tau == 0.0
    assert set(map(str, clf.classes_)) == set(ALL_LABELS)


def test_build_vector_dim(predictor):
    qwen, dino, pe = _fake_embeddings()
    x = build_vector(predictor.bundle, 'зенит ttl', qwen, dino, pe)
    assert x.shape == (1, TOTAL_DIM)


def test_predict_structure(predictor):
    qwen, dino, pe = _fake_embeddings()
    out = predictor.predict_from_embeddings('зенит ttl плёночный фотоаппарат', qwen, dino, pe)
    assert out.value in ALL_LABELS
    assert (out.decision == 'abstain') == (out.value == ABSTAIN)
    assert 0.0 <= out.confidence <= 1.0
    assert set(out.probabilities) == set(ALL_LABELS)
    assert abs(sum(out.probabilities.values()) - 1.0) < 1e-6


def test_decide_mapping():
    assert decide('SLR', 0.9, 0.0) == ('SLR', 'auto_fill', 'model_argmax')
    value, decision, _ = decide(ABSTAIN, 0.9, 0.0)
    assert (value, decision) == (ABSTAIN, 'abstain')


def test_no_photo_abstains():
    out = abstain(REASON_NO_PHOTO)
    assert out.value == ABSTAIN
    assert out.decision == 'abstain'
    assert out.reason == REASON_NO_PHOTO
    assert out.confidence == 0.0
