"""Тесты метрик: /metrics отдаёт счётчики, /predict их инкрементит (без загрузки моделей)"""

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_metrics_endpoint():
    client.get('/health')
    response = client.get('/metrics')
    assert response.status_code == 200
    assert 'predict_requests_total' in response.text
    assert 'http_requests_total' in response.text


def test_predict_increments_counter():
    client.post('/predict', data={'title': 'Зенит'})  # без фото -> abstain
    metrics = client.get('/metrics').text
    assert 'predict_requests_total{decision="abstain"' in metrics
