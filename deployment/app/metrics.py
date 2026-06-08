"""Prometheus-метрики сервиса (отдаются на /metrics)"""

from prometheus_client import Counter, Gauge, Histogram

# prometheus_client сам добавит суффикс _total -> метрика predict_requests_total
predict_requests = Counter('predict_requests', 'Ответы /predict по решению и классу', ['decision', 'value'])
predict_latency = Histogram('predict_latency_seconds', 'Время инференса по стадиям', ['stage'])
# уверенность модели - прокси дрейфа: падает, когда входы уходят от обучающих
predict_confidence = Histogram(
    'predict_confidence', 'Уверенность модели, когда она отработала',
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
model_loaded = Gauge('model_loaded', 'Модель и энкодеры загружены (1/0)')


def record(response):
    """Учесть ответ /predict в метриках"""
    predict_requests.labels(decision=response.decision, value=response.value).inc()
    if response.probabilities:                       # модель реально отработала, а не отказ по входу
        predict_confidence.observe(response.confidence)
