"""Тесты страницы и API (без загрузки моделей): форма, abstain, рендер результата"""

import io

from app.main import app
from app.schemas import PredictResponse
from fastapi.testclient import TestClient
from PIL import Image


class _FakePredictor:
    """Заглушка вместо энкодеров/модели - отдаёт фиксированный ответ"""

    def predict(self, title, description, image):
        return PredictResponse(
            value='SLR', decision='auto_fill', confidence=0.9, reason='model_argmax',
            probabilities={'SLR': 0.9, 'TLR': 0.0, 'rangefinder_viewfinder': 0.04,
                           'compact_point_and_shoot': 0.03, 'instant': 0.02, 'other_unknown': 0.01},
        )


client = TestClient(app)
app.state.predictor = _FakePredictor()


def _tiny_png():
    buffer = io.BytesIO()
    Image.new('RGB', (4, 4), 'white').save(buffer, format='png')
    return buffer.getvalue()


def test_index_has_form():
    response = client.get('/')
    assert response.status_code == 200
    html = response.text.lower()
    assert '</form>' in html
    assert 'method="post"' in html
    assert 'multipart/form-data' in html


def test_predict_no_photo_abstains():
    response = client.post('/predict', data={'title': 'Зенит'})
    assert response.status_code == 200
    body = response.json()
    assert body['decision'] == 'abstain'
    assert body['reason'] == 'photo_required'


def test_page_renders_result():
    response = client.post('/', data={'title': 'Зенит TTL'},
                           files={'photos': ('p.png', _tiny_png(), 'image/png')})
    assert response.status_code == 200
    assert 'SLR' in response.text
