"""Нагрузка на /predict: на каждый запрос случайное объявление (разные входы = без кэша)

Цель - критерий №2: сервис держит >=1 rps на разнообразных данных
Запуск: locust -f locustfile.py --host http://localhost:8010 (web-UI на :8089)
"""

import random
from pathlib import Path

from locust import HttpUser, between, task
from samples import SAMPLES

IMG_DIR = Path(__file__).parent / 'samples' / 'img'


class PredictUser(HttpUser):
    """Шлёт случайное объявление (заголовок + описание + фото) на /predict"""

    wait_time = between(0.1, 0.5)

    @task
    def predict(self):
        title, description, image_name = random.choice(SAMPLES)
        with open(IMG_DIR / image_name, 'rb') as photo:
            files = {'photos': (image_name, photo, 'image/jpeg')}
            data = {'title': title, 'description': description}
            self.client.post('/predict', data=data, files=files)
