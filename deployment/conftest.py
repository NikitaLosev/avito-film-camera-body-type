"""Фикстуры тестов прод-сервиса (conftest в корне deployment кладёт его в sys.path)"""

import pytest
from app.model import Predictor


@pytest.fixture(scope='session')
def predictor():
    """Загруженный один раз Predictor для тестов контракта"""
    return Predictor()
