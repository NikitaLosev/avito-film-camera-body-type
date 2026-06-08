"""Тест парсера avito (без сети, без torch): заголовок из h1, фото с CDN, нет фото -> ошибка"""

import pytest
from app.avito import _parse

_PAGE = (
    '<html><body>'
    '<h1>Instax mini 11</h1>'
    '<img src="https://10.img.avito.st/image/1/abc.jpg?cqp=xyz" alt="Instax mini 11, Коммунарка"/>'
    '</body></html>'
)
_NO_PHOTO = '<html><body><h1>Зенит TTL</h1></body></html>'


def test_parse_title_and_photo():
    title, image_url = _parse(_PAGE)
    assert title == 'Instax mini 11'
    assert image_url == 'https://10.img.avito.st/image/1/abc.jpg?cqp=xyz'


def test_parse_no_photo_raises():
    with pytest.raises(ValueError):
        _parse(_NO_PHOTO)
